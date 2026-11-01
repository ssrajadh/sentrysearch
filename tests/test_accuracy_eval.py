"""Offline tests for benchmarks/accuracy_eval.py (issue #78).

No model downloads, live API calls, or destructive database access: the
script is loaded by path, and the embedding/search steps are monkeypatched.
The throwaway collection tests run against real ChromaDB instances under
``tmp_path`` only.
"""

import builtins
import hashlib
import importlib.util
import io
import json
import math
import os
import shlex
import subprocess
import time
from pathlib import Path

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "accuracy_eval.py"
)


@pytest.fixture(scope="module")
def ae():
    """Load benchmarks/accuracy_eval.py as a module (by path, not import)."""
    spec = importlib.util.spec_from_file_location("accuracy_eval", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result(start=0.0, end=30.0, source="benchmark_video.mp4"):
    return {
        "source_file": f"/some/dir/{source}",
        "start_time": start,
        "end_time": end,
        "similarity_score": 0.9,
    }


class TestPrecisionAtK:
    def test_precision_at_1(self, ae):
        assert ae.precision_at_k([True], 1) == 1.0
        assert ae.precision_at_k([False], 1) == 0.0

    def test_precision_at_3(self, ae):
        assert ae.precision_at_k([True, False, True], 3) == pytest.approx(2 / 3)

    def test_precision_at_5(self, ae):
        relevance = [True, True, False, False, True]
        assert ae.precision_at_k(relevance, 5) == pytest.approx(3 / 5)

    def test_fewer_than_k_results_counts_missing_as_irrelevant(self, ae):
        # Standard P@k: denominator is k, unretrieved slots are non-relevant.
        assert ae.precision_at_k([True], 5) == 0.2
        assert ae.precision_at_k([], 5) == 0.0

    def test_k_must_be_positive(self, ae):
        with pytest.raises(ValueError):
            ae.precision_at_k([True], 0)


class TestReciprocalRank:
    def test_no_relevant_results(self, ae):
        assert ae.reciprocal_rank([]) == 0.0
        assert ae.reciprocal_rank([False, False, False]) == 0.0

    def test_relevant_at_rank_1(self, ae):
        assert ae.reciprocal_rank([True, False]) == 1.0

    def test_relevant_at_rank_2(self, ae):
        assert ae.reciprocal_rank([False, True, False]) == 0.5

    def test_relevant_at_rank_3(self, ae):
        assert ae.reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)

    def test_first_relevant_wins(self, ae):
        assert ae.reciprocal_rank([False, True, True]) == 0.5


class TestJudgmentKey:
    def test_matches_issue_format(self, ae):
        assert ae.judgment_key(124.0, 151.0) == "124.0-151.0"
        assert ae.judgment_key(233.0, 260.0) == "233.0-260.0"

    def test_rounds_to_one_decimal(self, ae):
        assert ae.judgment_key(124.04, 151.04) == "124.0-151.0"
        assert ae.judgment_key(124.06, 151.06) == "124.1-151.1"

    def test_stable_under_float_noise(self, ae):
        a = ae.judgment_key(30.0000000001, 60.0000000001)
        b = ae.judgment_key(29.9999999999, 59.9999999999)
        assert a == b == "30.0-60.0"


class TestJudgmentCache:
    def test_missing_file_is_empty_cache(self, ae, tmp_path):
        assert ae.load_judgments(tmp_path / "nope.json") == {}

    def test_roundtrip(self, ae, tmp_path):
        path = tmp_path / "j.json"
        data = {"car driving on road": {"0.0-30.0": True, "25.0-55.0": False}}
        ae.save_judgments(path, data)
        assert ae.load_judgments(path) == data

    def test_save_preserves_unknown_queries(self, ae, tmp_path):
        path = tmp_path / "j.json"
        ae.save_judgments(path, {"old query": {"0.0-30.0": True}})
        judgments = ae.load_judgments(path)
        judgments["new query"] = {"1.0-31.0": False}
        ae.save_judgments(path, judgments)
        loaded = ae.load_judgments(path)
        assert loaded["old query"] == {"0.0-30.0": True}
        assert loaded["new query"] == {"1.0-31.0": False}

    def test_malformed_json_raises_and_does_not_clobber(self, ae, tmp_path):
        path = tmp_path / "j.json"
        path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ae.AccuracyEvalError):
            ae.load_judgments(path)
        assert path.read_text(encoding="utf-8") == "{not valid json"

    def test_non_dict_root_raises(self, ae, tmp_path):
        path = tmp_path / "j.json"
        path.write_text('["not", "a", "dict"]', encoding="utf-8")
        with pytest.raises(ae.AccuracyEvalError):
            ae.load_judgments(path)

    def test_non_bool_verdict_raises(self, ae, tmp_path):
        path = tmp_path / "j.json"
        path.write_text('{"q": {"0.0-30.0": "yes"}}', encoding="utf-8")
        with pytest.raises(ae.AccuracyEvalError):
            ae.load_judgments(path)

    def test_non_dict_entry_raises(self, ae, tmp_path):
        path = tmp_path / "j.json"
        path.write_text('{"q": true}', encoding="utf-8")
        with pytest.raises(ae.AccuracyEvalError):
            ae.load_judgments(path)


class TestClipIdentity:
    """B1: the judgment cache is bound to the clip it was judged against."""

    def _identity(self, ae, tmp_path, content=b"bench-bytes"):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(content)
        return clip, ae.clip_identity(clip)

    def test_identity_records_url_and_sha256(self, ae, tmp_path):
        clip = tmp_path / "c.mp4"
        clip.write_bytes(b"hello")
        identity = ae.clip_identity(clip)
        assert identity["clip_url"] == ae.BENCHMARK_CLIP_URL
        assert identity["clip_sha256"] == hashlib.sha256(b"hello").hexdigest()

    def test_matching_identity_loads(self, ae, tmp_path):
        path = tmp_path / "j.json"
        _, identity = self._identity(ae, tmp_path)
        ae.save_judgments(path, {"q": {"0.0-30.0": True}},
                          clip_identity=identity)
        loaded = ae.load_judgments(path, clip_identity=identity)
        assert loaded == {"q": {"0.0-30.0": True}}
        # metadata never leaks into the query map
        assert ae._META_KEY not in loaded

    def test_mismatched_sha256_fails_loudly(self, ae, tmp_path):
        path = tmp_path / "j.json"
        _, identity = self._identity(ae, tmp_path, b"clip-a")
        ae.save_judgments(path, {"q": {"0.0-30.0": True}},
                          clip_identity=identity)
        other = dict(identity, clip_sha256=hashlib.sha256(b"clip-b").hexdigest())
        with pytest.raises(ae.AccuracyEvalError, match="different clip") as exc:
            ae.load_judgments(path, clip_identity=other)
        message = str(exc.value)
        assert "clip_sha256" in message
        assert "--clip" in message  # actionable: how to resolve it

    def test_mismatched_url_fails_loudly(self, ae, tmp_path):
        path = tmp_path / "j.json"
        _, identity = self._identity(ae, tmp_path)
        ae.save_judgments(path, {"q": {"0.0-30.0": True}},
                          clip_identity=identity)
        other = dict(identity, clip_url="https://example.com/other.mp4")
        with pytest.raises(ae.AccuracyEvalError, match="different clip"):
            ae.load_judgments(path, clip_identity=other)

    def test_legacy_cache_with_verdicts_rejected(self, ae, tmp_path):
        path = tmp_path / "j.json"
        path.write_text('{"q": {"0.0-30.0": true}}', encoding="utf-8")
        identity = {"clip_url": ae.BENCHMARK_CLIP_URL,
                    "clip_sha256": "0" * 64}
        with pytest.raises(ae.AccuracyEvalError, match="no clip identity") as exc:
            ae.load_judgments(path, clip_identity=identity)
        message = str(exc.value)
        # actionable: shows exactly what to add to keep the judgments
        assert '"_meta"' in message
        assert identity["clip_sha256"] in message

    def test_partial_identity_rejected(self, ae, tmp_path):
        # One of the two identity fields cannot bind the cache to the clip.
        path = tmp_path / "j.json"
        path.write_text(
            json.dumps({
                ae._META_KEY: {"clip_url": ae.BENCHMARK_CLIP_URL},
                "q": {"0.0-30.0": True},
            }),
            encoding="utf-8",
        )
        identity = {
            "clip_url": ae.BENCHMARK_CLIP_URL,
            "clip_sha256": "0" * 64,
        }
        with pytest.raises(ae.AccuracyEvalError, match="no clip identity") as exc:
            ae.load_judgments(path, clip_identity=identity)
        assert "clip_sha256 missing" in str(exc.value)

    def test_legacy_empty_cache_adopted(self, ae, tmp_path):
        path = tmp_path / "j.json"
        path.write_text("{}\n", encoding="utf-8")
        _, identity = self._identity(ae, tmp_path)
        assert ae.load_judgments(path, clip_identity=identity) == {}

    def test_empty_cache_with_stale_identity_adopted(self, ae, tmp_path):
        # Nothing judged yet => nothing to protect; the stale identity is
        # simply overwritten on the next save.
        path = tmp_path / "j.json"
        _, stale = self._identity(ae, tmp_path, b"old-clip")
        ae.save_judgments(path, {}, clip_identity=stale)
        _, current = self._identity(ae, tmp_path, b"new-clip")
        assert ae.load_judgments(path, clip_identity=current) == {}

    def test_save_without_identity_preserves_existing_meta(self, ae, tmp_path):
        path = tmp_path / "j.json"
        _, identity = self._identity(ae, tmp_path)
        ae.save_judgments(path, {"q": {"0.0-30.0": True}},
                          clip_identity=identity)
        judgments = ae.load_judgments(path, clip_identity=identity)
        judgments["q2"] = {"1.0-31.0": False}
        ae.save_judgments(path, judgments)  # no identity passed this time
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw[ae._META_KEY] == identity
        assert ae.load_judgments(path, clip_identity=identity) == judgments

    def test_malformed_meta_rejected(self, ae, tmp_path):
        path = tmp_path / "j.json"
        path.write_text(
            '{"_meta": "oops", "q": {"0.0-30.0": true}}', encoding="utf-8",
        )
        with pytest.raises(ae.AccuracyEvalError, match="_meta"):
            ae.load_judgments(path, clip_identity=None)

    def test_load_without_identity_skips_verification(self, ae, tmp_path):
        path = tmp_path / "j.json"
        ae.save_judgments(path, {"q": {"0.0-30.0": True}},
                          clip_identity={"clip_url": "u", "clip_sha256": "x"})
        assert ae.load_judgments(path) == {"q": {"0.0-30.0": True}}


class TestAtomicJudgmentWrites:
    """I1: a failed write can never truncate the committed judgments."""

    def _seed(self, ae, path):
        ae.save_judgments(path, {"q": {"0.0-30.0": True}})
        return path.read_text(encoding="utf-8")

    def test_successful_save_leaves_no_temp_file(self, ae, tmp_path):
        path = tmp_path / "j.json"
        ae.save_judgments(path, {"q": {"0.0-30.0": True}})
        assert path.exists()
        assert list(tmp_path.glob("*.tmp")) == []

    def test_failed_replace_keeps_original_and_cleans_temp(self, ae, tmp_path,
                                                           monkeypatch):
        path = tmp_path / "j.json"
        before = self._seed(ae, path)

        def fail_replace(self, target):
            raise OSError("replace failed")

        monkeypatch.setattr(Path, "replace", fail_replace)
        with pytest.raises(OSError, match="replace failed"):
            ae.save_judgments(path, {"q": {"0.0-30.0": False}})
        assert path.read_text(encoding="utf-8") == before
        assert list(tmp_path.glob("*.tmp")) == []

    def test_failed_write_keeps_original_and_cleans_temp(self, ae, tmp_path,
                                                         monkeypatch):
        path = tmp_path / "j.json"
        before = self._seed(ae, path)
        real_open = builtins.open

        def failing_open(file, *args, **kwargs):
            if str(file).endswith(".tmp"):
                raise OSError("no space left on device")
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", failing_open)
        with pytest.raises(OSError, match="no space left"):
            ae.save_judgments(path, {"q": {"0.0-30.0": False}})
        assert path.read_text(encoding="utf-8") == before
        assert list(tmp_path.glob("*.tmp")) == []


class TestVerdicts:
    def test_cached_verdict_skips_prompt(self, ae, tmp_path):
        path = tmp_path / "j.json"
        judgments = {"q": {ae.judgment_key(0.0, 30.0): True}}
        ae.save_judgments(path, judgments)

        def must_not_prompt(*_a, **_kw):
            raise AssertionError("prompt called despite cached verdict")

        verdict = ae.get_verdict(
            "q", _result(), ae.load_judgments(path), path,
            input_fn=must_not_prompt,
        )
        assert verdict is True

    def test_new_clip_prompts_and_writes_through(self, ae, tmp_path):
        path = tmp_path / "j.json"
        judgments = {}
        answers = iter(["y"])
        verdict = ae.get_verdict(
            "q", _result(), judgments, path,
            input_fn=lambda _prompt: next(answers),
        )
        assert verdict is True
        assert judgments == {"q": {"0.0-30.0": True}}
        # written through to disk immediately, not only at end of run
        assert ae.load_judgments(path) == {"q": {"0.0-30.0": True}}

    def test_empty_answer_defaults_to_no(self, ae, tmp_path):
        path = tmp_path / "j.json"
        verdict = ae.get_verdict(
            "q", _result(), {}, path, input_fn=lambda _prompt: "",
        )
        assert verdict is False

    def test_reprompts_on_garbage_input(self, ae, tmp_path):
        path = tmp_path / "j.json"
        answers = iter(["maybe", "yes"])
        verdict = ae.get_verdict(
            "q", _result(), {}, path,
            input_fn=lambda _prompt: next(answers),
        )
        assert verdict is True

    def test_eof_propagates_without_storing(self, ae, tmp_path):
        path = tmp_path / "j.json"
        judgments: dict = {}

        def eof(_prompt):
            raise EOFError

        with pytest.raises(EOFError):
            ae.get_verdict("q", _result(), judgments, path, input_fn=eof)
        assert judgments == {}


class TestPromptForVerdict:
    """I2: the human is shown where — and how — to watch the clip segment."""

    def test_shows_absolute_path_window_and_ffplay_command(self, ae, tmp_path):
        clip = tmp_path / "benchmark_video.mp4"
        out = io.StringIO()
        verdict = ae.prompt_for_verdict(
            "range rover", _result(124.0, 151.0),
            input_fn=lambda _p: "y", out=out, clip_path=clip,
        )
        text = out.getvalue()
        assert verdict is True
        assert str(clip) in text  # absolute path of the clip
        assert "124.0s-151.0s" in text
        assert f"ffplay -ss 124.0 -t 27.0 {shlex.quote(str(clip))}" in text

    def test_path_with_spaces_is_quoted(self, ae, tmp_path):
        clip = tmp_path / "my bench clip.mp4"
        out = io.StringIO()
        ae.prompt_for_verdict(
            "q", _result(), input_fn=lambda _p: "n", out=out, clip_path=clip,
        )
        quoted = shlex.quote(str(clip))
        assert quoted.startswith("'")  # spaces present, so it must be quoted
        assert f"ffplay -ss 0.0 -t 30.0 {quoted}" in out.getvalue()

    def test_relative_source_is_resolved_to_absolute(self, ae):
        out = io.StringIO()
        ae.prompt_for_verdict(
            "q",
            {"source_file": "relative/bench.mp4",
             "start_time": 0.0, "end_time": 30.0},
            input_fn=lambda _p: "y", out=out,
        )
        assert str(Path("relative/bench.mp4").absolute()) in out.getvalue()

    def test_clip_path_overrides_result_source_file(self, ae, tmp_path):
        clip = tmp_path / "real clip.mp4"
        out = io.StringIO()
        ae.prompt_for_verdict(
            "q", _result(source="stored_basename.mp4"),
            input_fn=lambda _p: "y", out=out, clip_path=clip,
        )
        text = out.getvalue()
        assert str(clip) in text
        assert "stored_basename.mp4" not in text

    def test_ffplay_command_is_printed_never_executed(self, ae, monkeypatch,
                                                      tmp_path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"x")

        def forbid(*_args, **_kwargs):
            raise AssertionError("prompt must not execute commands")

        monkeypatch.setattr(subprocess, "run", forbid)
        monkeypatch.setattr(subprocess, "Popen", forbid)
        monkeypatch.setattr(subprocess, "call", forbid)
        monkeypatch.setattr(os, "system", forbid)
        out = io.StringIO()
        verdict = ae.prompt_for_verdict(
            "q", _result(), input_fn=lambda _p: "y", out=out, clip_path=clip,
        )
        assert verdict is True
        assert "ffplay" in out.getvalue()


class TestScoreQuery:
    def test_scores_via_cache(self, ae):
        judgments = {"q": {"0.0-30.0": True, "25.0-55.0": False}}
        results = [_result(0.0, 30.0), _result(25.0, 55.0), _result(90.0, 120.0)]
        assert ae.score_query(results, judgments["q"]) == [True, False, False]

    def test_unjudged_defaults_to_false(self, ae):
        assert ae.score_query([_result()], {}) == [False]


class TestAggregateMetrics:
    def test_floor_and_fine_aggregated_separately(self, ae):
        scored = [
            ("floor a", [True, True, True]),
            ("floor b", [True, True, False]),
            ("fine a", [False, True, False]),
        ]
        floor = ae.aggregate_metrics(scored[:2])
        fine = ae.aggregate_metrics(scored[2:])
        assert floor["n_queries"] == 2
        assert floor["p@1"] == 1.0
        assert floor["p@3"] == pytest.approx((1.0 + 2 / 3) / 2)
        assert floor["mrr"] == 1.0
        assert fine["p@1"] == 0.0
        assert fine["mrr"] == 0.5

    def test_empty_group_is_none(self, ae):
        assert ae.aggregate_metrics([]) is None

    def test_no_relevant_results_scores_zero(self, ae):
        agg = ae.aggregate_metrics([("q", [False, False])])
        assert agg["p@1"] == 0.0
        assert agg["mrr"] == 0.0


class TestFormatReport:
    def test_report_has_groups_and_queries(self, ae):
        scored = [
            ("car driving on road", "floor", [True, True, False]),
            ("range rover", "fine", [False, True, False]),
        ]
        report = ae.format_report("local / qwen8b", scored)
        assert "local / qwen8b" in report
        assert "| floor | 1 |" in report
        assert "| fine-grained | 1 |" in report
        assert "| overall | 2 |" in report
        assert "| range rover | fine |" in report

    def test_empty_group_shows_na(self, ae):
        report = ae.format_report("gemini", [("car driving on road", "floor", [])])
        assert "| fine-grained | 0 | n/a |" in report


# ---------------------------------------------------------------------------
# Throwaway collection lifecycle (real ChromaDB under tmp_path)
# ---------------------------------------------------------------------------


class TestCollectionLifecycle:
    def test_prepare_creates_marked_collection(self, ae, tmp_path):
        store = ae.prepare_accuracy_store(tmp_path / "db", "gemini", None)
        names = {c.name for c in store._client.list_collections()}
        assert ae.ACCURACY_COLLECTION in names
        meta = store.collection.metadata or {}
        assert meta.get(ae.EVAL_MARKER_KEY) is True
        assert "dashcam_chunks" not in names  # production name untouched

    def test_prepare_replaces_leftover_marked_collection(self, ae, tmp_path):
        db = tmp_path / "db"
        ae.prepare_accuracy_store(db, "gemini", None)  # simulate crashed run
        store = ae.prepare_accuracy_store(db, "gemini", None)  # re-run
        assert store.collection.count() == 0

    def test_prepare_refuses_unmarked_preexisting_collection(self, ae, tmp_path):
        import chromadb

        db = tmp_path / "db"
        db.mkdir(parents=True)
        client = chromadb.PersistentClient(path=str(db))
        client.create_collection(
            ae.ACCURACY_COLLECTION, metadata={"user_data": True},
        )
        with pytest.raises(ae.AccuracyEvalError, match="ownership marker"):
            ae.prepare_accuracy_store(db, "gemini", None)
        # untouched
        remaining = client.get_collection(ae.ACCURACY_COLLECTION)
        assert (remaining.metadata or {}).get("user_data") is True

    def test_cleanup_deletes_marked_collection(self, ae, tmp_path):
        ae.prepare_accuracy_store(tmp_path / "db", "gemini", None)
        assert ae.cleanup_accuracy_collection(tmp_path / "db") is True
        client = ae._chroma_client(tmp_path / "db")
        assert client.list_collections() == []

    def test_cleanup_refuses_unmarked_collection(self, ae, tmp_path):
        import chromadb

        db = tmp_path / "db"
        db.mkdir(parents=True)
        client = chromadb.PersistentClient(path=str(db))
        client.create_collection(ae.ACCURACY_COLLECTION)
        assert ae.cleanup_accuracy_collection(db) is False
        assert ae.ACCURACY_COLLECTION in {c.name for c in client.list_collections()}

    def test_cleanup_missing_collection_is_noop(self, ae, tmp_path):
        assert ae.cleanup_accuracy_collection(tmp_path / "db") is False


# ---------------------------------------------------------------------------
# run_evaluation end-to-end with fakes
# ---------------------------------------------------------------------------


class TestRunEvaluation:
    def _fake_runtime(self, ae, monkeypatch, results_by_query=None,
                      index_error=None):
        """Wire fakes so run_evaluation never loads a model or calls a network."""
        monkeypatch.setattr(ae, "get_embedder", lambda *a, **kw: object())

        def fake_index(store, clip_path, embedder, **kw):
            if index_error is not None:
                raise index_error
            store.add_chunk("fake", [1.0, 0.0], {
                "source_file": str(clip_path), "start_time": 0.0, "end_time": 30.0,
            })
            return 1

        monkeypatch.setattr(ae, "index_clip", fake_index)
        canned = results_by_query or {}

        def fake_search(query, store, n_results=5, **kw):
            return canned.get(query, [_result()])

        monkeypatch.setattr(ae, "search_footage", fake_search)

    def _run(self, ae, tmp_path, monkeypatch, **kwargs):
        clip = tmp_path / "bench.mp4"
        clip.write_bytes(b"fake-video")
        self._fake_runtime(ae, monkeypatch, **kwargs.pop("runtime", {}))
        defaults = dict(
            backend="gemini",
            clip_path=clip,
            db_path=tmp_path / "db",
            judgments_path=tmp_path / "j.json",
            input_fn=lambda _p: "y",
        )
        defaults.update(kwargs)
        return ae.run_evaluation(**defaults)

    def test_success_report_and_cleanup(self, ae, tmp_path, monkeypatch):
        answers = []

        def input_fn(_prompt):
            answers.append(_prompt)
            return "y"

        report = self._run(
            ae, tmp_path, monkeypatch, input_fn=input_fn,
            workload=[("car driving on road", "floor"), ("range rover", "fine")],
        )
        assert isinstance(report, str)
        assert "| floor |" in report and "| fine-grained |" in report
        # every retrieved window was judged exactly once, then cached
        assert len(answers) == 2
        cache = json.loads((tmp_path / "j.json").read_text())
        assert cache["range rover"] == {"0.0-30.0": True}
        # the judged clip's identity was stamped for later runs to verify
        assert cache[ae._META_KEY]["clip_url"] == ae.BENCHMARK_CLIP_URL
        assert cache[ae._META_KEY]["clip_sha256"] == ae.file_sha256(
            tmp_path / "bench.mp4",
        )
        # collection removed on success
        assert ae.cleanup_accuracy_collection(tmp_path / "db") is False

    def test_cached_verdicts_reused_on_rerun(self, ae, tmp_path, monkeypatch):
        workload = [("car driving on road", "floor")]
        self._run(ae, tmp_path, monkeypatch, workload=workload)

        def must_not_prompt(_prompt):
            raise AssertionError("second run re-prompted for a judged clip")

        report = self._run(
            ae, tmp_path, monkeypatch, workload=workload,
            input_fn=must_not_prompt,
        )
        assert "| floor | 1 | 1.00 |" in report

    def test_cleanup_on_failure(self, ae, tmp_path, monkeypatch):
        self._fake_runtime(
            ae, monkeypatch, index_error=RuntimeError("boom"),
        )
        clip = tmp_path / "bench.mp4"
        clip.write_bytes(b"fake-video")
        with pytest.raises(RuntimeError, match="boom"):
            ae.run_evaluation(
                backend="gemini",
                clip_path=clip,
                db_path=tmp_path / "db",
                judgments_path=tmp_path / "j.json",
                input_fn=lambda _p: "y",
            )
        # the throwaway collection was created, then removed despite the error
        assert ae.cleanup_accuracy_collection(tmp_path / "db") is False

    def test_cleanup_runs_even_when_reset_fails(self, ae, tmp_path, monkeypatch):
        self._fake_runtime(ae, monkeypatch)

        def failing_reset():
            raise RuntimeError("reset exploded")

        monkeypatch.setattr(ae, "reset_embedder", failing_reset)
        clip = tmp_path / "bench.mp4"
        clip.write_bytes(b"fake-video")
        with pytest.raises(RuntimeError, match="reset exploded"):
            ae.run_evaluation(
                backend="gemini", clip_path=clip, db_path=tmp_path / "db",
                judgments_path=tmp_path / "j.json",
                input_fn=lambda _p: "y",
            )
        # the collection was still dropped — a reset failure can't leak it
        assert ae._chroma_client(tmp_path / "db").list_collections() == []

    def test_reset_runs_even_when_cleanup_fails(self, ae, tmp_path, monkeypatch):
        self._fake_runtime(ae, monkeypatch, index_error=RuntimeError("boom"))

        def failing_cleanup(_db_path):
            raise RuntimeError("cleanup exploded")

        resets = []
        monkeypatch.setattr(ae, "reset_embedder", lambda: resets.append(True))
        monkeypatch.setattr(ae, "cleanup_accuracy_collection", failing_cleanup)
        clip = tmp_path / "bench.mp4"
        clip.write_bytes(b"fake-video")
        with pytest.raises(RuntimeError, match="cleanup exploded"):
            ae.run_evaluation(
                backend="gemini", clip_path=clip, db_path=tmp_path / "db",
                judgments_path=tmp_path / "j.json",
                input_fn=lambda _p: "y",
            )
        assert resets == [True]

    def test_preexisting_unmarked_collection_blocks_run(self, ae, tmp_path,
                                                        monkeypatch):
        import chromadb

        db = tmp_path / "db"
        db.mkdir(parents=True)
        client = chromadb.PersistentClient(path=str(db))
        client.create_collection(ae.ACCURACY_COLLECTION)
        clip = tmp_path / "bench.mp4"
        clip.write_bytes(b"fake-video")
        self._fake_runtime(ae, monkeypatch)
        with pytest.raises(ae.AccuracyEvalError, match="ownership marker"):
            ae.run_evaluation(
                backend="gemini", clip_path=clip, db_path=db,
                judgments_path=tmp_path / "j.json",
                input_fn=lambda _p: "y",
            )
        assert ae.ACCURACY_COLLECTION in {c.name for c in client.list_collections()}

    def test_empty_workload_skips_everything(self, ae, tmp_path, monkeypatch):
        self._fake_runtime(ae, monkeypatch)
        clip = tmp_path / "bench.mp4"
        clip.write_bytes(b"fake-video")
        report = ae.run_evaluation(
            backend="gemini", clip_path=clip, db_path=tmp_path / "db",
            judgments_path=tmp_path / "j.json", workload=[],
            input_fn=lambda _p: "y",
        )
        assert report is None
        assert not (tmp_path / "db").exists() or (
            ae.cleanup_accuracy_collection(tmp_path / "db") is False
        )

    def test_malformed_judgments_abort_before_touching_db(self, ae, tmp_path,
                                                          monkeypatch):
        clip = tmp_path / "bench.mp4"
        clip.write_bytes(b"fake-video")
        jpath = tmp_path / "j.json"
        jpath.write_text("{broken", encoding="utf-8")
        self._fake_runtime(ae, monkeypatch)
        with pytest.raises(ae.AccuracyEvalError):
            ae.run_evaluation(
                backend="gemini", clip_path=clip, db_path=tmp_path / "db",
                judgments_path=jpath, input_fn=lambda _p: "y",
            )
        assert jpath.read_text(encoding="utf-8") == "{broken"
        assert not (tmp_path / "db").exists()

    def test_fewer_than_five_results_end_to_end(self, ae, tmp_path, monkeypatch):
        runtime = {
            "results_by_query": {
                "car driving on road": [_result(0.0, 30.0), _result(25.0, 55.0)],
            },
        }
        report = self._run(
            ae, tmp_path, monkeypatch, runtime=runtime,
            workload=[("car driving on road", "floor")],
        )
        # 2/5 relevant (both judged yes), missing slots count as non-relevant
        assert "| floor | 1 | 1.00 | 0.67 | 0.40 | 1.00 |" in report

    def test_wrong_clip_identity_aborts_before_touching_db(self, ae, tmp_path,
                                                           monkeypatch):
        clip = tmp_path / "bench.mp4"
        clip.write_bytes(b"fake-video")
        jpath = tmp_path / "j.json"
        ae.save_judgments(
            jpath, {"car driving on road": {"0.0-30.0": True}},
            clip_identity={"clip_url": ae.BENCHMARK_CLIP_URL,
                           "clip_sha256": "0" * 64},
        )
        self._fake_runtime(ae, monkeypatch)
        with pytest.raises(ae.AccuracyEvalError, match="different clip"):
            ae.run_evaluation(
                backend="gemini", clip_path=clip, db_path=tmp_path / "db",
                judgments_path=jpath, input_fn=lambda _p: "y",
            )
        assert not (tmp_path / "db").exists()  # never opened a database
        # the mismatched cache is left untouched for inspection
        assert "0" * 64 in jpath.read_text(encoding="utf-8")

    def test_legacy_cache_with_verdicts_aborts_before_touching_db(
        self, ae, tmp_path, monkeypatch,
    ):
        clip = tmp_path / "bench.mp4"
        clip.write_bytes(b"fake-video")
        jpath = tmp_path / "j.json"
        legacy = '{"car driving on road": {"0.0-30.0": true}}'
        jpath.write_text(legacy, encoding="utf-8")
        self._fake_runtime(ae, monkeypatch)
        with pytest.raises(ae.AccuracyEvalError, match="no clip identity"):
            ae.run_evaluation(
                backend="gemini", clip_path=clip, db_path=tmp_path / "db",
                judgments_path=jpath, input_fn=lambda _p: "y",
            )
        assert not (tmp_path / "db").exists()
        assert jpath.read_text(encoding="utf-8") == legacy

    def test_empty_legacy_cache_adopted_and_stamped(self, ae, tmp_path,
                                                    monkeypatch):
        jpath = tmp_path / "j.json"
        jpath.write_text("{}\n", encoding="utf-8")
        report = self._run(
            ae, tmp_path, monkeypatch,
            workload=[("car driving on road", "floor")],
        )
        assert isinstance(report, str)
        cache = json.loads(jpath.read_text(encoding="utf-8"))
        assert cache["_meta"]["clip_url"] == ae.BENCHMARK_CLIP_URL
        assert cache["_meta"]["clip_sha256"] == ae.file_sha256(
            tmp_path / "bench.mp4",
        )
        assert cache["car driving on road"] == {"0.0-30.0": True}

class TestBackendModelResolution:
    """Flag resolution is delegated to the shared helpers in
    sentrysearch.embedder, so the eval and `sentrysearch index` agree on
    which backend/model a given flag pair selects (issue #78).
    """

    def test_resolves_before_the_empty_workload_gate(self, ae, monkeypatch):
        # Resolution runs before the empty-workload early return, so an empty
        # workload proves the wiring without touching disk or the network.
        seen = {}

        def fake_resolve_backend(backend=None, model=None, dashscope_model=None):
            seen["backend"] = (backend, model, dashscope_model)
            return "local"

        def fake_resolve_model(backend, model=None, dashscope_model=None,
                               **_kwargs):
            seen["model"] = (backend, model, dashscope_model)
            return "qwen2b"

        monkeypatch.setattr(ae, "resolve_backend", fake_resolve_backend)
        monkeypatch.setattr(ae, "resolve_model", fake_resolve_model)

        assert ae.run_evaluation(workload=[], err=io.StringIO()) is None
        assert seen == {
            "backend": (None, None, None),
            "model": ("local", None, None),
        }

    def test_dashscope_model_is_passed_through_resolution(
            self, ae, monkeypatch):
        seen = {}

        def fake_resolve_backend(backend=None, model=None, dashscope_model=None):
            seen["backend"] = (backend, model, dashscope_model)
            return "qwen-cloud"

        def fake_resolve_model(backend, model=None, dashscope_model=None,
                               **_kwargs):
            seen["model"] = (backend, model, dashscope_model)
            return "qc-y"

        monkeypatch.setattr(ae, "resolve_backend", fake_resolve_backend)
        monkeypatch.setattr(ae, "resolve_model", fake_resolve_model)

        assert ae.run_evaluation(
            dashscope_model="qc-y", workload=[], err=io.StringIO(),
        ) is None
        assert seen == {
            "backend": (None, None, "qc-y"),
            "model": ("qwen-cloud", None, "qc-y"),
        }

    def test_conflicting_model_flags_are_rejected(self, ae):
        # Same rule as `sentrysearch index` (issue #78 CLI parity).
        with pytest.raises(ae.AccuracyEvalError, match="not both"):
            ae.run_evaluation(
                model="qwen2b", dashscope_model="qc-model",
                workload=[], err=io.StringIO(),
            )

    def test_gemini_is_the_default_and_has_no_model(self, ae):
        assert ae.resolve_backend() == "gemini"
        assert ae.resolve_backend("gemini", None) == "gemini"
        assert ae.resolve_model("gemini", None) is None

    def test_model_implies_local(self, ae):
        assert ae.resolve_backend(None, "qwen2b") == "local"
        assert ae.resolve_model("local", "qwen2b") == "qwen2b"

    def test_local_normalizes_hf_id(self, ae):
        assert ae.resolve_model(
            "local", "Qwen/Qwen3-VL-Embedding-2B",
        ) == "qwen2b"

    def test_dashscope_model_implies_qwen_cloud(self, ae):
        assert ae.resolve_backend(None, None, "qc-model") == "qwen-cloud"

    def test_qwen_cloud_prefers_the_dashscope_flag(self, ae):
        assert ae.resolve_model(
            "qwen-cloud", "ignored", "qc-model",
        ) == "qc-model"

    def test_qwen_cloud_ignores_the_model_flag(self, ae):
        # CLI parity: generic --model never selects a DashScope model.
        from sentrysearch.qwen_cloud_embedder import (
            default_dashscope_embedding_model,
        )

        assert (ae.resolve_model("qwen-cloud", "qc-model", None)
                == default_dashscope_embedding_model())

    def test_qwen_cloud_without_any_model_flag_uses_default(self, ae):
        from sentrysearch.qwen_cloud_embedder import (
            default_dashscope_embedding_model,
        )

        assert (ae.resolve_model("qwen-cloud", None, None)
                == default_dashscope_embedding_model())

    def test_litellm_falls_back_to_its_default(self, ae):
        from sentrysearch.litellm_embedder import default_litellm_model

        assert ae.resolve_model("litellm", None) == default_litellm_model()

    def test_mlx_resolves_the_full_model_reference(self, ae):
        from sentrysearch.mlx_embedder import resolve_model_ref

        assert ae.resolve_model("mlx", None) == resolve_model_ref(None)

    def test_unknown_backend_is_rejected_by_the_cli(self, ae, capsys):
        with pytest.raises(SystemExit) as excinfo:
            ae.main(["--backend", "bogus"])
        assert excinfo.value.code == 2
        assert "invalid choice" in capsys.readouterr().err


class TestEnsureBenchmarkClip:
    def test_existing_file_is_returned_as_is(self, ae, tmp_path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"data")
        assert ae.ensure_benchmark_clip(clip) == clip

    def test_missing_explicit_clip_errors(self, ae, tmp_path):
        with pytest.raises(ae.AccuracyEvalError, match="not found"):
            ae.ensure_benchmark_clip(tmp_path / "missing.mp4")


class TestIndexClip:
    def test_stores_every_chunk_and_cleans_temp_files(self, ae, tmp_path, monkeypatch):
        import sentrysearch.store as store_mod

        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake")
        chunk_dir = tmp_path / "chunks"
        chunk_dir.mkdir()
        chunk_file = chunk_dir / "chunk_000.mp4"
        chunk_file.write_bytes(b"chunk")

        monkeypatch.setattr(ae, "chunk_video", lambda path: [{
            "chunk_path": str(chunk_file),
            "source_file": str(clip),
            "start_time": 0.0,
            "end_time": 30.0,
        }])
        monkeypatch.setattr(ae, "preprocess_chunk", lambda p: p)

        calls = []

        class FakeEmbedder:
            def embed_video_chunk(self, path, verbose=False):
                calls.append(path)
                vec = [math.sin(i * 0.1) for i in range(3)]
                norm = math.sqrt(sum(x * x for x in vec))
                return [x / norm for x in vec]

        store = store_mod.SentryStore(db_path=tmp_path / "db")
        stored = ae.index_clip(store, clip, FakeEmbedder())
        assert stored == 1
        assert calls == [str(chunk_file)]
        assert store.get_stats()["total_chunks"] == 1
        # temp chunk dir removed even on success
        assert not chunk_dir.exists()

    def test_temp_files_removed_on_embedding_failure(self, ae, tmp_path,
                                                     monkeypatch):
        import sentrysearch.store as store_mod

        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake")
        chunk_dir = tmp_path / "chunks"
        chunk_dir.mkdir()
        (chunk_dir / "chunk_000.mp4").write_bytes(b"chunk")

        monkeypatch.setattr(ae, "chunk_video", lambda path: [{
            "chunk_path": str(chunk_dir / "chunk_000.mp4"),
            "source_file": str(clip),
            "start_time": 0.0,
            "end_time": 30.0,
        }])
        monkeypatch.setattr(ae, "preprocess_chunk", lambda p: p)
        # The shared retry loop backs off between attempts; skip the waits.
        monkeypatch.setattr(time, "sleep", lambda _seconds: None)

        attempts = []

        class BoomEmbedder:
            def embed_video_chunk(self, path, verbose=False):
                attempts.append(path)
                raise RuntimeError("embed failed")

        store = store_mod.SentryStore(db_path=tmp_path / "db")
        with pytest.raises(RuntimeError, match="embed failed"):
            ae.index_clip(store, clip, BoomEmbedder())
        # Every attempt failed, so index_clip raised instead of storing a
        # partial index — and the temp chunk dir is still cleaned up.
        assert len(attempts) == 3
        assert not chunk_dir.exists()

    def test_transient_embed_failure_is_retried(self, ae, tmp_path,
                                                 monkeypatch):
        import sentrysearch.store as store_mod

        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake")
        chunk_dir = tmp_path / "chunks"
        chunk_dir.mkdir()
        (chunk_dir / "chunk_000.mp4").write_bytes(b"chunk")

        monkeypatch.setattr(ae, "chunk_video", lambda path: [{
            "chunk_path": str(chunk_dir / "chunk_000.mp4"),
            "source_file": str(clip),
            "start_time": 0.0,
            "end_time": 30.0,
        }])
        monkeypatch.setattr(ae, "preprocess_chunk", lambda p: p)
        sleeps = []
        monkeypatch.setattr(time, "sleep", sleeps.append)

        vec = [math.sin(i * 0.1) for i in range(3)]
        norm = math.sqrt(sum(x * x for x in vec))
        vec = [x / norm for x in vec]

        calls = []

        class FlakyEmbedder:
            def embed_video_chunk(self, path, verbose=False):
                calls.append(path)
                if len(calls) == 1:
                    raise RuntimeError("429 rate limited")
                return vec

        store = store_mod.SentryStore(db_path=tmp_path / "db")
        err = io.StringIO()
        stored = ae.index_clip(store, clip, FlakyEmbedder(), err=err)

        assert stored == 1
        assert len(calls) == 2
        assert sleeps == [2]
        assert "attempt 1/3" in err.getvalue()
        assert store.get_stats()["total_chunks"] == 1
        assert not chunk_dir.exists()


class TestMainEntryPoint:
    """B2: EOF during interactive judging exits cleanly, not with a traceback."""

    def test_eof_during_prompt_exits_1_with_guidance(self, ae, monkeypatch,
                                                     capsys):
        def raise_eof(**_kwargs):
            raise EOFError

        monkeypatch.setattr(ae, "run_evaluation", raise_eof)
        assert ae.main([]) == 1
        err = capsys.readouterr().err
        assert "EOF" in err
        assert "judgment prompt input ended" in err
        assert "pre-populate" in err
        assert "terminal" in err

    def test_accuracy_error_still_exits_1(self, ae, monkeypatch, capsys):
        def raise_error(**_kwargs):
            raise ae.AccuracyEvalError("bad cache")

        monkeypatch.setattr(ae, "run_evaluation", raise_error)
        assert ae.main([]) == 1
        assert "Error: bad cache" in capsys.readouterr().err

    def test_conflicting_model_flags_exit_1_without_traceback(self, ae, capsys):
        # Raised in run_evaluation before any model/backend work — the same
        # message `sentrysearch index` prints via click.UsageError.
        assert ae.main(["--model", "x", "--dashscope-model", "y"]) == 1
        err = capsys.readouterr().err
        assert "not both" in err
        assert "Traceback" not in err

    def test_clean_run_exits_0(self, ae, monkeypatch):
        monkeypatch.setattr(ae, "run_evaluation", lambda **_kwargs: None)
        assert ae.main([]) == 0

    def test_backend_config_error_exits_1_without_traceback(
        self, ae, monkeypatch, capsys,
    ):
        """A backend setup failure is a user-facing error, not a crash."""
        from sentrysearch.litellm_embedder import LiteLLMConfigError

        def raise_error(**_kwargs):
            raise LiteLLMConfigError("The litellm package is not installed.")

        monkeypatch.setattr(ae, "run_evaluation", raise_error)
        assert ae.main(["--backend", "litellm"]) == 1
        err = capsys.readouterr().err
        assert "Error: The litellm package is not installed." in err
        assert "Traceback" not in err


class TestIssue68WorkloadImport:
    """Issue #68 imports the fixed workload instead of redefining it."""

    def test_documented_import_matches_the_module(self, ae):
        from benchmarks.accuracy_eval import (
            BENCHMARK_CLIP_PATH,
            BENCHMARK_CLIP_URL,
            FLOOR_QUERIES,
            FINE_QUERIES,
            WORKLOAD,
        )

        assert WORKLOAD == ae.WORKLOAD
        assert FLOOR_QUERIES == ae.FLOOR_QUERIES
        assert FINE_QUERIES == ae.FINE_QUERIES
        assert BENCHMARK_CLIP_PATH == ae.BENCHMARK_CLIP_PATH
        assert BENCHMARK_CLIP_URL == ae.BENCHMARK_CLIP_URL
        assert WORKLOAD
