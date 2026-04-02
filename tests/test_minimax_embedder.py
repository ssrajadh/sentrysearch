"""Tests for MiniMax response parsing (no API calls)."""

import pytest

from sentrysearch.minimax_embedder import (
    _assistant_text_from_chat_response,
    _embedding_from_response,
    _fallback_caption_policy_chunk,
    _resolve_vision_chat_model,
)


def test_embedding_from_native_vectors():
    out = _embedding_from_response(
        {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "vectors": [[0.1, 0.2, 0.3]],
        }
    )
    assert out == [0.1, 0.2, 0.3]


def test_embedding_from_openai_style():
    out = _embedding_from_response(
        {
            "data": [{"embedding": [1.0, 2.0], "index": 0}],
            "model": "x",
        }
    )
    assert out == [1.0, 2.0]


def test_embedding_from_top_level_embedding():
    out = _embedding_from_response({"embedding": [0.5]})
    assert out == [0.5]


def test_base_resp_error_raises():
    with pytest.raises(RuntimeError, match="status_code=1001"):
        _embedding_from_response(
            {
                "base_resp": {"status_code": 1001, "status_msg": "bad"},
            }
        )


def test_unparseable_raises():
    with pytest.raises(RuntimeError, match="Could not parse"):
        _embedding_from_response({"foo": 1})


def test_assistant_text_from_chat():
    text = _assistant_text_from_chat_response(
        {
            "choices": [
                {"message": {"role": "assistant", "content": "  A red car.  "}}
            ],
        }
    )
    assert text == "A red car."


def test_assistant_text_chat_base_resp_error():
    with pytest.raises(RuntimeError, match="chat API error"):
        _assistant_text_from_chat_response(
            {"base_resp": {"status_code": 2013, "status_msg": "bad"}}
        )


def test_resolve_vision_chat_model_rejects_hailuo():
    m, warn = _resolve_vision_chat_model("minimax-hailuo-2.3")
    assert m == "MiniMax-Text-01"
    assert warn and "Hailuo" in warn


def test_resolve_vision_chat_model_accepts_m27():
    m, warn = _resolve_vision_chat_model("MiniMax-M2.7")
    assert m == "MiniMax-M2.7"
    assert warn is None


def test_fallback_caption_policy_chunk_stable_shape():
    a = _fallback_caption_policy_chunk("/tmp/a.mp4")
    b = _fallback_caption_policy_chunk("/tmp/b.mp4")
    assert "segment_id=" in a and "segment_id=" in b
    assert a != b
