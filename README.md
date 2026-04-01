# SentrySearch

Semantic search over video footage. Type what you're looking for, get a trimmed clip back.

[OpenClaw Skill](https://clawhub.ai/ssrajadh/natural-language-video-search)

[<video src="https://github.com/ssrajadh/sentrysearch/raw/main/docs/demo.mp4" controls width="100%"></video>](https://github.com/user-attachments/assets/baf98fad-080b-48e1-97f5-a2db2cbd53f5)

## How it works

SentrySearch splits your videos into overlapping chunks, embeds each chunk as video using either Google's Gemini Embedding API or a local Qwen3-VL model, and stores the vectors in a local ChromaDB database. When you search, your text query is embedded into the same vector space and matched against the stored video embeddings. The top match is automatically trimmed from the original file and saved as a clip.

## Getting Started

1. Install [uv](https://docs.astral.sh/uv/) (if you don't have it):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # macOS/Linux
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows
```

2. Clone and install:

```bash
git clone https://github.com/ssrajadh/sentrysearch.git
cd sentrysearch
uv tool install .
```

3. Set up your API key (or [use a local model instead](#local-backend-no-api-key-needed)):

```bash
sentrysearch init
```

This prompts for your Gemini API key, writes it to `.env`, and validates it with a test embedding.

4. Index your footage:

```bash
sentrysearch index /path/to/footage
```

5. Search:

```bash
sentrysearch search "red truck running a stop sign"
```

ffmpeg is required for video chunking and trimming. If you don't have it system-wide, the bundled `imageio-ffmpeg` is used automatically.

> **Manual setup:** If you prefer not to use `sentrysearch init`, you can copy `.env.example` to `.env` and add your key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) manually.

## Usage

### Init

```bash
$ sentrysearch init
Enter your Gemini API key (get one at https://aistudio.google.com/apikey): ****
Validating API key...
Setup complete. You're ready to go — run `sentrysearch index <directory>` to get started.
```

If a key is already configured, you'll be asked whether to overwrite it.

> **Tip:** Set a spending limit at [aistudio.google.com/billing](https://aistudio.google.com/billing) to prevent accidental overspending.

### Index footage

```bash
$ sentrysearch index /path/to/video/footage
Indexing file 1/3: front_2024-01-15_14-30.mp4 [chunk 1/4]
Indexing file 1/3: front_2024-01-15_14-30.mp4 [chunk 2/4]
...
Indexed 12 new chunks from 3 files. Total: 12 chunks from 3 files.
```

Options:

- `--chunk-duration 30` — seconds per chunk
- `--overlap 5` — overlap between chunks
- `--no-preprocess` — skip downscaling/frame rate reduction (send raw chunks)
- `--target-resolution 480` — target height in pixels for preprocessing
- `--target-fps 5` — target frame rate for preprocessing
- `--no-skip-still` — embed all chunks, even ones with no visual change
- `--backend local` — use a local model instead of Gemini ([details below](#local-backend-no-api-key-needed))

### Search

```bash
$ sentrysearch search "red truck running a stop sign"
  #1 [0.87] front_2024-01-15_14-30.mp4 @ 02:15-02:45
  #2 [0.74] left_2024-01-15_14-30.mp4 @ 02:10-02:40
  #3 [0.61] front_2024-01-20_09-15.mp4 @ 00:30-01:00

Saved clip: ./match_front_2024-01-15_14-30_02m15s-02m45s.mp4
```

If the best result's similarity score is below the confidence threshold (default 0.41), you'll be prompted before trimming:

```
No confident match found (best score: 0.28). Show results anyway? [y/N]:
```

With `--no-trim`, low-confidence results are shown with a note instead of a prompt.

Options: `--results N`, `--output-dir DIR`, `--no-trim` to skip auto-trimming, `--threshold 0.5` to adjust the confidence cutoff, `--save-top N` to save the top N clips instead of just the best match. Backend and model are auto-detected from the index — pass `--backend` or `--model` only to override.

### Local Backend (no API key needed)

Index and search using a local Qwen3-VL-Embedding model instead of the Gemini API. Free, private, and runs entirely on your machine. For the best search quality, use the Gemini backend — the local 8B model is a solid alternative when you need offline/private search, and the 2B model is a fallback when hardware can't support 8B.

The model is **auto-detected from your hardware** — qwen8b for NVIDIA GPUs and Macs with 24 GB+ RAM, qwen2b for smaller Macs and CPU-only systems. You can override with `--model qwen2b` or `--model qwen8b`. Pick an install based on your hardware:

| Hardware | Install command | Auto-detected model | Notes |
|---|---|---|---|
| **Apple Silicon, 24 GB+ RAM** | `uv tool install ".[local]"` | qwen8b | Full float16 via MPS |
| **Apple Silicon, 16 GB RAM** | `uv tool install ".[local]"` | qwen2b | 8B won't fit; 2B uses ~6 GB |
| **Apple Silicon, 8 GB RAM** | `uv tool install ".[local]"` | qwen2b | Tight — may swap under load; Gemini API recommended instead |
| **NVIDIA, 18 GB+ VRAM** | `uv tool install ".[local]"` | qwen8b | Full bf16 precision |
| **NVIDIA, 8–16 GB VRAM** | `uv tool install ".[local-quantized]"` | qwen8b | 4-bit quantization (~6–8 GB) |

> **Won't work well:** Intel Macs and machines without a dedicated GPU. These fall back to CPU with float32 — too slow and memory-hungry for practical use. Use the **Gemini API backend** (the default) instead.

> **Not sure?** On Mac, use `".[local]"`. On NVIDIA, use `".[local-quantized]"` — 4-bit quantization works on the widest range of NVIDIA hardware with minimal quality loss. (bitsandbytes requires CUDA and does not work on Mac/MPS.)

**Mac prerequisite:** Install system FFmpeg (the local model's video processor requires it — the Gemini backend uses a bundled ffmpeg instead):

```bash
brew install ffmpeg
```

Index with `--backend local` and search — no extra flags needed:

```bash
sentrysearch index /path/to/footage --backend local
sentrysearch search "car running a red light"
```

The search command auto-detects the backend and model from whatever you indexed with. You can also use `--model` as a shorthand — it implies `--backend local`:

```bash
sentrysearch index /path/to/footage --model qwen2b   # same as --backend local --model qwen2b
sentrysearch search "car running a red light"          # auto-detects local/qwen2b from index
```

Options:
- `--model qwen2b` — smaller model, lower quality but only ~6 GB memory (also accepts full HuggingFace IDs)
- `--quantize` / `--no-quantize` — force 4-bit quantization on or off (default: auto-detect based on whether bitsandbytes is installed)

Notes:
- First run downloads the model (~16 GB for 8B, ~4 GB for 2B).
- Embeddings from different backends and models are **not compatible**. Each backend/model combination gets its own isolated index, so they can't accidentally mix. If you search with a model that has no indexed data, you'll be told which model was actually used.
- Speed varies by GPU core count — base M-series chips are slower than Pro/Max but produce identical results.

### Why the local model is fast

The local backend stays fast and memory-efficient through a few techniques that compound:

- **Preprocessing shrinks chunks before they hit the model.** Each 30s chunk is downscaled to 480p at 5fps via ffmpeg before embedding. A ~19 MB dashcam chunk becomes ~1 MB — a 95% reduction in pixels the model has to process. Model inference time scales with pixel count, not video duration, so this is the single biggest speedup.
- **Low frame sampling.** The video processor sends at most 32 frames per chunk to the model (`fps=1.0`, `max_frames=32`). A 30-second chunk produces ~30 frames — not hundreds.
- **MRL dimension truncation.** Qwen3-VL-Embedding supports [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147). Only the first 768 dimensions of each embedding are kept and L2-normalized, reducing storage and distance computation in ChromaDB.
- **Auto-quantization.** On NVIDIA GPUs with limited VRAM, the 8B model is automatically loaded in 4-bit (bitsandbytes) — dropping from ~18 GB to ~6-8 GB with minimal quality loss. A 4090 (24 GB) runs the full bf16 model with headroom to spare.
- **Still-frame skipping.** Chunks with no meaningful visual change (e.g. a parked car) are detected by comparing JPEG file sizes across sampled frames and skipped entirely — saving a full forward pass per chunk.

With all of this, expect ~2-5s per chunk on an A100 and ~3-8s on a T4. On a 4090, the 8B model in bf16 should be in the low single digits per chunk.

### Tesla Metadata Overlay

Burn speed, location, and time onto trimmed clips:

```bash
sentrysearch search "car cutting me off" --overlay
```

This extracts telemetry embedded in Tesla dashcam files (speed, GPS) and renders a HUD overlay. The overlay shows:

- **Top center:** speed and MPH label on a light gray card
- **Below card:** date and time (12-hour with AM/PM)
- **Top left:** city and road name (via reverse geocoding)

![tesla overlay](docs/tesla-overlay.png)

Requirements:

- Tesla firmware 2025.44.25 or later, HW3+
- SEI metadata is only present in driving footage (not parked/Sentry Mode)
- Reverse geocoding uses [OpenStreetMap's Nominatim API](https://nominatim.openstreetmap.org/) via geopy (optional)

Install with Tesla overlay support:

```bash
uv tool install ".[tesla]"
```

Without geopy, the overlay still works but omits the city/road name.

Source: [teslamotors/dashcam](https://github.com/teslamotors/dashcam)

### Managing the index

```bash
# Show index info (files marked [missing] no longer exist on disk)
sentrysearch stats

# Remove specific files by path substring
sentrysearch remove path/to/footage

# Wipe the entire index
sentrysearch reset
```

### Verbose mode

Add `--verbose` to either command for debug info (embedding dimensions, API response times, similarity scores).

## How is this possible?

Both Gemini Embedding 2 and Qwen3-VL-Embedding can natively embed video — raw video pixels are projected into the same vector space as text queries. There's no transcription, no frame captioning, no text middleman. A text query like "red truck at a stop sign" is directly comparable to a 30-second video clip at the vector level. This is what makes sub-second semantic search over hours of footage practical.

## Cost

Indexing 1 hour of footage costs ~$2.84 with Gemini's embedding API (default settings: 30s chunks, 5s overlap):

> 1 hour = 3,600 seconds of video = 3,600 frames processed by the model.
> 3,600 frames × $0.00079 = ~$2.84/hr

The Gemini API natively extracts and tokenizes exactly 1 frame per second from uploaded video, regardless of the file's actual frame rate. The preprocessing step (which downscales chunks to 480p at 5fps via ffmpeg) is a local/bandwidth optimization — it keeps payload sizes small so API requests are fast and don't timeout — but does not change the number of frames the API processes.

Two built-in optimizations help reduce costs in different ways:

- **Preprocessing** (on by default) — chunks are downscaled to 480p at 5fps before uploading. Since the API processes at 1fps regardless, this only reduces upload size and transfer time, not the number of frames billed. It primarily improves speed and prevents request timeouts.
- **Still-frame skipping** (on by default) — chunks with no meaningful visual change (e.g. a parked car) are skipped entirely. This saves real API calls and directly reduces cost. The savings depend on your footage — Sentry Mode recordings with hours of idle time benefit the most, while action-packed driving footage may have nothing to skip.

Search queries are negligible (text embedding only).

Tuning options:

- `--chunk-duration` / `--overlap` — longer chunks with less overlap = fewer API calls = lower cost
- `--no-skip-still` — embed every chunk even if nothing is happening
- `--target-resolution` / `--target-fps` — adjust preprocessing quality
- `--no-preprocess` — send raw chunks to the API

## Limitations & Future Work

- **Still-frame detection is heuristic** — it uses JPEG file size comparison across sampled frames. It may occasionally skip chunks with subtle motion or embed chunks that are truly static. Disable with `--no-skip-still` if you need every chunk indexed.
- **Search quality depends on chunk boundaries** — if an event spans two chunks, the overlapping window helps but isn't perfect. Smarter chunking (e.g. scene detection) could improve this.
- **Gemini Embedding 2 is in preview** — API behavior and pricing may change.

## Compatibility

This works with `.mp4` and `.mov` footage, not just Tesla Sentry Mode. The directory scanner recursively finds both file types regardless of folder structure.

## Requirements

- Python 3.11+
- `ffmpeg` on PATH, or use bundled ffmpeg via `imageio-ffmpeg` (installed by default)
- **Gemini backend:** Gemini API key ([get one free](https://aistudio.google.com/apikey))
- **Local backend:**
  - GPU with CUDA or Apple Metal (see [hardware table](#local-backend-no-api-key-needed) for VRAM/RAM requirements)
  - **macOS:** `brew install ffmpeg` (required by the video decoder)
  - **Linux/Windows:** no extra system dependencies
