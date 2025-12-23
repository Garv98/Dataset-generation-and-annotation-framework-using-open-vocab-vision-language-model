# ANN-DL — AMSDG-HV (Adaptive Multi‑Source Dataset Generation + Hybrid Verification)

Generate small-to-medium sized image datasets from the public web for ANN/DL labs.

This repo focuses on the **data side**: multi-source scraping, downloading, cleaning, deduplication, and (optional) semantic verification.

## Features

- **Multi-source scraping**: aggregates URLs from multiple providers for better diversity.
    - SerpAPI (Google Images)
    - Unsplash API
    - Google Images via Selenium (optional)
    - Bing HTML fallback
    - Flickr API (optional)
- **Hybrid 3-tier filtering/verification** (in `scripts/pipeline.py`):
    - Tier 1: URL/keyword heuristics (fast prefilter)
    - Tier 2: EXIF + simple noise heuristics (rejects corrupted/likely low-quality)
    - Tier 3: VLM semantic verification (currently **Qwen2‑VL‑2B‑Instruct**)
- **Dataset outputs**: writes curated images to `outputs/<query_slug>/` and saves metadata.
- **Debugability**: rejected images are moved to `data/_rejected_debug/` for inspection.

## Requirements

- Windows 10/11
- Python 3.9+ recommended
- (Optional) Chrome installed if using Selenium
- (Optional) API keys for SerpAPI / Unsplash / Flickr

## Setup (Windows)

### 1) Create & activate a virtual environment

```powershell
python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

If you want Tier-3 semantic verification with Qwen2‑VL:

```powershell
pip install transformers torch torchvision accelerate qwen-vl-utils python-dotenv
```

Note: on most Windows machines this will run on CPU unless you have CUDA set up.

### 3) Create `.env` (do not commit)

Create a `.env` file in the project root:

```ini
# Recommended
SERPAPI_KEY=your_serpapi_key
UNSPLASH_KEY=your_unsplash_access_key

# Optional
FLICKR_KEY=your_flickr_key

# 1 = enable Selenium Google Images scraping, 0 = disable
USE_SELENIUM=1
```

## Usage

### Main pipeline (interactive)

```powershell
cd scripts
python pipeline.py
```

You’ll be prompted for:

- **Query** (e.g., `person riding bicycle`, `industrial robot arm`, `red sports car side view`)
- **Target count** (final images to keep)
- **Minimum image size** (basic quality filter)
- **Task type** (auto/detection/classification/segmentation/captioning)

### Quick check: is Qwen2‑VL working?

```powershell
cd scripts
python test_qwen.py
```

### Diagnose scraping/download issues

```powershell
cd scripts
python diagnose_pipeline.py
```

## Outputs

Each run produces:

- `outputs/<query_slug>/`: final curated images
- `outputs/<query_slug>/metadata.json`: run metadata (sources, counts, etc.)
- `data/raw/`: temporary download/cache directory
- `data/_rejected_debug/`: rejected images (useful for tuning filters)

## Project layout

```
ANN-DL/
    README.md
    requirements.txt
    .env                      # local secrets (DO NOT COMMIT)
    data/
        raw/                    # temporary downloads
        _rejected_debug/        # rejected images for inspection
    outputs/                  # final datasets per query
    scripts/
        pipeline.py             # main multi-source + hybrid verification pipeline
        web_scraper.py          # standalone Bing-only scraper/downloader (older/simple)
        image_cleaner.py        # standalone cleaner/deduper (perceptual hashing)
        diagnose_pipeline.py    # stage-by-stage scraping/download diagnostic
        test_qwen.py            # sanity test: load Qwen2-VL and run a VQA prompt
    RESEARCH_NOTES.md
```

## Troubleshooting

- **Bing/Google blocking (CAPTCHA / empty results)**: try fewer requests, wait, change query wording, or rely on API sources (SerpAPI/Unsplash).
- **Selenium issues**: set `USE_SELENIUM=0` in `.env` to disable Selenium scraping.
- **Slow semantic verification**: Tier-3 (Qwen2‑VL) is the most expensive step; reduce target count or run with a GPU.
- **AWQ on Windows**: `autoawq` typically depends on `triton` (Linux-only). On Windows, use the non-AWQ model path (this repo uses `Qwen/Qwen2-VL-2B-Instruct`).

## Safety & ethics

- This project downloads publicly accessible images; you are responsible for complying with website terms, privacy rules, and dataset licensing.
- Do not commit secrets. Add `.env` to `.gitignore`.

## Credits

- Qwen2‑VL: Hugging Face model `Qwen/Qwen2-VL-2B-Instruct`
- Selenium + `webdriver-manager` for browser automation
- SerpAPI / Unsplash / Flickr for API-based image discovery
