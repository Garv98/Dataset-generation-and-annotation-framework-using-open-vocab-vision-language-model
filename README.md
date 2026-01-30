# A Cross-Modal Dataset Generation and Annotation Framework Leveraging Open-Vocabulary Vision–Language Models

Generate small-to-medium image datasets from the public web for ANN/DL labs. Focuses on multi-source scraping, downloading, cleaning, deduplication, and semantic verification.

## Features
- **Multi-source scraping**: Aggregates URLs for diversity.
  - SerpAPI (Google Images)
  - Unsplash API
  - Selenium/Google Images (optional)
  - Bing fallback
  - Flickr API (optional)
- **Hybrid 3-tier verification**:
  - Tier 1: URL heuristics
  - Tier 2: EXIF/quality checks
  - Tier 3: Semantic VLM (CLIP default for speed; Qwen2-VL optional for accuracy)
- **Outputs**: Curated images in `outputs/<query>/` with metadata.json.
- **Debug**: Rejected images in `data/_rejected_debug/`.

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

If you want Tier-3 semantic verification (with Qwen2‑VL for better accuracy):

```powershell
pip install transformers torch torchvision accelerate qwen-vl-utils python-dotenv
```

Note: on most Windows machines this will run on CPU unless you have CUDA set up.

### 3) Create `.env`

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

### Main pipeline (CLIP-based, fast)

```powershell
cd scripts
python pipeline.py --query "YOUR_QUERY" --num 50 --min_size 200 --task auto
```

- **Query** (e.g., `person riding bicycle`, `industrial robot arm`, `red sports car side view`)
- **Target count** (final images to keep)
- **Minimum image size** (basic quality filter)
- **Task type** (auto/detection/classification/segmentation/captioning)

### Accurate Variant (Qwen2-VL)

```powershell
cd scripts
python pipeline_qwen.py
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
        pipeline.py
        pipeline_qwen.py        # main multi-source + hybrid verification pipeline
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
