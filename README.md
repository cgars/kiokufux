# KiokuFux

KiokuFux MVP 1 is a local-first CLI prototype for indexing private photo archives and searching them by visual content and basic metadata. It does not modify original image files.

## What MVP 1 does

- Recursively scans `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, and `.webp` files.
- Generates stable photo IDs from SHA-256 file hashes.
- Extracts basic metadata with Pillow, including dimensions, file timestamps, MIME type, EXIF date, and GPS when available.
- Stores a resumable SQLite catalog in `.kiokufux/catalog.sqlite`.
- Generates JPEG thumbnails in `.kiokufux/thumbnails/` with EXIF orientation applied.
- Generates local embeddings in `.kiokufux/embeddings/`.
- Runs text-to-image semantic search using cosine similarity.
- Exports versioned `.kiokufux.json` sidecars next to photos.
- Logs scan errors to `.kiokufux/logs/kiokufux.log` and records unreadable images without stopping the scan.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For a stronger CLIP backend, optionally install OpenCLIP dependencies:

```bash
pip install -e '.[clip]'
```

If OpenCLIP is unavailable, KiokuFux falls back to a dependency-light local embedding backend so the MVP remains runnable offline.

## CLI usage

```bash
kiokufux -v init PATH
kiokufux scan PATH
kiokufux thumbnails PATH
kiokufux embed PATH
kiokufux search PATH "query text"
kiokufux search PATH "query text" --summary
kiokufux -v search PATH "query text" --summary
kiokufux tag PATH PHOTO_ID "family party"
kiokufux auto-tag PATH
kiokufux tag-proposals PATH [PHOTO_ID]
kiokufux accept-tag PATH PHOTO_ID TAG
kiokufux reject-tag PATH PHOTO_ID TAG
kiokufux tags PATH [PHOTO_ID]
kiokufux untag PATH PHOTO_ID "family party"
kiokufux export-sidecars PATH
```

## Example workflow

```bash
kiokufux init ./photos
kiokufux scan ./photos
kiokufux thumbnails ./photos
kiokufux embed ./photos
kiokufux search ./photos "red car in front of a house"
kiokufux search ./photos "red car in front of a house" --summary
kiokufux tag ./photos PHOTO_ID_FROM_SEARCH "family party"
kiokufux auto-tag ./photos
kiokufux tag-proposals ./photos PHOTO_ID_FROM_SEARCH
kiokufux accept-tag ./photos PHOTO_ID_FROM_SEARCH dog
kiokufux tags ./photos PHOTO_ID_FROM_SEARCH
kiokufux export-sidecars ./photos
```

The workspace is created at `./photos/.kiokufux/`:

```text
.kiokufux/
  catalog.sqlite
  thumbnails/
  embeddings/
  indexes/
  logs/
```

## Configuration

KiokuFux writes an AktenFuchs-style TOML configuration file at `.kiokufux/config.toml` during `kiokufux init`. CLI flags override configuration values for a single run. Example:

```toml
[thumbnails]
max_size = 512

[embeddings]
backend = "auto"
openclip_model = "ViT-B-32"
openclip_pretrained = "laion2b_s34b_b79k"

[search]
top_k = 10
min_raw_score = 0.20
min_robust_z = 1.0

[logging]
verbose = 0
```

Use the config file for stable project defaults, and command flags such as `--top-k`, `--embedding-backend`, `--openclip-model`, or `-v` for one-off overrides.

### Embedding backend configuration

By default, `kiokufux embed` and `kiokufux search` try OpenCLIP first and fall back to the lightweight local backend if OpenCLIP is unavailable. You can force a backend or choose a specific OpenCLIP model/weights pair:

```bash
kiokufux embed ./photos --embedding-backend openclip --openclip-model ViT-L-14 --openclip-pretrained datacomp_xl_s13b_b90k
kiokufux search ./photos "church" --embedding-backend openclip --openclip-model ViT-L-14 --openclip-pretrained datacomp_xl_s13b_b90k
kiokufux embed ./photos --embedding-backend simple
```

The same values can also be supplied with `KIOKUFUX_OPENCLIP_MODEL` and `KIOKUFUX_OPENCLIP_PRETRAINED`. Use the same backend/model options for search that you used when generating embeddings.

## Logging and privacy notices

KiokuFux writes command logs to `.kiokufux/logs/kiokufux.log`. Use `-v` to mirror informational logs to stderr, or `-vv` for debug logs. The verbose flag may be placed before or after the subcommand:

```bash
kiokufux -v scan ./photos
kiokufux scan ./photos -v
kiokufux search ./photos "church" --summary -vv
```

Every CLI command prints an online-services notice before doing work. The MVP does not send photo, metadata, or query data to online services. If the OpenCLIP backend is selected, OpenCLIP may contact the network only to download model weights when they are not already cached.

## Search score interpretation

Search results preserve the raw cosine similarity as `raw_score`, then add query-relative display fields: `rank`, `top_percent`, `robust_z`, `normalized_relative`, and a plain-language match label. Rank decides ordering, while a confidence gate decides whether the result is trustworthy: the best result must clear both a minimum raw-score threshold and a minimum robust z-score threshold (configurable with `--min-raw-score` and `--min-robust-z`). If the gate fails, KiokuFux prints `No confident matches found.` and `Showing closest available results.` and labels the best item `closest available · low confidence` rather than pretending it is a good match. The normalized relative value is only an ordering aid within the current query result set; it is not a probability and must not be read as “87% match.” Use `kiokufux search PATH "query text" --summary` to print only these search statistics plus the image file name.

## Tagging

MVP 1 supports local catalog tags without modifying original images. Use a `photo_id` from scan/search results:

```bash
kiokufux tag ./photos PHOTO_ID "family party" dog
kiokufux tags ./photos PHOTO_ID
kiokufux untag ./photos PHOTO_ID dog
```

Manual tags are stored in SQLite and exported into sidecars under `review.tags`. KiokuFux can also generate local AI-assisted tag proposals with `kiokufux auto-tag ./photos`; proposals remain pending until reviewed with `kiokufux accept-tag` or `kiokufux reject-tag`. Accepted AI proposals are stored as `source=auto` tags and exported under `semantic.auto_tags`; pending/rejected proposals are exported under `review.tag_proposals` for review. No image data is sent to online services for MVP1 auto-tagging.

## Sidecars

`kiokufux export-sidecars ./photos` writes files named like `image.jpg.kiokufux.json` next to each indexed image. Sidecars use schema `kiokufux.sidecar.v1` and include IDs, source paths, hashes, extracted metadata, semantic status, and review state.

## Limitations

- The fallback embedding backend is lightweight and local but not as semantically powerful as a downloaded OpenCLIP model.
- Search is a simple NumPy scan over stored vectors; FAISS or hnswlib can be added later.
- No face recognition, person clustering, event detection, historical place-name logic, GUI, accounts, sync, sharing, or hosted service is included.
- Metadata is read from images but never written back to originals.

## Suggested next MVPs

- MVP 2: stronger local model setup, optional approximate vector indexes, and richer auto-tags/captions.
- MVP 3: review workflows and safer bulk-edit tooling using sidecars only.
- MVP 4: optional desktop UI for browsing, search, and curation.
