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
kiokufux init PATH
kiokufux scan PATH
kiokufux thumbnails PATH
kiokufux embed PATH
kiokufux search PATH "query text"
kiokufux export-sidecars PATH
```

## Example workflow

```bash
kiokufux init ./photos
kiokufux scan ./photos
kiokufux thumbnails ./photos
kiokufux embed ./photos
kiokufux search ./photos "red car in front of a house"
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

### Embedding backend configuration

By default, `kiokufux embed` and `kiokufux search` try OpenCLIP first and fall back to the lightweight local backend if OpenCLIP is unavailable. You can force a backend or choose a specific OpenCLIP model/weights pair:

```bash
kiokufux embed ./photos --embedding-backend openclip --openclip-model ViT-L-14 --openclip-pretrained datacomp_xl_s13b_b90k
kiokufux search ./photos "church" --embedding-backend openclip --openclip-model ViT-L-14 --openclip-pretrained datacomp_xl_s13b_b90k
kiokufux embed ./photos --embedding-backend simple
```

The same values can also be supplied with `KIOKUFUX_OPENCLIP_MODEL` and `KIOKUFUX_OPENCLIP_PRETRAINED`. Use the same backend/model options for search that you used when generating embeddings.

## Search score interpretation

Search results preserve the raw cosine similarity as `raw_score`, then add query-relative display fields: `rank`, `top_percent`, `normalized_relative`, and a plain-language match label such as `very good match`, `good match`, `possible match`, or `weak match`. The normalized relative value is only an ordering aid within the current query result set; it is not a probability and must not be read as “87% match.”

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
