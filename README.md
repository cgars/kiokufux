# Fotofux / KiokuFux

Fotofux MVP 1 is a local-first CLI prototype for indexing private photo archives and searching them by visual content and basic metadata. It does not modify original image files.

## What MVP 1 does

- Recursively scans `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, and `.webp` files.
- Generates stable photo IDs from SHA-256 file hashes.
- Extracts basic metadata with Pillow, including dimensions, file timestamps, MIME type, EXIF date, and GPS when available.
- Stores a resumable SQLite catalog in `.fotofux/catalog.sqlite`.
- Generates JPEG thumbnails in `.fotofux/thumbnails/` with EXIF orientation applied.
- Generates local embeddings in `.fotofux/embeddings/`.
- Runs text-to-image semantic search using cosine similarity.
- Exports versioned `.fotofux.json` sidecars next to photos.
- Logs scan errors to `.fotofux/logs/fotofux.log` and records unreadable images without stopping the scan.

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

If OpenCLIP is unavailable, Fotofux falls back to a dependency-light local embedding backend so the MVP remains runnable offline.

## CLI usage

```bash
fotofux init PATH
fotofux scan PATH
fotofux thumbnails PATH
fotofux embed PATH
fotofux search PATH "query text"
fotofux export-sidecars PATH
```

## Example workflow

```bash
fotofux init ./photos
fotofux scan ./photos
fotofux thumbnails ./photos
fotofux embed ./photos
fotofux search ./photos "red car in front of a house"
fotofux export-sidecars ./photos
```

The workspace is created at `./photos/.fotofux/`:

```text
.fotofux/
  catalog.sqlite
  thumbnails/
  embeddings/
  indexes/
  logs/
```

## Sidecars

`fotofux export-sidecars ./photos` writes files named like `image.jpg.fotofux.json` next to each indexed image. Sidecars use schema `fotofux.sidecar.v1` and include IDs, source paths, hashes, extracted metadata, semantic status, and review state.

## Limitations

- The fallback embedding backend is lightweight and local but not as semantically powerful as a downloaded OpenCLIP model.
- Search is a simple NumPy scan over stored vectors; FAISS or hnswlib can be added later.
- No face recognition, person clustering, event detection, historical place-name logic, GUI, accounts, sync, sharing, or hosted service is included.
- Metadata is read from images but never written back to originals.

## Suggested next MVPs

- MVP 2: stronger local model setup, optional approximate vector indexes, and richer auto-tags/captions.
- MVP 3: review workflows and safer bulk-edit tooling using sidecars only.
- MVP 4: optional desktop UI for browsing, search, and curation.
