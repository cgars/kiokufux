# KiokuFux

KiokuFux MVP 1 is a local-first CLI prototype for indexing private photo archives and searching them by visual content and basic metadata. It does not modify original image files.

## What MVP 1 does

- Recursively scans `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, and `.webp` files.
- Generates stable photo IDs from SHA-256 file hashes.
- Extracts basic metadata with Pillow, including dimensions, file timestamps, MIME type, EXIF date, and GPS when available.
- Stores a resumable SQLite catalog in `.kiokufux/catalog.sqlite`.
- Generates JPEG thumbnails in `.kiokufux/thumbnails/` with EXIF orientation applied.
- Generates local embeddings in `.kiokufux/embeddings/`.
- Stores embedding artifact paths relative to the `.kiokufux/` workspace so archives remain movable.
- Runs text-to-image semantic search using cosine similarity.
- Exports versioned `.kiokufux.json` sidecars next to photos.
- Prints scan progress to stderr, logs scan errors to `.kiokufux/logs/kiokufux.log`, and records unreadable images without stopping the scan.

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
kiokufux rotate PATH PHOTO_ID_OR_7_CHAR_PREFIX --degrees 90
kiokufux rotate PATH --auto
kiokufux rotate PATH --auto --vlm-fallback
kiokufux rotate PATH --auto --vlm-only
kiokufux embed PATH
kiokufux search PATH "query text"
kiokufux search PATH "query text" --summary
kiokufux -v search PATH "query text" --summary
kiokufux tag PATH PHOTO_ID "family party"
kiokufux auto-tag PATH
kiokufux vlm-analyze PATH
kiokufux vlm-analyze PATH --vlm-backend ollama --ollama-url http://HOST:11434 --ollama-model MODEL
kiokufux descriptions PATH [PHOTO_ID]
kiokufux tag-summary PATH
kiokufux vocab-propose PATH
kiokufux vocab PATH
kiokufux vocab-accept PATH TAG [--category CATEGORY] [--scope core|collection-specific|optional]
kiokufux vocab-reject PATH TAG
kiokufux vocab-merge PATH ALIAS CANONICAL_TAG
kiokufux vocab-apply PATH
kiokufux tag-proposals PATH [PHOTO_ID]
kiokufux tag-review PATH [PHOTO_ID]
kiokufux accept-tag PATH PHOTO_ID_OR_7_CHAR_PREFIX TAG
kiokufux accept-tag PATH [PHOTO_ID_OR_7_CHAR_PREFIX] --all
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
kiokufux rotate ./photos PHOTO_ID_FROM_SEARCH --degrees 90
kiokufux embed ./photos
kiokufux search ./photos "red car in front of a house"
kiokufux search ./photos "red car in front of a house" --summary
kiokufux tag ./photos PHOTO_ID_FROM_SEARCH "family party"
kiokufux auto-tag ./photos
kiokufux vlm-analyze ./photos --vlm-backend fake
kiokufux vlm-analyze ./photos --vlm-backend ollama --ollama-url http://gaming-pc:11434 --ollama-model llava
kiokufux descriptions ./photos
kiokufux tag-summary ./photos
kiokufux vocab-propose ./photos
kiokufux vocab-accept ./photos garden --category place --scope core --alias yard
kiokufux vocab-merge ./photos backyard garden
kiokufux vocab-apply ./photos
kiokufux tag-review ./photos
kiokufux tag-proposals ./photos PHOTO_ID_FROM_SEARCH
kiokufux accept-tag ./photos PHOTO_ID_OR_7_CHAR_PREFIX dog
kiokufux accept-tag ./photos --all
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

[autotagging]
candidate_tags = "person, man, woman, child, baby, ... , text_in_image, handwriting"
top_k = 5
min_score = 0.20

[search]
top_k = 10
min_raw_score = 0.20
min_robust_z = 1.0

[logging]
verbose = 0
```

Use the config file for stable project defaults, and command flags such as `--top-k`, `--embedding-backend`, `--openclip-model`, or `-v` for one-off overrides. The generated config contains the full 100-tag default auto-tag vocabulary.

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

Manual tags are stored in SQLite and exported into sidecars under `review.tags`. KiokuFux can also generate local AI-assisted tag proposals with `kiokufux auto-tag ./photos`; this uses the configured embedding backend for zero-shot image/text similarity between each photo and candidate tag labels from `.kiokufux/config.toml` (or `--candidate-tags` for one run). Proposals remain pending until reviewed with `kiokufux tag-summary`, `kiokufux tag-review` (or `kiokufux tag-proposals`) and then accepted with `kiokufux accept-tag` or rejected with `kiokufux reject-tag`. `kiokufux tag-summary ./photos` prints recurring proposed tags aggregated by tag, source, status, image count, proposal count, average confidence, and maximum confidence so users can review collection-level concepts before inspecting individual photos.

KiokuFux includes a VLM analysis layer that stores structured per-image analysis and turns VLM candidate tags into pending review proposals. The default backend is `--vlm-backend fake`, a deterministic local backend used to wire and test the pipeline without adding heavyweight model dependencies. For real VLM analysis, `--vlm-backend ollama` can call a local or LAN Ollama server, for example `--ollama-url http://gaming-pc:11434 --ollama-model llava`, and sends images to Ollama's `/api/generate` endpoint with JSON output requested. VLM captions, complete descriptions, and analysis are exported into sidecars under `semantic.caption`, `semantic.description`, and `semantic.vlm`; VLM tag evidence is exported with review proposals. Use `kiokufux descriptions ./photos` to print VLM captions/descriptions in an aligned table.

For a remote Ollama machine, pass the server root URL, not the generate endpoint. For example, use `--ollama-url http://gaming-pc:11434`; KiokuFux appends `/api/generate` itself. If you get a 404, rerun with `-v` to print the endpoint being called and the current image, or `-vv` for debug-level skip messages.


KiokuFux also supports a local controlled vocabulary workflow. `kiokufux vocab-propose ./photos` promotes recurring pending tag proposals into vocabulary candidates. Review candidates with `kiokufux vocab ./photos`, accept canonical tags with categories/scopes via `kiokufux vocab-accept`, reject unwanted tags with `kiokufux vocab-reject`, and merge synonyms with `kiokufux vocab-merge ALIAS CANONICAL_TAG`. `kiokufux vocab-apply ./photos` then applies accepted vocabulary tags to matching pending proposals, using aliases to store canonical tags and using rejected vocabulary entries as persistent reject memory. This implements the curated-vocabulary part of the workflow while remaining local-first.

Running `kiokufux tag-review ./photos` without a photo ID prints a grouped list of all images that have proposed tags, showing each image filename and the first seven characters of its photo ID; adding a photo ID limits review output to that image. To accept every pending proposal, use `kiokufux accept-tag ./photos --all`; add either the full photo ID or its first seven characters before `--all` to accept all pending proposals for only that image. Single-proposal accepts also allow either the full photo ID or its first seven characters. Accepted AI proposals are stored as `source=auto` tags and exported under `semantic.auto_tags`; pending/rejected proposals are exported under `review.tag_proposals` for review. No image data is sent to online services for MVP1 auto-tagging, though OpenCLIP may download model weights if selected and uncached.

## Image rotation

`kiokufux rotate PATH PHOTO_ID_OR_7_CHAR_PREFIX --degrees 90|180|270` rotates an indexed image clockwise in place. Use `kiokufux rotate PATH --auto` to recursively process all indexed images under `PATH`, or `--auto` with a photo ID to check one image. Auto mode asks KiokuFux to choose a rotation from EXIF orientation first, then from any existing `vlm-analyze` caption/description that clearly mentions orientation, and finally from a conservative local image-content heuristic that looks for document/text-line structure when EXIF is absent. Add `--vlm-fallback` to run an additional VLM orientation check only for images where those cheaper signals are not confident, or `--vlm-only` to force a fresh VLM check that directly asks for the corrective action as `action_clockwise_degrees` and skips the EXIF/stored-text/local-heuristic path. Each auto-rotation candidate prints a decision, rotation basis, confidence, and reason. If auto-detection is not confident, no image is changed and KiokuFux asks you to choose `--degrees` manually.

By default rotation writes a same-folder `.bak` copy before changing the original; pass `--no-backup` only if you already have external backups. After rotation, KiokuFux refreshes the catalog metadata, clears the stale thumbnail path, deletes stale embeddings and VLM analyses, and marks the image for fresh thumbnail and embedding generation. Rerun `kiokufux thumbnails PATH` and `kiokufux embed PATH` when ready.

This is useful for fixing incorrectly oriented scans, but it is intentionally explicit because it edits the source image file. Automatic non-EXIF detection is intentionally conservative: existing VLM descriptions can help when they explicitly mention that an image is rotated, and the local heuristic works best for document-like/text-heavy images; arbitrary photos may still need manual review. For archival collections, prefer keeping backups or representing rotation in sidecars until you are sure destructive edits fit your workflow.

## Sidecars

`kiokufux export-sidecars ./photos` writes files named like `image.jpg.kiokufux.json` next to each indexed image. Sidecars use schema `kiokufux.sidecar.v1` and include IDs, source paths, hashes, extracted metadata, semantic status, and review state.

## Limitations

- The fallback embedding backend is lightweight and local but not as semantically powerful as a downloaded OpenCLIP model.
- Search is a simple NumPy scan over stored vectors; FAISS or hnswlib can be added later.
- No face recognition, person clustering, event detection, historical place-name logic, GUI, accounts, sync, sharing, or hosted service is included.
- Metadata is read from images but never written back to originals.

## Suggested next MVPs

- MVP 2: stronger local model setup, optional approximate vector indexes, and real local VLM backends for richer auto-tags/captions.
- MVP 3: review workflows and safer bulk-edit tooling using sidecars only.
- MVP 4: optional desktop UI for browsing, search, and curation.
