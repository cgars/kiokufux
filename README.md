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

For local recurring-face discovery, install the optional inference stack. The first
run may download the explicitly selected facenet-pytorch VGGFace2 weights; subsequent
processing is offline:

```bash
pip install -e '.[faces]'
kiokufux faces scan ./photos
kiokufux faces cluster ./photos
kiokufux faces review ./photos
```

Face groups are anonymous algorithmic suggestions. KiokuFux never assigns names or
creates people automatically: only a reviewed group explicitly confirmed by a user
becomes a stable, collection-local person, and its display name remains optional.

## CLI usage

Global options:

```bash
kiokufux [-v|--verbose] COMMAND ...
```

Complete command reference:

```bash
kiokufux init PATH
kiokufux scan PATH
kiokufux thumbnails PATH
kiokufux export-sidecars PATH
kiokufux export-gallery PATH OUTPUT [--title TITLE] [--query QUERY] [--tag TAG] [--faces none|confirmed|grouped|detected] [--face-boxes] [--person PERSON] [--face-group GROUP] [--unknown-faces] [--top-k TOP_K] [--min-tag-count MIN_TAG_COUNT] [--max-cloud-tags MAX_CLOUD_TAGS] [--image-max-size IMAGE_MAX_SIZE] [--overwrite] [--embedding-backend auto|openclip|simple] [--openclip-model MODEL] [--openclip-pretrained WEIGHTS]

kiokufux rotate PATH [PHOTO_ID_OR_7_CHAR_PREFIX] (--degrees 90|180|270 | --auto) [--no-backup] [--vlm-fallback] [--vlm-only] [--vlm-verify] [--vlm-compare] [--vlm-backend fake|ollama] [--ollama-url URL] [--ollama-model MODEL] [--vlm-timeout SECONDS]
kiokufux prompts [--topic all|rotation|vlm-analysis]

kiokufux embed PATH [--embedding-backend auto|openclip|simple] [--openclip-model MODEL] [--openclip-pretrained WEIGHTS]
kiokufux search PATH "query text" [--top-k TOP_K] [--summary] [--min-raw-score SCORE] [--min-robust-z SCORE] [--embedding-backend auto|openclip|simple] [--openclip-model MODEL] [--openclip-pretrained WEIGHTS]

kiokufux tag PATH PHOTO_ID TAG [TAG ...]
kiokufux untag PATH PHOTO_ID TAG [TAG ...]
kiokufux tags PATH [PHOTO_ID]
kiokufux auto-tag PATH [--candidate-tags TAGS] [--top-k TOP_K] [--min-score SCORE] [--embedding-backend auto|openclip|simple] [--openclip-model MODEL] [--openclip-pretrained WEIGHTS]
kiokufux tag-summary PATH [--status pending|accepted|rejected|all]
kiokufux tag-proposals PATH [PHOTO_ID] [--status pending|accepted|rejected|all]
kiokufux tag-review PATH [PHOTO_ID] [--status pending|accepted|rejected|all]
kiokufux accept-tag PATH [PHOTO_ID_OR_7_CHAR_PREFIX] [TAG] [--all] [--source SOURCE]
kiokufux reject-tag PATH PHOTO_ID TAG [--source SOURCE]

kiokufux vocab PATH [--status proposed|accepted|rejected|all]
kiokufux vocab-propose PATH [--min-photos MIN_PHOTOS]
kiokufux vocab-accept PATH TAG [--category CATEGORY] [--scope core|collection-specific|optional] [--parent PARENT] [--alias ALIAS] [--notes NOTES]
kiokufux vocab-reject PATH TAG [--notes NOTES]
kiokufux vocab-merge PATH ALIAS CANONICAL_TAG
kiokufux vocab-apply PATH [--source SOURCE]

kiokufux vlm-analyze PATH [--vlm-backend fake|ollama] [--ollama-url URL] [--ollama-model MODEL] [--vlm-timeout SECONDS] [--limit LIMIT] [--force]
kiokufux descriptions PATH [PHOTO_ID]
kiokufux vlm-descriptions PATH [PHOTO_ID]

kiokufux faces scan PATH [--device auto|cuda|cpu]
kiokufux faces cluster PATH
kiokufux faces review PATH [--host HOST] [--port PORT] [--no-open]
kiokufux faces reset-derived PATH
kiokufux faces remove-all PATH --yes
```

Notes:

- `tag-review` is an alias for `tag-proposals`.
- `vlm-descriptions` is an alias for `descriptions`.
- `accept-tag --all` accepts all pending proposals, optionally limited by `PHOTO_ID_OR_7_CHAR_PREFIX`; single-proposal acceptance uses `PHOTO_ID_OR_7_CHAR_PREFIX TAG`.
- `faces review` defaults to a loopback host; non-loopback hosts are rejected by the review server.
- The face reviewer uses the gallery's pine-and-paper visual system, responsive photographic cards, and context-sensitive actions for ungrouped detections, provisional groups, and confirmed people.

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
  faces.sqlite
  face-review.json
  people.json
  cache/face-thumbnails/
```

## Face-data privacy and recovery

Face detection, 512-dimensional embeddings, anonymous clustering, review, and person
metadata stay inside the collection workspace. Face embeddings are biometric-derived
data: protect and back up `.kiokufux/` with the same care as the photographs. The
review server refuses non-loopback hosts by default and does not perform cloud lookup,
telemetry, demographic inference, or automatic identification. Source photographs
are opened read-only and are never cropped or rewritten by the face workflow.

`faces reset-derived` deletes the rebuildable SQLite index and face-thumbnail cache
but preserves `people.json` and `face-review.json`; durable face locators let a
later scan reconnect unchanged reviewed occurrences where the face region can be
matched safely. If re-clustering creates a different provisional set, that new
group remains a provisional suggestion tied to its cluster run; confirmed people
keep their stable `person_id` and permanent friendly name. Undo is last-action
first: review actions are persisted in history, and a newer dependent action can
block undo until the dependency is resolved. KiokuFux rotations and external edits
create new content-derived photo IDs; old derived face rows must be invalidated,
exact quarter-turn rotations can transform durable boxes for review, and some
edits may require re-scanning and re-review to restore face occurrences.
when possible so a moved collection remains reviewable, and byte-identical files
follow KiokuFux catalog semantics as one logical photograph. `faces remove-all --yes` deletes
both derived and human-authored face data without deleting photographs. Model and
preprocessing identifiers are stored beside every compact float32 embedding, and
incompatible versions are clustered separately. The default backend uses
facenet-pytorch (MIT-licensed code) with its `vggface2` pretrained InceptionResnetV1;
review the upstream weights terms and VGGFace2 training-data provenance for your use
before installation. KiokuFux does not bundle those weights or use InsightFace models.

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

`kiokufux rotate PATH PHOTO_ID_OR_7_CHAR_PREFIX --degrees 90|180|270` rotates an indexed image clockwise in place. Use `kiokufux rotate PATH --auto` to recursively process all indexed images under `PATH`, or `--auto` with a photo ID to check one image. Auto mode asks KiokuFux to choose a rotation from EXIF orientation first, then from any existing `vlm-analyze` caption/description that clearly mentions orientation, and finally from a conservative local image-content heuristic that looks for document/text-line structure when EXIF is absent. Add `--vlm-fallback` to run an additional VLM orientation check only for images where those cheaper signals are not confident, or `--vlm-only` to force a fresh VLM check that directly asks for the corrective action as `action_clockwise_degrees` and skips the EXIF/stored-text/local-heuristic path. Use `--vlm-compare` with fresh VLM modes to show the VLM a contact sheet of 0/90/180/270-degree candidates and ask it to choose the upright version. Use `--vlm-verify` with fresh VLM modes to ask the VLM once after a VLM-driven rotation whether the result appears upright; verification never applies a second rotation, so it cannot loop. Use `kiokufux prompts --topic rotation` to print the exact VLM prompts used for direct and comparison rotation checks. Each auto-rotation candidate prints a decision, rotation basis, confidence, and reason. If auto-detection is not confident, no image is changed and KiokuFux asks you to choose `--degrees` manually.

By default rotation writes a same-folder `.bak` copy before changing the original; pass `--no-backup` only if you already have external backups. After rotation, KiokuFux refreshes the catalog metadata, clears the stale thumbnail path, deletes stale embeddings and VLM analyses, and marks the image for fresh thumbnail and embedding generation. Rerun `kiokufux thumbnails PATH` and `kiokufux embed PATH` when ready.

This is useful for fixing incorrectly oriented scans, but it is intentionally explicit because it edits the source image file. Automatic non-EXIF detection is intentionally conservative: existing VLM descriptions can help when they explicitly mention that an image is rotated, and the local heuristic works best for document-like/text-heavy images; arbitrary photos may still need manual review. For archival collections, prefer keeping backups or representing rotation in sidecars until you are sure destructive edits fit your workflow.

## Sidecars

`kiokufux export-sidecars ./photos` writes files named like `image.jpg.kiokufux.json` next to each indexed image. Sidecars use schema `kiokufux.sidecar.v2` and include IDs, source paths, hashes, extracted metadata, semantic status, review state, and a `faces` block when face data exists. Face sidecars are included by default for local export but never include embedding vectors, embedding BLOBs, raw model tensors, or face-thumbnail paths.

The `faces.scan_status` value distinguishes `not_scanned` from `scanned` with zero detections. Each occurrence exports a stable `face_id`, normalized oriented-image box coordinates, detector confidence, optional quality, and an `identity.status` of `confirmed`, `provisional`, `ungrouped`, `excluded`, `rejected`, or `conflict`. Provisional groups export a non-identifying run-scoped `friendly_name` such as `quiet_fox`, the `group_id`, `cluster_run_id`, review state, and conflict flag. Confirmed people export a stable `person_id`, a permanent non-identifying person `friendly_name`, and an optional user-entered `display_name`; assigned display names may be real names and are copied into local sidecars, so treat exported sidecars as private data. Existing schema-v1 `people.json` files are migrated atomically by assigning deterministic friendly names from stable `person_id` values, not from current provisional group IDs.

Compact example:

```json
{
  "schema": "kiokufux.sidecar.v2",
  "photo_id": "photo-sha256",
  "source_path": "photos/a.jpg",
  "file_hash": "photo-sha256",
  "metadata": {"width": 1600, "height": 1200, "datetime_original": null, "gps": {"lat": null, "lon": null}},
  "semantic": {"embedding_model": null, "auto_tags": [], "caption": null, "description": null, "vlm": null, "status": "pending"},
  "review": {"state": "unreviewed", "tags": [], "tag_proposals": []},
  "faces": {
    "scan_status": "scanned",
    "model_key": "facenet-pytorch:inception-resnet-v1-vggface2:facenet-pytorch:mtcnn-160-v1:512",
    "occurrences": [
      {
        "face_id": "face-uuid",
        "box": {"x1": 0.12, "y1": 0.08, "x2": 0.31, "y2": 0.42},
        "detection_confidence": 0.992,
        "quality": null,
        "identity": {"status": "confirmed", "person_id": "person-uuid", "friendly_name": "quiet_fox", "display_name": "Anna"}
      }
    ]
  }
}
```


## Limitations

- The fallback embedding backend is lightweight and local but not as semantically powerful as a downloaded OpenCLIP model.
- Search is a simple NumPy scan over stored vectors; FAISS or hnswlib can be added later.
- No event detection, historical place-name logic, accounts, sync, sharing, hosted service, cloud face lookup, or automatic real-world person identification is included.
- Metadata is read from images but never written back to originals.

## Suggested next MVPs

- MVP 2: stronger local model setup, optional approximate vector indexes, and real local VLM backends for richer auto-tags/captions.
- MVP 3: review workflows and safer bulk-edit tooling using sidecars only.
- MVP 4: optional desktop UI for browsing, search, and curation.

## Static HTML gallery export

`kiokufux export-gallery PATH OUTPUT` creates a standalone, offline-friendly gallery in `OUTPUT` with `index.html`, `gallery.json`, static CSS/JavaScript assets, copied images, and thumbnails. Open `index.html` directly in a browser; its data, CSS, and JavaScript are embedded, so no local web server is required. The gallery searches filenames, relative paths, VLM captions/descriptions, manual tags, and accepted automatic tags in the browser, and includes a frequency-weighted tag cloud for published tags. Its lightbox centers each photograph, supports zooming and panning in an independently scrollable media pane, keeps details independently scrollable, and collapses long descriptions behind a reader-controlled expansion. Exports from the earlier `fetch("gallery.json")` implementation are detected and replaced automatically; use `--overwrite` to regenerate any other existing export.

During export, source photographs are located from the current collection `PATH` and their indexed relative paths first. Windows and POSIX directory separators are normalized for the current operating system. The stored absolute path is only a compatibility fallback, so moving the whole collection between drives, mount points, or machines does not require a rescan when its internal directory structure remains unchanged. Relative paths are constrained to the collection root.

The exporter reports and times selection, optional face processing, output preparation, and image export separately. This makes slow mounted-drive operations visible instead of leaving the terminal apparently idle. With `--overwrite`, a valid previous gallery is refreshed incrementally: unchanged copied images and generated thumbnails are reused, changed files are overwritten, and media no longer selected is removed. If a source cannot be found or copied, the exporter records the photo and exact error in `.kiokufux/logs/kiokufux.log`; add `-v` to show the warning in the terminal. A thumbnail-generation failure no longer drops an otherwise exported photograph: the gallery uses its exported image as the preview instead.

Face information is excluded by default. The optional `--faces` modes progressively add privacy-safe People filters:

- `confirmed` publishes user-confirmed people using stable person IDs, display names, and friendly names.
- `grouped` also publishes non-conflicting provisional recurring groups under their anonymous friendly names and marks them as unconfirmed.
- `detected` also adds a single **Unknown people** category for photos containing usable ungrouped detections, including only the number found in each photo.

All enabled identities become searchable and appear in the photograph detail view. Add the separate, explicit `--face-boxes` option to publish normalized bounding rectangles for identities allowed by the selected face mode; the lightbox then offers a **Show faces** toggle. Without that option, bounding boxes remain private. Embeddings, face IDs, detector confidence, model metadata, rejected faces, excluded faces, and conflicting assignments are never published.

Use repeatable `--person` and `--face-group` selectors to export photos containing an exact confirmed person or provisional group; `--unknown-faces` selects photos with usable ungrouped detections. Identity selectors are combined with OR, while identity selection combines with query and tag filters using AND. These selectors can be used with `--faces none` to create a face-based gallery without publishing any identity metadata or selector values in the resulting files.

Examples:

```bash
kiokufux export-gallery ./photos ./gallery-export --title "Family Archive"
kiokufux export-gallery ./photos ./gallery-export --query "beach" --top-k 25 --overwrite
kiokufux export-gallery ./photos ./gallery-export --tag beach --tag family
kiokufux export-gallery ./photos ./gallery-export --faces confirmed
kiokufux export-gallery ./photos ./gallery-export --faces grouped
kiokufux export-gallery ./photos ./review-export --faces detected --unknown-faces
kiokufux export-gallery ./photos ./review-export --faces detected --face-boxes
kiokufux export-gallery ./photos ./anna-export --person Anna --faces confirmed
kiokufux export-gallery ./photos ./family-export --person Anna --person Bert
kiokufux export-gallery ./photos ./anonymous-group --face-group quiet_fox
```

Pending or rejected tag proposals are not included. Use `--min-tag-count` and `--max-cloud-tags` to tune the default cloud, and `--image-max-size` to export downscaled image derivatives instead of original-size copies.
