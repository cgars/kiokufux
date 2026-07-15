from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


VALID_ROTATION_DEGREES = {90, 180, 270}


@dataclass(frozen=True)
class RotationDetection:
    degrees: int | None
    confidence: float
    source: str
    reason: str


@dataclass(frozen=True)
class RotationResult:
    path: Path
    degrees: int
    backup_path: Path | None


def backup_path_for(path: Path) -> Path:
    candidate = path.with_name(f"{path.name}.bak")
    if not candidate.exists():
        return candidate
    index = 1
    while True:
        numbered = path.with_name(f"{path.name}.bak{index}")
        if not numbered.exists():
            return numbered
        index += 1


def _normalized_orientation_text(text: str) -> str:
    return " ".join(text.lower().replace("-", " ").split())


def detect_clockwise_rotation_from_description(text: str | None, source: str = "vlm-description") -> RotationDetection:
    if not text:
        return RotationDetection(None, 0.0, source, "no VLM description available")
    normalized = _normalized_orientation_text(text)
    if not any(marker in normalized for marker in ("rotated", "rotation", "orientation", "sideways", "upside down", "on its side", "turned")):
        return RotationDetection(None, 0.0, source, "VLM description does not mention image orientation")

    explicit = re.search(r"(?:rotate|rotated|rotation|turn|turned)\D{0,24}(90|180|270)\D{0,24}(clockwise|counterclockwise|anti clockwise|anticlockwise|left|right)", normalized)
    if explicit:
        degrees = int(explicit.group(1))
        direction = explicit.group(2)
        if direction in {"clockwise", "right"}:
            degrees = (360 - degrees) % 360
        if degrees in VALID_ROTATION_DEGREES:
            return RotationDetection(degrees, 0.86, source, "VLM description explicitly mentions the image rotation")

    if "upside down" in normalized or "180" in normalized:
        return RotationDetection(180, 0.82, source, "VLM description says the image is upside down")
    if "counterclockwise" in normalized or "anti clockwise" in normalized or "anticlockwise" in normalized or "rotated left" in normalized or "turned left" in normalized:
        return RotationDetection(90, 0.78, source, "VLM description says the image is rotated counterclockwise/left")
    if "clockwise" in normalized or "rotated right" in normalized or "turned right" in normalized:
        return RotationDetection(270, 0.78, source, "VLM description says the image is rotated clockwise/right")
    return RotationDetection(None, 0.20, source, "VLM description mentions orientation but not a usable direction")


def _coerce_rotation_degrees(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _action_text_rotation(text: str, source: str, confidence: float) -> RotationDetection:
    normalized = _normalized_orientation_text(text)
    if "180" in normalized or "upside down" in normalized:
        return RotationDetection(180, confidence, source, "VLM action says to rotate 180°")
    if "counterclockwise" in normalized or "anti clockwise" in normalized or "anticlockwise" in normalized or "left" in normalized:
        return RotationDetection(270, confidence, source, "VLM action says to rotate counterclockwise/left")
    if "clockwise" in normalized or "right" in normalized:
        return RotationDetection(90, confidence, source, "VLM action says to rotate clockwise/right")
    return RotationDetection(None, confidence, source, "VLM action text did not contain a usable rotation direction")


def detect_clockwise_rotation_from_vlm_response(raw: object, source: str = "fresh-vlm") -> RotationDetection:
    if not isinstance(raw, dict):
        return RotationDetection(None, 0.0, source, "VLM rotation response was not a JSON object")
    rotation = raw.get("rotation") if isinstance(raw.get("rotation"), dict) else raw
    degrees = _coerce_rotation_degrees(
        rotation.get("action_clockwise_degrees")
        or rotation.get("correction_clockwise_degrees")
        or rotation.get("clockwise_degrees")
        or rotation.get("rotation_degrees")
        or rotation.get("degrees")
    )
    needs_rotation = rotation.get("needs_rotation")
    confidence_value = rotation.get("confidence", raw.get("confidence", 0.75))
    try:
        confidence = max(0.0, min(1.0, float(confidence_value)))
    except (TypeError, ValueError):
        confidence = 0.75
    reason = str(rotation.get("reason") or raw.get("reason") or "fresh VLM rotation response")
    if needs_rotation is False or degrees == 0:
        return RotationDetection(None, confidence, source, reason)
    if degrees in VALID_ROTATION_DEGREES:
        return RotationDetection(degrees, confidence, source, reason)

    action_text = " ".join(str(rotation.get(key, "")) for key in ("action", "corrective_action", "fix", "recommended_action"))
    action_detection = _action_text_rotation(action_text, source, confidence) if action_text.strip() else None
    if action_detection is not None and action_detection.degrees is not None:
        return action_detection

    appearance_text = " ".join(str(rotation.get(key, "")) for key in ("orientation", "direction", "description", "reason"))
    text_detection = detect_clockwise_rotation_from_description(appearance_text, source=source)
    if text_detection.degrees is not None:
        return text_detection
    return RotationDetection(None, confidence, source, "VLM rotation response did not contain usable corrective rotation")

def _exif_clockwise_rotation(img: Image.Image) -> int | None:
    orientation = img.getexif().get(274)
    return {3: 180, 6: 90, 8: 270}.get(orientation)


def _textline_score(img: Image.Image) -> float:
    sample = ImageOps.grayscale(img)
    sample.thumbnail((256, 256), Image.Resampling.BILINEAR)
    pixels = list(sample.tobytes())
    if not pixels:
        return 0.0
    width, height = sample.size
    threshold = max(0, min(255, sum(pixels) / len(pixels) - 25))
    dark = [1 if pixel < threshold else 0 for pixel in pixels]
    row_counts = [sum(dark[y * width : (y + 1) * width]) for y in range(height)]
    col_counts = [sum(dark[x + y * width] for y in range(height)) for x in range(width)]
    row_score = sum(count * count for count in row_counts) / max(1, width * width * height)
    col_score = sum(count * count for count in col_counts) / max(1, height * height * width)
    ink = sum(dark) / max(1, width * height)
    if ink < 0.005 or ink > 0.65:
        return 0.0
    return row_score - col_score


def detect_clockwise_rotation(path: Path, description_text: str | None = None) -> RotationDetection:
    """Detect a likely clockwise rotation using EXIF, VLM text, then a conservative text-line heuristic."""
    with Image.open(path) as img:
        exif_degrees = _exif_clockwise_rotation(img)
        if exif_degrees is not None:
            return RotationDetection(exif_degrees, 1.0, "exif", f"EXIF orientation requests {exif_degrees}° clockwise rotation")

        description_detection = detect_clockwise_rotation_from_description(description_text, source="stored-vlm-description")
        if description_detection.degrees is not None:
            return description_detection

        base = ImageOps.exif_transpose(img)
        scores = {degrees: _textline_score(base.rotate(-degrees, expand=True)) for degrees in (0, 90, 270)}

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_degrees, best_score = ranked[0]
    baseline_score = scores[0]
    margin = best_score - baseline_score
    confidence = max(0.0, min(0.99, margin / max(abs(best_score), 0.001)))
    if best_degrees != 0 and best_score > 0.01 and confidence >= 0.20:
        return RotationDetection(
            best_degrees,
            confidence,
            "textline-heuristic",
            "image content has stronger horizontal line structure after rotation",
        )
    return RotationDetection(None, confidence, "textline-heuristic", "no confident non-EXIF rotation detected")


def rotate_image(path: Path, degrees: int, create_backup: bool = True) -> RotationResult:
    """Rotate an image clockwise in-place, optionally keeping a same-folder backup."""
    if degrees not in VALID_ROTATION_DEGREES:
        raise ValueError("Rotation degrees must be one of: 90, 180, 270")
    if not path.exists():
        raise FileNotFoundError(path)

    backup_path = backup_path_for(path) if create_backup else None
    if backup_path is not None:
        shutil.copy2(path, backup_path)

    with Image.open(path) as img:
        fmt = img.format
        exif = img.getexif()
        if 274 in exif:
            exif[274] = 1
        rotated = ImageOps.exif_transpose(img).rotate(-degrees, expand=True)
        save_kwargs = {}
        if fmt:
            save_kwargs["format"] = fmt
        if exif:
            save_kwargs["exif"] = exif.tobytes()
        rotated.save(path, **save_kwargs)

    return RotationResult(path=path, degrees=degrees, backup_path=backup_path)
