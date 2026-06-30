from __future__ import annotations

import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GPS_TAGS: dict[str, int] = {}


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _gps_float(value: Any, ref: str | None) -> float | None:
    try:
        parts = [float(x) for x in value]
        coord = parts[0] + parts[1] / 60 + parts[2] / 3600
        if ref in {"S", "W"}:
            coord *= -1
        return coord
    except Exception:
        return None


def extract_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    data: dict[str, Any] = {
        "file_size": stat.st_size,
        "mime_type": mimetypes.guess_type(path.name)[0],
        "created_at_file": _iso(stat.st_ctime),
        "modified_at_file": _iso(stat.st_mtime),
        "width": None,
        "height": None,
        "exif_datetime_original": None,
        "exif_gps_lat": None,
        "exif_gps_lon": None,
    }
    from PIL import Image, ExifTags
    global GPS_TAGS
    if not GPS_TAGS:
        GPS_TAGS = {v: k for k, v in ExifTags.GPSTAGS.items()}
    with Image.open(path) as img:
        data["width"], data["height"] = img.size
        exif = img.getexif()
        if exif:
            data["exif_datetime_original"] = exif.get(36867) or exif.get(306)
            gps = exif.get_ifd(ExifTags.IFD.GPSInfo) if hasattr(ExifTags, "IFD") else {}
            if gps:
                data["exif_gps_lat"] = _gps_float(gps.get(GPS_TAGS.get("GPSLatitude")), gps.get(GPS_TAGS.get("GPSLatitudeRef")))
                data["exif_gps_lon"] = _gps_float(gps.get(GPS_TAGS.get("GPSLongitude")), gps.get(GPS_TAGS.get("GPSLongitudeRef")))
    return data
