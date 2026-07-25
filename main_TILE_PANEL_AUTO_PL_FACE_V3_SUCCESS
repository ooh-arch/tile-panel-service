from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
from PIL import Image, ImageOps, ImageDraw, ImageColor
import cloudinary
import cloudinary.uploader
import requests
import io
import os
import math
import hashlib

SERVICE_VERSION = "TILE_PANEL_SERVICE_AUTO_PL_FACE_V3"

app = FastAPI(title="Tile Panel Service")

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

COLOR_MAP = {
    "light_gray": "#D3D3D3",
    "dark_gray": "#666666",
    "medium_gray": "#999999",
    "white": "#FFFFFF",
    "black": "#000000",
    "gray": "#999999",
    "beige": "#D8C7A8"
}


class PanelRequest(BaseModel):
    job_id: str
    panel_key: str

    tile_face_mode: str = "single"
    random_face_count: int = 1
    tile_face_urls: Union[List[str], str]
    random_layout_rule: str = "repeat"

    tile_width_cm: float
    tile_height_cm: float

    install_orientation: str = "horizontal"
    layout_pattern: str = "straight_grid"
    symmetry_mode: str = "center_equal_cut"
    start_anchor: str = "top"

    target_width_cm: float
    target_height_cm: float

    panel_grout_mm: float = 2
    panel_grout_color: str = "light_gray"

    output_ppcm: int = 10


def normalize_face_urls(value):
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    if isinstance(value, str):
        return [
            x.strip()
            for x in value.replace("\n", ",").replace("|", ",").split(",")
            if x.strip()
        ]

    return []


def parse_color(value: str):
    if not value:
        value = "light_gray"

    value = value.strip().lower()
    hex_value = COLOR_MAP.get(value, value)

    try:
        return ImageColor.getrgb(hex_value)
    except Exception:
        return (211, 211, 211)


def download_image(url: str) -> Image.Image:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def fit_to_tile(img: Image.Image, width_px: int, height_px: int) -> Image.Image:
    return ImageOps.fit(img, (width_px, height_px), method=Image.LANCZOS)


def _smooth_signal(values, radius: int = 2):
    if not values:
        return []

    smoothed = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        window = values[start:end]
        smoothed.append(sum(window) / max(1, len(window)))

    return smoothed


def _axis_boundary_signal(img: Image.Image, axis: str):
    """
    Build a lightweight seam signal without OpenCV or NumPy.

    The signal combines:
    - average pixel difference across neighboring columns/rows
    - average brightness deviation, which helps with light grout lines
    """
    analysis = img.convert("L")

    max_analysis_side = 900
    scale = min(
        1.0,
        max_analysis_side / max(analysis.width, analysis.height)
    )

    if scale < 1.0:
        analysis = analysis.resize(
            (
                max(2, round(analysis.width * scale)),
                max(2, round(analysis.height * scale))
            ),
            Image.Resampling.BILINEAR
        )

    pixels = analysis.load()
    width, height = analysis.size

    if axis == "x":
        length = width
        cross_length = height

        means = []
        gradients = []

        for x in range(width):
            total = 0.0
            diff_total = 0.0

            for y in range(height):
                value = pixels[x, y]
                total += value

                if x > 0:
                    diff_total += abs(value - pixels[x - 1, y])

            means.append(total / max(1, cross_length))
            gradients.append(diff_total / max(1, cross_length))
    else:
        length = height
        cross_length = width

        means = []
        gradients = []

        for y in range(height):
            total = 0.0
            diff_total = 0.0

            for x in range(width):
                value = pixels[x, y]
                total += value

                if y > 0:
                    diff_total += abs(value - pixels[x, y - 1])

            means.append(total / max(1, cross_length))
            gradients.append(diff_total / max(1, cross_length))

    mean_level = sum(means) / max(1, len(means))
    brightness_deviation = [abs(value - mean_level) for value in means]

    smooth_gradient = _smooth_signal(gradients, radius=1)
    smooth_brightness = _smooth_signal(brightness_deviation, radius=2)

    gradient_mean = sum(smooth_gradient) / max(1, len(smooth_gradient))
    brightness_mean = sum(smooth_brightness) / max(1, len(smooth_brightness))

    signal = []

    for gradient_value, brightness_value in zip(
        smooth_gradient,
        smooth_brightness
    ):
        gradient_score = gradient_value / max(0.001, gradient_mean)
        brightness_score = brightness_value / max(0.001, brightness_mean)

        signal.append(
            (gradient_score * 0.72) +
            (brightness_score * 0.28)
        )

    return signal, length



def _score_grid_counts(
    signal,
    axis_length: int,
    min_count: int = 1,
    max_count: int = 30
):
    """
    Return several plausible repeated-cell counts for one axis.

    Unlike the older V2 logic, this function does not lock X and Y
    independently. It returns ranked candidates so the final grid can be
    selected as a compatible X/Y pair using the real tile aspect ratio.
    """
    if axis_length < 40 or not signal:
        return [(1, 0.0)]

    max_count = min(max_count, max(2, axis_length // 18))

    baseline_sorted = sorted(signal)
    baseline_index = max(
        0,
        min(len(baseline_sorted) - 1, round(len(baseline_sorted) * 0.65))
    )
    baseline = max(0.001, baseline_sorted[baseline_index])

    candidates = [(1, 0.0)]

    for count in range(max(2, min_count), max_count + 1):
        spacing = axis_length / count

        if spacing < 18:
            continue

        local_radius = max(1, min(6, round(spacing * 0.04)))
        boundary_scores = []

        for boundary_index in range(1, count):
            expected = round(boundary_index * spacing)
            start = max(1, expected - local_radius)
            end = min(len(signal) - 1, expected + local_radius)

            if start > end:
                continue

            local_peak = max(signal[start:end + 1])
            boundary_scores.append(local_peak / baseline)

        if not boundary_scores:
            continue

        average_score = sum(boundary_scores) / len(boundary_scores)
        strong_ratio = (
            sum(1 for value in boundary_scores if value >= 1.22)
            / len(boundary_scores)
        )
        weak_ratio = (
            sum(1 for value in boundary_scores if value >= 1.08)
            / len(boundary_scores)
        )

        score = (
            average_score * 0.55 +
            strong_ratio * 0.30 +
            weak_ratio * 0.15
        )

        score -= max(0, count - 16) * 0.008
        candidates.append((count, score))

    candidates.sort(key=lambda item: item[1], reverse=True)

    # Keep enough alternatives for pairwise aspect-ratio matching.
    return candidates[:12]


def _infer_panel_grid(
    img: Image.Image,
    tile_width_cm: float,
    tile_height_cm: float
):
    """
    Infer a repeated face grid from a legacy _pl image.

    V3 selects the X/Y grid as a pair. This prevents a strong but incompatible
    X count and Y count from being chosen independently, which caused square
    60x60 legacy panels to be rejected as 1.6667:1 cells in V2.
    """
    x_signal, x_length = _axis_boundary_signal(img, "x")
    y_signal, y_length = _axis_boundary_signal(img, "y")

    x_candidates = _score_grid_counts(x_signal, x_length)
    y_candidates = _score_grid_counts(y_signal, y_length)

    tile_ratio = tile_width_cm / max(1e-6, tile_height_cm)

    best_pair = None
    best_pair_score = float("-inf")

    for cols, x_score in x_candidates:
        for rows, y_score in y_candidates:
            if cols <= 1 or rows <= 1:
                continue

            total_faces = cols * rows

            if total_faces < 4 or total_faces > 900:
                continue

            source_cell_ratio = (
                (img.width / cols) /
                max(1e-6, (img.height / rows))
            )

            ratio_error = abs(
                math.log(
                    max(1e-6, source_cell_ratio) /
                    max(1e-6, tile_ratio)
                )
            )

            # Reject clearly impossible cell shapes.
            if ratio_error > 0.42:
                continue

            # Prefer seam evidence, then real tile aspect compatibility.
            aspect_score = max(0.0, 1.0 - (ratio_error / 0.42))
            pair_score = (
                x_score * 0.38 +
                y_score * 0.38 +
                aspect_score * 0.52
            )

            # Mild penalty against implausibly dense grids.
            pair_score -= max(0, total_faces - 144) * 0.0015

            if pair_score > best_pair_score:
                best_pair_score = pair_score
                best_pair = {
                    "cols": cols,
                    "rows": rows,
                    "x_score": x_score,
                    "y_score": y_score,
                    "source_cell_ratio": source_cell_ratio,
                    "tile_ratio": tile_ratio,
                    "ratio_error": ratio_error,
                    "total_faces": total_faces,
                    "pair_score": pair_score
                }

    if best_pair is None:
        return 1, 1, {
            "detected": False,
            "reason": "no_compatible_grid_pair",
            "tile_ratio": round(tile_ratio, 4),
            "x_candidates": [
                {"count": count, "score": round(score, 4)}
                for count, score in x_candidates[:6]
            ],
            "y_candidates": [
                {"count": count, "score": round(score, 4)}
                for count, score in y_candidates[:6]
            ]
        }

    # Conservative minimum confidence, but lower than V2 because pairwise
    # aspect matching itself is a strong validation signal.
    if (
        best_pair["x_score"] < 0.88 or
        best_pair["y_score"] < 0.88 or
        best_pair["pair_score"] < 1.15
    ):
        return 1, 1, {
            "detected": False,
            "reason": "grid_pair_confidence_low",
            "x_score": round(best_pair["x_score"], 4),
            "y_score": round(best_pair["y_score"], 4),
            "pair_score": round(best_pair["pair_score"], 4),
            "source_cell_ratio": round(best_pair["source_cell_ratio"], 4),
            "tile_ratio": round(best_pair["tile_ratio"], 4),
            "cols_candidate": best_pair["cols"],
            "rows_candidate": best_pair["rows"]
        }

    return best_pair["cols"], best_pair["rows"], {
        "detected": True,
        "reason": "compatible_grid_pair_detected",
        "x_score": round(best_pair["x_score"], 4),
        "y_score": round(best_pair["y_score"], 4),
        "pair_score": round(best_pair["pair_score"], 4),
        "source_cell_ratio": round(best_pair["source_cell_ratio"], 4),
        "tile_ratio": round(best_pair["tile_ratio"], 4),
        "ratio_error": round(best_pair["ratio_error"], 4),
        "cols": best_pair["cols"],
        "rows": best_pair["rows"],
        "total_faces": best_pair["total_faces"],
        "x_candidates": [
            {"count": count, "score": round(score, 4)}
            for count, score in x_candidates[:6]
        ],
        "y_candidates": [
            {"count": count, "score": round(score, 4)}
            for count, score in y_candidates[:6]
        ]
    }


def _extract_panel_faces(

    img: Image.Image,
    tile_width_cm: float,
    tile_height_cm: float,
    output_width_px: int,
    output_height_px: int,
    max_faces: int = 120
):
    """
    Extract individual tile faces from a multi-tile _pl panel.

    The crop boundaries are calculated from the inferred grid. A tiny inset
    removes legacy grout pixels from each face before resizing.
    """
    cols, rows, detection = _infer_panel_grid(
        img,
        tile_width_cm,
        tile_height_cm
    )

    if cols == 1 and rows == 1:
        return [
            fit_to_tile(img, output_width_px, output_height_px)
        ], detection

    face_images = []
    cell_width = img.width / cols
    cell_height = img.height / rows

    inset_x = max(0, round(cell_width * 0.008))
    inset_y = max(0, round(cell_height * 0.008))

    for row in range(rows):
        for col in range(cols):
            left = round(col * cell_width) + inset_x
            top = round(row * cell_height) + inset_y
            right = round((col + 1) * cell_width) - inset_x
            bottom = round((row + 1) * cell_height) - inset_y

            if right <= left or bottom <= top:
                continue

            face = img.crop((left, top, right, bottom))
            face_images.append(
                fit_to_tile(face, output_width_px, output_height_px)
            )

            if len(face_images) >= max_faces:
                break

        if len(face_images) >= max_faces:
            break

    if not face_images:
        return [
            fit_to_tile(img, output_width_px, output_height_px)
        ], {
            **detection,
            "detected": False,
            "reason": "no_valid_face_crops"
        }

    detection["extracted_face_count"] = len(face_images)
    detection["max_faces"] = max_faces

    return face_images, detection


def pick_face_index(row: int, col: int, face_count: int, rule: str, seed_key: str) -> int:
    if face_count <= 1:
        return 0

    rule = (rule or "repeat").lower()

    if rule == "repeat":
        return (row * face_count + col) % face_count

    seed = f"{seed_key}-{row}-{col}"
    h = hashlib.md5(seed.encode("utf-8")).hexdigest()
    return int(h, 16) % face_count


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "tile-panel-service",
        "service_version": SERVICE_VERSION
    }


@app.post("/create-panel")
def create_panel(req: PanelRequest):
    try:
        face_urls = normalize_face_urls(req.tile_face_urls)

        if not face_urls:
            raise HTTPException(status_code=400, detail="tile_face_urls is empty")

        if req.layout_pattern != "straight_grid":
            raise HTTPException(
                status_code=400,
                detail="Only straight_grid is supported in V1"
            )

        ppcm = int(req.output_ppcm or 10)

        tile_w_px = max(1, round(req.tile_width_cm * ppcm))
        tile_h_px = max(1, round(req.tile_height_cm * ppcm))

        target_w_px = max(1, round(req.target_width_cm * ppcm))
        target_h_px = max(1, round(req.target_height_cm * ppcm))

        grout_px = max(1, round((req.panel_grout_mm / 10.0) * ppcm))

        prepared_faces = []
        face_extraction_meta = []

        for url in face_urls:
            img = download_image(url)

            extracted_faces, extraction_meta = _extract_panel_faces(
                img=img,
                tile_width_cm=req.tile_width_cm,
                tile_height_cm=req.tile_height_cm,
                output_width_px=tile_w_px,
                output_height_px=tile_h_px,
                max_faces=max(1, min(120, int(req.random_face_count or 120)))
                if int(req.random_face_count or 1) > 1
                else 120
            )

            prepared_faces.extend(extracted_faces)
            face_extraction_meta.append(
                {
                    "source_url": url,
                    **extraction_meta
                }
            )

        cols = math.ceil(req.target_width_cm / req.tile_width_cm)
        rows = math.ceil(req.target_height_cm / req.tile_height_cm)

        full_w_px = cols * tile_w_px
        full_h_px = rows * tile_h_px

        mosaic = Image.new(
            "RGB",
            (full_w_px, full_h_px),
            parse_color(req.panel_grout_color)
        )

        face_count = len(prepared_faces)
        seed_key = f"{req.job_id}-{req.panel_key}"

        for row in range(rows):
            for col in range(cols):
                idx = pick_face_index(
                    row,
                    col,
                    face_count,
                    req.random_layout_rule,
                    seed_key
                )

                tile_img = prepared_faces[idx]
                x = col * tile_w_px
                y = row * tile_h_px

                mosaic.paste(tile_img, (x, y))

        excess_w_px = max(0, full_w_px - target_w_px)
        excess_h_px = max(0, full_h_px - target_h_px)

        if req.symmetry_mode == "center_equal_cut":
            left_crop_px = round(excess_w_px / 2)
        else:
            left_crop_px = 0

        if req.start_anchor == "top":
            top_crop_px = 0
        else:
            top_crop_px = round(excess_h_px / 2)

        panel = mosaic.crop(
            (
                left_crop_px,
                top_crop_px,
                left_crop_px + target_w_px,
                top_crop_px + target_h_px
            )
        )

        grout_rgb = parse_color(req.panel_grout_color)
        draw = ImageDraw.Draw(panel)

        for n in range(1, cols):
            x = n * tile_w_px - left_crop_px

            if 0 < x < target_w_px:
                draw.rectangle(
                    [
                        x - grout_px // 2,
                        0,
                        x + math.ceil(grout_px / 2) - 1,
                        target_h_px
                    ],
                    fill=grout_rgb
                )

        for n in range(1, rows):
            y = n * tile_h_px - top_crop_px

            if 0 < y < target_h_px:
                draw.rectangle(
                    [
                        0,
                        y - grout_px // 2,
                        target_w_px,
                        y + math.ceil(grout_px / 2) - 1
                    ],
                    fill=grout_rgb
                )

        buffer = io.BytesIO()
        panel.save(buffer, format="PNG")
        buffer.seek(0)

        upload_result = cloudinary.uploader.upload(
            buffer,
            folder=f"tile_panels/{req.job_id}",
            public_id=f"{req.panel_key}",
            overwrite=True,
            resource_type="image"
        )

        left_cut_cm = round(left_crop_px / ppcm, 2)
        right_cut_cm = round((excess_w_px - left_crop_px) / ppcm, 2)
        top_cut_cm = round(top_crop_px / ppcm, 2)
        bottom_cut_cm = round((excess_h_px - top_crop_px) / ppcm, 2)

        panel_meta = {
            "tile_width_cm": req.tile_width_cm,
            "tile_height_cm": req.tile_height_cm,
            "target_width_cm": req.target_width_cm,
            "target_height_cm": req.target_height_cm,
            "calculated_cols": cols,
            "calculated_rows": rows,
            "calculated_tile_count": cols * rows,
            "left_cut_cm": left_cut_cm,
            "right_cut_cm": right_cut_cm,
            "top_cut_cm": top_cut_cm,
            "bottom_cut_cm": bottom_cut_cm,
            "output_width_px": target_w_px,
            "output_height_px": target_h_px,
            "grout_px": grout_px,
            "random_layout_rule": req.random_layout_rule,
            "tile_face_mode": req.tile_face_mode,
            "random_face_count": req.random_face_count,
            "face_url_count": len(face_urls),
            "prepared_face_count": len(prepared_faces),
            "face_extraction": face_extraction_meta,
            "service_version": SERVICE_VERSION
        }

        return {
            "ok": True,
            "job_id": req.job_id,
            "panel_key": req.panel_key,
            "stage_tile_panel_url": upload_result.get("secure_url", ""),
            "panel_status": "panel_ready",
            "panel_version": "auto_pl_face_v3",
            "panel_meta": panel_meta
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
