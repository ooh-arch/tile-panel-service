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
        "service": "tile-panel-service"
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

        for url in face_urls:
            img = download_image(url)
            prepared_faces.append(fit_to_tile(img, tile_w_px, tile_h_px))

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
            "face_url_count": len(face_urls)
        }

        return {
            "ok": True,
            "job_id": req.job_id,
            "panel_key": req.panel_key,
            "stage_tile_panel_url": upload_result.get("secure_url", ""),
            "panel_status": "panel_ready",
            "panel_version": "v1",
            "panel_meta": panel_meta
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
