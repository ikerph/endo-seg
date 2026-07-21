"""
Download of Kvasir-SEG dataset and its conversion into the
YOLO segmentation format.

Ultralytics expects this structure:

data/kvasir-seg/
├── dataset.yaml
├── images/train/<stem>.jpg   (and val/ and test/)
└── labels/train/<stem>.txt   (and val/ and test/) one polygon per line: "0 x1 y1 x2 y2..."
"""

import random
import shutil
import ssl
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import cv2

DATASET_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"

ROOT_DIR = Path(__file__).resolve().parent
RAW_DIR = ROOT_DIR / "data" / "_raw"       # download + extraction
ZIP_PATH = RAW_DIR / "kvasir-seg.zip"
OUT_DIR = ROOT_DIR / "data" / "kvasir-seg" # YOLO-formatted output

SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42
MIN_AREA_FRAC = 0.0005  # ignore mask specks smaller than this fraction of the image


def download_kvasir() -> Path:
    """Download and extract Kvasir-SEG into data/_raw/. Returns extracted root."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse a possible previous extraction, looks for a folder containing images/ and masks/.
    for images in RAW_DIR.rglob("images"):
        if images.is_dir() and (images.parent / "masks").is_dir():
            print(f"Using Kvasir-SEG at {images.parent}")
            return images.parent

    if not ZIP_PATH.exists():
        print(f"Downloading {DATASET_URL}")
        req = urllib.request.Request(DATASET_URL, headers={"User-Agent": "endo-tector"})
        try:
            resp = urllib.request.urlopen(req)
        except urllib.error.URLError:
            # Unverified download (it's a public dataset, it is not a problem)
            ctx = ssl._create_unverified_context()
            resp = urllib.request.urlopen(req, context=ctx)
        with resp, open(ZIP_PATH, "wb") as f:
            shutil.copyfileobj(resp, f)

    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(RAW_DIR)

    for images in RAW_DIR.rglob("images"):
        if images.is_dir() and (images.parent / "masks").is_dir():
            return images.parent
    raise RuntimeError(f"Images or masks not found at {RAW_DIR}")


def mask_to_yolo_polygons(mask_path: Path) -> list[str]:
    """Convert one binary mask into YOLO polygon label lines (normalized coords)."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    h, w = mask.shape
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)  # pixel > thresh 127 -> polyp

    # Outer contour of each polyp (a mask can contain several polyps).
    # MASK CONTOUR; List of positions (x,y) of mask border corners
    # [CV2] RETR_EXTERNAL: Just store outside borders of the polyp, not internal hollows
    # [CV2] CHAIN_APPROX_SIMPLE: Straight line compression (stores lines instead of each coord)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = MIN_AREA_FRAC * h * w
    lines = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:  # noise skip
            continue
        pts = contour.reshape(-1, 2)  # (N, 2) array of absolute (x, y) vertices
        if len(pts) < 3:  # a valid polygon needs at least 3 vertices
            continue
        coord_parts = []
        for x, y in pts:  # absolute pixels -> coords relative to image size
            coord_parts.append(f"{x / w:.6f} {y / h:.6f}")
        # Class id 0 (polyp) + flattened polygon: "0 0.412 0.232 0.321 ..."
        lines.append("0 " + " ".join(coord_parts))
    return lines


def build_dataset() -> None:
    """Download, split 80/10/10, convert every mask, and build the YOLO dataset structure."""
    src_root = download_kvasir()
    images_dir = src_root / "images"
    masks_dir = src_root / "masks"

    image_of = {}
    for p in images_dir.iterdir():
        is_image = p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        has_mask = (masks_dir / p.name).exists()
        if is_image and has_mask:
            image_of[p.stem] = p # Valid img + mask

    # Reproducible split: sort, shuffle with a fixed seed, slice 80/10/10.
    stems = sorted(image_of)
    random.Random(SEED).shuffle(stems)
    n_train = int(SPLITS["train"] * len(stems))
    n_val = int(SPLITS["val"] * len(stems))
    split_stems = {
        "train": stems[:n_train],
        "val": stems[n_train:n_train + n_val],
        "test": stems[n_train + n_val:],
    }

    # Clean old outputs.
    for sub in ("images", "labels"):
        shutil.rmtree(OUT_DIR / sub, ignore_errors=True)

    for split, split_items in split_stems.items():
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        for stem in split_items: # Copy and convert mask into polygon-label
            img_src = image_of[stem]
            shutil.copy2(img_src, OUT_DIR / "images" / split / img_src.name)
            lines = mask_to_yolo_polygons(masks_dir / img_src.name)
            (OUT_DIR / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lines), encoding="utf-8"
            )

    # dataset.yaml, the config that Ultralytics reads at training.
    (OUT_DIR / "dataset.yaml").write_text(
        f"path: {OUT_DIR.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: polyp\n",
        encoding="utf-8",
    )

    print("\nDataset ready:")

if __name__ == "__main__":
    build_dataset()
