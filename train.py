r"""
Fine-tune YOLO26-seg on prepared data of Kvasir-SEG.

REQUIRES:
Dataset downloaded and structured in the Ultralytics layout that
data_prep.py provides

CHECK METRICS:
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; 
YOLO('models/best.pt').val(data='data/kvasir-seg/dataset.yaml', split='test')"

"""

import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent
DATA_YAML = ROOT_DIR / "data" / "kvasir-seg" / "dataset.yaml"
MODELS_DIR = ROOT_DIR / "models"


def main() -> None:
    if not DATA_YAML.exists():
        raise SystemExit()

    # COCO-pretrained segmentation checkpoint.
    model = YOLO("yolo26n-seg.pt")
    # patience: early-stop if val mAP plateaus for 25 epochs.
    model.train(data=str(DATA_YAML),
                epochs=100,
                imgsz=640,
                patience=25,
                seed=42)

    # Copy the best checkpoint to where app.py expects it.
    MODELS_DIR.mkdir(exist_ok=True)
    shutil.copy2(model.trainer.best, MODELS_DIR / "best.pt")
    print(f"\nDone. Weights at {MODELS_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
