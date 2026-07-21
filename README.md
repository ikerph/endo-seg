# EndoTector

**Polyp segmentation pipeline: dataset → training → demo**

<p align="center"><img src="https://raw.githubusercontent.com/ikerph/endo-tector/main/assets/gifs/demo_yoloseg.gif" alt="YOLO26-seg segmenting a polyp in a real colonoscopy video" width="560"></p>

> **Research tool — NOT a medical device.** 

The pipeline works end-to-end. Moreover, it achieves great results (mAP@50 ~0.95, median Dice ~0.96) despite being a light model (6 MB). It opens the door to real-time use.

YOLO-seg provides `box + mask` at the same time.

On the other hand, the test set has only 100 samples, so merging Kvasir-SEG with other datasets would help the model generalize.

## Pipeline

```
data_prep.py          train.py                 app.py
download Kvasir-SEG   fine-tune YOLO26-seg     Gradio demo:
+ masks, polygons   -> 100 epochs          ->  image in, mask +
+ 80/10/10 split        models/best.pt         box
```

`YOLO26-seg` fine-tuned on [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/).

## Results

Test split (100 images) on an RTX 3060 (6 GB):

| Metric      | Box   | Mask  |
| ----------- | ----- | ----- |
| Precision   | 0.902 | 0.902 |
| Recall      | 0.906 | 0.906 |
| mAP@50      | 0.948 | 0.948 |
| mAP@50-95   | 0.792 | 0.805 |

Mean **Dice 0.888** (median 0.961) at conf 0.25. Reproduce the mAP with `yolo segment val model=models/best.pt data=data/kvasir-seg/dataset.yaml split=test`.

## Test the project

The trained weights (`models/best.pt`) come with the repo, so the demo runs out of the box:

```bash
# Environment (PowerShell)
py -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt

# Demo, opens the Gradio UI in your browser
./.venv/Scripts/python.exe app.py
```

Reproduce the model (optional, needs a GPU for the training of `YOLO-seg`):

```bash
./.venv/Scripts/python.exe data_prep.py   # download Kvasir-SEG + build YOLO labels
./.venv/Scripts/python.exe train.py        # fine-tune, models/best.pt
```

## License

Data, CC-BY-4.0 (Simula): training on [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/). Demo clip from [Hyper-Kvasir](https://datasets.simula.no/hyper-kvasir/).

---

<sub>Built and engineered by Iker Pacheco Herrero, Biomedical Engineer.</sub>
