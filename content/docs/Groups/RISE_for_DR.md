# RISE‑based DR Classifier — Detailed Code Walk‑Through

## Table of Contents
1. [Repository Structure](#repo)
2. [Data Pipeline `data.py`](#data)
3. [Aux Dataset `data_sport.py`](#datasport)
4. [Model Definition `model.py`](#model)
5. [Training Loop `train.py`](#train)
6. [RISE Explainer `rise.py`](#rise)
7. [Saliency CLI `explain.py`](#explain)
8. [RISE Examples](#examples)

---
## Method overview
To generate a saliency map for model's prediction, RISE queries black-box model on multiple randomly masked versions of input.
After all the queries are done we average all the masks with respect to their scores to produce the final saliency map. The idea behind this is that whenever a mask preserves important parts of the image it gets higher score, and consequently has a higher weight in the sum.
![](https://eclique.github.io/rep-imgs/RISE/rise-overview.png)
---
## [Link to our repository](https://github.com/AlexeyShulmin/xai/tree/main)

<a name="repo"></a>
## 1 · Repository Structure
```
src/
 ├ data.py          # APTOS CSV → eager‑loading dataset
 ├ data_sport.py    # tiny CIFAR‑like dataset for fast tests
 ├ model.py         # ResNet‑50 factory helper
 ├ train.py         # CLI training script (AdamW + cosine LR)
 ├ rise.py          # memory‑efficient RISE implementation
 └ explain.py       # generates & overlays saliency maps
```

---

<a name="data"></a>
## 2 · Data Pipeline — `data.py`
### Responsibilities
* Read **`train.csv`** (`id_code, diagnosis` or `filepaths, label`).
* Resolve each id to an image inside `img_dir` (tries `.png`, `.jpg`, `.jpeg`).
* **Eager‑load & transform** once → tensors cached in RAM.
* Provide `get_loaders(...)` that returns train / val `DataLoader`s.

### Key Classes & Functions
| Symbol | Purpose |
|--------|---------|
| `RetinopathyDataset` | Implements `__len__`, `__getitem__`; holds `self.images`, `self.labels`. |
| `_resolve_path` | Gracefully locates file whether path is absolute, relative, or stem only. |
| `DEFAULT_TRANSFORM` | `Resize` → `CenterCrop` 224 × 224 → `ToTensor` → Normalize. |
| `get_loaders` | Splits dataset `split_ratio` (default 0.8) and constructs `DataLoader`s. |

> **Usage example**  
> ```python
> train_loader, val_loader, classes = get_loaders(
>     csv_path="data/train.csv", img_dir="data", batch_size=32)
> ```

---

<a name="datasport"></a>
## 3 · Aux Dataset — `data_sport.py`
A minimal wrapper around **CIFAR‑10 formatted** datasets used to debug the pipeline quickly.  Interface mirrors `RetinopathyDataset` so you can swap loaders by changing one import.

---

<a name="model"></a>
## 4 · Model Definition — `model.py`
```python
model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
model.fc = nn.Linear(model.fc.in_features, num_classes)
```
Options:
* `pretrained=True/False`
* Function returns **unfrozen** model ready for fine‑tuning.

---

<a name="train"></a>
## 5 · Training Loop — `train.py`
CLI flags:
```
--csv        path to CSV
--img-dir    images root
--epochs     default 10
--batch-size default 16
--lr         default 1e‑4
--weights    checkpoint path
```

### Internals
1. Build loaders via `get_loaders`.  
2. Optimiser = **AdamW**, scheduler = **CosineAnnealingLR**.  
3. Loop: `train_one_epoch` → `evaluate`.  
4. Save best model when `val_acc` improves.  
5. `torch.manual_seed(42)` for reproducibility.

---

<a name="rise"></a>
## 6 · RISE Explainer — `rise.py`
### Improvements over original paper
* **Coarse masks** (`s×s`, default 7) ↑ bilinear → contiguous saliency blobs.
* **Mask streaming** (`batch` arg) → can run with GPU memory < 4 GB.
* **Optional Gaussian blur** for nicer overlays.

### Core Logic
```python
masks = self._upsample(torch.bernoulli(...))   # (N,1,H,W)
masked_batch = x * masks                       # broadcast
probs = softmax(model(masked_batch), 1)        # (N,C)
weights = probs[:, target].view(N,1,1,1)
saliency = (weights * masks).sum(0) / N
saliency = saliency / saliency.max()
```

---

<a name="explain"></a>
## 7 · Saliency CLI — `explain.py`
Command:
```bash
python -m src.explain \
  --weights models/best_resnet50.pt \
  --images data/val_samples \
  --outdir outputs/maps \
  --N 8000 --batch 128
```
Process:
1. For each image → preprocess (resize).  
2. Call `RISE.explain` (GPU or CPU).  
3. Overlay heat‑map with `matplotlib` (`jet` colormap, `alpha=0.5`).  
4. Save PNG as `{stem}_rise.png` in `outdir`.

---

<a name="examples"></a>
## 8 · RISE Example Gallery


| Original | RISE |
|----------|------|
| <img src="/RISE_images/air_hockey.jpg" width="200"/> | <img src="/RISE_images/air_hockey_rise.png" width="200"/> |
| <img src="/RISE_images/ampute_football.jpg" width="200"/> | <img src="/RISE_images/ampute_football_rise.png" width="200"/> |
| <img src="/RISE_images/archery.jpg" width="200"/> | <img src="/RISE_images/archery_rise.png" width="200"/> |
| <img src="/RISE_images/arm_wrestling.jpg" width="200"/> | <img src="/RISE_images/arm_wrestling_rise.png" width="200"/> |
| <img src="/RISE_images/axe_throwing.jpg" width="200"/> | <img src="/RISE_images/axe_throwing_rise.png" width="200"/> |


