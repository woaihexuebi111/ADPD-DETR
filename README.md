# ADPD-DETR Supplementary Source Code

This source-only package implements the manuscript's reconstruction-guided
teacher and the deployable ADPD-DETR and L-ADPD-DETR students. It includes the
modified Ultralytics/RT-DETR framework, training, DRD/TFD/PD distillation, COCO
evaluation and model profiling. Datasets, weights and historical result files
are not included. This revised implementation has not been retrained to verify
the numerical results in the manuscript; it is not a result-reproduction capsule.

## Models

| Model | YAML under `ultralytics/cfg/models/ADPD/` | Backbone | Difference branch | Scales |
|---|---|---|---|---|
| Teacher | `ADPD-Teacher-R50.yaml` | ResNet-50 | Reconstruction ADPM on P2 | P2-P5 |
| ADPD-DETR | `ADPD-DETR-R50.yaml` | ResNet-50 | RH-free DDPM on P2 | P2-P5 |
| L-ADPD-DETR | `L-ADPD-DETR-FasterNet.yaml` | FasterNet-based LFENet/LFEB | RH-free DDPM on P3 | P3-P5 |

Both students remove image reconstruction. The lightweight backbone is a
modified FasterNet-based implementation, not the unmodified official
FasterNet-T1 or FasterNet-L. Model YAMLs default to one detection class;
training uses the class count from the dataset YAML. The teacher checkpoint
must have the same detection classes as its student.

## Installation

The manuscript reference platform is Ubuntu 22.04, Python 3.10, PyTorch 2.3.0,
CUDA 12.1, and an NVIDIA RTX 4090. This is the target environment, not a claim
that the entire reference platform has been revalidated for this revision.

```bash
conda env create -f environment.yml
conda activate adpd-detr
```

For a Python 3.10 virtual environment, install the CUDA 12.1 PyTorch wheels
before the remaining pinned dependencies:

```bash
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Run commands from the repository root. Use the bundled `ultralytics` directory;
do not replace it with a separately installed upstream framework. Load only
trusted checkpoints, because PyTorch model checkpoints can contain executable
serialized Python objects.

## Data

Edit `path` in `configs/datasets/complex_sea.yaml` after obtaining the data.
The directory format is `images/{train,val,test}` and
`labels/{train,val,test}`. YOLO label rows contain normalized coordinates:

```text
class_id center_x center_y width height
```

The self-constructed data and compatible checkpoints are available by request
subject to the conditions in `DATA_ACCESS.md`. Contact the corresponding
author using the address in the manuscript. No access credentials are bundled.

Condition-adaptive filtration is used only by the frozen teacher during
distillation. The filename must retain the supplied environment code:

| Codes | Condition | q |
|---|---|---:|
| E01 | Glare | 0.99 |
| E02 | Rainy | 0.96 |
| E03 | Fog | 0.93 |
| E06 | Night | 0.95 |
| E10 | Oil pollution | 0.91 |
| E11 | White waves | 0.94 |
| E12 | Wake | 0.92 |
| E13 / E14 / E15 | Birds / buoys / floating objects | 0.93 |

For other conditions or filenames without a code, the fallback is 0.97.
For SeaDronesSee distillation use `--no-condition-q --default-condition-q 0.91`
and a dataset YAML with its five detection classes. This package does not
include the SeaDronesSee conversion, original ignored-region annotations or a
cross-paper metric-normalization procedure. It must not be used to claim
verification of the SeaDronesSee table without those evaluation artifacts.

## Training

All three entries default to 100 epochs, 640 x 640 input, batch 4, workers 4,
seed 42, AdamW, learning rate 1e-4 and weight decay 1e-4. Mosaic and MixUp are
disabled. AMP is off by default. In this bundled trainer, `warmup_epochs=2000`
is interpreted as 2000 warmup iterations, not 2000 epochs.

```bash
python scripts/train_teacher.py --data configs/datasets/complex_sea.yaml --device 0
python scripts/train_adpd_distillation.py --teacher runs/adpd_teacher/weights/best.pt --data configs/datasets/complex_sea.yaml --device 0
python scripts/train_l_adpd_distillation.py --teacher runs/adpd_teacher/weights/best.pt --data configs/datasets/complex_sea.yaml --device 0
```

Use `--init-weights /path/to/trusted_initialization.pt` if an initialization
checkpoint is available; otherwise the selected student starts from its YAML
initialization. Only single-device distillation is supported. Checkpoints go
under `runs/`, and student run metadata under `reports/`.

The teacher uses detection loss plus 0.1 times reconstruction MSE. The student
uses detection loss plus these three terms:

- DRD: 0.05 times difference-response Smooth L1 plus 0.05 times filtration
  Smooth L1. Teacher D is min-max normalized per image; binary B is resized
  directly. Student maps retain gradients.
- TFD: weight 0.05, with spatial weights clipped to [1, 2], alpha=0.5,
  beta=0.2 and gamma=1.0. Ground-truth boxes below 32 x 32 pixels at the
  training input scale form the small-target prior, expanded by 1.5.
- PD: classification weight 0.05 and regression weight 0.40, top 100 teacher
  queries with confidence at least 0.5. Classification uses BCE without
  temperature scaling. Regression uses teacher-confidence normalization.

The main student aligns its enhanced P2 with teacher P2. The lightweight student
uses layer 18, the existing 1x1 projection of DDPM-enhanced P3 to 256 channels,
and spatially resizes teacher P2 to this scale. This is one TFD term, with no
extra P2/P3 mixture or duplicated loss. No auxiliary BoxKD is used.

Disable a distillation branch with `--no-drd`, `--no-tfd` or `--no-pd` for
ablation. These options change supervision, not the deployed student structure.
Best checkpoint selection uses actual pycocotools bbox AP on the validation
split, not 0.1*AP50+0.9*AP. Training logs retain framework precision/recall
statistics; AP columns and checkpoint selection use COCO values.

## Evaluation

```bash
python scripts/evaluate.py --weights /path/to/student_best.pt --data configs/datasets/complex_sea.yaml --split val --imgsz 640 --device 0 --output outputs/evaluation
```

Use `--split test` only for final held-out evaluation. `--output` is a directory.
It receives ground truth, predictions and `<checkpoint>_<split>_metrics.json`.
Metrics are reported as fractions in [0, 1], not percentage values. AP uses
IoU 0.50:0.05:0.95, COCO maxDets=100, and APS uses original-image annotation
area below 32 squared pixels. Predictions are evaluated in memory without
appending old text files. There is no numerical conversion to another paper's
results and no hard-coded manuscript AP values.

The framework's `RTDETR.predict` API remains available for image inference;
there is no additional top-level `predict.py` script.

## Profiling

```bash
python scripts/model_profile.py --model ultralytics/cfg/models/ADPD/ADPD-DETR-R50.yaml --imgsz 640
python scripts/model_profile.py --model ultralytics/cfg/models/ADPD/L-ADPD-DETR-FasterNet.yaml --imgsz 640
```

Profiling uses the fused inference model and pinned THOP; both parameter count
and GFLOPs are computed, not substituted from the manuscript. Operator
coverage, fusion and class count must match when comparing profiles. Full
retraining and validation of manuscript AP, GFLOPs and FPS remain necessary;
source alignment alone does not establish numerical reproduction.

## Contents and license

This revision does not
include `MODEL_ARCHITECTURE.md`, a separate prediction script, historical
metrics, checkpoints, private data or local test artifacts.

The modified framework is derived from Ultralytics 8.0.201 under AGPL-3.0.
See `LICENSE` and `THIRD_PARTY_NOTICES.md`. Cite the associated article when
its bibliographic information is available.
