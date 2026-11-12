# Dataset access

Run the commands below from the repository root.

## Complex Sea-Condition Dataset

The dataset is published on Mendeley Data as Version 1. The dataset page provides all three split archives. [Mendeley Data dataset DOI](https://doi.org/10.17632/28zjh6bmjv.1).

The same split archives are also available through the Baidu Netdisk links below.

| Split | Download archive | Images | Objects | Baidu Netdisk | Extraction code |
|---|---|---:|---:|---|---|
| Training | `ComplexSea_train.zip` | 1992 | 7427 | [Baidu Netdisk](https://pan.baidu.com/s/1ezwpyF_55x1CSxWu0rnJ8w?pwd=c6zy) | `c6zy` |
| Validation | `ComplexSea_val.zip` | 380 | 1353 | [Baidu Netdisk](https://pan.baidu.com/s/1FAapaAyH85IhuMBusiz_yA?pwd=g7n2) | `g7n2` |
| Test | `ComplexSea_test.zip` | 429 | 1667 | [Baidu Netdisk](https://pan.baidu.com/s/16HGiINk9d9mdoXvUzI_2eg?pwd=3sne) | `3sne` |

Download all three archives and extract them into the same parent directory. They assemble a shared `formal_complex_sea_public/` root containing `images/train`, `images/val`, `images/test` and their matching label directories. Preserve the supplied partitions and file names.

Verify the downloaded split archive checksums against [data/downloads.json](../data/downloads.json).

Select the extracted directory containing `images/` and `labels/` when supplying a dataset root. The single detection class is `0: person_overboard` (man overboard).

```text
dataset_root/
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
```

| Split | Images | Objects |
|---|---:|---:|
| Train | 1992 | 7427 |
| Val | 380 | 1353 |
| Test | 429 | 1667 |
| Total | 2801 | 10447 |

YOLO rows are `class_id center_x center_y width height`, normalized to image dimensions. Preserve file names and the supplied split; filename environment codes are used for condition-adaptive teacher guidance. `data/splits/manifest.json` and the split lists describe the public ZIP version. Do not merge the validation and test sets.

```bash
python scripts/verify_dataset.py --data-root /path/to/dataset_root
python scripts/prepare_data.py --dataset complex_sea --source-root /path/to/dataset_root --output-root prepared_data/complex_sea
```

The Complex Sea-Condition Dataset is distributed under CC BY 4.0. Cite the dataset when using the data.

Liu, D., 2026. Complex Sea-Condition Dataset. Mendeley Data, V1. https://doi.org/10.17632/28zjh6bmjv.1.

## SeaDronesSee ODv2

Official data: https://seadronessee.cs.uni-tuebingen.de/

Official evaluation resources: https://github.com/Ben93kie/SeaDronesSee/tree/main/OD/evaluation%20of%20OD%20V2

Download Object Detection V2 training and validation images and COCO annotations. Arrange the extracted official files as:

```text
SeaDronesSee_ODv2/
  images/train/
  images/val/
  annotations/instances_train.json
  annotations/instances_val.json
```

Keep image names consistent with the annotation JSON. Run:

```bash
python scripts/prepare_data.py --dataset seadronessee --source-root /path/to/SeaDronesSee_ODv2 --output-root prepared_data/seadronessee --download-evaluation
```

This creates training-format labels, a portable dataset YAML, and the official crowd-aware validation ground truth. Images are hardlinked, so put source and prepared data on the same filesystem. Use `--image-mode copy` if hardlinks are unavailable; source images are never edited. For offline use, replace `--download-evaluation` with `--evaluation-annotations /path/to/instances_val_iscrowd.json`.

| Model output slot | Meaning | Official evaluation ID |
|---|---|---|
| 0 | Reserved ignored slot | Excluded |
| 1 | Swimmer | 1 |
| 2 | Boat | 2 |
| 3 | Jetski | 3 |
| 4 | Life-saving appliances | 4 |
| 5 | Buoy | 5 |

The default `--class-layout original` preparation preserves official category IDs 1–5 and reserves output slot 0, producing six model output slots. The optional `foreground` layout maps the five foreground categories to model IDs 0–4. Use the same class layout and dataset YAML for teacher training, student training and evaluation.

Final local scoring uses official validation ground truth with crowd regions retained. The official test split has no public ground truth and is scored through the benchmark service. Follow the official dataset license and citation instructions.

## Training and evaluation

Training and evaluation commands are provided in [Training guide](TRAINING.md).
