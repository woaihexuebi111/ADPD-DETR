# Training and evaluation workflow

Run the commands below from the repository root.

## Dataset preparation

Prepare ComplexSea using the commands in [README.md](../README.md). Download SeaDronesSee ODv2 images and annotations into the structure shown in [Data access](DATA_ACCESS.md), then run the following command.

```bash
python scripts/prepare_data.py --dataset seadronessee --source-root /path/to/SeaDronesSee_ODv2 --output-root prepared_data/seadronessee --download-evaluation
```

The default `original` layout preserves category IDs 1–5 and reserves output slot 0, producing six model output slots. To use five foreground output slots, select `--class-layout foreground`. Train the teacher and both students with the resulting dataset YAML, then evaluate using the same YAML.

## Public COCO initialization

```bash
python scripts/prepare_initialization.py --download --output-root initialization
```

For offline preparation use `--official /path/to/rtdetr_r50vd_6x_coco_from_paddle.pth` in place of `--download`. This converts compatible official RT-DETR R50 COCO tensors into OWRT and current teacher initialization files. Its preparation reports record transferred and newly initialized tensors. Pass `--init-weights initialization/teacher_coco.pt` when training the teacher. New task-specific and shape-changed tensors are initialized by the target configuration. The training commands below initialize the teacher from the converted COCO checkpoint and the main student from compatible tensors in the trained teacher checkpoint. The lightweight student uses its model YAML initialization.

## ComplexSea training

```bash
python scripts/train_teacher.py --data prepared_data/complex_sea/data.yaml --init-weights initialization/teacher_coco.pt --device 0 --name complex_teacher
python scripts/train_adpd_distillation.py --teacher runs/complex_teacher/weights/best.pt --data prepared_data/complex_sea/data.yaml --init-weights runs/complex_teacher/weights/best.pt --device 0 --name complex_main
python scripts/train_l_adpd_distillation.py --teacher runs/complex_teacher/weights/best.pt --data prepared_data/complex_sea/data.yaml --device 0 --name complex_light
```

## SeaDronesSee training

```bash
python scripts/train_teacher.py --data prepared_data/seadronessee/data.yaml --init-weights initialization/teacher_coco.pt --device 0 --name sea_teacher
python scripts/train_adpd_distillation.py --teacher runs/sea_teacher/weights/best.pt --data prepared_data/seadronessee/data.yaml --init-weights runs/sea_teacher/weights/best.pt --no-condition-q --default-condition-q 0.91 --device 0 --name sea_main
python scripts/train_l_adpd_distillation.py --teacher runs/sea_teacher/weights/best.pt --data prepared_data/seadronessee/data.yaml --no-condition-q --default-condition-q 0.91 --device 0 --name sea_light
```

Each command defaults to 100 epochs, input size 640, batch 4, workers 4, seed 42, AdamW, learning rate 1e-4 and weight decay 1e-4. AMP is disabled. In this bundled trainer, `warmup_epochs=2000` controls 2000 warmup iterations. Change the device and batch size according to available memory and record those changes.

Each dataset uses one teacher training stage, one main-student distillation stage and one lightweight-student distillation stage. Both students use the trained teacher for supervision. The `--init-weights` argument controls the initial student parameters independently of teacher supervision.

## Distillation

The teacher uses detection loss plus reconstruction MSE with coefficient 0.1. The student objective contains DRD, TFD and PD. DRD weights are 0.05 for difference responses and 0.05 for filtration responses. TFD weight is 0.05, with alpha 0.5, beta 0.2, gamma 1.0 and spatial weights clipped to [1, 2]. The small-object prior uses boxes below 32 squared pixels at the training input scale, expanded by 1.5. PD uses classification weight 0.05 and regression weight 0.40, retaining the top 100 teacher queries above confidence 0.5.

The main student's enhanced P2 aligns with teacher P2. The P3-DDPM lightweight student aligns its layer-18 projected P3 feature with resized teacher P2. The teacher remains frozen during distillation. Deployment runs the student alone.

For ComplexSea, condition quantiles are selected from filename codes. E01 uses 0.99, E02 0.96, E03 0.93, E06 0.95, E10 0.91, E11 0.94, E12 0.92, and E13/E14/E15 0.93. The fallback is 0.97. SeaDronesSee uses the fixed 0.91 quantile supplied above.

Training selects checkpoints using validation COCO bbox AP. Prepared SeaDronesSee YAMLs select the official crowd-aware validator for checkpoint selection and standalone evaluation.

## Evaluation

```bash
python scripts/evaluate.py --weights runs/complex_main/weights/best.pt --data prepared_data/complex_sea/data.yaml --split test --device 0 --output outputs/complex_main_test
python scripts/evaluate.py --weights runs/complex_light/weights/best.pt --data prepared_data/complex_sea/data.yaml --split test --device 0 --output outputs/complex_light_test
python scripts/evaluate.py --weights runs/sea_main/weights/best.pt --data prepared_data/seadronessee/data.yaml --split val --device 0 --output outputs/sea_main_val
python scripts/evaluate.py --weights runs/sea_light/weights/best.pt --data prepared_data/seadronessee/data.yaml --split val --device 0 --output outputs/sea_light_val
```

Keep training metadata, results.csv, selected checkpoint, dataset preparation report and evaluation JSON for each experiment. The ComplexSea commands above evaluate the held-out test partition. Use a separate `--split val` command and output directory for validation evaluation. The output JSON records the evaluated partition.
