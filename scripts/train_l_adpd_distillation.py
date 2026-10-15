"""Train the three-scale L-ADPD-DETR student with the same DRD/TFD/PD objective."""

import sys

from train_adpd_distillation import ROOT, main


if __name__ == "__main__":
    # Layer 18 projects DDPM-enhanced P3 to the teacher P2' channel width.
    defaults = ["--student", str(ROOT / "ultralytics/cfg/models/ADPD/L-ADPD-DETR-FasterNet.yaml"),
                "--student-feature-layer", "18", "--teacher-feature-layer", "8",
                "--name", "l_adpd_student"]
    main(defaults + sys.argv[1:])
