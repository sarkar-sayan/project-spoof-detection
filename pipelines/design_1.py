"""Design 1: threshold-based verification using pretrained backbone ensemble.

Loads pretrained CNN and LBP models plus an SVM (all defined in `models.py`)
and performs subject-specific threshold estimation and verification.
"""

import glob
import os
import pickle
from typing import List, Tuple

import torch  # type: ignore
import torch.nn.functional as F  # type: ignore

import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from lib.train_ensemble import get_models  # type: ignore
from lib.utils import PreProcess_img, Return_Scores_all, load_models  # type: ignore





def main() -> None:
    model1, model3, model9, clf = load_models(SAVED_MODELS_DIR)
    originals = get_models(model1, model3, model9)
    modelsvm = [model1, clf, model9]

    subject = "043-M"
    input_image_path = os.path.join(BASE_DIR, "datasets", "Registration")
    th: List[float] = [0.0, 0.0, 0.0]
    frr: float = 0.0
    sfar: float = 0.0

    for root, dirs, _ in os.walk(input_image_path):
        if os.path.basename(root) == subject:
            for subdir in dirs:
                if subdir == "enrol":
                    subdir_path = os.path.join(root, subdir)
                    image_files = glob.glob(os.path.join(subdir_path, "*.png"))
                    for image_file in image_files:
                        image = PreProcess_img(image_file)
                        score: List[float]
                        score, _ = Return_Scores_all(
                            originals, modelsvm, image, model9, clf
                        )
                        for i in range(len(score)):
                            th[i] = th[i] + score[i] / 2  # type: ignore

    print(f"Threshold Values for Subject {subject} are: {th}")

    print("Remaining Images Test")
    for root, dirs, _ in os.walk(input_image_path):
        if os.path.basename(root) == subject:
            for subdir in dirs:
                if subdir == "probe_spoof":
                    subdir_path = os.path.join(root, subdir)
                    image_files = glob.glob(os.path.join(subdir_path, "*.png"))
                    for image_file in image_files:
                        image = PreProcess_img(image_file)
                        accept: int = 0
                        score, _ = Return_Scores_all(
                            originals, modelsvm, image, model9, clf
                        )
                        for i in range(len(score)):
                            if score[i] > th[i]:
                                accept = accept + 1  # type: ignore
                        if accept >= 2:
                            print(
                                "Image Accepted. Same Subject ",
                                os.path.basename(root),
                            )
                            sfar = sfar + 1.0  # type: ignore
                        else:
                            print("Image Rejected. Try Again")
                    print("Spoof Probes Accepted:", sfar)
                    print("SFAR% for Same User is:", sfar / len(image_files))  # type: ignore
                    sfar = 0.0
                elif subdir == "probe_real":
                    subdir_path = os.path.join(root, subdir)
                    image_files = glob.glob(os.path.join(subdir_path, "*.png"))
                    for image_file in image_files:
                        image = PreProcess_img(image_file)
                        accept: int = 0
                        score, _ = Return_Scores_all(
                            originals, modelsvm, image, model9, clf
                        )
                        for i in range(len(score)):
                            if score[i] > th[i]:
                                accept = accept + 1  # type: ignore
                        if accept >= 2:
                            print(
                                "Image Accepted. Same Subject ",
                                os.path.basename(root),
                            )
                        else:
                            print("Image Rejected. Try Again")
                            frr = frr + 1.0  # type: ignore
                    print("Genuine Probes Rejected:", frr)
                    print("FRR% for Same User is:", frr / len(image_files))  # type: ignore
                    frr = 0.0


if __name__ == "__main__":
    main()

