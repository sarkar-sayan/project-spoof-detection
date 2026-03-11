"""Design 2: FRR, FAR, EER, and SFAR evaluation for the ensemble.

Uses pretrained models and SVM from `models.py` to evaluate normal operation
and spoofing attack scenarios.
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





def Real_Count(predictions: List[int]) -> int:
    return sum(1 for p in predictions if p == 0)


def main() -> None:
    model1, model3, model9, clf = load_models(SAVED_MODELS_DIR)
    originals = get_models(model1, model3, model9)
    modelsvm = [model1, clf, model9]

    print("NORMAL OPERATION MODE")
    nom_path = os.path.join(BASE_DIR, "datasets", "test", "enrol", "nom")
    image_files = glob.glob(os.path.join(nom_path, "*.png"))
    total1 = len(image_files)
    print("Total Genuine Inputs (Enrolment): ", total1)
    frr = 0
    for image_file in image_files:
        image = PreProcess_img(image_file)
        _, prediction = Return_Scores_all(originals, modelsvm, image, model9, clf)
        r = Real_Count(prediction)
        if r < 2:
            frr += 1  # type: ignore
    print("Genuine Inputs Rejected:", frr)
    print("FRR % is :", (frr * 100) / total1)  # type: ignore

    nom_path = os.path.join(BASE_DIR, "datasets", "test", "probe", "nom")
    image_files = glob.glob(os.path.join(nom_path, "*.png"))
    total2 = len(image_files)
    print("Total Real Probe Inputs by Attacker(Nom): ", total2)
    far = 0
    for image_file in image_files:
        image = PreProcess_img(image_file)
        _, prediction = Return_Scores_all(originals, modelsvm, image, model9, clf)
        r = Real_Count(prediction)
        if r >= 2:
            far += 1  # type: ignore
    print("Real Probes Accepted:", far)
    print("FAR % is :", (far * 100) / total2)  # type: ignore

    eer = ((frr / total1) + (far / total2)) / 2  # type: ignore
    print("EER % is :", eer * 100)

    print("\nSPOOFING ATTACK MODE")
    spoof_path = os.path.join(BASE_DIR, "datasets", "test", "probe", "attack")
    image_files = glob.glob(os.path.join(spoof_path, "*.png"))
    total3 = len(image_files)
    print("Total Spoof Inputs: ", total3)
    nom_count = 0
    for image_file in image_files:
        image = PreProcess_img(image_file)
        _, prediction = Return_Scores_all(originals, modelsvm, image, model9, clf)
        r = Real_Count(prediction)
        if r >= 2:
            nom_count += 1  # type: ignore
    print("Spoof Images Accepted:", nom_count)
    print("SFAR % is :", (nom_count * 100) / total3)  # type: ignore


if __name__ == "__main__":
    main()

