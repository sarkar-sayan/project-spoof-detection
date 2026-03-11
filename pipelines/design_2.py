"""Design 2: FRR, FAR, EER, and SFAR evaluation for the ensemble.

Uses pretrained models and SVM from `models.py` to evaluate normal operation
and spoofing attack scenarios.
"""

import glob
import os
import pickle
from typing import List, Tuple

import torch
import torch.nn.functional as F

from models import (
    PreProcess_img,
    get_LBP,
    get_model1,
    get_model3,
    get_models,
    get_weighted_score_img,
    normalisation,
    gray_transform,
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")


def load_models() -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, object]:
    model1 = get_model1()
    model1.load_state_dict(torch.load(os.path.join(SAVED_MODELS_DIR, "model1.pth")))
    model1.eval()

    model3 = get_model3()
    model3.load_state_dict(torch.load(os.path.join(SAVED_MODELS_DIR, "model3.pth")))
    model3.eval()

    model9 = get_LBP()
    model9.load_state_dict(torch.load(os.path.join(SAVED_MODELS_DIR, "model9.pth")))
    model9.eval()

    with open(os.path.join(SAVED_MODELS_DIR, "svm.pk1"), "rb") as file:
        clf = pickle.load(file)

    return model1, model3, model9, clf


def Nom_Score(model: torch.nn.Module, image: torch.Tensor) -> Tuple[float, int]:
    model.eval()
    with torch.no_grad():
        output = model(image)
    probabilities = F.softmax(output, dim=1)
    confidence_scores, predicted_classes = torch.max(probabilities, dim=1)
    class_index = predicted_classes.item()
    cs = confidence_scores.item()
    if class_index != 0:
        return 1 - cs, class_index
    return cs, class_index


def Return_Scores_all(
    originals: List[torch.nn.Module],
    models_list: List[object],
    image: torch.Tensor,
    lbp_model: torch.nn.Module,
    clf: object,
) -> Tuple[List[float], List[int]]:
    scores: List[float] = []
    preds: List[int] = []
    for model in models_list:
        if model is lbp_model:
            img_gray = gray_transform(image)
            c, p = Nom_Score(model, img_gray)
        elif model is clf:
            img = get_weighted_score_img(originals, image)
            conf = clf.decision_function(img)
            conf = normalisation(conf)
            c = conf.tolist()[0]
            pred = clf.predict(img)
            p = pred.tolist()[0]
        else:
            c, p = Nom_Score(model, image)
        scores.append(c)
        preds.append(p)
    return scores, preds


def Real_Count(predictions: List[int]) -> int:
    return sum(1 for p in predictions if p == 0)


def main() -> None:
    model1, model3, model9, clf = load_models()
    originals = get_models(model1, model3, model9)
    modelsvm = [model1, clf, model9]

    print("NORMAL OPERATION MODE")
    nom_path = "/content/drive/MyDrive/DataSets/Palmvein_h/test/enrol/nom"
    image_files = glob.glob(os.path.join(nom_path, "*.png"))
    total1 = len(image_files)
    print("Total Genuine Inputs (Enrolment): ", total1)
    frr = 0
    for image_file in image_files:
        image = PreProcess_img(image_file)
        _, prediction = Return_Scores_all(originals, modelsvm, image, model9, clf)
        r = Real_Count(prediction)
        if r < 2:
            frr += 1
    print("Genuine Inputs Rejected:", frr)
    print("FRR % is :", (frr * 100) / total1)

    nom_path = "/content/drive/MyDrive/DataSets/Palmvein_h/test/probe/nom"
    image_files = glob.glob(os.path.join(nom_path, "*.png"))
    total2 = len(image_files)
    print("Total Real Probe Inputs by Attacker(Nom): ", total2)
    far = 0
    for image_file in image_files:
        image = PreProcess_img(image_file)
        _, prediction = Return_Scores_all(originals, modelsvm, image, model9, clf)
        r = Real_Count(prediction)
        if r >= 2:
            far += 1
    print("Real Probes Accepted:", far)
    print("FAR % is :", (far * 100) / total2)

    eer = ((frr / total1) + (far / total2)) / 2
    print("EER % is :", eer * 100)

    print("\nSPOOFING ATTACK MODE")
    spoof_path = "/content/drive/MyDrive/DataSets/Palmvein_h/test/probe/attack"
    image_files = glob.glob(os.path.join(spoof_path, "*.png"))
    total3 = len(image_files)
    print("Total Spoof Inputs: ", total3)
    nom_count = 0
    for image_file in image_files:
        image = PreProcess_img(image_file)
        _, prediction = Return_Scores_all(originals, modelsvm, image, model9, clf)
        r = Real_Count(prediction)
        if r >= 2:
            nom_count += 1
    print("Spoof Images Accepted:", nom_count)
    print("SFAR % is :", (nom_count * 100) / total3)


if __name__ == "__main__":
    main()

