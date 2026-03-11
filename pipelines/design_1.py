"""Design 1: threshold-based verification using pretrained backbone ensemble.

Loads pretrained CNN and LBP models plus an SVM (all defined in `models.py`)
and performs subject-specific threshold estimation and verification.
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


def main() -> None:
    model1, model3, model9, clf = load_models()
    originals = get_models(model1, model3, model9)
    modelsvm = [model1, clf, model9]

    subject = "043-M"
    input_image_path = "/content/drive/MyDrive/DataSets/Palmvein_h/Registration"
    th = [0.0, 0.0, 0.0]
    frr = 0
    sfar = 0

    for root, dirs, _ in os.walk(input_image_path):
        if os.path.basename(root) == subject:
            for subdir in dirs:
                if subdir == "enrol":
                    subdir_path = os.path.join(root, subdir)
                    image_files = glob.glob(os.path.join(subdir_path, "*.png"))
                    for image_file in image_files:
                        image = PreProcess_img(image_file)
                        score, _ = Return_Scores_all(
                            originals, modelsvm, image, model9, clf
                        )
                        for i in range(len(score)):
                            th[i] = th[i] + score[i] / 2

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
                        accept = 0
                        score, _ = Return_Scores_all(
                            originals, modelsvm, image, model9, clf
                        )
                        for i in range(len(score)):
                            if score[i] > th[i]:
                                accept += 1
                        if accept >= 2:
                            print(
                                "Image Accepted. Same Subject ",
                                os.path.basename(root),
                            )
                            sfar += 1
                        else:
                            print("Image Rejected. Try Again")
                    print("Spoof Probes Accepted:", sfar)
                    print("SFAR% for Same User is:", sfar / len(image_files))
                    sfar = 0
                elif subdir == "probe_real":
                    subdir_path = os.path.join(root, subdir)
                    image_files = glob.glob(os.path.join(subdir_path, "*.png"))
                    for image_file in image_files:
                        image = PreProcess_img(image_file)
                        accept = 0
                        score, _ = Return_Scores_all(
                            originals, modelsvm, image, model9, clf
                        )
                        for i in range(len(score)):
                            if score[i] > th[i]:
                                accept += 1
                        if accept >= 2:
                            print(
                                "Image Accepted. Same Subject ",
                                os.path.basename(root),
                            )
                        else:
                            print("Image Rejected. Try Again")
                            frr += 1
                    print("Genuine Probes Rejected:", frr)
                    print("FRR% for Same User is:", frr / len(image_files))
                    frr = 0


if __name__ == "__main__":
    main()

