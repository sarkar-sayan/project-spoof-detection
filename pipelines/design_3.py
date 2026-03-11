"""Design 3: feature-similarity-based verification.

Uses pretrained CNN and LBP models from `models.py` to extract features for
enrolment templates and probes, then compares them via Euclidean distance
with per-model thresholds.
"""

import glob
import os
from typing import List

import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances

from models import (
    PreProcess_img,
    get_LBP,
    get_model1,
    get_model3,
    get_models,
    gray_transform,
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")


def load_models() -> List[torch.nn.Module]:
    model1 = get_model1()
    model1.load_state_dict(torch.load(os.path.join(SAVED_MODELS_DIR, "model1.pth")))
    model1.eval()

    model3 = get_model3()
    model3.load_state_dict(torch.load(os.path.join(SAVED_MODELS_DIR, "model3.pth")))
    model3.eval()

    model9 = get_LBP()
    model9.load_state_dict(torch.load(os.path.join(SAVED_MODELS_DIR, "model9.pth")))
    model9.eval()

    return get_models(model1, model3, model9)


def extract_features(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        features = model(image)
    return features


def extract(models_list: List[torch.nn.Module], image: torch.Tensor) -> List[torch.Tensor]:
    feature_list: List[torch.Tensor] = []
    for model in models_list:
        if model.__class__.__name__ == "LBPNet":
            img = gray_transform(image)
            feature = extract_features(model, img)
        else:
            feature = extract_features(model, image)
        feature_list.append(feature)
    return feature_list


def compare_features(features1: torch.Tensor, features2: torch.Tensor) -> np.ndarray:
    return euclidean_distances(features1.cpu().numpy(), features2.cpu().numpy())


def Return_SinglePredClass(
    model: torch.nn.Module,
    image: torch.Tensor,
    feat_l: torch.Tensor,
    feat_r: torch.Tensor,
    th: float,
) -> int:
    x = 1
    feat = extract_features(model, image)
    s_left = compare_features(feat, feat_l)
    s_right = compare_features(feat, feat_r)
    if s_left < s_right:
        if s_left < th:
            x -= 1
    else:
        if s_right < th:
            x -= 1
    return x


def returnPredList(
    models_list: List[torch.nn.Module],
    image: torch.Tensor,
    features1: List[torch.Tensor],
    features2: List[torch.Tensor],
    threshold: List[float],
) -> List[int]:
    pred_list: List[int] = []
    for i, model in enumerate(models_list):
        if model.__class__.__name__ == "LBPNet":
            img = gray_transform(image)
            p = Return_SinglePredClass(model, img, features1[2], features2[2], threshold[2])
        elif "resnet" in model.__class__.__name__.lower():
            p = Return_SinglePredClass(model, image, features1[1], features2[1], threshold[1])
        else:
            p = Return_SinglePredClass(model, image, features1[0], features2[0], threshold[0])
        pred_list.append(p)
    return pred_list


def returnPredClass(pred_list: List[int]) -> int:
    r = sum(1 for x in pred_list if x == 0)
    return 0 if r >= 2 else 1


def main() -> None:
    models_list = load_models()

    subject = "043-M"
    input_image_path = "/content/drive/MyDrive/DataSets/Palmvein_h/Registration"
    threshold = [0.3, 0.3, 1.0]

    for root, dirs, _ in os.walk(input_image_path):
        if os.path.basename(root) == subject:
            for subdir in dirs:
                if subdir == "enrol":
                    subdir_path = os.path.join(root, subdir)
                    image_files = glob.glob(os.path.join(subdir_path, "*.png"))
                    for image_file in image_files:
                        if image_file.endswith("_L_1_1.png"):
                            image = PreProcess_img(image_file)
                            features1 = extract(models_list, image)
                        elif image_file.endswith("_R_1_1.png"):
                            image = PreProcess_img(image_file)
                            features2 = extract(models_list, image)
                    print(
                        "Threshold Features for Subject ",
                        os.path.basename(root),
                        "computed.",
                    )
                    print("Threshold Value for Similarity check is ", threshold)

    print("Remaining Images Test")
    frr = 0
    sfar = 0
    for root, dirs, _ in os.walk(input_image_path):
        if os.path.basename(root) == subject:
            for subdir in dirs:
                if subdir == "probe_spoof":
                    subdir_path = os.path.join(root, subdir)
                    image_files = glob.glob(os.path.join(subdir_path, "*.png"))
                    for image_file in image_files:
                        image = PreProcess_img(image_file)
                        pred = returnPredList(models_list, image, features1, features2, threshold)
                        pred_class = returnPredClass(pred)
                        if pred_class == 0:
                            print("Image Accepted. Same Subject ", os.path.basename(root))
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
                        pred = returnPredList(models_list, image, features1, features2, threshold)
                        pred_class = returnPredClass(pred)
                        if pred_class == 0:
                            print("Image Accepted. Same Subject ", os.path.basename(root))
                        else:
                            print("Image Rejected. Try Again")
                            frr += 1
                    print("Genuine Probes Rejected:", frr)
                    print("FRR% for Same User is:", frr / len(image_files))
                    frr = 0


if __name__ == "__main__":
    main()

