"""Training script using shared model definitions from `models.py`.

Trains DenseNet, ResNet, VGG, and LBP models for spoof detection, then fits
an SVM on concatenated logits and saves all weights under `saved_models/`.
"""

import os
import pickle

import torch
from sklearn import svm
from sklearn.metrics import accuracy_score

from models import (
    Cal_Confidence,
    PreProcess_img,
    criterion,
    data_transforms,
    get_LBP,
    get_TVT,
    get_model1,
    get_model3,
    get_model7,
    get_models,
    get_weighted_score_ft,
    get_weighted_score_img,
    lbp_transforms,
    lr,
    normalisation,
    num_epoch,
    test_acc,
    train_model,
)


DATA_ROOT = "/content/drive/MyDrive/DataSets/Palmvein_h/"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)


def main() -> None:
    trainset, valset, testset = get_TVT(DATA_ROOT, data_transforms)

    img_path = os.path.join(
        DATA_ROOT,
        "test/probe/nom/041_R_2_4.png",
    )
    img = PreProcess_img(img_path)

    model1 = get_model1()
    opt1 = torch.optim.Adam(model1.parameters(), lr=lr)
    sch1 = torch.optim.lr_scheduler.StepLR(opt1, step_size=9, gamma=0.3)
    model1 = train_model(trainset, valset, model1, criterion, opt1, sch1, num_epoch)

    torch.save(model1.state_dict(), os.path.join(SAVED_MODELS_DIR, "model1.pth"))

    model1 = get_model1()
    model1.load_state_dict(torch.load(os.path.join(SAVED_MODELS_DIR, "model1.pth")))
    model1.eval()

    Cal_Confidence(model1, img)
    print(test_acc(model1, testset))

    model3 = get_model3()
    opt3 = torch.optim.Adam(model3.parameters(), lr=lr)
    sch3 = torch.optim.lr_scheduler.StepLR(opt3, step_size=9, gamma=0.4)
    model3 = train_model(trainset, valset, model3, criterion, opt3, sch3, num_epoch)

    torch.save(model3.state_dict(), os.path.join(SAVED_MODELS_DIR, "model3.pth"))

    model3 = get_model3()
    model3.load_state_dict(torch.load(os.path.join(SAVED_MODELS_DIR, "model3.pth")))
    model3.eval()

    Cal_Confidence(model3, img)
    print(test_acc(model3, testset))

    model7 = get_model7()
    opt7 = torch.optim.Adam(model7.parameters(), lr=lr)
    sch7 = torch.optim.lr_scheduler.StepLR(opt7, step_size=9, gamma=0.4)
    model7 = train_model(trainset, valset, model7, criterion, opt7, sch7, num_epoch)

    torch.save(model7.state_dict(), os.path.join(SAVED_MODELS_DIR, "model7.pth"))

    model7 = get_model7()
    model7.load_state_dict(torch.load(os.path.join(SAVED_MODELS_DIR, "model7.pth")))
    model7.eval()

    Cal_Confidence(model7, img)
    print(test_acc(model7, testset))

    trainset1, valset1, testset1 = get_TVT(DATA_ROOT, lbp_transforms)
    model9 = get_LBP()
    opt9 = torch.optim.Adam(model9.parameters(), lr=lr)
    sch9 = torch.optim.lr_scheduler.StepLR(opt9, step_size=9, gamma=0.3)
    model9 = train_model(trainset1, valset1, model9, criterion, opt9, sch9, num_epoch)

    torch.save(model9.state_dict(), os.path.join(SAVED_MODELS_DIR, "model9.pth"))

    model9 = get_LBP()
    model9.load_state_dict(torch.load(os.path.join(SAVED_MODELS_DIR, "model9.pth")))
    model9.eval()

    gray_img = lbp_transforms.transforms[1](img.squeeze(0)).unsqueeze(0)
    Cal_Confidence(model9, gray_img)
    print(test_acc(model9, testset1))

    ensemble_models = get_models(model1, model3, model9)

    train_X, train_Y = get_weighted_score_ft(ensemble_models, trainset)
    test_X, test_Y = get_weighted_score_ft(ensemble_models, testset)

    clf = svm.SVC(kernel="poly", break_ties=True).fit(train_X, train_Y)

    svm_path = os.path.join(SAVED_MODELS_DIR, "svm.pk1")
    with open(svm_path, "wb") as file:
        pickle.dump(clf, file)

    with open(svm_path, "rb") as file:
        clf = pickle.load(file)

    pred = clf.predict(test_X)
    acc = accuracy_score(test_Y, pred)
    print(acc)
    clf.score(train_X, train_Y)

    img_X = get_weighted_score_img(ensemble_models, img)
    pred1 = clf.predict(img_X)
    print(pred1)

    confidence_scores = clf.decision_function(test_X)
    print(confidence_scores.max())
    print(confidence_scores.min())

    conf1 = clf.decision_function(img_X)
    print(conf1)

    norm = normalisation(conf1)
    print(norm)


if __name__ == "__main__":
    main()

