from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(".")

from src.domain.box import Box
from src.train import ActionClassifier


def show_model_performance(model_path: Path):
    """
    学習させた重みの混同行列と学習曲線を表示します．
    """

    label_names = ["None", "pull the rope", "ride on swing"]

    data = torch.load(model_path, map_location="cpu")

    mat = confusion_matrix(data["test_true_y"], data["test_pred_y"])
    df = pd.DataFrame(mat, columns=label_names, index=label_names)
    print(df)
    print(classification_report(data["test_true_y"], data["test_pred_y"]))

    fig, ax = plt.subplots()
    ax.plot(range(1, data["epoch"] + 1), data["train_accs"], color="k")
    ax.plot(range(1, data["epoch"] + 1), data["test_accs"], color="r")
    plt.show()


def test_model(model_path: Path, deepsort_output_directory_path: Path):
    """
    学習済みモデルの適用結果を表示します．
    """

    csv_paths = deepsort_output_directory_path.glob("*.csv")
    jpg_paths = deepsort_output_directory_path.glob("*.jpg")

    model_path = Path("./data/model/epoch_70_model.pth")
    classifier = ActionClassifier(pretrained_model_path=model_path)
    action_label = {0: "None", 1: "Pull the rope", 2: "Ride on swing "}

    for csv_path, jpg_path in zip(csv_paths, jpg_paths):
        boxes = Box.read_csv(csv_path)

        if not boxes:
            continue

        img = cv2.imread(str(jpg_path))

        for box in boxes:
            person_img = img[box.ymin : box.ymax, box.xmin : box.xmax]
            label = classifier.predict([person_img])[0]

            cv2.rectangle(img, (box.xmin, box.ymin), (box.xmax, box.ymax), (0, 0, 0), 3)
            cv2.putText(img, action_label[label], (box.xmin, box.ymin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)

        cv2.imshow("img", img)
        cv2.waitKey(1)


model_path = Path("./data/model/epoch_70_model.pth")
deepsort_output_directory_path = Path("./data/boxes/sample")

show_model_performance(model_path)
# test_model(model_path, deepsort_output_directory_path)
