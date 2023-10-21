from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

sys.path.append(".")
from src.domain.teacher import Teacher

device = "cuda" if torch.cuda.is_available() else "cpu"


class TeacherDataset(data.Dataset):
    def __init__(self, teachers: list[Teacher], transform=None):
        super().__init__()
        self.teachers = teachers
        self.transform = transform

    def __len__(self):
        return len(self.teachers)

    def __getitem__(self, index: int):
        teacher = self.teachers[index]
        if self.teachers is not None:
            return self.transform(teacher.image), teacher.label
        return teacher.image, teacher.label


class ActionClassifier(nn.Module):  # ProgrammingClassifier
    def __init__(self, pretrained_model_path: str = None):
        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2, 2),
        )

        self.block_3 = nn.Sequential(
            nn.Linear(3136, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 3),
        )

        if pretrained_model_path is not None:
            self.load_state_dict(torch.load(pretrained_model_path, map_location=device)["model_state_dict"])
            self.eval()

    def forward(self, x):
        y = self.block_1(x)
        y = self.block_2(y)
        y = y.view(x.shape[0], -1)
        y = self.block_3(y)
        return y

    def predict(self, images: list[np.ndarray]):
        tensors = [val_transform(image) for image in images]
        result = self(torch.stack(tensors))
        return torch.argmax(result, dim=1).tolist()


def train_transform(img: np.ndarray) -> tuple[torch.Tensor]:
    transformer = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomAffine(30, (0.1, 0.1), (0.7, 1.3)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.7),
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
        ]
    )

    return transformer(img)


def val_transform(img: np.ndarray) -> tuple[torch.Tensor]:
    transformer = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
        ]
    )

    return transformer(img)


def collate_fn(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    imgs = []
    labels = []
    for sample, t in batch:
        imgs.append(sample)
        labels.append(t)
    return torch.stack(imgs).to("cuda"), torch.Tensor(labels).long().to("cuda")


def model_compile(
    model: nn.Module,
    train_loader: data.DataLoader,
    val_loader: data.DataLoader,
    max_epoch: int,
    optim: optim.SGD,
    criterion: nn.Module,
    save_dir: Path,
    checkpoints: Sequence[int] | None = None,
) -> None:
    """
    毎回学習する際ののひな型を書くのが面倒なので関数にしたもの．モデルの入力が画像のみの時なら使える.
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoints = [max_epoch] if checkpoints is None else checkpoints

    train_accs = []
    test_accs = []
    for epoch in range(1, max_epoch + 1):
        sum_acc = 0
        model.train()
        for imgs, t in tqdm(train_loader, total=len(train_loader)):
            pred_y = model(imgs)
            loss = criterion(pred_y, t)
            model.zero_grad()
            loss.backward()
            optim.step()
            sum_acc += torch.sum(t == torch.argmax(pred_y, dim=1))
        print(f"train acc:{sum_acc/len(train_loader.dataset)}")
        train_accs.append(float(sum_acc / len(train_loader.dataset)))
        sum_acc = 0
        model.eval()
        with torch.no_grad():
            for imgs, t in val_loader:
                pred_y = model(imgs)
                sum_acc += torch.sum(t == torch.argmax(pred_y, dim=1))
            print(f"test acc:{sum_acc/len(val_loader.dataset)} epoch {epoch}/{max_epoch} done.")
            test_accs.append(float(sum_acc / len(val_loader.dataset)))
            if epoch in checkpoints:
                ts = []
                preds_ys = []
                for imgs, t in val_loader:
                    ts += t.tolist()
                    preds_ys += torch.argmax(model(imgs), dim=1).tolist()
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "train_accs": train_accs,
                        "test_accs": test_accs,
                        "test_pred_y": preds_ys,
                        "test_true_y": ts,
                    },
                    str(save_dir / f"epoch_{epoch}_model.pth"),
                )


if __name__ == "__main__":
    base_dir = Path("./data/teachers")
    teachers: list[Teacher] = []

    for path in base_dir.glob("*.pickle"):
        teachers.extend(Teacher.load_instances(path))

    train, test = train_test_split(teachers)

    batch_size = 16
    max_epoch = 100

    train_set = TeacherDataset(train, train_transform)
    test_set = TeacherDataset(test, val_transform)
    train_loader = data.DataLoader(train_set, batch_size, collate_fn=collate_fn)
    test_loader = data.DataLoader(test_set, batch_size, collate_fn=collate_fn)

    model = ActionClassifier().to(device)
    optim_ = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    checkpoints = [10, 30, 50, 70, 100]

    model_compile(
        model,
        train_loader,
        test_loader,
        max_epoch,
        optim_,
        criterion,
        Path("./data/model"),
        checkpoints,
    )
