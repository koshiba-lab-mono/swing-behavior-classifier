import pickle
from pathlib import Path

import numpy as np

from .box import Box


class Teacher:
    def __init__(self, image: np.ndarray, label: int):
        """
        教師データの構造体．

        image: np.ndarray 画像
        label: int 教師ラベル 0: 見学 1: ブランコ紐を引っ張っている 3: ブランコに乗っている

        save_instances, load_instancesにより, バイナリ形式で保存，読み込みをします．

        """

        self.image = image
        self.label = label

    @classmethod
    def load_instances(path: Path):
        with open(path, "rb") as f:
            instances: list[Teacher] = pickle.load(f)
        return instances

    @classmethod
    def save_instances(teachers, path: Path):
        with open(path, "wb") as f:
            pickle.dump(teachers, f)
