from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


def as_positive_num(num: int):
    if num < 0:
        return 0
    return num


class Box:
    def __init__(self, id: int, xmin: int, ymin: int, xmax: int, ymax: int):
        self.id = id

        self.xmin = as_positive_num(xmin)
        self.ymin = as_positive_num(ymin)
        self.xmax = as_positive_num(xmax)
        self.ymax = as_positive_num(ymax)

    @classmethod
    def read_csv(self, path: Path) -> list[Box]:
        try:
            table = pd.read_csv(path).values
        except EmptyDataError:
            return []

        return [Box(*row) for row in table]
