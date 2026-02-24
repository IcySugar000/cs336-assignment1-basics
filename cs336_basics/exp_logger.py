import os
import json
from typing import Any

from loguru import logger


class ExpLogger:
    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.data: dict[int, dict[str, Any]] = {}

    def log(self, step: int, data: dict[str, Any]):
        self.data[step] = data

    def save(self):
        path = f"checkpoints/exp_logger/{self.exp_name}.json"
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f)

    def load(self):
        path = f"checkpoints/exp_logger/{self.exp_name}.json"
        if not os.path.exists(path):
            logger.error(f"Exp logger path not exists: {path}, skipping loading data")
            return
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            self.data = {int(k): v for k, v in data.items()}


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    for exp in ["base"]:
        exp_logger = ExpLogger(exp)
        exp_logger.load()
        x = list(exp_logger.data.keys())
        y = [exp_logger.data[i]["loss"] for i in x]
        plt.plot(x, y, label=exp)
    plt.legend()
    plt.show()
