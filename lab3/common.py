from __future__ import annotations

import csv
import hashlib
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
IMG_SIZE = 160
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
EPOCHS = 8
TRAIN_PER_CLASS = 200
BATCH_TRAIN = 32
BATCH_EVAL = 64
WEIGHT_DECAY = 1e-4
TOP_MODULES = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]
RUNS = [("scratch", 1), ("frozen", 1), ("frozen", 2), ("frozen", 3),
        ("unfreeze", 1), ("unfreeze", 2), ("unfreeze", 3)]

REGIMES = {
    "scratch": {
        "weights": None,
        "trainable": TOP_MODULES,
        "groups": [(TOP_MODULES, 3e-3)],
        "eval_modules": [],
    },
    "frozen": {
        "weights": "IMAGENET1K_V1",
        "trainable": ["fc"],
        "groups": [(["fc"], 3e-3)],
        "eval_modules": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
    },
    "unfreeze": {
        "weights": "IMAGENET1K_V1",
        "trainable": ["layer4", "fc"],
        "groups": [(["fc"], 3e-3), (["layer4"], 3e-4)],
        "eval_modules": ["conv1", "bn1", "layer1", "layer2", "layer3"],
    },
}

TRAIN_TF = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
EVAL_TF = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])


# ---------- поділ ----------

def class_files(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.is_file())


def build_split(data: Path) -> list[dict[str, str]]:
    train_root = data / "seg_train" / "seg_train"
    test_root = data / "seg_test" / "seg_test"
    rows = []
    rng = np.random.default_rng(0)
    for label in CLASSES:
        files = class_files(train_root / label)
        idx = np.arange(len(files))
        rng.shuffle(idx)
        rows += [{"file": files[i], "label": label, "part": "train"} for i in idx[:TRAIN_PER_CLASS]]
    rng = np.random.default_rng(0)
    val_rows, test_rows = [], []
    for label in CLASSES:
        files = class_files(test_root / label)
        idx = np.arange(len(files))
        rng.shuffle(idx)
        cut = len(files) // 2
        val_rows += [{"file": files[i], "label": label, "part": "validation"} for i in idx[:cut]]
        test_rows += [{"file": files[i], "label": label, "part": "test"} for i in idx[cut:]]
    return rows + val_rows + test_rows


def write_split(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "part"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def read_split(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------- дані ----------

class SplitDataset(Dataset):
    def __init__(self, data: Path, split_csv: Path, part: str, transform):
        sub = "seg_train/seg_train" if part == "train" else "seg_test/seg_test"
        rows = [r for r in read_split(split_csv) if r["part"] == part]
        self.files = [r["file"] for r in rows]
        self.paths = [data / sub / r["label"] / r["file"] for r in rows]
        self.labels = np.array([CLASSES.index(r["label"]) for r in rows], dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        return self.transform(img), int(self.labels[i])


def train_loader(data: Path, split_csv: Path) -> DataLoader:
    ds = SplitDataset(data, split_csv, "train", TRAIN_TF)
    return DataLoader(ds, batch_size=BATCH_TRAIN, shuffle=True, drop_last=False, num_workers=0)


def eval_loader(data: Path, split_csv: Path, part: str) -> DataLoader:
    ds = SplitDataset(data, split_csv, part, EVAL_TF)
    return DataLoader(ds, batch_size=BATCH_EVAL, shuffle=False, num_workers=0)


# ---------- модель ----------

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple MPS"
    return platform.processor() or platform.machine() or "cpu"


def build_model(regime: str, seed: int, device: torch.device):
    cfg = REGIMES[regime]
    torch.manual_seed(seed)
    np.random.seed(seed)
    weights = ResNet18_Weights.IMAGENET1K_V1 if cfg["weights"] else None
    model = torchvision.models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(512, len(CLASSES))
    for name, p in model.named_parameters():
        p.requires_grad = name.split(".")[0] in cfg["trainable"]
    model = model.to(device=device, dtype=torch.float32)
    groups = [
        {"params": [p for m in mods for p in getattr(model, m).parameters()], "lr": lr}
        for mods, lr in cfg["groups"]
    ]
    opt = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    return model, opt


def set_train_mode(model: torch.nn.Module, regime: str) -> None:
    model.train()
    for name in REGIMES[regime]["eval_modules"]:
        getattr(model, name).eval()


def count(params) -> int:
    return sum(p.numel() for p in params)


# ---------- навчання й оцінка ----------

@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out = [model(x.to(device)).float().cpu() for x, _ in loader]
    return torch.cat(out).numpy()


def macro_f1(labels: np.ndarray, logits: np.ndarray) -> float:
    return float(f1_score(labels, logits.argmax(1), average="macro", labels=list(range(len(CLASSES)))))


def train_epoch(model, opt, loader, regime, device) -> float:
    set_train_mode(model, regime)
    loss_fn = torch.nn.CrossEntropyLoss()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


def git_info(cwd: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--", "*.py", "pyproject.toml", "uv.lock"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def denormalize(x: torch.Tensor) -> np.ndarray:
    img = x.permute(1, 2, 0).numpy() * np.array(STD) + np.array(MEAN)
    return np.clip(img, 0, 1)
