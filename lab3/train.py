from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402

import common as C  # noqa: E402

EXPECTED_SPLIT = {
    "buildings": {"train": (200, 1996761), "validation": (218, 4837323), "test": (219, 4837559)},
    "forest": {"train": (200, 2032625), "validation": (237, 5276962), "test": (237, 5291326)},
    "glacier": {"train": (200, 2183923), "validation": (276, 6135652), "test": (277, 6157250)},
    "mountain": {"train": (200, 1950345), "validation": (262, 5789535), "test": (263, 5849765)},
    "sea": {"train": (200, 1987613), "validation": (255, 5655068), "test": (255, 5668011)},
    "street": {"train": (200, 1841817), "validation": (250, 5541845), "test": (251, 5547533)},
}
EXPECTED_TRAINABLE = {"scratch": 11_179_590, "frozen": 3_078, "unfreeze": 8_396_806}
PARTS = ["train", "validation", "test"]


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ---------- пункт 1 ----------

def check_split(rows: list[dict[str, str]]) -> dict:
    table, ok = {}, True
    for label in C.CLASSES:
        table[label] = {}
        for part in PARTS:
            files = [r["file"] for r in rows if r["label"] == label and r["part"] == part]
            got = (len(files), sum(int(Path(f).stem) for f in files))
            exp = EXPECTED_SPLIT[label][part]
            table[label][part] = {"got": got, "expected": exp, "match": got == exp}
            ok &= got == exp
    parts_of = {}
    for r in rows:
        parts_of.setdefault(r["file"], set()).add(r["part"])
    overlap = sorted(f for f, p in parts_of.items() if len(p) > 1)
    return {"table": table, "all_match": ok, "overlap_files": overlap, "n_rows": len(rows),
            "n_unique_files": len(parts_of)}


def batch_figure(data: Path, split_csv: Path, out: Path) -> None:
    torch.manual_seed(0)
    x, y = next(iter(C.train_loader(data, split_csv)))
    fig, axes = plt.subplots(4, 4, figsize=(8, 8.6))
    for ax, img, lab in zip(axes.flat, x[:16], y[:16]):
        ax.imshow(C.denormalize(img))
        ax.set_title(C.CLASSES[int(lab)], fontsize=10)
        ax.axis("off")
    fig.suptitle("Перші 16 зображень навчального пакета після перетворень", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def initial_loss(regime: str, data: Path, split_csv: Path, device) -> float:
    model, _ = C.build_model(regime, 1, device)
    x, y = next(iter(C.train_loader(data, split_csv)))
    C.set_train_mode(model, regime)
    with torch.no_grad():
        return float(torch.nn.functional.cross_entropy(model(x.to(device)), y.to(device)))


# ---------- пункт 2 ----------

def param_report(device) -> dict:
    report = {}
    for regime in C.REGIMES:
        model, opt = C.build_model(regime, 1, device)
        trainable = C.count(p for p in model.parameters() if p.requires_grad)
        in_opt = [p for g in opt.param_groups for p in g["params"]]
        frozen_in_opt = sum(not p.requires_grad for p in in_opt)
        report[regime] = {
            "total": C.count(model.parameters()),
            "trainable": trainable,
            "expected_trainable": EXPECTED_TRAINABLE[regime],
            "groups": [
                {"modules": mods, "params": C.count(g["params"]), "lr": g["lr"]}
                for (mods, _), g in zip(C.REGIMES[regime]["groups"], opt.param_groups)
            ],
            "params_in_optimizer": C.count(in_opt),
            "frozen_tensors_in_optimizer": frozen_in_opt,
            "match": trainable == EXPECTED_TRAINABLE[regime] and frozen_in_opt == 0
            and C.count(in_opt) == trainable,
        }
    return report


def five_steps(regime: str, data: Path, split_csv: Path, device, mode_fn) -> tuple:
    model, opt = C.build_model(regime, 1, device)
    loader = C.train_loader(data, split_csv)
    before = copy.deepcopy(model.state_dict())
    mode_fn(model)
    loss_fn = torch.nn.CrossEntropyLoss()
    losses = []
    for step, (x, y) in enumerate(loader):
        if step == 5:
            break
        opt.zero_grad()
        loss = loss_fn(model(x.to(device)), y.to(device))
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return model, before, model.state_dict(), losses


def change_table(regime: str, before: dict, after: dict) -> list[dict]:
    frozen = set(C.REGIMES[regime]["eval_modules"])
    rows = []
    for mod in C.TOP_MODULES:
        keys = [k for k in before if k.split(".")[0] == mod]
        param_keys = [k for k in keys if not k.endswith(("running_mean", "running_var", "num_batches_tracked"))]
        stat_keys = [k for k in keys if k.endswith(("running_mean", "running_var"))]
        p_changed = any(not torch.equal(before[k], after[k]) for k in param_keys)
        s_changed = None if not stat_keys else any(not torch.equal(before[k], after[k]) for k in stat_keys)
        exp_p = mod not in frozen
        exp_s = None if not stat_keys else mod not in frozen
        rows.append({
            "module": mod,
            "params_changed": p_changed,
            "bn_stats_changed": s_changed,
            "expected_params": exp_p,
            "expected_bn": exp_s,
            "match": p_changed == exp_p and s_changed == exp_s,
        })
    return rows


def bn_delta(before: dict, after: dict, prefix: str = "layer1.0.bn1") -> dict:
    return {
        stat: float((after[f"{prefix}.{stat}"] - before[f"{prefix}.{stat}"]).abs().max())
        for stat in ("running_mean", "running_var")
    }


def run_checks(data: Path, split_csv: Path, out: Path, device) -> dict:
    checks = {}
    checks["initial_loss"] = {
        r: initial_loss(r, data, split_csv, device) for r in ("scratch", "frozen")
    }
    checks["ln6"] = math.log(6)
    checks["params"] = param_report(device)

    changes = {}
    for regime in ("frozen", "unfreeze"):
        _, before, after, losses = five_steps(
            regime, data, split_csv, device, lambda m, r=regime: C.set_train_mode(m, r))
        changes[regime] = {"rows": change_table(regime, before, after), "losses": losses,
                           "layer1.0.bn1": bn_delta(before, after)}
    checks["changes"] = changes

    _, before, after, losses = five_steps("frozen", data, split_csv, device, lambda m: m.train())
    checks["bn_bug"] = {
        "rows": change_table("frozen", before, after),
        "losses": losses,
        "layer1.0.bn1": bn_delta(before, after),
    }
    checks["all_pass"] = (
        all(v["match"] for v in checks["params"].values())
        and all(r["match"] for c in changes.values() for r in c["rows"])
    )
    save_json(out / "checks.json", checks)
    return checks


# ---------- пункт 3 ----------

def settings(regime: str, seed: int) -> dict:
    cfg = C.REGIMES[regime]
    return {
        "regime": regime,
        "seed": seed,
        "classes": C.CLASSES,
        "transforms": {
            "train": [
                {"RandomResizedCrop": {"size": C.IMG_SIZE, "scale": [0.7, 1.0], "ratio": [0.75, 4 / 3],
                                       "interpolation": "bilinear", "antialias": True}},
                {"RandomHorizontalFlip": {"p": 0.5}},
                "ToTensor",
                {"Normalize": {"mean": C.MEAN, "std": C.STD}},
            ],
            "eval": [
                {"Resize": {"size": [C.IMG_SIZE, C.IMG_SIZE], "interpolation": "bilinear", "antialias": True}},
                "ToTensor",
                {"Normalize": {"mean": C.MEAN, "std": C.STD}},
            ],
            "image_mode": "RGB (PIL convert)",
        },
        "model": {
            "architecture": "torchvision.models.resnet18",
            "weights": cfg["weights"],
            "fc": "Linear(512, 6), default init",
            "trainable_modules": cfg["trainable"],
            "eval_modules_during_training": cfg["eval_modules"],
            "bn_momentum": 0.1,
            "bn_eps": 1e-5,
            "dtype": "float32",
        },
        "optimizer": {
            "name": "AdamW",
            "groups": [{"modules": m, "lr": lr} for m, lr in cfg["groups"]],
            "weight_decay": C.WEIGHT_DECAY,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "amsgrad": False,
        },
        "loss": "CrossEntropyLoss(reduction='mean')",
        "loop": {
            "epochs": C.EPOCHS,
            "train_batch_size": C.BATCH_TRAIN,
            "shuffle": True,
            "drop_last": False,
            "num_workers": 0,
            "eval_batch_size": C.BATCH_EVAL,
            "steps_per_epoch": math.ceil(1200 / C.BATCH_TRAIN),
            "total_steps": C.EPOCHS * math.ceil(1200 / C.BATCH_TRAIN),
            "lr_schedule": None,
            "grad_clip": None,
            "amp": False,
            "label_smoothing": 0.0,
            "class_weights": None,
            "early_stopping": False,
            "checkpoint": "after epoch 8",
        },
    }


def one_run(regime: str, seed: int, data: Path, split_csv: Path, out: Path, device) -> dict:
    run_dir = out / "runs" / f"{regime}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model, opt = C.build_model(regime, seed, device)
    loader = C.train_loader(data, split_csv)
    val_loader = C.eval_loader(data, split_csv, "validation")
    y_val = val_loader.dataset.labels
    history = []
    start = time.perf_counter()
    for epoch in range(1, C.EPOCHS + 1):
        loss = C.train_epoch(model, opt, loader, regime, device)
        logits = C.predict(model, val_loader, device)
        f1 = C.macro_f1(y_val, logits)
        history.append({"epoch": epoch, "train_loss": loss, "val_macro_f1": f1})
        print(f"  {regime} seed{seed} епоха {epoch}: loss={loss:.4f} F1={f1:.4f}", flush=True)
    seconds = time.perf_counter() - start
    np.save(run_dir / "val_logits.npy", logits.astype(np.float32))
    torch.save(model.state_dict(), run_dir / "model.pt")
    log = {
        **settings(regime, seed),
        "git": C.git_info(Path(__file__).parent),
        "split_csv_sha256": C.sha256(split_csv),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
        },
        "device": {"type": device.type, "name": C.device_name(device),
                   "torch_threads": torch.get_num_threads()},
        "history": history,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_macro_f1": history[-1]["val_macro_f1"],
        "train_seconds": seconds,
    }
    save_json(run_dir / "run.json", log)
    return log


# ---------- пункт 5 ----------

def select(logs: dict[tuple[str, int], dict]) -> tuple[str, int]:
    means = {}
    for regime in C.REGIMES:
        vals = [logs[(r, s)]["final_val_macro_f1"] for r, s in C.RUNS if r == regime]
        means[regime] = sum(vals) / len(vals)
    best_regime = max(C.REGIMES, key=lambda r: means[r])
    seeds = [s for r, s in C.RUNS if r == best_regime]
    best_seed = max(seeds, key=lambda s: (logs[(best_regime, s)]["final_val_macro_f1"], -s))
    return best_regime, best_seed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="тека з seg_train/ і seg_test/")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent, help="куди писати результати")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    device = C.pick_device()
    print(f"Пристрій: {device} ({C.device_name(device)})")

    print("\n[1] Поділ")
    split_csv = out / "split.csv"
    rows = C.build_split(args.data)
    C.write_split(rows, split_csv)
    rows = C.read_split(split_csv)
    sc = check_split(rows)
    save_json(out / "split_check.json", sc)
    for label, parts in sc["table"].items():
        cells = "  ".join(f"{p}: {v['got'][0]} / {v['got'][1]} {'✓' if v['match'] else '✗'}"
                          for p, v in parts.items())
        print(f"  {label:<10} {cells}")
    print(f"  усі 18 клітинок збігаються: {sc['all_match']}; спільних файлів між частинами: "
          f"{len(sc['overlap_files'])}")
    if not sc["all_match"] or sc["overlap_files"]:
        sys.exit("Поділ не збігається з контрольною таблицею.")

    batch_figure(args.data, split_csv, out / "figures" / "train_batch.png")

    print("\n[1–2] Перевірки до навчання")
    checks = run_checks(args.data, split_csv, out, device)
    for r, v in checks["initial_loss"].items():
        print(f"  L0 {r}: {v:.4f} (ln 6 = {checks['ln6']:.4f})")
    for r, v in checks["params"].items():
        groups = "; ".join(f"{'+'.join(g['modules'])}: {g['params']} @ {g['lr']:g}" for g in v["groups"])
        print(f"  {r}: усього {v['total']}, навчаються {v['trainable']} [{groups}] "
              f"{'✓' if v['match'] else '✗'}")
    for regime, c in checks["changes"].items():
        for row in c["rows"]:
            print(f"  {regime:<8} {row['module']:<7} params={row['params_changed']!s:<5} "
                  f"bn={row['bn_stats_changed']!s:<5} {'✓' if row['match'] else '✗'}")
    bug = checks["bn_bug"]
    print(f"  дослід model.train(): layer1.0.bn1 {bug['layer1.0.bn1']}")
    for row in bug["rows"]:
        print(f"    {row['module']:<7} params={row['params_changed']!s:<5} bn={row['bn_stats_changed']!s}")
    if not checks["all_pass"]:
        sys.exit("Перевірки режимів не збіглися з очікуваними.")

    print("\n[3] Сім запусків")
    logs = {}
    for regime, seed in C.RUNS:
        logs[(regime, seed)] = one_run(regime, seed, args.data, split_csv, out, device)

    regime, seed = select(logs)
    state = torch.load(out / "runs" / f"{regime}_seed{seed}" / "model.pt", map_location="cpu",
                       weights_only=True)
    torch.save({"state_dict": state, "regime": regime, "seed": seed, "classes": C.CLASSES,
                "img_size": C.IMG_SIZE}, out / "lab3_model.pt")
    print(f"\n[5] Вибрано {regime}, зерно {seed}; збережено lab3_model.pt")


if __name__ == "__main__":
    main()
