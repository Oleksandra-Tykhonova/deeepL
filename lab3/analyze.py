from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

import common as C  # noqa: E402
from train import select  # noqa: E402


def load_logs(root: Path) -> dict[tuple[str, int], dict]:
    logs = {}
    for regime, seed in C.RUNS:
        path = root / "runs" / f"{regime}_seed{seed}" / "run.json"
        logs[(regime, seed)] = json.loads(path.read_text(encoding="utf-8"))
    return logs


def check_hash(logs: dict, split_csv: Path) -> str:
    current = C.sha256(split_csv)
    bad = [f"{r}_seed{s}" for (r, s), log in logs.items() if log["split_csv_sha256"] != current]
    if bad:
        sys.exit(f"Хеш split.csv не збігається в журналах: {', '.join(bad)}")
    return current


def effects(logs: dict) -> dict:
    F = {k: v["final_val_macro_f1"] for k, v in logs.items()}
    res = {}
    for regime in ("frozen", "unfreeze"):
        vals = [F[(regime, s)] for s in (1, 2, 3)]
        res[regime] = {"mean": float(np.mean(vals)), "range": max(vals) - min(vals)}
    R = max(res["frozen"]["range"], res["unfreeze"]["range"])
    d_pre = res["frozen"]["mean"] - F[("scratch", 1)]
    d_unf = res["unfreeze"]["mean"] - res["frozen"]["mean"]
    return {
        "F_scratch_1": F[("scratch", 1)],
        **{f"{r}_{k}": v for r, d in res.items() for k, v in d.items()},
        "R": R,
        "delta_pre": d_pre,
        "delta_unf": d_unf,
        "delta_total": d_pre + d_unf,
        "verdict_pre": "|Δ| > R" if abs(d_pre) > R else "|Δ| ≤ R",
        "verdict_unf": "|Δ| > R" if abs(d_unf) > R else "|Δ| ≤ R",
    }


def curves(logs: dict, out: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = {"scratch": "tab:red", "frozen": "tab:blue", "unfreeze": "tab:green"}
    styles = {1: "-", 2: "--", 3: ":"}
    for (regime, seed), log in logs.items():
        ep = [h["epoch"] for h in log["history"]]
        kw = dict(color=colors[regime], ls=styles[seed], marker="o", ms=3, label=f"{regime}, зерно {seed}")
        a1.plot(ep, [h["train_loss"] for h in log["history"]], **kw)
        a2.plot(ep, [h["val_macro_f1"] for h in log["history"]], **kw)
    a1.set(xlabel="Епоха", ylabel="Втрата на train", title="Втрата на train")
    a2.set(xlabel="Епоха", ylabel="macro-F1 на validation", title="macro-F1 на validation")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.set_xticks(range(1, C.EPOCHS + 1))
    a2.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=130)
    plt.close(fig)


def check_selection(model_path: Path, logs: dict, data: Path, split_csv: Path) -> dict:
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    regime, seed = select(logs)
    if (ckpt["regime"], ckpt["seed"]) != (regime, seed):
        sys.exit(f"lab3_model.pt ({ckpt['regime']}, {ckpt['seed']}) не відповідає правилу вибору "
                 f"({regime}, {seed}).")
    if ckpt["classes"] != C.CLASSES or ckpt["img_size"] != C.IMG_SIZE:
        sys.exit("lab3_model.pt має інші класи або img_size.")
    model = build_eval_model(ckpt)
    val = C.eval_loader(data, split_csv, "validation")
    logits = C.predict(model, val, torch.device("cpu"))
    saved = np.load(model_path.parent / "runs" / f"{regime}_seed{seed}" / "val_logits.npy")
    return {
        "regime": regime,
        "seed": seed,
        "logged_val_f1": logs[(regime, seed)]["final_val_macro_f1"],
        "recomputed_val_f1_cpu": C.macro_f1(val.dataset.labels, logits),
        "max_abs_logit_diff_vs_saved": float(np.abs(logits - saved).max()),
        "model": model,
    }


def build_eval_model(ckpt: dict) -> torch.nn.Module:
    import torchvision
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def evaluate_test(model, data: Path, split_csv: Path, out: Path) -> dict:
    loader = C.eval_loader(data, split_csv, "test")
    logits = C.predict(model, loader, torch.device("cpu"))
    np.save(out / "test_logits.npy", logits.astype(np.float32))
    y = loader.dataset.labels
    pred = logits.argmax(1)
    probs = torch.softmax(torch.from_numpy(logits), 1).numpy()
    cm = confusion_matrix(y, pred, labels=list(range(len(C.CLASSES))))
    acc = float((pred == y).mean())
    n = len(y)
    return {
        "macro_f1": C.macro_f1(y, logits),
        "accuracy": acc,
        "n": n,
        "se_accuracy": math.sqrt(acc * (1 - acc) / n),
        "confusion": cm,
        "errors": top_errors(cm, loader.dataset, y, pred, probs),
    }


def top_errors(cm, ds, y, pred, probs, k: int = 5) -> list[dict]:
    pairs = [(int(cm[i, j]), i, j) for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j] > 0]
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    files_of = {
        (i, j): sorted((ds.files[n], n) for n in np.flatnonzero((y == i) & (pred == j)))
        for _, i, j in pairs
    }
    chosen, rank = [], 0
    while len(chosen) < k:
        added = False
        for cnt, i, j in pairs:
            if len(chosen) < k and rank < len(files_of[(i, j)]):
                name, n = files_of[(i, j)][rank]
                chosen.append({"file": name, "true": C.CLASSES[i], "pred": C.CLASSES[j],
                               "prob": float(probs[n, j]), "pair_count": cnt, "path": str(ds.paths[n])})
                added = True
        if not added:
            break
        rank += 1
    return chosen


def confusion_figure(cm: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(6), C.CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(6), C.CLASSES)
    ax.set(xlabel="Передбачений клас", ylabel="Справжній клас", title="Матриця плутанини, test")
    thr = cm.max() / 2
    for i in range(6):
        for j in range(6):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def errors_figure(errors: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, len(errors), figsize=(3.2 * len(errors), 3.8))
    for ax, e in zip(np.atleast_1d(axes), errors):
        with Image.open(e["path"]) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"{e['file']}\n{e['true']} → {e['pred']} ({e['prob']:.2f})", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="тека з seg_train/ і seg_test/")
    ap.add_argument("--root", type=Path, default=Path(__file__).parent,
                    help="тека з split.csv, runs/ і lab3_model.pt")
    args = ap.parse_args()
    root = args.root
    figs = root / "figures"
    figs.mkdir(exist_ok=True)
    split_csv = root / "split.csv"

    logs = load_logs(root)
    digest = check_hash(logs, split_csv)
    print(f"SHA-256 split.csv однаковий у всіх 7 журналах: {digest}")

    print("\n[3] Сім запусків")
    print(f"{'#':>2} {'Режим':<9} {'Зерно':>5} {'Втрата':>8} {'macro-F1':>9} {'Час, с':>8}")
    table = []
    for n, ((regime, seed), log) in enumerate(logs.items(), 1):
        row = {"n": n, "regime": regime, "seed": seed, "train_loss": log["final_train_loss"],
               "val_f1": log["final_val_macro_f1"], "seconds": log["train_seconds"],
               "commit": log["git"]["commit"], "dirty": log["git"]["dirty"]}
        table.append(row)
        print(f"{n:>2} {regime:<9} {seed:>5} {row['train_loss']:>8.4f} {row['val_f1']:>9.4f} "
              f"{row['seconds']:>8.1f}")
    curves(logs, figs)

    print("\n[4] Розклад ефектів")
    eff = effects(logs)
    for k, v in eff.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n[5] Перевірка вибору")
    sel = check_selection(root / "lab3_model.pt", logs, args.data, split_csv)
    model = sel.pop("model")
    for k, v in sel.items():
        print(f"  {k}: {v}")

    print("\n[5] Test (CPU, один раз)")
    test = evaluate_test(model, args.data, split_csv, root)
    print(f"  macro-F1 = {test['macro_f1']:.4f}, accuracy = {test['accuracy']:.4f}, "
          f"SE = {test['se_accuracy']:.4f} (n = {test['n']})")
    print("  Матриця плутанини (рядки — справжній, стовпці — передбачений):")
    print("  " + " ".join(f"{c[:8]:>9}" for c in [""] + C.CLASSES))
    for c, r in zip(C.CLASSES, test["confusion"]):
        print("  " + f"{c[:8]:>9} " + " ".join(f"{int(v):>9}" for v in r))
    confusion_figure(test["confusion"], figs / "confusion_test.png")
    print("  П'ять характерних помилок:")
    for e in test["errors"]:
        print(f"    {e['file']:<10} {e['true']:<10} → {e['pred']:<10} p={e['prob']:.4f}  "
              f"таких помилок: {e['pair_count']}")
    errors_figure(test["errors"], figs / "test_errors.png")

    result = {
        "split_sha256": digest,
        "runs": table,
        "effects": eff,
        "selection": sel,
        "test": {**{k: v for k, v in test.items() if k not in ("confusion", "errors")},
                 "confusion": test["confusion"].tolist(),
                 "errors": [{k: v for k, v in e.items() if k != "path"} for e in test["errors"]]},
    }
    (root / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
    print(f"\nРезультати: {root / 'analysis.json'}")


if __name__ == "__main__":
    main()
