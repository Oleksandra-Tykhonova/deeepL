"""ЛР2, пункти 2–4: аналіз порогів зі збережених прогнозів (без CSV і без навчання).

Запуск:
    uv run python analysis.py --saved results --out outputs

Послідовність:
  2) validation: PR/ROC-криві, ROC-AUC, повна таблиця кандидатів, вибір t*;
  3) перевірка пошуку на наборах A/B/C незалежним перебором;
  4) лише якщо всі перевірки пройдено — оцінка test за зафіксованим t* і розбір помилок.
"""

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
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from lab2_starter import load_predictions, split_summary  # noqa: E402
from thresholds import (  # noqa: E402
    COLUMNS,
    brute_force_choice,
    brute_force_table,
    choose_threshold,
    confusion,
    threshold_table,
)

# Набори для перевірки з умови завдання.
CHECK_SETS: dict[str, tuple[list[float], list[int]]] = {
    "A": ([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], [0, 1, 0, 1, 0, 0]),
    "B": ([0.9, 0.6, 0.3], [0, 0, 0]),
    "C": ([0.8] + [0.5] * 11 + [0.2], [1] + [1] + [0] * 10 + [0]),
}


def fmt_t(t: float) -> str:
    """Поріг для відображення (лише у звіті; у CSV/JSON — повна точність)."""
    return "+inf" if math.isinf(t) else f"{t:.6g}"


def t_json(t: float) -> str:
    """Точне текстове подання порога: repr float64 (float32-оцінки відтворюються точно)."""
    return "inf" if math.isinf(t) else repr(float(t))


def to_csv_exact(df: pd.DataFrame, path: Path) -> None:
    """CSV без округлення: float64 через repr (найкоротше точне подання), inf як 'inf'."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = [repr(float(v)) for v in df[col]]
    df.to_csv(path, index=False)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def ratio(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


# ---------------------------------------------------------------- пункт 2


def validation_part(y: np.ndarray, s: np.ndarray, out: Path) -> dict:
    s64 = s.astype(np.float64)
    roc_auc = float(roc_auc_score(y, s64))
    ap = float(average_precision_score(y, s64))

    table = threshold_table(s64, y)
    to_csv_exact(table, out / "validation_thresholds.csv")
    best = choose_threshold(table)
    t_star = float(best["threshold"])

    # Контроль: прямий підрахунок у t* має збігатися з рядком таблиці.
    direct = confusion(s64, y, t_star)
    for k in ("TP", "FP", "FN", "TN", "C"):
        assert int(best[k]) == direct[k], f"Розбіжність у {k} при t*"

    # Контроль швидкої таблиці прямим правилом на випадковій підмножині кандидатів.
    rng = np.random.default_rng(0)
    sample = rng.choice(len(table), size=min(300, len(table)), replace=False)
    for i in sample:
        row = table.iloc[i]
        d = confusion(s64, y, float(row["threshold"]))
        assert all(int(row[k]) == d[k] for k in ("TP", "FP", "FN", "TN", "C"))

    rules = [
        ("Стандартний поріг", 0.5),
        ("Мінімум вартості t*", t_star),
        ("Усі негативні", math.inf),
    ]
    comparison = []
    for name, t in rules:
        c = confusion(s64, y, t)
        comparison.append({"Правило": name, "Поріг": fmt_t(t), **{k: c[k] for k in COLUMNS[1:]}})
    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(out / "validation_comparison.csv", index=False)

    # Кількість кандидатів з мінімальною вартістю (для пояснення нічиєї).
    n_ties = int((table["C"] == table["C"].min()).sum())

    plot_curves(y, s64, t_star, out)
    (out / "threshold.json").write_text(
        json.dumps(
            {
                "t_star": t_json(t_star),
                "t_star_hex": float(t_star).hex() if math.isfinite(t_star) else "inf",
                "rule": "fraud if score >= t_star",
                "cost": "C = 10*FN + FP",
                "chosen_on": "validation",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "roc_auc": roc_auc,
        "average_precision": ap,
        "n_candidates": len(table),
        "n_unique_scores": len(table) - 1,
        "n_min_cost_ties": n_ties,
        "t_star": t_star,
        "comparison": comparison_df,
        "min_cost": int(best["C"]),
    }


def plot_curves(y: np.ndarray, s: np.ndarray, t_star: float, out: Path) -> None:
    prec, rec, _ = precision_recall_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    points = [("t = 0.5", 0.5, "o"), (f"t* = {fmt_t(t_star)}", t_star, "*")]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, drawstyle="steps-post", label="PR-крива")
    ax.axhline(y.mean(), ls="--", lw=1, color="grey", label=f"частка позитивів = {y.mean():.4f}")
    for lab, t, m in points:
        c = confusion(s, y, t)
        p = ratio(c["TP"], c["TP"] + c["FP"])
        r = ratio(c["TP"], c["TP"] + c["FN"])
        if p is not None and r is not None:
            ax.plot(r, p, m, ms=10, label=f"{lab}: P={p:.3f}, R={r:.3f}")
    ax.set(xlabel="Повнота (recall)", ylabel="Точність (precision)", title="PR-крива, validation")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "pr_curve_validation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    auc = roc_auc_score(y, s)
    ax.plot(fpr, tpr, label=f"ROC-крива, AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="grey", label="випадковий класифікатор")
    for lab, t, m in points:
        c = confusion(s, y, t)
        ax.plot(c["FP"] / (c["FP"] + c["TN"]), c["TP"] / (c["TP"] + c["FN"]), m, ms=10, label=lab)
    ax.set(xlabel="FPR = FP / (FP + TN)", ylabel="TPR = TP / (TP + FN)", title="ROC-крива, validation")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "roc_curve_validation.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- пункт 3


def run_checks(out: Path) -> tuple[bool, dict[str, dict]]:
    results = {}
    all_ok = True
    for name, (s, y) in CHECK_SETS.items():
        slow = brute_force_table(s, y)
        t_slow = brute_force_choice(slow)
        fast = threshold_table(np.array(s), np.array(y))
        t_fast = float(choose_threshold(fast)["threshold"])
        same_thr = np.array_equal(slow["threshold"].to_numpy(), fast["threshold"].to_numpy())
        same_cnt = all(
            np.array_equal(slow[k].to_numpy(dtype=np.int64), fast[k].to_numpy(dtype=np.int64))
            for k in COLUMNS[1:]
        )
        ok = bool(same_thr and same_cnt and t_slow == t_fast)
        all_ok &= ok
        merged = slow.copy()
        merged["threshold"] = merged["threshold"].map(fmt_t)
        merged["C (швидко)"] = fast["C"].to_numpy()
        merged["збіг рядка"] = [
            "так" if all(int(slow.iloc[i][k]) == int(fast.iloc[i][k]) for k in COLUMNS[1:]) else "НІ"
            for i in range(len(slow))
        ]
        merged.to_csv(out / f"check_{name}.csv", index=False)
        results[name] = {
            "table": merged,
            "t_slow": t_slow,
            "t_fast": t_fast,
            "ok": ok,
        }
        print(f"  Набір {name}: t*(перебір)={fmt_t(t_slow)}, t*(швидко)={fmt_t(t_fast)}, "
              f"{'OK' if ok else 'ПОМИЛКА'}")
    return all_ok, results


# ---------------------------------------------------------------- пункт 4


def test_part(data: dict[str, np.ndarray], t_star: float, out: Path) -> dict:
    y = data["test_y"]
    s = data["test_scores"].astype(np.float64)
    c = confusion(s, y, t_star)
    precision = ratio(c["TP"], c["TP"] + c["FP"])
    recall = ratio(c["TP"], c["TP"] + c["FN"])
    c05 = confusion(s, y, 0.5)
    cinf = confusion(s, y, math.inf)

    rows = pd.DataFrame(
        {
            "row": data["test_row_ids"].astype(np.int64),
            "label": y,
            "score": s,
            "amount": data["test_amount"],
        }
    )
    pred = rows["score"] >= t_star
    rows["error"] = np.where(pred & (rows["label"] == 0), "FP",
                             np.where(~pred & (rows["label"] == 1), "FN", ""))
    rows["distance"] = (rows["score"] - t_star).abs()
    errors = rows[rows["error"] != ""].sort_values("row")
    to_csv_exact(errors, out / "test_errors_all.csv")
    first = pd.concat(
        [errors[errors["error"] == "FP"].head(5), errors[errors["error"] == "FN"].head(5)]
    )
    to_csv_exact(first, out / "test_errors_first5.csv")

    def agg(kind: str) -> dict:
        part = errors[errors["error"] == kind]
        if part.empty:
            return {"n": 0}
        return {
            "n": len(part),
            "amount_median": float(part["amount"].median()),
            "amount_mean": float(part["amount"].mean()),
            "distance_median": float(part["distance"].median()),
        }

    tn_amount = rows[(~pred) & (rows["label"] == 0)]["amount"]
    tp_amount = rows[pred & (rows["label"] == 1)]["amount"]
    return {
        "confusion": c,
        "confusion_05": c05,
        "confusion_inf": cinf,
        "precision": precision,
        "recall": recall,
        "first_errors": first,
        "fp_stats": agg("FP"),
        "fn_stats": agg("FN"),
        "tn_amount_median": float(tn_amount.median()) if len(tn_amount) else None,
        "tp_amount_median": float(tp_amount.median()) if len(tp_amount) else None,
    }


# ---------------------------------------------------------------- звіт


def to_jsonable(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return t_json(float(obj)) if not math.isfinite(obj) else float(obj)
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--saved", type=Path, default=Path("results"),
                        help="Тека зі збереженими прогнозами (predictions.npz)")
    parser.add_argument("--out", type=Path, default=Path("outputs"),
                        help="Тека для таблиць, графіків і підсумків аналізу")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    data = load_predictions(args.saved)
    summary = split_summary(data)
    summary.to_csv(out / "split_summary.csv", index=False)
    print("Поділ:\n" + summary.to_string(index=False))

    print("\n[2] Validation: криві, таблиця кандидатів, вибір t*")
    val = validation_part(data["validation_y"], data["validation_scores"], out)
    print(f"  ROC-AUC = {val['roc_auc']:.6f}, AP = {val['average_precision']:.6f}")
    print(f"  Кандидатів: {val['n_candidates']} (унікальних оцінок {val['n_unique_scores']} + inf)")
    print(f"  t* = {t_json(val['t_star'])}  (C = {val['min_cost']})")
    print(val["comparison"].to_string(index=False))

    print("\n[3] Перевірка пошуку на наборах A/B/C")
    ok, checks = run_checks(out)
    if not ok:
        print("Перевірки не пройдено — тестову оцінку не виконано.", file=sys.stderr)
        sys.exit(1)

    print("\n[4] Test із зафіксованим t*")
    test = test_part(data, val["t_star"], out)
    c = test["confusion"]
    print(f"  TP={c['TP']} FP={c['FP']} FN={c['FN']} TN={c['TN']} C={c['C']}")
    print(f"  precision={test['precision']}, recall={test['recall']}")
    print(test["first_errors"].to_string(index=False))

    full = {
        "split": summary,
        "validation": val,
        "checks": {k: {kk: vv for kk, vv in v.items()} for k, v in checks.items()},
        "test": test,
    }
    (out / "summary.json").write_text(
        json.dumps(to_jsonable(full), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nРезультати збережено: {out.resolve()}")


if __name__ == "__main__":
    main()
