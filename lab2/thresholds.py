"""Підрахунок матриці плутанини, вартості помилок і пошук порога.

Правило рішення: прогнозуємо шахрайство (клас 1), якщо s >= t.
Вартість: C(t) = 10 * FN(t) + FP(t).

Модуль містить дві незалежні реалізації:
* ``threshold_table`` — швидка, через сортування та кумулятивні суми, O(n log n);
* ``brute_force_table`` — повільна, пряме застосування правила для кожного кандидата
  (лише для перевірки на малих наборах).
Вони не використовують результатів одна одної.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

FN_COST = 10
FP_COST = 1
COLUMNS = ["threshold", "TP", "FP", "FN", "TN", "C"]


def cost(fn: int, fp: int) -> int:
    return FN_COST * fn + FP_COST * fp


def _check(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64)  # float32 -> float64 без втрат
    y = np.asarray(labels).astype(np.int64)
    if s.ndim != 1 or s.shape != y.shape:
        raise ValueError("Оцінки й мітки мають бути одновимірними масивами однакової довжини.")
    if not np.isfinite(s).all():
        raise ValueError("Оцінки мають бути скінченними числами.")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Мітки мають бути 0/1.")
    return s, y


def confusion(scores: np.ndarray, labels: np.ndarray, t: float) -> dict[str, float | int]:
    """Матриця плутанини та вартість для одного порога за правилом s >= t."""
    s, y = _check(scores, labels)
    pred = s >= t  # для t = +inf усі прогнози негативні
    tp = int(np.count_nonzero(pred & (y == 1)))
    fp = int(np.count_nonzero(pred & (y == 0)))
    fn = int(np.count_nonzero(~pred & (y == 1)))
    tn = int(np.count_nonzero(~pred & (y == 0)))
    return {"threshold": float(t), "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": cost(fn, fp)}


def threshold_table(scores: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Усі кандидати (унікальні оцінки + +inf) з TP, FP, FN, TN, C.

    Сортуємо оцінки за спаданням, рахуємо кумулятивні суми позитивів і негативів
    і беремо значення в кінці кожної групи рівних оцінок: для порога t = u
    позитивними є всі об'єкти з s >= u, тобто вся група u і все, що вище.
    Рядки впорядковані від +inf до найменшої оцінки.
    """
    s, y = _check(scores, labels)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    order = np.argsort(-s, kind="stable")
    s_sorted = s[order]
    y_sorted = y[order]
    cum_tp = np.cumsum(y_sorted)
    cum_fp = np.cumsum(1 - y_sorted)
    if len(s_sorted):
        group_end = np.flatnonzero(np.r_[s_sorted[1:] != s_sorted[:-1], True])
    else:
        group_end = np.array([], dtype=np.int64)
    thr = np.r_[np.inf, s_sorted[group_end]]
    tp = np.r_[0, cum_tp[group_end]].astype(np.int64)
    fp = np.r_[0, cum_fp[group_end]].astype(np.int64)
    fn = n_pos - tp
    tn = n_neg - fp
    return pd.DataFrame(
        {"threshold": thr, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": FN_COST * fn + FP_COST * fp}
    )


def choose_threshold(table: pd.DataFrame) -> pd.Series:
    """Рядок з мінімальною C; за нічиєї — найбільший поріг (+inf найбільший)."""
    best = table["C"].min()
    tied = table[table["C"] == best]
    return tied.loc[tied["threshold"].idxmax()]


# ---------- незалежна повільна реалізація для перевірки ----------


def brute_force_table(scores: list[float], labels: list[int]) -> pd.DataFrame:
    """Перебір: кандидати будуються окремо, кожен поріг — повний прохід вибіркою."""
    candidates = [math.inf] + sorted(set(float(v) for v in scores), reverse=True)
    rows = []
    for t in candidates:
        tp = fp = fn = tn = 0
        for s, y in zip(scores, labels, strict=True):
            positive = float(s) >= t
            if positive and y == 1:
                tp += 1
            elif positive and y == 0:
                fp += 1
            elif not positive and y == 1:
                fn += 1
            else:
                tn += 1
        rows.append({"threshold": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": 10 * fn + fp})
    return pd.DataFrame(rows, columns=COLUMNS)


def brute_force_choice(rows: pd.DataFrame) -> float:
    best_cost = min(int(c) for c in rows["C"])
    return max(float(t) for t, c in zip(rows["threshold"], rows["C"]) if int(c) == best_cost)
