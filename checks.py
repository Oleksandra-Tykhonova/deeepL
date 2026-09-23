import numpy as np

from model import forward, cross_entropy_loss


def compare_table(rows, tolerance):
    header = f"{'величина':<14}{'макс. |numpy - pytorch|':<26}{'скінченне':<12}{'перевірку пройдено'}"
    print(header)
    print("-" * len(header))
    for name, np_val, torch_val in rows:
        diff = np.abs(np.asarray(np_val) - np.asarray(torch_val)).max()
        finite = bool(np.isfinite(np_val).all() and np.isfinite(torch_val).all())
        passed = bool(finite and diff <= tolerance)
        print(f"{name:<14}{diff:<26.3e}{str(finite):<12}{passed}")


def numeric_grad_entry(W1, b1, W2, b2, X, y, array, index, eps=1e-6):
    original = array[index]

    array[index] = original + eps
    logits_plus, _ = forward(X, W1, b1, W2, b2)
    loss_plus, _ = cross_entropy_loss(logits_plus, y)

    array[index] = original - eps
    logits_minus, _ = forward(X, W1, b1, W2, b2)
    loss_minus, _ = cross_entropy_loss(logits_minus, y)

    array[index] = original
    return (loss_plus - loss_minus) / (2 * eps)


def numeric_check_table(W1, b1, W2, b2, X, y, gradients, tolerance, eps=1e-6):
    dW1, db1, dW2, db2 = gradients
    entries = [
        ("W1[0,0]", W1, (0, 0), dW1[0, 0]),
        ("b1[0]", b1, (0,), db1[0]),
        ("W2[0,0]", W2, (0, 0), dW2[0, 0]),
        ("b2[0]", b2, (0,), db2[0]),
    ]

    header = f"{'параметр':<10}{'backward()':<22}{'чисельна похідна':<22}{'абс. різниця':<18}{'перевірку пройдено'}"
    print(header)
    print("-" * len(header))
    for name, array, index, g_manual in entries:
        g_num = numeric_grad_entry(W1, b1, W2, b2, X, y, array, index, eps)
        diff = abs(g_num - g_manual)
        passed = diff <= tolerance
        print(f"{name:<10}{g_manual:<22.10f}{g_num:<22.10f}{diff:<18.3e}{passed}")
