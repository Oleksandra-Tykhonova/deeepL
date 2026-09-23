import argparse

from data import load_split_standardized
from model import init_params, forward, cross_entropy_loss, backward
from torch_reference import build_torch_model, torch_forward_backward
from checks import compare_table, numeric_check_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bug", action="store_true")
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_split_standardized()
    W1, b1, W2, b2 = init_params()

    logits, cache = forward(X_train, W1, b1, W2, b2)
    loss, log_probs = cross_entropy_loss(logits, y_train)
    dW1, db1, dW2, db2 = backward(y_train, W2, cache, log_probs, divide_by_n=not args.bug)

    mode = "з навмисною помилкою (без ділення на n)" if args.bug else "коректна реалізація"
    print(f"режим: {mode}")
    print("loss:", loss)
    print()

    torch_model = build_torch_model(W1, b1, W2, b2)
    loss_t, dW1_t, db1_t, dW2_t, db2_t = torch_forward_backward(torch_model, X_train, y_train)

    print("звірка з PyTorch (поріг 1e-12):")
    compare_table(
        [
            ("втрата", loss, loss_t),
            ("градієнт W1", dW1, dW1_t),
            ("градієнт b1", db1, db1_t),
            ("градієнт W2", dW2, dW2_t),
            ("градієнт b2", db2, db2_t),
        ],
        1e-12,
    )
    print()

    print("чисельна перевірка (поріг 1e-7):")
    numeric_check_table(W1, b1, W2, b2, X_train, y_train, (dW1, db1, dW2, db2), 1e-7)


if __name__ == "__main__":
    main()
