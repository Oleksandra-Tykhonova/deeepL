import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


class TorchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def build_torch_model(W1, b1, W2, b2):
    model = TorchNet()
    with torch.no_grad():
        model.fc1.weight.copy_(torch.from_numpy(W1.T))
        model.fc1.bias.copy_(torch.from_numpy(b1))
        model.fc2.weight.copy_(torch.from_numpy(W2.T))
        model.fc2.bias.copy_(torch.from_numpy(b2))
    return model


def torch_forward_backward(model, X, y):
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y).long()

    logits = model(X_t)
    loss = nn.functional.cross_entropy(logits, y_t)

    model.zero_grad()
    loss.backward()

    dW1 = model.fc1.weight.grad.numpy().T
    db1 = model.fc1.bias.grad.numpy()
    dW2 = model.fc2.weight.grad.numpy().T
    db2 = model.fc2.bias.grad.numpy()

    return loss.item(), dW1, db1, dW2, db2
