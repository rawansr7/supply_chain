import numpy as np
import torch
from torch import nn

from .. import config as C
from .base import Forecaster, scaled_windows

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONTEXT = 52
LAYERS = 2
DROPOUT = 0.1
STEPS = 5000
BATCH = 256


class Net(nn.Module):
    def __init__(self, hidden, outputs):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, LAYERS, batch_first=True, dropout=DROPOUT)
        self.head = nn.Linear(hidden, outputs)

    def forward(self, x):
        return self.head(self.lstm(x.unsqueeze(-1))[0][:, -1])


class LSTMGlobal(Forecaster):
    grids = {"statistical": [{"hidden": n, "lr": r} for n in (32, 64, 128, 256) for r in (3e-4, 1e-3, 3e-3, 1e-2)]}

    def fit(self, Y):
        torch.manual_seed(C.SEED)
        X, F = (torch.tensor(a, device=DEVICE) for a in scaled_windows(Y.to_numpy(np.float32), CONTEXT, self.horizon))
        levels = torch.tensor(self.levels, device=DEVICE)
        self.net = Net(self.params["hidden"], self.horizon * len(self.levels)).to(DEVICE)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.params["lr"])
        for _ in range(STEPS):
            batch = torch.randint(len(X), (BATCH,), device=DEVICE)
            diff = F[batch, :, None] - self.net(X[batch]).view(BATCH, self.horizon, -1)
            loss = torch.maximum(levels * diff, (levels - 1) * diff).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.net.eval()
        return self

    def predict(self, Y):
        context = Y.to_numpy(np.float32)[:, -CONTEXT:]
        scale = 1 + context.mean(1, keepdims=True)
        with torch.no_grad():
            Q = self.net(torch.tensor(context / scale, device=DEVICE)).view(len(context), self.horizon, -1)
        return np.clip(np.sort(Q.cpu().numpy(), -1) * scale[..., None], 0, None)
