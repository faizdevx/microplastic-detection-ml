"""Small CNN on 48x48 particle patches (CPU, a few minutes)."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class PatchNet(nn.Module):
    def __init__(self, emb: int = 64):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, emb),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(emb, 1)

    def forward(self, x):
        z = self.features(x)
        return self.head(z).squeeze(1), z


def _tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).permute(0, 3, 1, 2).float().div_(255.0)


def fit(X: np.ndarray, y: np.ndarray, w: np.ndarray, epochs: int = 3, seed: int = 0,
        batch: int = 256, lr: float = 2e-3) -> PatchNet:
    torch.manual_seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    rng = np.random.default_rng(seed)
    net = PatchNet()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
    yt = torch.from_numpy(y.astype(np.float32))
    wt = torch.from_numpy((w / w.mean()).astype(np.float32))
    n = len(X)
    net.train()
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb = _tensor(X[idx])
            if rng.random() < 0.5:
                xb = torch.flip(xb, dims=[3])
            if rng.random() < 0.5:
                xb = torch.flip(xb, dims=[2])
            xb = torch.rot90(xb, int(rng.integers(0, 4)), dims=[2, 3])
            logit, _ = net(xb)
            loss = (nn.functional.binary_cross_entropy_with_logits(
                logit, yt[idx], reduction="none") * wt[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


@torch.no_grad()
def predict(net: PatchNet, X: np.ndarray, batch: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    net.eval()
    probs, embs = [], []
    for s in range(0, len(X), batch):
        logit, z = net(_tensor(X[s:s + batch]))
        probs.append(torch.sigmoid(logit).numpy())
        embs.append(z.numpy())
    return np.concatenate(probs), np.concatenate(embs)
