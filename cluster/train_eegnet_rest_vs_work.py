"""EEGNet-8,2 rest vs working for the demo page.

Same people split, clips and calibration as the LaBraM demo model (labram_cut3):
  train UN_101-108, validation UN_109-110 (epoch pick), test UN_111-112.
  4-s clips at 200 Hz (data/processed/*_clips_4s_200hz.npz), first half of each session's rest video = calibration only.
Calibration here: each channel is z-scored with the session's calibration clips (mean/std over time and clips).
Scoring: official 60-s windows (15 clips averaged, > 0.5 = working), balanced accuracy.
Writes demo_outputs/eegnet_rest_vs_work/{result.json, eegnet_demo.pt, eegnet_minutes.json}.
"""
import json, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

OUT = "demo_outputs/eegnet_rest_vs_work"
TRAIN = [f"UN_{i}" for i in range(101, 109)]
VAL, TEST = ["UN_109", "UN_110"], ["UN_111", "UN_112"]
SHOW = {"UN_109": "validation", "UN_110": "validation", "UN_111": "test", "UN_112": "test"}
SEEDS, EPOCHS, STEPS, BATCH = [0, 1, 2], 30, 150, 128
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device", dev)

task = np.load("data/processed/raw_clips_4s_200hz.npz")
rest = np.load("data/processed/rest_clips_4s_200hz.npz")
X = np.concatenate([rest["X"], task["X"]]).astype(np.float32)
y = np.r_[np.zeros(len(rest["X"]), np.int64), np.ones(len(task["X"]), np.int64)]
person = np.r_[rest["person"], task["person"]]
session = np.r_[rest["session"], task["session"]]
rec = np.r_[[f"{p}|{s}|rest" for p, s in zip(rest["person"], rest["session"])],
            [f"{p}|{s}|{t}|{l}" for p, s, t, l in zip(task["person"], task["session"], task["task"], task["y"])]]
assert set(person) <= set(TRAIN + VAL + TEST)

cal_of, use_of = {}, {}
for p, s in sorted(set(zip(person, session))):
    sess = np.where((person == p) & (session == s))[0]
    r = sess[y[sess] == 0]
    if len(r) == 0:
        continue
    cal_of[(p, s)] = r[: len(r) // 2]
    use_of[(p, s)] = np.setdiff1d(sess, cal_of[(p, s)])

# per-session channel z-score from calibration clips
Xc = np.empty_like(X)
for (p, s), cal in cal_of.items():
    sess = (person == p) & (session == s)
    m = X[cal].mean(axis=(0, 2), keepdims=True)[0]
    sd = X[cal].std(axis=(0, 2), keepdims=True)[0] + 1e-6
    Xc[sess] = (X[sess] - m) / sd
Xc = np.clip(Xc, -20, 20)


class EEGNet(nn.Module):
    """EEGNet-8,2 (Lawhern et al. 2018), 4 channels x 800 samples at 200 Hz."""
    def __init__(self, C=4, T=800, F1=8, D=2, F2=16, drop=0.5):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, F1, (1, 100), padding=(0, 50), bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(drop),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(drop), nn.Flatten())
        with torch.no_grad():
            n = self.f(torch.zeros(1, 1, C, T)).shape[1]
        self.head = nn.Linear(n, 2)

    def forward(self, x):
        return self.head(self.f(x.unsqueeze(1)))


def stretch(prob, label, rids, k):
    pr, tr = [], []
    for rid in pd.unique(rids):
        ii = np.where(rids == rid)[0]
        for st in range(0, len(ii) - k + 1, k):
            pr.append(prob[ii[st:st + k]].mean() > 0.5)
            tr.append(label[ii[st]])
    pr, tr = np.array(pr), np.array(tr)
    r, w = (pr[tr == 0] == 0).mean(), (pr[tr == 1] == 1).mean()
    return round(float((r + w) / 2), 4), round(float(r), 4), round(float(w), 4)


@torch.no_grad()
def predict(net, idx):
    net.eval()
    out = [torch.softmax(net(torch.from_numpy(Xc[idx[i:i + 1024]]).to(dev)), 1)[:, 1].cpu().numpy()
           for i in range(0, len(idx), 1024)]
    return np.concatenate(out) if out else np.zeros(0)


def used(people):
    return np.sort(np.concatenate([u for (p, s), u in use_of.items() if p in people]))


tr_idx = np.sort(np.concatenate([np.where((person == p) & (session == s))[0]
                                 for (p, s) in use_of if p in TRAIN]))
tr_idx = np.setdiff1d(tr_idx, np.concatenate([cal_of[k] for k in cal_of if k[0] in TRAIN]))
tr0, tr1 = tr_idx[y[tr_idx] == 0], tr_idx[y[tr_idx] == 1]
va = used(VAL)
print("train clips rest/work", len(tr0), len(tr1), "val", len(va))

best_overall = None
history = []
for seed in SEEDS:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = EEGNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    for ep in range(1, EPOCHS + 1):
        net.train()
        L = []
        for _ in range(STEPS):
            ii = np.r_[rng.choice(tr0, BATCH // 2), rng.choice(tr1, BATCH // 2)]
            xb = torch.from_numpy(Xc[ii]).to(dev)
            xb = xb * (1 + 0.1 * torch.randn(len(ii), 4, 1, device=dev))  # channel gain jitter
            loss = lossf(net(xb), torch.from_numpy(y[ii]).to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
            L.append(loss.item())
        b60 = stretch(predict(net, va), y[va], rec[va], 15)
        history.append({"seed": seed, "epoch": ep, "loss": round(float(np.mean(L)), 4), "val60": b60})
        print(f"seed {seed} ep {ep} loss {np.mean(L):.3f} val60 {b60}", flush=True)
        if best_overall is None or b60[0] > best_overall[0]:
            best_overall = (b60[0], seed, ep, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})

val_best, seed, ep, sd = best_overall
net = EEGNet().to(dev)
net.load_state_dict(sd)
p_all = np.zeros(len(X))
allu = used(TRAIN + VAL + TEST)
p_all[allu] = predict(net, allu)

res = {"pick": {"seed": seed, "epoch": ep, "val60": val_best}, "history": history}
for split, ppl in [("validation", VAL), ("test", TEST)]:
    for who in ["ALL"] + ppl:
        ii = used(ppl if who == "ALL" else [who])
        res[f"{split}|{who}|60s"] = stretch(p_all[ii], y[ii], rec[ii], 15)
        res[f"{split}|{who}|4s"] = stretch(p_all[ii], y[ii], rec[ii], 1)
print({k: v for k, v in res.items() if "|" in k})

# per-minute rows in the same order as export_demo_extras.py / demo_predictions.csv
rows = []
for (p, s), cal in cal_of.items():
    if p not in SHOW:
        continue
    idx_all = np.where((person == p) & (session == s))[0]
    idx_all = np.setdiff1d(idx_all, cal)
    for rid in pd.unique(rec[idx_all]):
        idx = idx_all[rec[idx_all] == rid]
        for st in range(0, len(idx) - 15 + 1, 15):
            rows.append(round(float(p_all[idx[st:st + 15]].mean()), 4))
csv = pd.read_csv("results/demo_predictions.csv")
assert len(rows) == len(csv), (len(rows), len(csv))
res["n_minutes"] = len(rows)

os.makedirs(OUT, exist_ok=True)
torch.save({"state_dict": sd, "arch": "EEGNet-8,2 C=4 T=800", "calibration": "per-session channel z-score from rest calibration clips"},
           f"{OUT}/eegnet_demo.pt")
json.dump({"eegnet_prob": rows}, open(f"{OUT}/eegnet_minutes.json", "w"))
json.dump(res, open(f"{OUT}/result.json", "w"), indent=1)
print("saved", OUT)
