"""Easy-vs-hard answers for the demo page (MA416 team 10). Read-only on team data and models.

For every working minute in results/demo_predictions.csv (same row order as export_demo_predictions.py),
the chance that the minute is the HARD level of its task, from four models trained on UN_101-108:
  labram : models/labram_cut3_easy_vs_hard/best.pt (code/LaBraM/labram_cut3_easy_vs_hard.py), as saved
  eegnet : models/eegnet_ours/eegnet_easy_vs_hard.pt (code/Experiments/eegnet_probe_rest_vs_work.py), as saved
  simple : same 20 calibrated band-power numbers + logistic regression as the demo's rest-vs-working model,
           refit on easy vs hard task clips
  motion : same 10 head-motion numbers + logistic regression as demo_extras, refit on easy vs hard
Rest minutes get null. Scores use the team's 60-s rule (15 clips averaged, > 0.5 = hard).
Writes demo_outputs/easy_vs_hard/easy_hard.json
"""
import json, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import welch, resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "code/LaBraM/LaBraM-official")
from modeling_finetune import labram_base_patch200_200

OUT = "demo_outputs/easy_vs_hard"
os.makedirs(OUT, exist_ok=True)
TRAIN = [f"UN_{i}" for i in range(101, 109)]
SHOW = {"UN_109": "validation", "UN_110": "validation", "UN_111": "test", "UN_112": "test"}
BANDS = [(1, 4), (4, 7), (8, 12), (13, 30), (30, 45)]
CH = ["TP9", "AF7", "AF8", "TP10"]
INPUT_CHANS = [0, 48, 5, 13, 58]
RAW = "data/raw/UNIVERSE"
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))

task = np.load("data/processed/raw_clips_4s_200hz.npz")
rest = np.load("data/processed/rest_clips_4s_200hz.npz")
X = np.concatenate([rest["X"], task["X"]]).astype(np.float32)
y = np.r_[np.zeros(len(rest["X"]), int), np.ones(len(task["X"]), int)]
hard = np.r_[np.full(len(rest["X"]), -1), task["y"]].astype(int)
person = np.r_[rest["person"], task["person"]]
session = np.r_[rest["session"], task["session"]]
rec = np.r_[[f"{p}|{s}|rest" for p, s in zip(rest["person"], rest["session"])],
            [f"{p}|{s}|{t}|{l}" for p, s, t, l in zip(task["person"], task["session"], task["task"], task["y"])]]
del task, rest

is_cal = np.zeros(len(y), bool)
cal_of = {}
for p, s in sorted(set(zip(person, session))):
    r = np.where((person == p) & (session == s) & (y == 0))[0]
    if len(r) > 0:
        is_cal[r[: len(r) // 2]] = True
        cal_of[(p, s)] = r[: len(r) // 2]
has_rest = np.array([(p, s) in cal_of for p, s in zip(person, session)])
show_task = np.isin(person, list(SHOW)) & (y == 1) & has_rest
train_task = np.isin(person, TRAIN) & (y == 1) & has_rest
prob = {k: np.full(len(X), np.nan) for k in ["labram", "eegnet", "simple", "motion"]}


def calibrate(F):
    F = F.copy()
    for (p, s), cal in cal_of.items():
        sess = (person == p) & (session == s)
        F[sess] = (F[sess] - F[cal].mean(0)) / (F[cal].std(0) + 1e-6)
    return F


# ---- simple: calibrated band power, refit on easy vs hard
f, psd = welch(X, fs=200, nperseg=400, axis=-1)
F = calibrate(np.log(np.stack([psd[..., (f >= lo) & (f < hi)].mean(-1) for lo, hi in BANDS], -1)).reshape(len(X), -1))
del psd
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(F[train_task], hard[train_task])
prob["simple"][show_task] = clf.predict_proba(F[show_task])[:, 1]
print("simple done", flush=True)


# ---- motion: same 10 features as demo_extras, refit on easy vs hard
def folder(rid):
    parts = rid.split("|")
    name = "relaxation_video" if parts[2] == "rest" else parts[2] + ("_hard" if parts[3] == "1" else "_easy")
    return f"{RAW}/{parts[0]}/{parts[1]}/Labeled/{name}"


def load(d, name, n):
    return pd.read_pickle(f"{d}/{name}.pickle").interpolate().bfill().ffill().to_numpy()[:n]


need = has_rest & (np.isin(person, TRAIN) | np.isin(person, list(SHOW)))
M = np.full((len(X), 10), np.nan)
for rid in pd.unique(rec[need]):
    idx = np.where(rec == rid)[0]
    d = folder(rid)
    n = min(len(pd.read_pickle(f"{d}/muse_{c}_RAW.pickle")) for c in CH)
    k = int(np.ceil(n * 25 / 32)) // 800
    assert k == len(idx), (rid, k, len(idx))
    G = np.stack([load(d, f"muse_GYRO_{a}", n) for a in "XYZ"]).astype(float)
    A = np.stack([load(d, f"muse_ACC_{a}", n) for a in "XYZ"]).astype(float)
    G = G[:, :k * 1024].reshape(3, k, 1024).transpose(1, 0, 2)
    A = A[:, :k * 1024].reshape(3, k, 1024).transpose(1, 0, 2)
    gm, am = np.linalg.norm(G, axis=1), np.linalg.norm(A, axis=1)
    cols = [np.log(G[:, j].std(-1) + 1e-6) for j in range(3)] + [np.log(A[:, j].std(-1) + 1e-6) for j in range(3)]
    cols += [np.log(gm.mean(-1) + 1e-6), np.log(gm.std(-1) + 1e-6), np.log(am.std(-1) + 1e-6),
             np.log(np.abs(np.diff(am, axis=-1)).mean(-1) * 256 + 1e-6)]
    M[idx] = np.stack(cols, -1)
Mc = np.full_like(M, np.nan)
for (p, s), cal in cal_of.items():
    sess = (person == p) & (session == s)
    if np.isnan(M[sess]).any():
        continue
    Mc[sess] = (M[sess] - M[cal].mean(0)) / (M[cal].std(0) + 1e-6)
assert not np.isnan(Mc[train_task | show_task]).any()
clf_m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(Mc[train_task], hard[train_task])
prob["motion"][show_task] = clf_m.predict_proba(Mc[show_task])[:, 1]
print("motion done", flush=True)


# ---- LaBraM easy vs hard, saved model, embedding minus the session's calibration mean
class CutLaBraM(nn.Module):
    def __init__(self):
        super().__init__()
        bb = labram_base_patch200_200(init_values=0.1, qkv_bias=False, use_mean_pooling=False, num_classes=0)
        bb.blocks = nn.ModuleList(list(bb.blocks)[:3])
        bb.norm = nn.Identity()
        self.backbone = bb
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(200, 2)

    def embed(self, x):
        return self.backbone.forward_features(x.reshape(-1, 4, 4, 200), input_chans=INPUT_CHANS, return_patch_tokens=True).mean(1)


lab = CutLaBraM()
lab.load_state_dict(torch.load("models/labram_cut3_easy_vs_hard/best.pt", map_location="cpu"))
lab.eval()
with torch.no_grad():
    for (p, s), cal in cal_of.items():
        if p not in SHOW:
            continue
        base = lab.embed(torch.from_numpy(X[cal])).mean(0, keepdim=True)
        use = np.where((person == p) & (session == s) & (y == 1))[0]
        for i in range(0, len(use), 256):
            b = use[i:i + 256]
            prob["labram"][b] = torch.softmax(lab.head(lab.embed(torch.from_numpy(X[b])) - base), 1)[:, 1].numpy()
print("labram done", flush=True)


# ---- EEGNet easy vs hard, saved model; input exactly as eegnet_probe_rest_vs_work.py builds it
class EEGNet(nn.Module):
    def __init__(self, Chans=4, Samples=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), padding="same", bias=False), nn.BatchNorm2d(8, eps=0.001, momentum=0.01),
            nn.Conv2d(8, 16, (Chans, 1), groups=8, bias=False), nn.BatchNorm2d(16, eps=0.001, momentum=0.01),
            nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 16), padding="same", groups=16, bias=False), nn.Conv2d(16, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16, eps=0.001, momentum=0.01), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(0.5))
        self.fc1 = nn.Linear(16 * (Samples // 4 // 8), 2)

    def forward(self, x):
        return self.fc1(self.features(x).flatten(start_dim=1))


def to_eegnet(idx):
    Z = np.concatenate([resample_poly(X[idx[i:i + 2000]], 16, 25, axis=-1) for i in range(0, len(idx), 2000)]).astype(np.float32)
    Z -= Z.mean(1, keepdims=True)
    return Z.reshape(len(Z), 4, 2, 256).transpose(0, 2, 1, 3).copy()


tr_all = np.where(np.isin(person, TRAIN) & (y == 1))[0]  # the probe script's scaling set (all training task clips)
Ztr = to_eegnet(tr_all)
mean, std = Ztr.mean(axis=(0, 1, 3), keepdims=True), Ztr.std(axis=(0, 1, 3), keepdims=True)
del Ztr
net = EEGNet()
net.load_state_dict(torch.load("models/eegnet_ours/eegnet_easy_vs_hard.pt", map_location="cpu"))
net.eval()
sidx = np.where(show_task)[0]
Z = ((to_eegnet(sidx) - mean) / std).astype(np.float32)
with torch.no_grad():
    pw = torch.softmax(net(torch.from_numpy(Z.reshape(-1, 1, 4, 256))), 1)[:, 1].numpy().reshape(-1, 2).mean(1)
prob["eegnet"][sidx] = pw
print("eegnet done", flush=True)


# ---- scores, team 60-s rule: balanced over easy and hard minutes
def stretch(pr, lab_, rids, k):
    P, T = [], []
    for rid in pd.unique(rids):
        ii = np.where(rids == rid)[0]
        for st in range(0, len(ii) - k + 1, k):
            P.append(pr[ii[st:st + k]].mean() > 0.5)
            T.append(lab_[ii[st]])
    P, T = np.array(P), np.array(T)
    e, h = (P[T == 0] == 0).mean(), (P[T == 1] == 1).mean()
    return [round((e + h) / 2, 4), round(e, 4), round(h, 4), int((T == 0).sum()), int((T == 1).sum())]


metrics = {}
for split, ppl in [("validation", ["UN_109", "UN_110"]), ("test", ["UN_111", "UN_112"])]:
    for who in ["ALL"] + ppl:
        m = np.isin(person, ppl if who == "ALL" else [who]) & show_task
        for name in prob:
            metrics[f"{split}|{who}|{name}"] = stretch(prob[name][m], hard[m], rec[m], 15)
for k_, v in metrics.items():
    print(k_, v)

# ---- rows in demo order
rows = []
for (p, s), cal in cal_of.items():
    if p not in SHOW:
        continue
    idx_all = np.where((person == p) & (session == s) & ~is_cal)[0]
    for rid in pd.unique(rec[idx_all]):
        idx = idx_all[rec[idx_all] == rid]
        for st in range(0, len(idx) - 15 + 1, 15):
            ch = idx[st:st + 15]
            rows.append(None if y[ch[0]] == 0 else {k: round(float(prob[k][ch].mean()), 5) for k in prob})
csv = pd.read_csv("results/demo_predictions.csv")
assert len(rows) == len(csv), (len(rows), len(csv))
assert all((r is None) == (t == "Rest") for r, t in zip(rows, csv.truth))
json.dump({"rows": rows, "metrics": metrics}, open(f"{OUT}/easy_hard.json", "w"), separators=(",", ":"))
print("saved", OUT, len(rows))
