"""Demo-site extras. Read-only on teammates' files.

Rebuilds the exact rows of results/demo_predictions.csv and adds:
  * live: the simple model's per-4-s-clip calibrated features + scaler + logistic weights,
          so the browser can recompute simple_prob itself;
  * motion_prob: a head-motion-only model (Muse gyro + accelerometer, 10 features per 4-s clip,
          same people split UN_101-108 train, same first-half-of-rest calibration, same
          logistic regression), averaged per minute exactly like the EEG models;
  * rec_start_row: first Labeled-pickle row of each recording, used only to order the timeline.
Writes code/Experiments/demo_site/out/demo_extras.json and metrics.json.
"""
import json, os
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = "code/Experiments/demo_site/out"
os.makedirs(OUT, exist_ok=True)
TRAIN = [f"UN_{i}" for i in range(101, 109)]
SHOW = {"UN_109": "validation", "UN_110": "validation", "UN_111": "test", "UN_112": "test"}
BANDS = [(1, 4), (4, 7), (8, 12), (13, 30), (30, 45)]
CH = ["TP9", "AF7", "AF8", "TP10"]
RAW = "data/raw/UNIVERSE"

task = np.load("data/processed/raw_clips_4s_200hz.npz")
rest = np.load("data/processed/rest_clips_4s_200hz.npz")
X = np.concatenate([rest["X"], task["X"]]).astype(np.float32)
y = np.r_[np.zeros(len(rest["X"]), int), np.ones(len(task["X"]), int)]
person = np.r_[rest["person"], task["person"]]
session = np.r_[rest["session"], task["session"]]
rec = np.r_[[f"{p}|{s}|rest" for p, s in zip(rest["person"], rest["session"])],
            [f"{p}|{s}|{t}|{l}" for p, s, t, l in zip(task["person"], task["session"], task["task"], task["y"])]]
print("clips", len(X), "people", sorted(set(person)))

is_cal = np.zeros(len(y), bool)
cal_of = {}
for p, s in sorted(set(zip(person, session))):
    r = np.where((person == p) & (session == s) & (y == 0))[0]
    if len(r) > 0:
        is_cal[r[: len(r) // 2]] = True
        cal_of[(p, s)] = r[: len(r) // 2]
has_rest = np.array([(p, s) in cal_of for p, s in zip(person, session)])
train = np.isin(person, TRAIN) & has_rest


def calibrate(F):
    F = F.copy()
    for (p, s), cal in cal_of.items():
        sess = (person == p) & (session == s)
        F[sess] = (F[sess] - F[cal].mean(0)) / (F[cal].std(0) + 1e-6)
    return F


# ---- simple EEG model, identical to export_demo_predictions.py
f, psd = welch(X, fs=200, nperseg=400, axis=-1)
F = np.log(np.stack([psd[..., (f >= lo) & (f < hi)].mean(-1) for lo, hi in BANDS], -1)).reshape(len(X), -1)
del psd
Fc = calibrate(F)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(Fc[train], y[train])
p_simple = clf.predict_proba(Fc)[:, 1]


# ---- head-motion features per clip, same pickle rows as the EEG clips
def folder(rid):
    parts = rid.split("|")
    name = "relaxation_video" if parts[2] == "rest" else parts[2] + ("_hard" if parts[3] == "1" else "_easy")
    return f"{RAW}/{parts[0]}/{parts[1]}/Labeled/{name}"


def load(d, name, n):
    return pd.read_pickle(f"{d}/{name}.pickle").interpolate().bfill().ffill().to_numpy()[:n]


FS = 256
M = np.full((len(X), 10), np.nan)
start_row, n_rows = {}, {}
for rid in pd.unique(rec):
    idx = np.where(rec == rid)[0]
    d = folder(rid)
    eeg = [pd.read_pickle(f"{d}/muse_{c}_RAW.pickle") for c in CH]
    n = min(len(e) for e in eeg)
    k = int(np.ceil(n * 25 / 32)) // 800
    assert k == len(idx), (rid, k, len(idx))
    start_row[rid] = int(eeg[0].index[0])
    n_rows[rid] = int(n)
    G = np.stack([load(d, f"muse_GYRO_{a}", n) for a in "XYZ"]).astype(float)
    A = np.stack([load(d, f"muse_ACC_{a}", n) for a in "XYZ"]).astype(float)
    G = G[:, :k * 1024].reshape(3, k, 1024).transpose(1, 0, 2)
    A = A[:, :k * 1024].reshape(3, k, 1024).transpose(1, 0, 2)
    gm, am = np.linalg.norm(G, axis=1), np.linalg.norm(A, axis=1)
    cols = [np.log(G[:, j].std(-1) + 1e-6) for j in range(3)] + [np.log(A[:, j].std(-1) + 1e-6) for j in range(3)]
    cols += [np.log(gm.mean(-1) + 1e-6), np.log(gm.std(-1) + 1e-6), np.log(am.std(-1) + 1e-6),
             np.log(np.abs(np.diff(am, axis=-1)).mean(-1) * FS + 1e-6)]
    M[idx] = np.stack(cols, -1)
assert not np.isnan(M).any()
Mc = calibrate(M)
clf_m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(Mc[train], y[train])
p_motion = clf_m.predict_proba(Mc)[:, 1]


# ---- official-style score (same as stretch_scores in the team scripts)
def stretch(prob, label, rids, k):
    pr, tr = [], []
    for rid in pd.unique(rids):
        ii = np.where(rids == rid)[0]
        for st in range(0, len(ii) - k + 1, k):
            pr.append(prob[ii[st:st + k]].mean() > 0.5)
            tr.append(label[ii[st]])
    pr, tr = np.array(pr), np.array(tr)
    r, w = (pr[tr == 0] == 0).mean(), (pr[tr == 1] == 1).mean()
    return round((r + w) / 2, 4), round(r, 4), round(w, 4)


metrics = {}
for split, ppl in [("validation", ["UN_109", "UN_110"]), ("test", ["UN_111", "UN_112"])]:
    for who in ["ALL"] + ppl:
        m = np.isin(person, ppl if who == "ALL" else [who]) & has_rest & ~is_cal
        for name, pr in [("simple", p_simple), ("motion", p_motion)]:
            metrics[f"{split}|{who}|{name}|60s"] = stretch(pr[m], y[m], rec[m], 15)
            metrics[f"{split}|{who}|{name}|4s"] = stretch(pr[m], y[m], rec[m], 1)
for k_, v in metrics.items():
    print(k_, v)

# ---- rebuild demo rows in the same order as export_demo_predictions.py
csv = pd.read_csv("results/demo_predictions.csv")
rows = []
for (p, s), cal in cal_of.items():
    if p not in SHOW:
        continue
    idx_all = np.where((person == p) & (session == s) & ~is_cal)[0]
    for rid in pd.unique(rec[idx_all]):
        idx = idx_all[rec[idx_all] == rid]
        parts = rid.split("|")
        activity = "Rest (relaxation video)" if parts[2] == "rest" else f"{parts[2]} ({'hard' if parts[3] == '1' else 'easy'})"
        n_cal = len(cal) if parts[2] == "rest" else 0
        for mi, st in enumerate(range(0, len(idx) - 15 + 1, 15), 1):
            ch = idx[st:st + 15]
            rows.append({"session_id": f"{p} {s} ({SHOW[p]})", "activity": activity, "minute_in_activity": mi,
                         "rec": rid, "rec_start_row": start_row[rid], "rec_rows": n_rows[rid],
                         "rec_clips": int((rec == rid).sum()), "first_clip_in_rec": int(n_cal + st),
                         "simple_prob_recomputed": float(p_simple[ch].mean()),
                         "motion_prob": float(p_motion[ch].mean()),
                         "feat": np.round(Fc[ch], 4).tolist()})
assert len(rows) == len(csv), (len(rows), len(csv))
for r, (_, c) in zip(rows, csv.iterrows()):
    assert r["session_id"] == c.session_id and r["activity"] == c.activity and r["minute_in_activity"] == c.minute_in_activity
diff = np.abs(np.array([r["simple_prob_recomputed"] for r in rows]) - csv.simple_prob.to_numpy()).max()
print("max |simple recomputed - csv| =", diff)

sc, lr = clf.named_steps["standardscaler"], clf.named_steps["logisticregression"]
live = {"feature_names": [f"{c} {b}" for c in CH for b in ["delta", "theta", "alpha", "beta", "gamma"]],
        "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
        "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}
mlr = clf_m.named_steps["logisticregression"]
json.dump({"rows": rows, "live_simple": live, "max_simple_diff": float(diff),
           "motion_coef": mlr.coef_[0].tolist(),
           "motion_features": ["gyro x sd", "gyro y sd", "gyro z sd", "acc x sd", "acc y sd", "acc z sd",
                               "gyro magnitude mean", "gyro magnitude sd", "acc magnitude sd", "acc jerk"]},
          open(f"{OUT}/demo_extras.json", "w"))
json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=1)
print("saved", OUT)
