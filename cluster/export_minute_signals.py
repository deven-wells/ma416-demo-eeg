"""Per-minute signal snippets for the demo page. Read-only on team data.

For every row of results/demo_predictions.csv (same order as export_demo_extras.py), writes:
  raw  : the minute's middle 4-s clip (clip 8 of 15) exactly as LaBraM / EEGNet receive it
         (0.1-75 Hz band-pass, 50 Hz notch, 200 Hz), in microvolts, down-sampled to 100 Hz for display,
         int16 little-endian, base64, shape 4 x 400 (TP9, AF7, AF8, TP10)
  spec : the minute's average power spectrum per sensor, log10(uV^2/Hz), 1..45 Hz in 1-Hz steps (4 x 45)
  rest_spec : the same spectrum averaged over the session's calibration clips (first half of rest video)
Writes demo_outputs/minute_signals/signals.json
"""
import base64, json, os
import numpy as np
import pandas as pd
from scipy.signal import welch, resample_poly

OUT = "demo_outputs/minute_signals"
SHOW = {"UN_109", "UN_110", "UN_111", "UN_112"}

task = np.load("data/processed/raw_clips_4s_200hz.npz")
rest = np.load("data/processed/rest_clips_4s_200hz.npz")
X = np.concatenate([rest["X"], task["X"]]).astype(np.float32) * 100.0  # back to microvolts
y = np.r_[np.zeros(len(rest["X"]), int), np.ones(len(task["X"]), int)]
person = np.r_[rest["person"], task["person"]]
session = np.r_[rest["session"], task["session"]]
rec = np.r_[[f"{p}|{s}|rest" for p, s in zip(rest["person"], rest["session"])],
            [f"{p}|{s}|{t}|{l}" for p, s, t, l in zip(task["person"], task["session"], task["task"], task["y"])]]

cal_of = {}
for p, s in sorted(set(zip(person, session))):
    r = np.where((person == p) & (session == s) & (y == 0))[0]
    if len(r) > 0:
        cal_of[(p, s)] = r[: len(r) // 2]


def spectrum(idx):
    f, psd = welch(X[idx], fs=200, nperseg=400, axis=-1)  # 0.5 Hz bins
    psd = psd.mean(0)  # (4, F)
    out = [[float(np.log10(psd[c, (f >= hz - 0.5) & (f < hz + 0.5)].mean() + 1e-12)) for hz in range(1, 46)] for c in range(4)]
    return np.round(out, 3).tolist()


rows, rest_spec = [], {}
for (p, s), cal in cal_of.items():
    if p not in SHOW:
        continue
    rest_spec[f"{p} {s}"] = spectrum(cal)
    idx_all = np.setdiff1d(np.where((person == p) & (session == s))[0], cal)
    for rid in pd.unique(rec[idx_all]):
        idx = idx_all[rec[idx_all] == rid]
        for st in range(0, len(idx) - 15 + 1, 15):
            ch = idx[st:st + 15]
            clip = resample_poly(X[ch[7]], 1, 2, axis=-1)  # 4 x 400 at 100 Hz
            q = np.clip(np.round(clip), -32767, 32767).astype("<i2")
            rows.append({"raw": base64.b64encode(q.tobytes()).decode(), "spec": spectrum(ch)})

csv = pd.read_csv("results/demo_predictions.csv")
assert len(rows) == len(csv), (len(rows), len(csv))
os.makedirs(OUT, exist_ok=True)
json.dump({"fs": 100, "channels": ["TP9", "AF7", "AF8", "TP10"], "hz": list(range(1, 46)),
           "rows": rows, "rest_spec": rest_spec}, open(f"{OUT}/signals.json", "w"), separators=(",", ":"))
print("rows", len(rows), "MB", os.path.getsize(f"{OUT}/signals.json") / 1e6)
