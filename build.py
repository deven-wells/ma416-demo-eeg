"""Build docs/data/demo.json from the cluster exports in data_in/ (stdlib only).

Inputs (copied off the cluster, see README):
  data_in/demo_predictions.csv  - Umesh's per-minute export (LaBraM + simple model)
  data_in/demo_extras.json      - cluster/export_demo_extras.py output (live features,
                                  head-motion probabilities, recording order)
Run:  python build.py
"""
import csv, json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "data_in")
OUT = os.path.join(HERE, "docs", "data", "demo.json")

TASK_NAMES = {"arithmetix": "Mental arithmetic", "n_back": "N-back memory", "stroop": "Stroop colour-word",
              "sudoku": "Sudoku", "rest": "Rest (relaxation video)"}

rows = list(csv.DictReader(open(os.path.join(IN, "demo_predictions.csv"), newline="")))
ext = json.load(open(os.path.join(IN, "demo_extras.json")))
assert len(rows) == len(ext["rows"])
eeg_net = json.load(open(os.path.join(IN, "eegnet_minutes.json")))["eegnet_prob"]  # cluster/train_eegnet_rest_vs_work.py, same row order
assert len(eeg_net) == len(rows)
eh = json.load(open(os.path.join(IN, "easy_hard.json")))  # cluster/export_easy_vs_hard.py, same row order; None for rest
assert len(eh["rows"]) == len(rows)

sessions = {}
for row_i, (r, e, pe) in enumerate(zip(rows, ext["rows"], eeg_net)):
    assert r["session_id"] == e["session_id"] and r["activity"] == e["activity"]
    assert int(r["minute_in_activity"]) == e["minute_in_activity"]
    assert abs(float(r["simple_prob"]) - e["simple_prob_recomputed"]) < 1e-9
    person, lab = r["session_id"].split()[:2]
    split = r["session_id"].split("(")[1].rstrip(")")
    sid = f"{person} {lab}"
    s = sessions.setdefault(sid, {"id": sid, "person": person, "lab": lab, "split": split, "recs": {}})
    parts = e["rec"].split("|")
    task = parts[2]
    level = None if task == "rest" else ("hard" if parts[3] == "1" else "easy")
    rec = s["recs"].setdefault(e["rec"], {
        "task": task, "task_name": TASK_NAMES[task], "level": level,
        "truth": r["truth"], "start_row": e["rec_start_row"], "rows": e["rec_rows"],
        "clips": e["rec_clips"], "minutes": []})
    rec["minutes"].append({
        "m": int(r["minute_in_activity"]),
        "row": row_i,  # index into demo_predictions.csv and signals.json
        "first_clip": e["first_clip_in_rec"],
        "labram": round(float(r["labram_prob"]), 5),
        "eegnet": round(float(pe), 5),
        "simple": round(float(r["simple_prob"]), 5),
        "motion": round(e["motion_prob"], 5),
        "eh": eh["rows"][row_i],  # easy-vs-hard chance of HARD per model (None for rest minutes)
        # 15 clips x 20 calibrated band-power features (4 sensors x 5 bands)
        "feat": [[round(v, 3) for v in clip] for clip in e["feat"]],
    })

out_sessions = []
for s in sessions.values():
    recs = sorted(s["recs"].values(), key=lambda x: x["start_row"])
    t0 = recs[0]["start_row"]
    for x in recs:
        # pickle row / 256 Hz, approximate (rows are not a perfect clock, see Data Pipeline Audit)
        x["approx_start_min"] = round((x["start_row"] - t0) / 256 / 60, 1)
    s["recs"] = recs
    out_sessions.append(s)
out_sessions.sort(key=lambda s: (s["split"] != "test", s["id"]))

metrics = json.load(open(os.path.join(IN, "metrics.json")))
live = ext["live_simple"]
doc = {
    "sessions": out_sessions,
    "live_simple": live,
    "motion_cluster_metrics": metrics,
    "motion_features": ext["motion_features"],
    "eh_cluster_metrics": eh["metrics"],
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(doc, open(OUT, "w"), separators=(",", ":"))

# self-check: live model in pure Python reproduces the CSV simple_prob
worst = 0.0
for s in out_sessions:
    for x in s["recs"]:
        for m in x["minutes"]:
            ps = []
            for clip in m["feat"]:
                z = sum(c * (v - mu) / sd for c, v, mu, sd in zip(live["coef"], clip, live["scaler_mean"], live["scaler_scale"]))
                ps.append(1 / (1 + math.exp(-(z + live["intercept"]))))
            worst = max(worst, abs(sum(ps) / len(ps) - m["simple"]))
print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB); sessions={len(out_sessions)}; "
      f"live-vs-csv max diff after 3-dp rounding = {worst:.5f}")
