# Rest or Working? EEG demo (MA416 team 10)

A static web page for the CSSE/MA416 Deep Learning term project (Rose-Hulman, Fall 2026, team 10).
It follows one minute of 4-sensor Muse S EEG from raw wave to model answer, and asks whether models can tell
**rest** (watching a relaxation video) from **working** (arithmetic, n-back, Stroop, sudoku).

Live page: https://deven-wells.github.io/ma416-demo-eeg/ (the old `/v2/` link now redirects there).

The page does **rest vs working** as its main question, with an easy-vs-hard switch that shows how much harder
that second question is. It does not measure workload, and it does not use any data beyond four UNIVERSE participants.

## What the page shows

- **How it works.** "Two ways into the same signal": a diagram of the raw-wave path (LaBraM, EEGNet) and the
  band-power path (Simple model).
- **Pick a person and session**, then **Inside one minute.** The minute's raw wave, its spectrum against the person's
  own rest, the 20 band-power numbers (5 bands × 4 sensors), and each model's answer. A rest-vs-working /
  easy-vs-hard switch changes the question. An optional quiz hides the answer until you guess; on phones the guess
  bar sits at the bottom of the screen.
- **Session timeline.** Every scored minute of one session, in recording order, with each model's probability line.
- **Scoreboard.** Balanced accuracy per person and overall, computed in the browser and checked against the team's
  official result files (see below).
- **How to read this.** The limits: rest vs working only, head motion as a strong rival explanation, calibration from
  the same video, the tiny test set, and rest always coming first in a session.

## Models

| Line | What it is | Where it runs |
|---|---|---|
| LaBraM | First 3 layers of the pretrained LaBraM EEG transformer plus a 2-class head, with a rest-calibration offset (`models/labram_cut3/best.pt`) | Precomputed on the cluster |
| EEGNet | EEGNet-8,2 trained on the same people, clips and calibration as LaBraM (`cluster/train_eegnet_rest_vs_work.py`) | Precomputed on the cluster |
| Simple | Log band power (5 bands × 4 sensors), z-scored against the first half of the session's rest video, then StandardScaler and logistic regression | **Live in the browser** from saved features and weights |
| Head motion | Same recipe as Simple, but with 10 Muse gyroscope/accelerometer features instead of EEG | Precomputed on the cluster |

All models were trained on UNIVERSE UN_101 to UN_108. UN_109 and UN_110 are validation (used to pick the LaBraM epoch).
UN_111 and UN_112 are the final test. UN_112 has one lab day, so only that session is scored.

Final-test balanced accuracy, rest vs working: LaBraM 0.805, head motion 0.785, Simple 0.773, EEGNet 0.643.
Easy vs hard: LaBraM 0.577, head motion 0.513, Simple 0.509, EEGNet 0.496.

## The scoring rule, and why the page matches the official numbers

The official metric scores **60-second windows**. Each recording is cut into back-to-back runs of 15 four-second
clips, the 15 probabilities are averaged, and an average above 0.5 counts as "working". Balanced accuracy is
the mean of rest-correct and working-correct. Every row of `demo_predictions.csv` is exactly one of those windows,
so the page reproduces the official numbers to 4 decimals:

| | Page | Official file |
|---|---|---|
| LaBraM, test (both) | 0.8053 | `results/final_test_labram_cut3.csv` |
| LaBraM, UN_111 / UN_112 | 0.7758 / 0.8861 | same |
| LaBraM, validation | 0.9249 | `results/labram_cut3_rest_vs_work.csv`, winning epoch |
| Simple, test (both) | 0.7732 | `results/final_test_rest_vs_work_v2.csv` ("Simple + calibration") |
| Head motion, test (both) | 0.7848 | `code/Experiments/demo_site/out/metrics.json` (this page's run) |

The 4-second scores (LaBraM 0.648) cannot be recomputed from per-minute rows. The "simple 24-person model"
(0.62 in `results/rest_vs_work_simple_24.csv`) is a different model and is not shown.

## Data source and licence

UNIVERSE: Anders, C., Moontaha, S., Real, S., & Arnrich, B. (2023). *UNIVERSE: UNobtrusIVE measuRement of
mental workload and stress in uncontrolled environments.* Zenodo. https://doi.org/10.5281/zenodo.10371068,
licensed CC BY 4.0. For UN_109 to UN_112, this repository holds only:
- per-minute labels and model probabilities
- calibrated band-power summaries
- one filtered 4-second EEG snippet per minute for display (downsampled to 100 Hz)

It holds nothing from UN_113 to UN_124.

## Rebuild

The inputs live on the course cluster in `/work/cssema416/202710/10`. Run everything from the team root through Slurm.

1. `results/demo_predictions.csv` comes from `code/Experiments/export_demo_predictions.py` (team code, Umesh Sarma).
2. `code/Experiments/demo_site/export_demo_extras.py` (copy in `cluster/`) rebuilds the same rows and adds the
   live-model features and weights, the head-motion probabilities, and recording order:
   `srun -p cpu --time=00:30:00 --mem=16G envs/eeg416/bin/python code/Experiments/demo_site/export_demo_extras.py`.
   It asserts that the rows match the CSV.
3. The other three exports are in `cluster/` and were run on the cluster in these folders:
   - `train_eegnet_rest_vs_work.py` → `code/Experiments/demo_eegnet_rest_vs_work/`
   - `export_minute_signals.py` → `code/Experiments/demo_minute_signals/`
   - `export_easy_vs_hard.py` → `code/Experiments/demo_easy_vs_hard/`
4. Copy the outputs into `data_in/`, copy `signals.json` to `docs/data/`, then run `python build.py`. It writes
   `docs/data/demo.json` and checks the live model against the CSV.
5. Serve `docs/` with any static server (`python -m http.server --directory docs`). GitHub Pages serves it as is.
   Bump the `?v=` number in `docs/index.html` after changing `app.js` or `style.css`.

## AI use

This website was built with Claude Code (Anthropic's AI coding assistant), as the course allows with attribution.
The AI wrote the page (`docs/`), `build.py`, and the export scripts
`cluster/export_demo_extras.py`, `cluster/export_minute_signals.py`, and `cluster/export_easy_vs_hard.py`.
The LaBraM and simple-model predictions come from the team's own export script.
The page checks its scores against the team's official result files.
