/* Rest or Working? MA416 team 10 demo. Plain JS, no dependencies. */
(() => {
  "use strict";

  const MODELS = [
    { key: "labram", name: "LaBraM", long: "LaBraM (3-layer cut, pretrained EEG transformer)", how: "Precomputed on the course cluster", color: "--labram", short: "deep learning on the raw wave" },
    { key: "eegnet", name: "EEGNet", long: "EEGNet-8,2 (compact CNN on raw 4-channel EEG, retrained for rest vs working)", how: "Precomputed on the course cluster", color: "--eegnet", short: "deep learning on the raw wave" },
    { key: "simple", name: "Simple", long: "Simple model (band power + logistic regression)", how: "Computed live in your browser", color: "--simple", live: true, short: "20 band-power numbers" },
    { key: "motion", name: "Head motion", long: "Head motion only (gyroscope + accelerometer)", how: "Precomputed on the course cluster", color: "--motion", short: "movement only, no EEG" },
  ];
  // Official numbers from the team's results files (60 s windows), used to check this page's maths.
  const OFFICIAL = {
    labram: { test: { ALL: 0.8053, UN_111: 0.7758, UN_112: 0.8861 }, validation: { ALL: 0.9249 },
      src: { test: "team results: LaBraM final test", validation: "team results: LaBraM training log (winning epoch)" } },
    eegnet: { test: { ALL: 0.6432, UN_111: 0.6645, UN_112: 0.6013 }, validation: { ALL: 0.8482 },
      src: { test: "team results: EEGNet run for this page", validation: "team results: EEGNet run for this page (picked epoch)" } },
    simple: { test: { ALL: 0.7732, UN_111: 0.7871, UN_112: 0.7421 }, validation: {},
      src: { test: "team results: simple-model final test (\"Simple + calibration\")" } },
  };
  const SENSORS = ["TP9", "AF7", "AF8", "TP10"];
  const BANDS = [["delta", "1–4 Hz"], ["theta", "4–7 Hz"], ["alpha", "8–12 Hz"], ["beta", "13–30 Hz"], ["gamma", "30–45 Hz"]];
  const TASK_SHORT = { arithmetix: "Arithmetic", n_back: "N-back", stroop: "Stroop", sudoku: "Sudoku", rest: "Rest" };

  const $ = (s, el = document) => el.querySelector(s);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const pct = (x) => (x * 100).toFixed(1) + "%";
  // Two questions the same page can ask. "eh" only scores task minutes (rest has no easy/hard level).
  const Q = {
    rw: { label: "Rest vs working", neg: "Rest", pos: "Working", key1: "r", key2: "w",
      truth: (m) => m.truth, p: (m, k) => m[k], scored: () => true },
    eh: { label: "Easy vs hard", neg: "Easy", pos: "Hard", key1: "e", key2: "h",
      truth: (m) => (m.rec.level === "hard" ? "Hard" : "Easy"), p: (m, k) => (m.eh ? m.eh[k] : null), scored: (m) => !!m.eh },
  };
  const q = () => Q[state.q];
  const P = (m, k) => q().p(m, k);
  const right = (m, p) => (q().truth(m) === q().pos) === (p > 0.5);
  const isRight = (truth, p) => (truth === "Working") === (p > 0.5);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  let D, ALL = [], sessionsById = {};
  const state = { q: "rw", session: null, scope: "session", pick: "any", quiz: false, current: null, guessed: null,
    show: { labram: true, eegnet: true, simple: false, motion: false }, tally: { n: 0, you: 0, labram: 0, eegnet: 0, simple: 0, motion: 0 } };

  // ---------- theme
  const themeBtn = $("#theme");
  try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) {}
  themeBtn.addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("theme", document.documentElement.dataset.theme); } catch (e) {}
    drawTimeline();
  });

  // ---------- live simple model
  function liveSimple(feat) {
    const L = D.live_simple;
    const clipP = feat.map((clip) => {
      let z = L.intercept;
      for (let i = 0; i < clip.length; i++) z += L.coef[i] * (clip[i] - L.scaler_mean[i]) / L.scaler_scale[i];
      return 1 / (1 + Math.exp(-z));
    });
    return { clipP, mean: clipP.reduce((a, b) => a + b, 0) / clipP.length };
  }

  // ---------- load
  fetch("data/demo.json").then((r) => r.json()).then((d) => { D = d; init(); })
    .catch((e) => { document.querySelector("main").insertAdjacentHTML("afterbegin", `<p class="card">Could not load data/demo.json (${esc(e.message)}). If you opened this file directly, serve the folder with a local web server.</p>`); });

  function init() {
    for (const s of D.sessions) {
      sessionsById[s.id] = s;
      s.minutes = [];
      for (const rec of s.recs) {
        rec.calClips = rec.task === "rest" ? rec.minutes[0].first_clip : 0;
        for (const m of rec.minutes) {
          const o = { ...m, sid: s.id, person: s.person, split: s.split, rec, truth: rec.truth };
          s.minutes.push(o); ALL.push(o);
        }
      }
    }
    state.session = D.sessions.find((s) => s.id === "UN_111 Lab1").id;
    buildSessionButtons();
    buildQuestionToggles();
    buildGuessControls();
    buildModelToggle();
    fillScores();
    selectSession(state.session, true);
    window.addEventListener("resize", debounce(drawTimeline, 120));
    document.addEventListener("keydown", onKey);
  }

  function buildQuestionToggles() {
    document.querySelectorAll(".qseg").forEach((box) => {
      box.innerHTML = Object.entries(Q).map(([id, o]) => `<button data-q="${id}">${o.label}</button>`).join("");
    });
    document.querySelectorAll("[data-q]").forEach((b) => b.addEventListener("click", () => setQuestion(b.dataset.q)));
    syncQuestion();
  }
  function setQuestion(id) {
    if (state.q === id) return;
    state.q = id;
    state.tally = { n: 0, you: 0, labram: 0, eegnet: 0, simple: 0, motion: 0 };
    syncQuestion(); renderTally(); fillScores(); drawTimeline(); newMinute();
  }
  function syncQuestion() {
    const o = q();
    document.querySelectorAll("[data-q]").forEach((b) => b.classList.toggle("on", b.dataset.q === state.q));
    document.body.dataset.q = state.q;
    const set = (sel, t) => document.querySelectorAll(sel).forEach((el) => (el.textContent = t));
    set('[data-pick="neg"]', `${o.neg === "Easy" ? "An" : "A"} ${o.neg.toLowerCase()} minute`);
    set('[data-pick="pos"]', `A ${o.pos.toLowerCase()} minute`);
    set('[data-pick="any"]', state.q === "rw" ? "Any minute" : "Any task minute");
    document.querySelectorAll("[data-guess]").forEach((b) => {
      const neg = b.dataset.guess === "neg";
      b.innerHTML = `<span class="key">${(neg ? o.key1 : o.key2).toUpperCase()}</span> ${neg ? o.neg : o.pos}`;
    });
    set("[data-qtext]", state.q === "rw" ? "working" : "hard");
    document.querySelectorAll("[data-qshow]").forEach((el) => el.classList.toggle("hidden", el.dataset.qshow !== state.q));
  }

  function debounce(f, ms) { let t; return () => { clearTimeout(t); t = setTimeout(f, ms); }; }

  // ---------- session picker
  function buildSessionButtons() {
    const box = $("#sessionButtons");
    for (const split of ["test", "validation"]) {
      const g = document.createElement("div");
      g.className = "seg-group";
      g.innerHTML = `<span class="eyebrow">${split === "test" ? "Final test people" : "Validation people (used to pick the model)"}</span>`;
      const seg = document.createElement("div");
      seg.className = "seg"; seg.setAttribute("role", "group");
      for (const s of D.sessions.filter((x) => x.split === split)) {
        const b = document.createElement("button");
        b.textContent = s.id; b.dataset.sid = s.id;
        b.addEventListener("click", () => selectSession(s.id));
        seg.appendChild(b);
      }
      g.appendChild(seg); box.appendChild(g);
    }
  }
  function selectSession(sid, first) {
    state.session = sid;
    document.querySelectorAll("#sessionButtons button").forEach((b) => b.classList.toggle("on", b.dataset.sid === sid));
    drawTimeline();
    if (first || state.scope === "session") newMinute();
  }

  // ---------- inside one minute (raw wave -> spectrum -> 20 numbers -> models), optional quiz
  let SIG = null, specSensor = -1; // -1 = all four sensors, one row each
  fetch("data/signals.json").then((r) => (r.ok ? r.json() : null)).then((d) => {
    if (!d) { SIG = false; if (state.current) { drawRaw(state.current); drawSpec(state.current); } return; }
    SIG = d;
    if (state.current) { drawRaw(state.current); drawSpec(state.current); }
  }).catch(() => { SIG = false; });

  const sigOf = (m) => (SIG && SIG.rows[m.row]) || null;
  function buildGuessControls() {
    document.querySelectorAll("[data-scope]").forEach((b) => b.addEventListener("click", () => {
      state.scope = b.dataset.scope;
      document.querySelectorAll("[data-scope]").forEach((x) => x.classList.toggle("on", x === b));
      newMinute();
    }));
    document.querySelectorAll("[data-pick]").forEach((b) => b.addEventListener("click", () => { state.pick = b.dataset.pick; newMinute(); }));
    document.querySelectorAll("[data-guess]").forEach((b) => b.addEventListener("click", () => guess(b.dataset.guess === "neg" ? q().neg : q().pos)));
    $("#quiz").addEventListener("change", (e) => { state.quiz = e.target.checked; $("#tally").classList.toggle("hidden", !state.quiz); newMinute(); });
    $("#nextBtn").addEventListener("click", newMinute);
    const box = $("#specSensor");
    [["All 4", -1], ...SENSORS.map((x, i) => [x, i])].forEach(([s, i]) => {
      const b = document.createElement("button");
      b.textContent = s; b.dataset.si = i;
      b.addEventListener("click", () => { specSensor = i; drawSpec(state.current); });
      box.appendChild(b);
    });
    renderTally();
  }
  function pool() { return state.scope === "all" ? ALL : sessionsById[state.session].minutes; }
  function newMinute() {
    const p = pool();
    const o = q(), sc = pool().filter(o.scored);
    const rest = sc.filter((m) => o.truth(m) === o.neg), work = sc.filter((m) => o.truth(m) === o.pos);
    let src = state.pick === "neg" ? rest : state.pick === "pos" ? work : sc;
    if (state.quiz) src = Math.random() < 0.5 && rest.length ? rest : work; // quiz: half rest, half working
    if (!src.length) src = sc.length ? sc : p;
    document.querySelectorAll("[data-pick]").forEach((x) => x.classList.toggle("on", !state.quiz && x.dataset.pick === state.pick));
    showMinute(src[Math.floor(Math.random() * src.length)], !state.quiz);
  }
  function showMinute(m, revealNow) {
    state.current = m; state.guessed = null;
    $("#mSession").textContent = m.sid + (m.split === "test" ? " · test person" : " · validation person");
    $("#mLabel").textContent = revealNow ? minuteName(m) : "Mystery minute";
    drawRaw(m); drawSpec(m); drawGrid(m);
    $("#guessButtons").classList.toggle("hidden", revealNow);
    updateDock();
    document.querySelectorAll("[data-guess]").forEach((b) => { b.disabled = false; b.classList.remove("chosen"); });
    $("#revealEmpty").classList.toggle("hidden", revealNow);
    $("#revealBody").classList.toggle("hidden", !revealNow);
    if (revealNow) reveal(null);
  }
  // phone: a bar pinned to the bottom with the guess buttons while the quiz waits for a guess
  function updateDock() {
    const r = document.querySelector("#minute").getBoundingClientRect();
    const minuteInView = r.top < innerHeight * 0.7 && r.bottom > 120;
    const on = state.quiz && !state.guessed && minuteInView && innerWidth < 860;
    $("#guessDock").classList.toggle("hidden", !on);
    document.body.classList.toggle("dock-on", on);
  }
  window.addEventListener("scroll", updateDock, { passive: true });
  window.addEventListener("resize", debounce(updateDock, 120));

  function minuteName(m) { return `${TASK_SHORT[m.rec.task]}${m.rec.level ? " (" + m.rec.level + ")" : ""}, minute ${m.m}`; }

  // step 1: raw wave
  function drawRaw(m) {
    const box = $("#rawPlot");
    const sg = sigOf(m);
    if (!sg) { box.innerHTML = `<p class="muted small">${SIG === null ? "Loading the signal…" : "No signal for this minute."}</p>`; return; }
    const bin = atob(sg.raw), n = bin.length / 2, buf = new Int16Array(n);
    for (let i = 0; i < n; i++) { const v = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8); buf[i] = v > 32767 ? v - 65536 : v; }
    const T = n / 4, W = 560, rowH = 74, H = rowH * 4 + 22, x0 = 46;
    const scale = 50; // uV per half row
    let svg = `<svg viewBox="0 0 ${W} ${H}" class="rawsvg" role="img" aria-label="Four EEG traces, 4 seconds">`;
    for (let c = 0; c < 4; c++) {
      const y0 = rowH * c + rowH / 2;
      let d = "";
      for (let i = 0; i < T; i++) {
        const v = Math.max(-scale * 1.45, Math.min(scale * 1.45, buf[c * T + i]));
        d += (i ? "L" : "M") + (x0 + (i / (T - 1)) * (W - x0 - 6)).toFixed(1) + " " + (y0 - (v / scale) * (rowH / 2)).toFixed(1);
      }
      svg += `<text x="0" y="${y0 - rowH / 4 + 4}" class="ax">${SENSORS[c]}</text><line x1="${x0}" x2="${W - 6}" y1="${y0}" y2="${y0}" class="zero"/><path d="${d}" class="trace"/>`;
    }
    for (let s = 0; s <= 4; s++) { const x = x0 + (s / 4) * (W - x0 - 6); svg += `<text x="${x}" y="${H - 4}" class="ax mid">${s} s</text>`; }
    svg += `<line x1="${x0 - 8}" x2="${x0 - 8}" y1="${H - 22 - rowH / 2}" y2="${H - 22}" class="sbar"/><text x="${x0 - 12}" y="${H - 22 - rowH / 4 + 3}" class="ax end">50 µV</text></svg>`;
    box.innerHTML = svg;
  }

  // step 2: spectrum of one sensor vs own rest, with the 5 bands shaded
  function drawSpec(m) {
    document.querySelectorAll("#specSensor button").forEach((b) => b.classList.toggle("on", +b.dataset.si === specSensor));
    const box = $("#specPlot");
    const sg = m && sigOf(m);
    if (!sg) { box.innerHTML = `<p class="muted small">${SIG === null ? "Loading…" : "No spectrum for this minute."}</p>`; return; }
    const hz = SIG.hz, chans = specSensor < 0 ? [0, 1, 2, 3] : [specSensor], one = chans.length === 1;
    const W = 560, x0 = 48, xr = W - 12, rowH = one ? 200 : 74, gap = one ? 0 : 8, top = 20, bottom = 32;
    const H = top + chans.length * rowH + (chans.length - 1) * gap + bottom;
    const X = (f) => x0 + ((f - 1) / 44) * (xr - x0);
    const B = [[1, 4], [4, 7], [8, 12], [13, 30], [30, 45]];
    let svg = `<svg viewBox="0 0 ${W} ${H}" class="specsvg" role="img" aria-label="Power spectrum by wave speed, ${one ? SENSORS[specSensor] : "all four sensors"}, this minute against the same person at rest">`;
    B.forEach(([a, b], i) => {
      svg += `<rect x="${X(a) + 1}" y="${top}" width="${X(b) - X(a) - 2}" height="${H - top - bottom}" rx="3" class="bandbg"/><text x="${(X(a) + X(b)) / 2}" y="${top - 6}" class="ax mid band-lbl">${BANDS[i][0]}</text>`;
    });
    chans.forEach((c, k) => {
      const y1 = top + k * (rowH + gap) + 6, y2 = y1 + rowH - 12;
      const cur = sg.spec[c], ref = SIG.rest_spec[m.sid][c];
      const lo = Math.min(...cur, ...ref) - 0.1, hi = Math.max(...cur, ...ref) + 0.1;
      const Y = (v) => y2 - ((v - lo) / (hi - lo)) * (y2 - y1);
      const line = (arr) => arr.map((v, i) => (i ? "L" : "M") + X(hz[i]).toFixed(1) + " " + Y(v).toFixed(1)).join("");
      if (k) svg += `<line x1="0" x2="${W}" y1="${y1 - 6 - gap / 2}" y2="${y1 - 6 - gap / 2}" class="row-sep"/>`;
      if (one) svg += `<text x="${x0 - 6}" y="${y1 + 8}" class="ax end">more</text><text x="${x0 - 6}" y="${y2}" class="ax end">less</text>`;
      else svg += `<text x="${x0 - 8}" y="${(y1 + y2) / 2 + 4}" class="ax end sensor-lbl">${SENSORS[c]}</text>`;
      svg += `<path d="${line(ref)}" class="spec-ref"/><path d="${line(cur)}" class="spec-cur"/>`;
    });
    [1, 10, 20, 30, 45].forEach((f, i, a) => (svg += `<text x="${X(f)}" y="${H - 14}" class="ax ${i === a.length - 1 ? "end" : "mid"} small">${f} Hz</text>`));
    svg += `<text x="${(x0 + xr) / 2}" y="${H - 1}" class="ax mid small">wave speed (slow → fast)</text></svg>`;
    box.innerHTML = svg;
  }

  // step 3: the 20 calibrated band-power numbers
  function drawGrid(m) {
    const avg = new Array(20).fill(0);
    for (const clip of m.feat) clip.forEach((v, i) => (avg[i] += v / m.feat.length));
    const lo = css("--div-lo"), mid = css("--div-mid"), hi = css("--div-hi");
    let h = `<span></span>` + BANDS.map(([b, hz]) => `<span class="hd">${b}<br>${hz}</span>`).join("");
    SENSORS.forEach((s, si) => {
      h += `<span class="rl">${s}</span>`;
      BANDS.forEach(([b], bi) => {
        const v = avg[si * 5 + bi];
        const t = Math.max(-1, Math.min(1, v / 3));
        const bg = t < 0 ? mix(mid, lo, -t) : mix(mid, hi, t);
        const dark = Math.abs(t) > 0.55;
        h += `<span class="cell" style="background:${bg};color:${dark ? "#fff" : "var(--ink-2)"}" title="${s} ${b}: ${v >= 0 ? "+" : ""}${v.toFixed(1)} vs own rest (in rest-spread units)">${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(1)}</span>`;
      });
    });
    $("#grid").innerHTML = h;
  }
  function mix(a, b, t) {
    const pa = hex(a), pb = hex(b);
    const c = pa.map((x, i) => Math.round(x + (pb[i] - x) * t));
    return `rgb(${c.join(",")})`;
  }
  function hex(h) { h = h.replace("#", ""); return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)); }

  function guess(g) {
    if (!state.current || state.guessed || !state.quiz) return;
    state.guessed = g;
    document.querySelectorAll("[data-guess]").forEach((b) => { b.disabled = true; b.classList.toggle("chosen", (b.dataset.guess === "neg" ? q().neg : q().pos) === g); });
    const m = state.current, t = state.tally;
    t.n++; if (g === q().truth(m)) t.you++;
    for (const md of MODELS) if (right(m, P(m, md.key))) t[md.key]++;
    renderTally();
    updateDock();
    if (innerWidth < 860) $("#reveal").scrollIntoView({ behavior: "smooth", block: "start" });
    $("#mLabel").textContent = minuteName(m);
    $("#revealEmpty").classList.add("hidden");
    $("#revealBody").classList.remove("hidden");
    reveal(g);
  }
  function reveal(g) {
    const m = state.current;
    const tt = $("#truthText");
    const o = q(), tr = o.truth(m);
    tt.textContent = `It was ${tr.toLowerCase()}`;
    tt.className = "truth " + (tr === o.neg ? "rest" : "work");
    $("#truthDetail").textContent = m.rec.task === "rest"
      ? `Relaxation video, scored minute ${m.m} (second half of the video). About ${Math.round(m.rec.approx_start_min + (m.first_clip * 4) / 60)} min into the session.`
      : `${m.rec.task_name}, ${m.rec.level} level, minute ${m.m} of the task. About ${Math.round(m.rec.approx_start_min + (m.first_clip * 4) / 60)} min into the session.`;
    $("#youVerdict").innerHTML = g ? `<span class="muted small">You said ${g.toLowerCase()}</span><br>${verdict(g === tr)}` : "";
    const live = liveSimple(m.feat);
    $("#modelCards").innerHTML = MODELS.map((md) => {
      const p = md.live && state.q === "rw" ? live.mean : P(m, md.key);
      const call = (p > 0.5 ? o.pos : o.neg).toLowerCase();
      return `<div class="mc">
        <div class="mc-top"><span class="mc-name"><span class="dot" style="background:var(${md.color})"></span>${md.name}</span>${verdict(right(m, p))}</div>
        <div class="pbar"><span class="fill" data-w="${(p * 100).toFixed(1)}" style="width:0;background:var(${md.color})"></span><span class="mid"></span></div>
        <div class="mc-foot"><span>Says <b>${call}</b> (${pct(p)} chance of ${o.pos.toLowerCase()})</span><span class="muted small">${md.short}</span></div>
      </div>`;
    }).join("");
    requestAnimationFrame(() => requestAnimationFrame(() => {
      document.querySelectorAll("#modelCards .fill").forEach((f) => (f.style.width = f.dataset.w + "%"));
    }));
  }
  function verdict(ok) { return `<span class="verdict ${ok ? "ok" : "no"}"><span class="ic" aria-hidden="true">${ok ? "✓" : "✗"}</span>${ok ? "Right" : "Wrong"}</span>`; }
  function renderTally() {
    const t = state.tally;
    const item = (label, k, color) => `<div class="t">${color ? `<span class="chip-dot" style="background:var(${color})"></span>` : ""}<span>${label}</span><b>${t[k]}/${t.n}</b></div>`;
    $("#tally").innerHTML = `<span class="eyebrow">Quiz score in this browser (half ${q().neg.toLowerCase()}, half ${q().pos.toLowerCase()})</span>` + item("You", "you") +
      MODELS.map((m) => item(m.name, m.key, m.color)).join("") + `<button class="reset" id="resetTally">Reset</button>`;
    $("#resetTally").addEventListener("click", () => { state.tally = { n: 0, you: 0, labram: 0, eegnet: 0, simple: 0, motion: 0 }; renderTally(); });
  }
  function onKey(e) {
    if (e.target.closest && e.target.closest("input,textarea,select")) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (k === q().key1) guess(q().neg);
    else if (k === q().key2) guess(q().pos);
    else if (k === "n") { e.preventDefault(); newMinute(); }
  }

  // ---------- timeline
  function buildModelToggle() {
    const box = $("#modelToggle");
    const opts = [...MODELS.map((m) => ({ id: m.key, label: m.name, color: m.color })), { id: "both", label: "LaBraM vs EEGNet" }, { id: "all", label: "All four" }];
    for (const o of opts) {
      const b = document.createElement("button");
      b.dataset.mt = o.id;
      b.innerHTML = (o.color ? `<span class="chip-dot" style="background:var(${o.color})"></span>` : "") + o.label;
      b.addEventListener("click", () => {
        if (o.id === "both") state.show = { labram: true, eegnet: true, simple: false, motion: false };
        else if (o.id === "all") state.show = { labram: true, eegnet: true, simple: true, motion: true };
        else state.show = Object.fromEntries(MODELS.map((m) => [m.key, m.key === o.id]));
        syncToggle(); drawTimeline();
      });
      box.appendChild(b);
    }
    syncToggle();
  }
  function syncToggle() {
    const s = state.show;
    const on = MODELS.filter((m) => s[m.key]).map((m) => m.key);
    const id = on.length === MODELS.length ? "all" : on.join() === "labram,eegnet" ? "both" : on[0];
    document.querySelectorAll("[data-mt]").forEach((b) => b.classList.toggle("on", b.dataset.mt === id));
  }

  function drawTimeline() {
    if (!D) return;
    const s = sessionsById[state.session];
    const svg = $("#tl");
    const W = Math.max(760, $("#tlScroll").clientWidth);
    const shown = MODELS.filter((m) => state.show[m.key]);
    const L = 108, R = 14, top = 8;
    const stripY = top + 18, stripH = 30;
    const chartY = stripY + stripH + 14, chartH = 150;
    const tickY0 = chartY + chartH + 12, tickH = 16, tickGap = 4;
    const axisY = tickY0 + shown.length * (tickH + tickGap) + 6;
    const H = axisY + 26;
    const recEnd = (r) => r.approx_start_min + (r.clips * 4) / 60;
    const tMax = Math.max(...s.recs.map(recEnd));
    const x = (t) => L + (t / tMax) * (W - L - R);
    const y = (p) => chartY + (1 - p) * chartH;
    const mStart = (r, m) => r.approx_start_min + (m.first_clip * 4) / 60;

    let h = `<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="${css("--cal-band")}"/><line x1="0" y1="0" x2="0" y2="6" stroke="${css("--ink-3")}" stroke-width="1.2" opacity=".55"/></pattern></defs>`;
    // lane labels
    const lab = (txt, yy, col) => `<text x="${L - 10}" y="${yy}" text-anchor="end" font-size="12.5" fill="${col || css("--ink-2")}" font-weight="600">${txt}</text>`;
    h += lab("Activity", stripY + stripH / 2 + 4);
    h += lab("Chance of", chartY + chartH / 2 - 4);
    h += lab(q().pos.toLowerCase(), chartY + chartH / 2 + 11);
    // strip
    for (const r of s.recs) {
      const x0 = x(r.approx_start_min), x1 = x(recEnd(r));
      const isRest = r.task === "rest";
      h += `<rect x="${x0}" y="${stripY}" width="${Math.max(1, x1 - x0)}" height="${stripH}" rx="4" fill="${isRest ? css("--rest-band") : css("--work-band")}"/>`;
      if (isRest && r.calClips) {
        const xc = x(r.approx_start_min + (r.calClips * 4) / 60);
        h += `<rect x="${x0}" y="${stripY}" width="${xc - x0}" height="${stripH}" rx="4" fill="url(#hatch)"/>`;
        h += `<text x="${(x0 + xc) / 2}" y="${stripY - 5}" text-anchor="middle" font-size="11" fill="${css("--ink-3")}">calibration</text>`;
      }
      const label = TASK_SHORT[r.task] + (r.level ? (r.level === "hard" ? " ▲" : " ▽") : "");
      const cx = isRest && r.calClips ? (x(r.approx_start_min + (r.calClips * 4) / 60) + x1) / 2 : (x0 + x1) / 2;
      if (x1 - x0 > 34) h += `<text x="${cx}" y="${stripY + stripH / 2 + 4}" text-anchor="middle" font-size="12" font-weight="600" fill="${isRest ? css("--rest-ink") : css("--work-ink")}">${label}</text>`;
    }
    // chart grid
    for (const p of [0, 0.25, 0.5, 0.75, 1]) {
      h += `<line x1="${L}" x2="${W - R}" y1="${y(p)}" y2="${y(p)}" stroke="${p === 0.5 ? css("--ink-3") : css("--line")}" stroke-width="1" ${p === 0.5 ? `stroke-dasharray="5 4"` : ""}/>`;
      if (p !== 0.5) h += `<text x="${L + 3}" y="${y(p) + (p === 1 ? 11 : -3)}" font-size="10.5" fill="${css("--ink-3")}">${p * 100}%</text>`;
    }
    // rest shading in chart
    for (const r of s.recs) if (r.task === "rest" && state.q === "rw") {
      const x0 = x(mStart(r, r.minutes[0])), x1 = x(mStart(r, r.minutes[r.minutes.length - 1]) + 1);
      h += `<rect x="${x0}" y="${chartY}" width="${x1 - x0}" height="${chartH}" fill="${css("--rest-band")}" opacity=".55"/>`;
    }
    // lines (break between recordings)
    for (const md of shown) {
      const col = css(md.color);
      for (const r of s.recs) {
        const mins = s.minutes.filter((o) => o.rec === r && q().scored(o));
        if (!mins.length) continue;
        const pts = mins.map((m) => [x(mStart(r, m) + 0.5), y(P(m, md.key))]);
        h += `<polyline points="${pts.map((p) => p.join(",")).join(" ")}" fill="none" stroke="${col}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
        for (const p of pts) h += `<circle cx="${p[0]}" cy="${p[1]}" r="2.6" fill="${col}"/>`;
      }
    }
    // tick rows
    const good = css("--good"), bad = css("--bad");
    shown.forEach((md, i) => {
      const yy = tickY0 + i * (tickH + tickGap);
      h += lab(md.name, yy + tickH - 4, css("--ink"));
      h += `<circle cx="8" cy="${yy + tickH / 2}" r="4" fill="${css(md.color)}"/>`;
      for (const m of s.minutes.filter(q().scored)) {
        const r = m.rec;
        const ok = right(m, P(m, md.key));
        const x0 = x(mStart(r, m)), w = Math.max(2, x(mStart(r, m) + 1) - x0 - 1.5);
        // right = short green bar, wrong = full-height red bar (shape + colour)
        if (ok) h += `<rect x="${x0 + 0.75}" y="${yy + tickH / 2 - 3}" width="${w}" height="6" rx="2" fill="${good}" opacity=".8"/>`;
        else h += `<rect x="${x0 + 0.75}" y="${yy}" width="${w}" height="${tickH}" rx="2" fill="${bad}"/>`;
        if (!ok && w >= 9) h += `<text x="${x0 + 0.75 + w / 2}" y="${yy + tickH - 4}" text-anchor="middle" font-size="10" fill="#fff" font-weight="700">✗</text>`;
      }
    });
    // axis
    for (let t = 0; t <= tMax; t += 15) {
      h += `<line x1="${x(t)}" x2="${x(t)}" y1="${axisY - 4}" y2="${axisY}" stroke="${css("--ink-3")}"/>`;
      h += `<text x="${x(t)}" y="${axisY + 14}" text-anchor="middle" font-size="11" fill="${css("--ink-3")}">${t}</text>`;
    }
    h += `<text x="${L - 10}" y="${axisY + 14}" text-anchor="end" font-size="11" fill="${css("--ink-3")}">min (approx.)</text>`;
    // hover targets
    for (const r of s.recs) for (const m of r.minutes) {
      const x0 = x(mStart(r, m)), x1 = x(mStart(r, m) + 1);
      h += `<rect class="hit" data-k="${r.start_row}:${m.m}" x="${x0}" y="${stripY}" width="${Math.max(6, x1 - x0)}" height="${axisY - stripY}" fill="transparent" style="cursor:pointer"/>`;
    }
    h += `<line id="cross" x1="0" x2="0" y1="${stripY}" y2="${axisY}" stroke="${css("--ink")}" stroke-width="1" opacity="0"/>`;

    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("height", H);
    svg.style.width = W + "px";
    svg.innerHTML = h;

    const lookup = {};
    for (const r of s.recs) for (const m of r.minutes) lookup[r.start_row + ":" + m.m] = s.minutes.find((o) => o.rec === r && o.m === m.m);
    const tip = $("#tip"), cross = $("#cross");
    svg.querySelectorAll(".hit").forEach((el) => {
      const m = lookup[el.dataset.k];
      const show = (ev) => {
        const bx = +el.getAttribute("x") + +el.getAttribute("width") / 2;
        cross.setAttribute("x1", bx); cross.setAttribute("x2", bx); cross.setAttribute("opacity", 0.35);
        tip.innerHTML = `<h4>${TASK_SHORT[m.rec.task]}${m.rec.level ? " (" + m.rec.level + ")" : ""} · minute ${m.m}</h4>
          <div class="muted" style="margin-bottom:6px">${q().scored(m) ? `Truth: <b style="color:var(--ink)">${q().truth(m)}</b>` : "Rest minute: not part of easy vs hard"}</div>` +
          (q().scored(m) ? MODELS.filter((md) => state.show[md.key]).map((md) => `<div class="row"><span><span class="chip-dot" style="background:var(${md.color})"></span>${md.name}</span><span><b>${pct(P(m, md.key))}</b> ${right(m, P(m, md.key)) ? "✓" : "✗"}</span></div>`).join("") : "") +
          `<div class="muted small" style="margin-top:6px">Click to open in "Inside one minute"</div>`;
        tip.classList.remove("hidden");
        const px = ev.clientX ?? (ev.touches && ev.touches[0].clientX), py = ev.clientY ?? (ev.touches && ev.touches[0].clientY);
        const tw = tip.offsetWidth, th = tip.offsetHeight;
        tip.style.left = Math.min(window.innerWidth - tw - 8, Math.max(8, px + 14)) + "px";
        tip.style.top = Math.max(8, py - th - 12) + "px";
      };
      el.addEventListener("mousemove", show);
      el.addEventListener("mouseleave", () => { tip.classList.add("hidden"); cross.setAttribute("opacity", 0); });
      el.addEventListener("click", () => { if (!q().scored(m)) return; tip.classList.add("hidden"); showMinute(m, true); $("#minute").scrollIntoView(); });
    });

    // legend + summary
    $("#tlLegend").innerHTML = `<span class="lg"><i style="background:var(--rest-band)"></i>Rest (scored)</span>
      <span class="lg"><i style="background:repeating-linear-gradient(45deg,var(--cal-band) 0 3px,var(--ink-3) 3px 4px)"></i>Calibration, not scored</span>
      <span class="lg"><i style="background:var(--work-band)"></i>Working (▽ easy, ▲ hard)</span>` +
      shown.map((md) => `<span class="lg"><i class="ln" style="background:var(${md.color})"></i>${md.name}</span>`).join("") +
      `<span class="lg"><i class="ln" style="background:none;border-top:2px dashed var(--ink-3)"></i>50%: above = says ${q().pos.toLowerCase()}</span>` +
      `<span class="lg"><i style="background:var(--good);height:6px"></i>model right</span><span class="lg"><i style="background:var(--bad);height:14px"></i>model wrong</span>`;
    const scm = s.minutes.filter(q().scored);
    $("#tlSummary").textContent = `${s.id}: ${scm.length} scored minutes (${scm.filter((m) => q().truth(m) === q().neg).length} ${q().neg.toLowerCase()}). ` +
      shown.map((md) => `${md.name} ${pct(score(s.minutes, md.key).ba)}`).join(" · ") + " balanced accuracy";
  }

  // ---------- scores
  function score(ms, key, qid) {
    const o = Q[qid || state.q];
    ms = ms.filter(o.scored);
    const r = ms.filter((m) => o.truth(m) === o.neg), w = ms.filter((m) => o.truth(m) === o.pos);
    const rOk = r.filter((m) => !(o.p(m, key) > 0.5)).length, wOk = w.filter((m) => o.p(m, key) > 0.5).length;
    return { ba: (rOk / r.length + wOk / w.length) / 2, rOk, rN: r.length, wOk, wN: w.length };
  }
  const OFFICIAL_EH = {
    labram: { test: { ALL: 0.5769, UN_111: 0.6011, UN_112: 0.5321 }, validation: { ALL: 0.5629 },
      src: { test: "team results: LaBraM easy-vs-hard final test", validation: "team results: LaBraM easy-vs-hard training log (winning epoch)" } },
  };
  function fillScores() {
    const box = $("#scoreTables");
    const o = q(), eh = state.q === "eh";
    // the two questions side by side, final test people
    const testMin = ALL.filter((m) => m.split === "test");
    let html = `<div class="card score-card both-q"><h3 class="h3">Same models, two questions (final test, UN_111 + UN_112)</h3>
      <p class="muted small" style="margin:0">Balanced accuracy over 60-second windows. 50% is a coin flip. Easy vs hard only uses task minutes and a separate model of each kind trained for that question.</p>
      <table class="sc"><thead><tr><th>Model</th><th>Rest vs working</th><th>Easy vs hard</th></tr></thead><tbody>` +
      MODELS.map((md) => {
        const a = score(testMin, md.key, "rw"), b = score(testMin, md.key, "eh");
        return `<tr><td class="who"><span class="chip-dot" style="background:var(${md.color})"></span>${md.name}<span class="sub">${md.short}</span></td>
          <td><span class="ba">${pct(a.ba)}</span><span class="sub">rest ${a.rOk}/${a.rN} · working ${a.wOk}/${a.wN}</span></td>
          <td><span class="ba ${b.ba < 0.6 ? "coin" : ""}">${pct(b.ba)}</span><span class="sub">easy ${b.rOk}/${b.rN} · hard ${b.wOk}/${b.wN}</span></td></tr>`;
      }).join("") + `</tbody></table></div>`;

    html += `<h3 class="h3 score-sub">Details for <span class="qname">${o.label.toLowerCase()}</span></h3><div class="score-wrap">`;
    const checks = [];
    for (const split of ["test", "validation"]) {
      const people = [...new Set(ALL.filter((m) => m.split === split).map((m) => m.person))];
      html += `<div class="card score-card"><h3 class="h3">${split === "test" ? "Final test: UN_111 + UN_112" : "Validation: UN_109 + UN_110"}</h3>
        <p class="muted small" style="margin:0">${split === "test" ? "People the models never trained on and that were not used to pick epochs (but see \"Tiny test set\" below)." : "Used to pick the deep models' best training epoch, so expect these to look better than the test."}</p>
        <table class="sc"><thead><tr><th>Person</th>${MODELS.map((m) => `<th><span class="chip-dot" style="background:var(${m.color})"></span>${m.name}</th>`).join("")}</tr></thead><tbody>`;
      for (const who of [...people, "ALL"]) {
        const ms = ALL.filter((m) => m.split === split && (who === "ALL" || m.person === who)).filter(o.scored);
        html += `<tr class="${who === "ALL" ? "all" : ""}"><td class="who">${who === "ALL" ? "Both people" : who}<span class="sub">${ms.length} min</span></td>`;
        for (const md of MODELS) {
          const sc = score(ms, md.key);
          html += `<td><span class="ba">${pct(sc.ba)}</span><span class="sub">${o.neg.toLowerCase()} ${sc.rOk}/${sc.rN}<br>${o.pos.toLowerCase()} ${sc.wOk}/${sc.wN}</span></td>`;
          const OFF = eh ? OFFICIAL_EH : OFFICIAL;
          const off = OFF[md.key] && OFF[md.key][split] && OFF[md.key][split][who];
          if (off !== undefined) checks.push({ md, split, who, mine: sc.ba, off, src: OFF[md.key].src[split], official: true });
          const c = eh ? D.eh_cluster_metrics[`${split}|${who}|${md.key}`] : md.key === "motion" && D.motion_cluster_metrics[`${split}|${who}|motion|60s`];
          if (c) checks.push({ md, split, who, mine: sc.ba, off: c[0], src: eh ? "this page's easy-vs-hard cluster run" : "this page's head-motion cluster run" });
        }
        html += `</tr>`;
      }
      html += `</tbody></table></div>`;
    }
    html += `</div>`;
    box.innerHTML = html;

    const allOk = checks.every((c) => Math.abs(c.mine - c.off) < 5e-5);
    const lt = score(testMin, "labram", "rw").ba;
    const mt = score(testMin, "motion", "rw").ba;
    document.querySelectorAll('[data-fill="motionTest"]').forEach((el) => (el.textContent = `${pct(mt)} balanced accuracy, against ${pct(lt)} for LaBraM`));
    const nOff = checks.filter((c) => c.official).length, nRun = checks.length - nOff;
    const head = `<p><span class="check ${allOk ? "" : "bad"}">${allOk ? "✓ Matches the official results" : "✗ Mismatch with official results"}</span> `;
    $("#reproNote").innerHTML = (eh
      ? head + `(${nOff} official LaBraM easy-vs-hard numbers, plus ${nRun} numbers from this page's own cluster run, all equal to 4 decimals). Easy vs hard is scored the same way as rest vs working: 60-second windows, average the 15 four-second probabilities, above 50% counts as "hard". Only task minutes count, and each task's easy and hard levels are separate recordings.</p>
      <p>Models: <strong>LaBraM</strong> is the team's 3-layer cut LaBraM trained for easy vs hard (same design and calibration as the rest-vs-working one). <strong>EEGNet</strong> is the team's EEGNet-8,2 trained for easy vs hard on the same people (picked by validation loss). It reads 2-second windows at 128 samples/s, scaled with the training people's averages rather than the person's own rest. The <strong>simple</strong> and <strong>head-motion</strong> models use the same inputs as on the rest-vs-working side, refit for easy vs hard; in this mode the simple model's numbers are precomputed, not recomputed in your browser.</p>`
      : head + `(${nOff} official LaBraM, EEGNet and simple-model numbers, plus ${nRun} head-motion numbers from this page's own cluster run, all equal to 4 decimals). The official metric scores <strong>60-second windows</strong>: each recording is cut into back-to-back runs of 15 four-second clips, the 15 probabilities are averaged, and above 50% counts as "working". Every row of the per-minute file is exactly one of those windows, so the minute-level scores here equal the official 60-second scores (LaBraM final test <strong>${pct(lt)}</strong>). The official 4-second score (LaBraM 64.8%) can't be recomputed from minutes.</p>
      <p>The simple model on this page is the calibrated band-power model trained on UN_101–UN_108 (official test 77.3%). A different simple model, trained on a larger group of people and tested on 4 (UN_111, UN_112, UN_123, UN_124), scored 62%; it is not shown here. The head-motion line was added for this page with the same people, calibration, model type and 60-second windows; it was not part of the official runs.</p>`) +
      `<details><summary class="small">Checks</summary><ul class="small">${checks.map((c) => `<li>${c.md.name}, ${c.split}, ${c.who}: page ${c.mine.toFixed(4)} vs ${c.off.toFixed(4)} in <code>${esc(c.src)}</code></li>`).join("")}</ul></details>`;
  }
})();
