/* screening.js — SecureScan live screening console */

const LOG_STORAGE_KEY = "securescan_log_v1";
const SOUND_STORAGE_KEY = "securescan_sound_v1";

const CATEGORY = {
  gun: "critical", knife: "critical", blade: "critical", shuriken: "critical",
  spring: "suspicious", paperclip: "suspicious", zipper: "suspicious",
  bottle: "safe", screw: "safe", headset: "safe", spectacles: "safe",
};
const categoryOf = (name) => CATEGORY[name] || "safe";

const state = {
  queue: [],
  currentIndex: -1,
  isPlaying: false,
  intervalMs: 1800,
  confidence: 35,
  log: [],
  timerHandle: null,
  processing: false,
  soundEnabled: true,
  activeFilter: "all",
  searchQuery: "",
  classInfo: {},
  nonVisualCategories: [],
  mode: "queue",          // "queue" | "live"
  liveStream: null,
  liveFrameCount: 0,
};

let els = {};
let audioCtx = null;

document.addEventListener("DOMContentLoaded", init);

function init() {
  els = {
    stage: document.getElementById("stage"),
    stagePlaceholder: document.getElementById("stagePlaceholder"),
    stageImage: document.getElementById("stageImage"),
    stageCaption: document.getElementById("stageCaption"),
    stageFilename: document.getElementById("stageFilename"),
    stageCounter: document.getElementById("stageCounter"),
    hudReadoutTop: document.getElementById("hudReadoutTop"),
    hudThreshold: document.getElementById("hudThreshold"),
    progressFill: document.getElementById("progressFill"),
    screeningStatus: document.getElementById("screeningStatus"),
    playPauseBtn: document.getElementById("playPauseBtn"),
    playPauseLabel: document.getElementById("playPauseLabel"),
    prevBtn: document.getElementById("prevBtn"),
    nextBtn: document.getElementById("nextBtn"),
    speedSelect: document.getElementById("speedSelect"),
    confSlider: document.getElementById("confSlider"),
    confVal: document.getElementById("confVal"),
    loadSampleBtn: document.getElementById("loadSampleBtn"),
    liveCameraBtn: document.getElementById("liveCameraBtn"),
    liveVideo: document.getElementById("liveVideo"),
    captureCanvas: document.getElementById("captureCanvas"),
    selectFolderBtn: document.getElementById("selectFolderBtn"),
    selectFilesBtn: document.getElementById("selectFilesBtn"),
    folderInput: document.getElementById("folderInput"),
    filesInput: document.getElementById("filesInput"),
    resultEmpty: document.getElementById("resultEmpty"),
    resultContent: document.getElementById("resultContent"),
    resultImage: document.getElementById("resultImage"),
    resultImageWrap: document.getElementById("resultImageWrap"),
    resultBadge: document.getElementById("resultBadge"),
    resultFilename: document.getElementById("resultFilename"),
    resultTime: document.getElementById("resultTime"),
    resultChips: document.getElementById("resultChips"),
    logList: document.getElementById("logList"),
    logSummary: document.getElementById("logSummary"),
    logFilters: document.getElementById("logFilters"),
    logSearch: document.getElementById("logSearch"),
    exportLogBtn: document.getElementById("exportLogBtn"),
    clearLogBtn: document.getElementById("clearLogBtn"),
    modelStatusPill: document.getElementById("modelStatusPill"),
    modelStatusText: document.getElementById("modelStatusText"),
    soundToggle: document.getElementById("soundToggle"),
    toastStack: document.getElementById("toastStack"),
    lightbox: document.getElementById("lightbox"),
    lightboxImage: document.getElementById("lightboxImage"),
    lightboxClose: document.getElementById("lightboxClose"),
    statScreened: document.getElementById("statScreened"),
    statFlagged: document.getElementById("statFlagged"),
    statAvgMs: document.getElementById("statAvgMs"),
    statClearRate: document.getElementById("statClearRate"),
    resultReviewBadge: document.getElementById("resultReviewBadge"),
    referenceGuideBtn: document.getElementById("referenceGuideBtn"),
    referenceGuide: document.getElementById("referenceGuide"),
    referenceGuideBody: document.getElementById("referenceGuideBody"),
    referenceGuideClose: document.getElementById("referenceGuideClose"),
  };

  state.confidence = parseInt(els.confSlider.value, 10) || 35;
  state.intervalMs = parseInt(els.speedSelect.value, 10) || 1800;
  loadSoundPref();
  updateHudThreshold();

  loadLogFromStorage();
  renderLog();
  updateStats();
  checkModelStatus();
  bindEvents();
}

/* ── Model status ─────────────────────────────────────────────────────── */
async function checkModelStatus() {
  try {
    const res = await fetch("/api/model-status");
    const data = await res.json();
    state.classInfo = data.class_info || {};
    state.nonVisualCategories = data.non_visual_categories || [];
    if (data.ready) {
      els.modelStatusPill.className = "model-pill model-pill-ready";
      els.modelStatusText.textContent =
        `ENGINE READY · ${data.models_loaded}/${data.models_expected} MODELS · ${data.classes_covered.length} CLASSES`;
    } else {
      els.modelStatusPill.className = "model-pill model-pill-error";
      els.modelStatusText.textContent = "ENGINE UNAVAILABLE";
      showToast("Detection engine unavailable — check server logs.", "error");
    }
  } catch (e) {
    els.modelStatusPill.className = "model-pill model-pill-error";
    els.modelStatusText.textContent = "CANNOT REACH SERVER";
  }
}

/* ── Reference Guide ──────────────────────────────────────────────────── */
function openReferenceGuide() {
  renderReferenceGuide();
  els.referenceGuide.hidden = false;
}
function closeReferenceGuide() {
  els.referenceGuide.hidden = true;
}

function renderReferenceGuide() {
  const entries = Object.entries(state.classInfo);
  if (!entries.length) {
    els.referenceGuideBody.innerHTML = `<p class="ref-loading">Engine metadata unavailable — check /api/model-status.</p>`;
    return;
  }

  const order = { critical: 0, suspicious: 1, safe: 2 };
  entries.sort((a, b) => (order[a[1].category] ?? 9) - (order[b[1].category] ?? 9));

  const classRows = entries.map(([name, info]) => `
    <div class="ref-class-row">
      <span class="ref-class-icon">${info.icon}</span>
      <div class="ref-class-info">
        <span class="ref-class-name">
          ${escapeHTML(info.label || name)}
          <span class="badge chip-${info.category}" style="font-size:.62rem;padding:.1rem .4rem;border-radius:8px;">${info.category}</span>
        </span>
        <p class="ref-class-desc">${escapeHTML(info.description)}</p>
      </div>
      <span class="ref-class-tag ${info.trained ? "tag-trained" : "tag-untrained"}">
        ${info.trained ? "AI-DETECTABLE" : "LABEL ONLY"}
      </span>
    </div>`).join("");

  const nonVisualRows = state.nonVisualCategories.map((c) => `
    <div class="ref-nonvisual-row">
      <span class="ref-nonvisual-icon">${c.icon}</span>
      <div>
        <div class="ref-nonvisual-name">${escapeHTML(c.label)}</div>
        <div class="ref-nonvisual-note">${escapeHTML(c.note)}</div>
      </div>
    </div>`).join("");

  els.referenceGuideBody.innerHTML = `
    <div class="ref-section">
      <div class="ref-section-title">What this AI can actually see</div>
      <p class="ref-section-note">
        Every item below comes from the two trained models' own embedded class
        lists, not the dataset's aspirational label file. "Label only" classes
        are declared but were never shown a single training example — retraining
        with labelled images is the only way to close that gap, not a settings change.
      </p>
      ${classRows}
    </div>
    <div class="ref-section">
      <div class="ref-section-title">⚠ Categories no image classifier here covers</div>
      <p class="ref-section-note">
        These require trace detection, document checks, or human judgement —
        not visual classification. Listed so this tool doesn't imply coverage
        it doesn't have.
      </p>
      ${nonVisualRows}
    </div>`;
}

/* ── Event bindings ───────────────────────────────────────────────────── */
function bindEvents() {
  els.loadSampleBtn.addEventListener("click", loadSampleBatch);
  els.liveCameraBtn.addEventListener("click", startLiveCamera);
  els.selectFolderBtn.addEventListener("click", () => els.folderInput.click());
  els.selectFilesBtn.addEventListener("click", () => els.filesInput.click());
  els.folderInput.addEventListener("change", (e) => loadFilesIntoQueue(e.target.files));
  els.filesInput.addEventListener("change", (e) => loadFilesIntoQueue(e.target.files));

  els.stage.addEventListener("dragover", (e) => { e.preventDefault(); els.stage.classList.add("drag-over"); });
  els.stage.addEventListener("dragleave", () => els.stage.classList.remove("drag-over"));
  els.stage.addEventListener("drop", (e) => {
    e.preventDefault();
    els.stage.classList.remove("drag-over");
    loadFilesIntoQueue(e.dataTransfer.files);
  });

  els.playPauseBtn.addEventListener("click", togglePlay);
  els.prevBtn.addEventListener("click", () => {
    if (state.mode === "live") stopLiveCamera();
    else stepManual(-1);
  });
  els.nextBtn.addEventListener("click", () => stepManual(1));
  els.speedSelect.addEventListener("change", (e) => { state.intervalMs = parseInt(e.target.value, 10); });
  els.confSlider.addEventListener("input", (e) => {
    state.confidence = parseInt(e.target.value, 10);
    els.confVal.textContent = state.confidence + "%";
    updateHudThreshold();
  });

  els.exportLogBtn.addEventListener("click", exportLogCSV);
  els.clearLogBtn.addEventListener("click", clearLog);

  els.soundToggle.addEventListener("click", toggleSound);

  els.referenceGuideBtn.addEventListener("click", openReferenceGuide);
  els.referenceGuideClose.addEventListener("click", closeReferenceGuide);
  els.referenceGuide.addEventListener("click", (e) => { if (e.target === els.referenceGuide) closeReferenceGuide(); });

  els.resultImageWrap.addEventListener("click", () => {
    if (els.resultImage.src) openLightbox(els.resultImage.src);
  });
  els.lightboxClose.addEventListener("click", closeLightbox);
  els.lightbox.addEventListener("click", (e) => { if (e.target === els.lightbox) closeLightbox(); });

  els.logFilters.querySelectorAll(".filter-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      els.logFilters.querySelectorAll(".filter-chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      state.activeFilter = chip.dataset.filter;
      renderLog();
    });
  });
  els.logSearch.addEventListener("input", (e) => {
    state.searchQuery = e.target.value.trim().toLowerCase();
    renderLog();
  });

  document.addEventListener("keydown", handleKeydown);
  document.addEventListener("click", unlockAudioOnce, { once: true });
  window.addEventListener("pagehide", releaseCamera);
}

function handleKeydown(e) {
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "select" || tag === "textarea") return;

  if (e.key === "Escape") { closeLightbox(); closeReferenceGuide(); return; }
  if (!els.lightbox.hidden || !els.referenceGuide.hidden) return; // ignore other shortcuts while a modal is open

  if (e.code === "Space") { e.preventDefault(); togglePlay(); }
  else if (e.key === "ArrowLeft") { stepManual(-1); }
  else if (e.key === "ArrowRight") { stepManual(1); }
}

/* ── Sound ─────────────────────────────────────────────────────────────── */
function unlockAudioOnce() {
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === "suspended") audioCtx.resume();
  } catch (e) { /* Web Audio unavailable — sound alerts simply won't play */ }
}

function loadSoundPref() {
  const saved = localStorage.getItem(SOUND_STORAGE_KEY);
  state.soundEnabled = saved === null ? true : saved === "1";
  reflectSoundUI();
}

function toggleSound() {
  state.soundEnabled = !state.soundEnabled;
  localStorage.setItem(SOUND_STORAGE_KEY, state.soundEnabled ? "1" : "0");
  reflectSoundUI();
  if (state.soundEnabled) { unlockAudioOnce(); playTone(880, 0.08); }
}

function reflectSoundUI() {
  els.soundToggle.setAttribute("aria-pressed", String(state.soundEnabled));
  els.soundToggle.querySelector(".ico-sound-on").style.display = state.soundEnabled ? "" : "none";
  els.soundToggle.querySelector(".ico-sound-off").style.display = state.soundEnabled ? "none" : "";
}

function playTone(freq, duration, delay = 0) {
  if (!state.soundEnabled || !audioCtx) return;
  const t0 = audioCtx.currentTime + delay;
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = "sine";
  osc.frequency.value = freq;
  gain.gain.setValueAtTime(0.0001, t0);
  gain.gain.exponentialRampToValueAtTime(0.18, t0 + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.0001, t0 + duration);
  osc.connect(gain).connect(audioCtx.destination);
  osc.start(t0);
  osc.stop(t0 + duration + 0.02);
}

function playThreatAlert() {
  if (!state.soundEnabled) return;
  unlockAudioOnce();
  playTone(660, 0.12, 0);
  playTone(660, 0.12, 0.18);
}

function playReviewAlert() {
  if (!state.soundEnabled) return;
  unlockAudioOnce();
  playTone(520, 0.1, 0);
}

/* ── Queue loading ────────────────────────────────────────────────────── */
function stopCompletely() {
  state.isPlaying = false;
  if (state.timerHandle) { clearTimeout(state.timerHandle); state.timerHandle = null; }
  state.queue.forEach((item) => {
    if (item.source === "file" && item.previewURL) URL.revokeObjectURL(item.previewURL);
  });
  if (state.mode === "live") releaseCamera();
  state.mode = "queue";
}

/* ── Live camera ──────────────────────────────────────────────────────── */
async function startLiveCamera() {
  if (state.mode === "live") return; // already running
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showToast("Camera access needs HTTPS (or localhost). This page isn't in a secure context.", "error");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 480 } },
    });
    stopCompletely(); // tear down any queue-mode playback first
    state.mode = "live";
    state.liveStream = stream;
    state.liveFrameCount = 0;

    els.liveVideo.srcObject = stream;
    els.stage.classList.add("live-active");
    els.stagePlaceholder.style.display = "none";
    els.stageImage.style.display = "none";
    els.liveVideo.style.display = "block";
    els.stageCaption.style.display = "flex";
    els.hudReadoutTop.style.display = "flex";
    els.stageFilename.textContent = "Live camera feed";
    els.stageCounter.textContent = "LIVE";
    els.progressFill.style.width = "100%";

    els.playPauseBtn.disabled = false;
    els.prevBtn.disabled = false;
    els.prevBtn.title = "Stop Camera";
    els.nextBtn.disabled = true; // no meaning in live mode

    showToast("Live camera started — capturing at the selected speed interval.");
    startPlaying();
  } catch (err) {
    showToast("Camera access denied or unavailable: " + err.message, "error");
  }
}

function releaseCamera() {
  if (state.liveStream) {
    state.liveStream.getTracks().forEach((t) => t.stop());
    state.liveStream = null;
  }
  els.liveVideo.srcObject = null;
}

function stopLiveCamera() {
  pausePlaying();
  releaseCamera();
  state.mode = "queue";

  els.stage.classList.remove("live-active");
  els.liveVideo.style.display = "none";
  els.stageCaption.style.display = "none";
  els.hudReadoutTop.style.display = "none";
  els.stagePlaceholder.style.display = "flex";
  els.progressFill.style.width = "0%";
  els.prevBtn.title = "Previous (←)";
  els.prevBtn.disabled = true;
  els.nextBtn.disabled = true;
  els.playPauseBtn.disabled = true;
  setStatus("idle");
  setPlayButton("start");
  showToast("Live camera stopped.");
}

async function captureLiveFrame() {
  if (state.processing || !state.liveStream) return;
  const video = els.liveVideo;
  if (video.readyState < 2) return; // not enough data yet

  state.processing = true;
  state.liveFrameCount++;
  els.stageCounter.textContent = `LIVE · #${state.liveFrameCount}`;
  els.stage.classList.add("scanning");
  setStatus(state.isPlaying ? "scanning" : "paused");

  const canvas = els.captureCanvas;
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);

  try {
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.85));
    if (!blob) throw new Error("Could not capture frame");

    const fd = new FormData();
    fd.append("confidence", (state.confidence / 100).toFixed(2));
    fd.append("file", blob, `live-frame-${state.liveFrameCount}.jpg`);

    const res = await fetch("/api/screen", { method: "POST", body: fd });
    const data = await res.json();
    if (data.ok) {
      data.filename = `Live frame #${state.liveFrameCount}`;
      showResult(data);
      addLogEntry(data);
    }
  } catch (e) {
    console.warn("Live frame capture failed:", e);
  } finally {
    els.stage.classList.remove("scanning");
    state.processing = false;
  }
}

async function loadSampleBatch() {
  try {
    const res = await fetch("/api/samples");
    const data = await res.json();
    if (!data.samples || !data.samples.length) {
      showToast("No sample images found on the server.", "error");
      return;
    }
    stopCompletely();
    state.queue = data.samples.map((s, i) => ({
      id: "sample-" + i, name: s.name, source: "sample", sampleName: s.name, previewURL: s.url,
    }));
    state.currentIndex = -1;
    resetStageForNewQueue();
    showToast(`Loaded ${state.queue.length} sample images.`);
    startPlaying();
  } catch (e) {
    showToast("Could not load sample images: " + e.message, "error");
  }
}

function loadFilesIntoQueue(fileList) {
  const files = Array.from(fileList).filter((f) => f.type.startsWith("image/"));
  if (!files.length) {
    showToast("No image files found in that selection.", "warn");
    return;
  }
  stopCompletely();
  state.queue = files.map((f, i) => ({
    id: "file-" + i + "-" + f.name, name: f.name, source: "file", file: f,
    previewURL: URL.createObjectURL(f),
  }));
  state.currentIndex = -1;
  resetStageForNewQueue();
  showToast(`Loaded ${state.queue.length} image${state.queue.length === 1 ? "" : "s"} for screening.`);
  startPlaying();
}

function resetStageForNewQueue() {
  els.stage.classList.remove("live-active");
  els.liveVideo.style.display = "none";
  els.stagePlaceholder.style.display = "none";
  els.stageImage.style.display = "block";
  els.stageCaption.style.display = "flex";
  els.hudReadoutTop.style.display = "flex";
  els.playPauseBtn.disabled = false;
  els.prevBtn.disabled = false;
  els.prevBtn.title = "Previous (←)";
  els.nextBtn.disabled = false;
  setPlayButton("start");
  updateProgress();
}

/* ── Playback engine ──────────────────────────────────────────────────── */
function setPlayButton(mode) {
  const play = els.playPauseBtn.querySelector(".ico-play");
  const pause = els.playPauseBtn.querySelector(".ico-pause");
  const labels = { start: "Start", pause: "Pause", resume: "Resume", restart: "Restart" };
  els.playPauseLabel.textContent = labels[mode] || "Start";
  const showPause = mode === "pause";
  play.style.display = showPause ? "none" : "";
  pause.style.display = showPause ? "" : "none";
}

function startPlaying() {
  state.isPlaying = true;
  setStatus("scanning");
  setPlayButton("pause");
  if (state.mode === "live") liveLoop();
  else advanceLoop();
}

function pausePlaying() {
  state.isPlaying = false;
  setStatus("paused");
  setPlayButton("resume");
  if (state.timerHandle) { clearTimeout(state.timerHandle); state.timerHandle = null; }
}

function togglePlay() {
  if (state.mode === "live") {
    if (state.isPlaying) pausePlaying();
    else startPlaying();
    return;
  }
  if (state.queue.length === 0) return;
  if (state.isPlaying) {
    pausePlaying();
  } else {
    if (state.currentIndex >= state.queue.length - 1) state.currentIndex = -1;
    startPlaying();
  }
}

async function liveLoop() {
  if (!state.isPlaying || state.mode !== "live") return;
  await captureLiveFrame();
  if (!state.isPlaying || state.mode !== "live") return;
  state.timerHandle = setTimeout(liveLoop, state.intervalMs);
}

async function advanceLoop() {
  if (!state.isPlaying) return;
  if (state.currentIndex >= state.queue.length - 1) {
    state.isPlaying = false;
    setStatus("done");
    setPlayButton("restart");
    showToast(`Batch complete — ${state.queue.length} images screened.`);
    return;
  }
  state.currentIndex++;
  await processCurrent();
  if (!state.isPlaying) return;
  state.timerHandle = setTimeout(advanceLoop, state.intervalMs);
}

async function stepManual(direction) {
  if (state.mode === "live" || state.queue.length === 0) return;
  pausePlaying();
  const newIndex = state.currentIndex + direction;
  if (newIndex < 0 || newIndex >= state.queue.length) return;
  state.currentIndex = newIndex;
  await processCurrent();
}

async function processCurrent() {
  if (state.processing) return;
  state.processing = true;
  const item = state.queue[state.currentIndex];

  els.stageImage.src = item.previewURL;
  els.stageFilename.textContent = item.name;
  els.stageCounter.textContent = `${pad(state.currentIndex + 1)} / ${pad(state.queue.length)}`;
  els.stage.classList.add("scanning");
  setStatus(state.isPlaying ? "scanning" : "paused");
  updateProgress();

  const minDelay = new Promise((r) => setTimeout(r, 650));

  try {
    const fd = new FormData();
    fd.append("confidence", (state.confidence / 100).toFixed(2));
    if (item.source === "sample") fd.append("sample", item.sampleName);
    else fd.append("file", item.file, item.name);

    const [res] = await Promise.all([fetch("/api/screen", { method: "POST", body: fd }), minDelay]);
    const data = await res.json();

    if (data.ok) {
      showResult(data);
      addLogEntry(data);
    } else {
      await minDelay;
      showError(item.name, data.error || "Detection failed");
      showToast(`Failed to screen ${item.name}: ${data.error || "unknown error"}`, "error");
    }
  } catch (e) {
    await minDelay;
    showError(item.name, "Network error: " + e.message);
    showToast("Network error while screening " + item.name, "error");
  } finally {
    els.stage.classList.remove("scanning");
    state.processing = false;
  }
}

function pad(n) { return String(n).padStart(2, "0"); }

function setStatus(kind) {
  const map = {
    idle: ["IDLE", "status-idle"], scanning: ["SCANNING", "status-scanning"],
    paused: ["PAUSED", "status-paused"], done: ["COMPLETE", "status-done"],
  };
  const [label, cls] = map[kind] || map.idle;
  els.screeningStatus.textContent = label;
  els.screeningStatus.className = "status-badge " + cls;
}

function updateProgress() {
  const pct = state.queue.length ? ((state.currentIndex + 1) / state.queue.length) * 100 : 0;
  els.progressFill.style.width = pct + "%";
}

function updateHudThreshold() {
  if (els.hudThreshold) els.hudThreshold.textContent = `CONF ≥ ${state.confidence}%`;
}

/* ── Result panel ─────────────────────────────────────────────────────── */
function badgeLabel(level) {
  return level === "critical" ? "🔴 CRITICAL THREAT" : level === "suspicious" ? "🟠 SUSPICIOUS" : "🟢 SAFE";
}

function chipsHTML(objects, scores, tier = "confirmed") {
  if (!objects || !objects.length) {
    return tier === "confirmed" ? `<span class="chip chip-safe">No objects detected</span>` : "";
  }
  return objects.map((obj) => {
    const conf = scores && scores[obj] !== undefined ? ` ${scores[obj]}%` : "";
    if (tier === "possible") {
      return `<span class="chip chip-review" title="Below confirmation threshold — flagged for review">${escapeHTML(obj)}?${conf}</span>`;
    }
    const cat = categoryOf(obj);
    return `<span class="chip chip-${cat}">${escapeHTML(obj)}${conf}</span>`;
  }).join("");
}

function combinedChipsHTML(confirmedObjs, possibleObjs, scores) {
  const confirmed = chipsHTML(confirmedObjs, scores, "confirmed");
  const possible = chipsHTML(possibleObjs, scores, "possible");
  return confirmed + possible;
}

function showResult(data) {
  els.resultEmpty.style.display = "none";
  els.resultContent.style.display = "flex";
  els.resultImage.src = data.annotated_image;
  els.resultBadge.textContent = badgeLabel(data.security_level);
  els.resultBadge.className = "security-badge security-" + data.security_level;
  els.resultFilename.textContent = data.filename;
  els.resultTime.textContent = new Date().toLocaleTimeString();
  els.resultChips.innerHTML = combinedChipsHTML(data.detected_objects, data.possible_objects, data.confidence_scores);
  els.resultReviewBadge.style.display = data.needs_review ? "inline-flex" : "none";

  if (data.security_level === "critical") {
    triggerThreatFlash();
    playThreatAlert();
    showToast(`⚠ THREAT: ${data.threat_items.join(", ")} detected in ${data.filename}`, "error");
  } else if (data.security_level === "suspicious") {
    showToast(`Suspicious item flagged in ${data.filename}: ${data.threat_items.join(", ")}`, "warn");
  } else if (data.needs_review) {
    // No confirmed threat, but a borderline weapon/suspicious signal exists --
    // a real improvement over silently hiding it: surfaced with a lighter tone
    // instead of the full alarm, since it's explicitly unconfirmed.
    playReviewAlert();
    showToast(`Possible ${data.possible_threat_items.join(", ")} in ${data.filename} — below confirmation threshold, review recommended.`, "warn");
  }
}

function triggerThreatFlash() {
  els.stage.classList.remove("threat-flash");
  void els.stage.offsetWidth; // restart animation even if already mid-flash
  els.stage.classList.add("threat-flash");
  setTimeout(() => els.stage.classList.remove("threat-flash"), 1700);
}

function showResultFromLog(entry) {
  els.resultEmpty.style.display = "none";
  els.resultContent.style.display = "flex";
  els.resultImage.src = entry.thumb;
  els.resultBadge.textContent = badgeLabel(entry.security_level);
  els.resultBadge.className = "security-badge security-" + entry.security_level;
  els.resultFilename.textContent = entry.filename;
  els.resultTime.textContent = new Date(entry.timestamp).toLocaleTimeString();
  els.resultChips.innerHTML = combinedChipsHTML(entry.detected_objects, entry.possible_objects || [], entry.confidence_scores);
  els.resultReviewBadge.style.display = entry.needs_review ? "inline-flex" : "none";
}

function showError(filename, message) {
  els.resultEmpty.style.display = "none";
  els.resultContent.style.display = "flex";
  els.resultImage.src = "";
  els.resultBadge.textContent = "⚠️ ERROR";
  els.resultBadge.className = "security-badge security-suspicious";
  els.resultFilename.textContent = filename;
  els.resultTime.textContent = new Date().toLocaleTimeString();
  els.resultChips.innerHTML = `<span class="chip chip-suspicious">${escapeHTML(message)}</span>`;
}

/* ── Lightbox ─────────────────────────────────────────────────────────── */
function openLightbox(src) {
  els.lightboxImage.src = src;
  els.lightbox.hidden = false;
}
function closeLightbox() {
  els.lightbox.hidden = true;
  els.lightboxImage.src = "";
}

/* ── Toasts ───────────────────────────────────────────────────────────── */
function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = "toast" + (type === "error" ? " toast-error" : type === "warn" ? " toast-warn" : "");
  toast.textContent = message;
  els.toastStack.appendChild(toast);
  setTimeout(() => {
    toast.classList.add("toast-out");
    setTimeout(() => toast.remove(), 220);
  }, 3600);
}

/* ── Screening log ────────────────────────────────────────────────────── */
function addLogEntry(data) {
  const entry = {
    id: Date.now() + "-" + Math.random().toString(36).slice(2, 7),
    filename: data.filename,
    timestamp: new Date().toISOString(),
    security_level: data.security_level,
    detected_objects: data.detected_objects,
    threat_items: data.threat_items,
    possible_objects: data.possible_objects || [],
    possible_threat_items: data.possible_threat_items || [],
    needs_review: !!data.needs_review,
    confidence_scores: data.confidence_scores,
    thumb: data.annotated_image,
    duration_ms: data.duration_ms,
  };
  state.log.unshift(entry);
  if (state.log.length > 300) state.log.length = 300;
  saveLogToStorage();
  renderLog();
  updateStats();
}

function filteredLog() {
  return state.log.filter((e) => {
    if (state.activeFilter !== "all" && e.security_level !== state.activeFilter) return false;
    if (state.searchQuery && !e.filename.toLowerCase().includes(state.searchQuery)) return false;
    return true;
  });
}

function renderLog() {
  const rows = filteredLog();
  if (!rows.length) {
    els.logList.innerHTML = `<div class="log-empty">${state.log.length ? "No entries match this filter." : "No images screened yet."}</div>`;
    return;
  }
  els.logList.innerHTML = rows.map(rowHTML).join("");
  els.logList.querySelectorAll(".log-row").forEach((row) => {
    row.addEventListener("click", () => {
      const entry = state.log.find((e) => e.id === row.dataset.id);
      if (entry) showResultFromLog(entry);
    });
  });
}

function rowHTML(entry) {
  const rowClass = "log-row-" + entry.security_level + (entry.needs_review ? " log-row-needs-review" : "");
  const objs = entry.detected_objects.length ? entry.detected_objects.join(", ") : "No objects detected";
  const time = new Date(entry.timestamp).toLocaleTimeString();
  const reviewTag = entry.needs_review ? `<span class="log-review-tag">REVIEW</span>` : "";
  return `
    <div class="log-row ${rowClass}" data-id="${entry.id}">
      <img class="log-thumb" src="${entry.thumb}" alt="">
      <div class="log-meta">
        <span class="log-name">${escapeHTML(entry.filename)}</span>
        <span class="log-sub">${escapeHTML(objs)}</span>
      </div>
      <span class="log-time">${time}</span>
      ${reviewTag}
      <span class="log-badge security-${entry.security_level}">${entry.security_level.toUpperCase()}</span>
    </div>`;
}

function updateStats() {
  const total = state.log.length;
  const flagged = state.log.filter((e) => e.security_level !== "safe").length;
  const avgMs = total ? Math.round(state.log.reduce((sum, e) => sum + (e.duration_ms || 0), 0) / total) : null;
  const clearRate = total ? Math.round(((total - flagged) / total) * 100) : null;

  els.statScreened.textContent = total;
  els.statFlagged.textContent = flagged;
  els.statAvgMs.textContent = avgMs === null ? "—" : avgMs;
  els.statClearRate.textContent = clearRate === null ? "—" : clearRate + "%";
  els.logSummary.textContent = `${total} screened · ${flagged} flagged`;
}

function clearLog() {
  if (!state.log.length) return;
  if (!confirm("Clear the entire screening log? This cannot be undone.")) return;
  state.log = [];
  saveLogToStorage();
  renderLog();
  updateStats();
  showToast("Screening log cleared.");
}

function exportLogCSV() {
  if (!state.log.length) { showToast("Log is empty.", "warn"); return; }
  const header = ["Timestamp", "Filename", "Security Level", "Needs Review", "Detected Objects",
                  "Threat Items", "Possible Objects (below threshold)", "Duration (ms)"];
  const rows = state.log.map((e) => [
    e.timestamp, e.filename, e.security_level, e.needs_review ? "yes" : "no",
    e.detected_objects.join("; "), e.threat_items.join("; "),
    (e.possible_objects || []).join("; "), e.duration_ms,
  ]);
  const csv = [header, ...rows].map((r) => r.map(csvEscape).join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `securescan_log_${Date.now()}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  showToast("Log exported as CSV.");
}

function csvEscape(val) {
  const s = String(val ?? "");
  return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

function escapeHTML(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function saveLogToStorage() {
  try { localStorage.setItem(LOG_STORAGE_KEY, JSON.stringify(state.log)); }
  catch (e) { /* storage unavailable — log still works for this session */ }
}

function loadLogFromStorage() {
  try {
    const raw = localStorage.getItem(LOG_STORAGE_KEY);
    if (raw) state.log = JSON.parse(raw);
  } catch (e) { state.log = []; }
}
