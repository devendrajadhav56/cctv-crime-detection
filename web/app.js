const STORAGE_KEY = "cctv-api-url";

const els = {
  apiUrl: document.getElementById("api-url"),
  health: document.getElementById("health"),
  drop: document.getElementById("drop"),
  file: document.getElementById("file"),
  browse: document.getElementById("browse"),
  player: document.getElementById("player"),
  badge: document.getElementById("badge"),
  progressWrap: document.getElementById("progress-wrap"),
  stage: document.getElementById("stage"),
  pct: document.getElementById("pct"),
  bar: document.getElementById("bar"),
  timeline: document.getElementById("timeline"),
  error: document.getElementById("error"),
  download: document.getElementById("download"),
  statFile: document.getElementById("stat-file"),
  statStatus: document.getElementById("stat-status"),
  statWindows: document.getElementById("stat-windows"),
  statMix: document.getElementById("stat-mix"),
  statClasses: document.getElementById("stat-classes"),
};

let windows = [];
let objectUrl = null;
let pollTimer = null;

function apiBase() {
  const raw = els.apiUrl.value.trim().replace(/\/$/, "");
  return raw;
}

function jobUrl(path) {
  const base = apiBase();
  if (!base) return path;
  return `${base}${path}`;
}

function showError(message) {
  els.error.hidden = !message;
  els.error.textContent = message || "";
}

function setHealth(data, ok) {
  els.health.classList.remove("online", "offline", "dry");
  if (!ok) {
    els.health.classList.add("offline");
    els.health.textContent = "offline";
    return;
  }
  if (data.dry_run) {
    els.health.classList.add("dry");
    els.health.textContent = `dry-run · ${data.device}`;
    return;
  }
  els.health.classList.add("online");
  els.health.textContent = data.cuda ? `live · ${data.device}` : `cpu · ${data.device}`;
}

async function ping() {
  try {
    const response = await fetch(jobUrl("/v1/health"));
    if (!response.ok) throw new Error("health failed");
    setHealth(await response.json(), true);
  } catch {
    setHealth(null, false);
  }
}

function formatTime(seconds) {
  const total = Math.max(0, Math.floor(seconds));
  const minutes = String(Math.floor(total / 60)).padStart(2, "0");
  const secs = String(total % 60).padStart(2, "0");
  return `${minutes}:${secs}`;
}

// Probability this window is anomalous (any non-"normal" label), regardless
// of which backend or which specific class produced it.
function anomalyScore(row) {
  if (row.label == null || row.confidence == null) return 0;
  return row.label === "normal" ? 1 - row.confidence : row.confidence;
}

function displayAt(timeSec) {
  const covering = windows.filter((row) => timeSec >= row.start_sec && timeSec < row.end_sec);
  if (!covering.length) return { label: "NORMAL", score: 0 };
  const best = covering.reduce((a, b) => (anomalyScore(b) > anomalyScore(a) ? b : a));
  const score = anomalyScore(best);
  if (score >= 0.5) return { label: best.display_label || best.label, score };
  return { label: "NORMAL", score: 1 - score };
}

function classBreakdown(rows) {
  const counts = new Map();
  rows.forEach((row) => {
    const key = row.label || "?";
    counts.set(key, (counts.get(key) || 0) + 1);
  });
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([label, count]) => `${label} ×${count}`)
    .join(" · ");
}

function renderTimeline(rows) {
  els.timeline.innerHTML = "";
  rows.forEach((row) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `seg ${row.label !== "normal" ? "crime" : "normal"}`;
    button.title = `${formatTime(row.start_sec)}–${formatTime(row.end_sec)} ${row.display_label || row.label}`;
    button.addEventListener("click", () => {
      els.player.currentTime = row.start_sec + 0.05;
      els.player.play();
    });
    els.timeline.appendChild(button);
  });
}

function updateBadge() {
  if (els.player.hidden || !windows.length) return;
  const { label } = displayAt(els.player.currentTime);
  els.badge.hidden = false;
  els.badge.textContent = label;
  els.badge.classList.toggle("crime", label !== "NORMAL");
  els.badge.classList.toggle("normal", label === "NORMAL");
  const segs = [...els.timeline.children];
  segs.forEach((el, index) => {
    const row = windows[index];
    el.classList.toggle("active", els.player.currentTime >= row.start_sec && els.player.currentTime < row.end_sec);
  });
}

function setProgress(job) {
  const running = job.status === "queued" || job.status === "running";
  els.progressWrap.hidden = !running && job.status !== "done";
  els.stage.textContent = job.stage || job.status;
  const pct = Math.round((job.progress || 0) * 100);
  els.pct.textContent = `${pct}%`;
  els.bar.style.width = `${pct}%`;
  els.statStatus.textContent = job.status;
}

async function pollJob(jobId) {
  const response = await fetch(jobUrl(`/v1/jobs/${jobId}`));
  if (!response.ok) throw new Error("Could not read job status.");
  const job = await response.json();
  setProgress(job);
  if (job.status === "error") throw new Error(job.error || "Job failed.");
  if (job.status !== "done") {
    pollTimer = setTimeout(() => pollJob(jobId).catch((err) => showError(err.message)), 700);
    return;
  }
  windows = job.windows || [];
  els.statWindows.textContent = String(windows.length);
  const anomalous = windows.filter((row) => row.label !== "normal").length;
  els.statMix.textContent = `${anomalous} / ${windows.length - anomalous}`;
  els.statClasses.textContent = classBreakdown(windows);
  renderTimeline(windows);

  if (objectUrl) URL.revokeObjectURL(objectUrl);
  const videoResponse = await fetch(jobUrl(job.video_url));
  if (!videoResponse.ok) throw new Error("Could not download labelled video.");
  const blob = await videoResponse.blob();
  objectUrl = URL.createObjectURL(new Blob([blob], { type: "video/mp4" }));
  els.drop.hidden = true;
  els.player.hidden = false;
  els.player.src = objectUrl;
  els.player.load();
  els.download.hidden = false;
  els.download.href = objectUrl;
  els.download.download = "labelled.mp4";
  els.badge.hidden = false;
  try {
    await els.player.play();
  } catch {
    // Autoplay can be blocked; controls still work.
  }
}

async function upload(file) {
  showError("");
  if (pollTimer) clearTimeout(pollTimer);
  els.statFile.textContent = file.name;
  els.statStatus.textContent = "uploading";
  els.drop.hidden = false;
  els.player.hidden = true;
  els.badge.hidden = true;
  els.download.hidden = true;
  const body = new FormData();
  body.append("video", file, file.name);
  const response = await fetch(jobUrl("/v1/jobs"), { method: "POST", body });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || "Upload failed.");
  }
  const job = await response.json();
  setProgress(job);
  await pollJob(job.job_id);
}

function takeFile(file) {
  if (!file) return;
  upload(file).catch((err) => showError(err.message));
}

els.apiUrl.value = localStorage.getItem(STORAGE_KEY) || "";
els.apiUrl.addEventListener("change", () => {
  localStorage.setItem(STORAGE_KEY, els.apiUrl.value.trim());
  ping();
});

els.browse.addEventListener("click", (event) => {
  event.preventDefault();
  event.stopPropagation();
  els.file.click();
});
els.drop.addEventListener("click", () => els.file.click());
els.file.addEventListener("change", () => takeFile(els.file.files[0]));
els.drop.addEventListener("dragover", (event) => event.preventDefault());
els.drop.addEventListener("drop", (event) => {
  event.preventDefault();
  takeFile(event.dataTransfer.files[0]);
});
els.player.addEventListener("timeupdate", updateBadge);
els.player.addEventListener("error", () => {
  showError("The browser could not decode the labelled file. Restart the GB10 server on the latest code (H.264) and run the job again.");
});

ping();
setInterval(ping, 5000);
