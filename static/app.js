const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const API_KEY_STORAGE = "cosmos-studio-api-key";
const state = { health: null, jobs: [], gallery: [], config: null };

const MODES = {
  cosmos_image: { hint: "Generate a single Cosmos3 frame from text.", size: "1280x720", frames: 1, fps: 24, steps: 35, guidance: 4, flow: 3 },
  cosmos_text_video: { hint: "Generate a new Cosmos3 world sequence from text.", size: "1280x720", frames: 189, fps: 24, steps: 35, guidance: 6, flow: 10 },
  cosmos_image_video: { hint: "Anchor the first frame to an image, then generate motion.", size: "1280x720", frames: 189, fps: 24, steps: 35, guidance: 6, flow: 10 },
  cosmos_video_continue: { hint: "Use selected opening latent frames as context; this is continuation, not full-frame editing.", size: "1280x720", frames: 189, fps: 24, steps: 35, guidance: 6, flow: 10 },
  vace_text_video: { hint: "Generate video through the VACE model without media conditioning.", size: "832x480", frames: 81, fps: 16, steps: 30, guidance: 5, flow: 3 },
  vace_reference_video: { hint: "Guide subject or object identity with one or more reference images.", size: "832x480", frames: 81, fps: 16, steps: 30, guidance: 5, flow: 3 },
  vace_control_edit: { hint: "Regenerate a whole control clip or only white mask regions; black mask regions are preserved.", size: "832x480", frames: 81, fps: 16, steps: 30, guidance: 5, flow: 3 },
};

function randomSeed() {
  const bytes = new Uint32Array(2);
  crypto.getRandomValues(bytes);
  return String((BigInt(bytes[0] & 0x7fffffff) << 32n) | BigInt(bytes[1]));
}

function key() { return $("#apiKey").value.trim(); }
function authHeaders(extra = {}) { return key() ? { ...extra, Authorization: `Bearer ${key()}` } : extra; }

async function apiFetch(url, options = {}) {
  const response = await fetch(url, { ...options, headers: authHeaders(options.headers || {}) });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { const body = await response.json(); detail = body.detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return response.json();
}

function restoreKey() {
  const persistent = localStorage.getItem(API_KEY_STORAGE);
  const temporary = sessionStorage.getItem(API_KEY_STORAGE);
  $("#apiKey").value = persistent || temporary || "";
  $("#rememberKey").checked = Boolean(persistent);
}

function saveKey() {
  localStorage.removeItem(API_KEY_STORAGE); sessionStorage.removeItem(API_KEY_STORAGE);
  if (key()) ($("#rememberKey").checked ? localStorage : sessionStorage).setItem(API_KEY_STORAGE, key());
  $("#saveKey").textContent = "Saved";
  setTimeout(() => $("#saveKey").textContent = "Save key", 900);
  refreshAll();
}

function updateMode(resetValues = true) {
  const mode = $("#mode").value;
  const config = MODES[mode];
  const isVideo = mode !== "cosmos_image";
  const isVace = mode.startsWith("vace_");
  $("#modeHint").textContent = config.hint;
  $$('[data-modes]').forEach((element) => element.classList.toggle("visible", element.dataset.modes.split(" ").includes(mode)));
  $$('[data-video]').forEach((element) => element.hidden = !isVideo);
  $$('[data-image]').forEach((element) => element.hidden = isVideo);
  $$('[data-vace]').forEach((element) => element.hidden = !isVace);
  $$('[data-cosmos-video]').forEach((element) => element.hidden = !(mode.startsWith("cosmos_") && isVideo));
  $("#enhancePrompt").checked = !isVace;
  const sizes = isVace ? ["832x480", "480x832", "1280x720", "720x1280"] : ["1280x720", "720x1280", "1024x1024", "960x544", "544x960"];
  $("#size").replaceChildren(...sizes.map((size) => new Option(size, size)));
  if (resetValues) {
    $("#size").value = config.size; $("#numFrames").value = config.frames; $("#fps").value = config.fps;
    $("#steps").value = config.steps; $("#guidance").value = config.guidance; $("#flowShift").value = config.flow;
  }
}

async function enhanceNow() {
  const prompt = $("#prompt").value.trim();
  if (!prompt) return showMessage("Enter a prompt first.", true);
  $("#enhanceButton").disabled = true; showMessage("Enhancing prompt…");
  try {
    const result = await apiFetch("/v1/prompt/enhance", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, mode: $("#mode").value, size: $("#size").value }),
    });
    $("#structuredPrompt").value = JSON.stringify(result.structured_prompt, null, 2);
    $("#structuredWrap").classList.remove("hidden"); $("#enhancePrompt").checked = false;
    showMessage(`Enhanced in ${result.enhancement_seconds}s.`);
  } catch (error) { showMessage(error.message, true); }
  finally { $("#enhanceButton").disabled = false; }
}

function showMessage(message, error = false) { $("#formMessage").textContent = message; $("#formMessage").classList.toggle("error", error); }

async function enqueue(event) {
  event.preventDefault();
  const form = new FormData($("#generationForm"));
  if (!form.get("seed")) { $("#seed").value = randomSeed(); form.set("seed", $("#seed").value); }
  for (const checkbox of ["enhance_prompt", "enhancement_fallback_to_plain", "enable_sound"]) form.set(checkbox, $(`[name="${checkbox}"]`).checked ? "true" : "false");
  if (!$("#structuredPrompt").value.trim()) form.delete("structured_prompt");
  showMessage("Saving uploads and assigning queue position…");
  try {
    const result = await apiFetch("/v1/jobs", { method: "POST", body: form });
    showMessage(`Queued ${result.id.slice(0, 8)} with seed ${result.seed}.`);
    if ($("#advanceSeed").checked) $("#seed").value = randomSeed();
    await refreshJobs(); location.hash = "queue";
  } catch (error) { showMessage(error.message, true); }
}

function escapeHtml(text) { const div = document.createElement("div"); div.textContent = text ?? ""; return div.innerHTML; }
function formatTime(seconds) { return seconds ? new Date(seconds * 1000).toLocaleString() : ""; }
function modeLabel(mode) { return mode.replaceAll("_", " "); }

function renderJobs() {
  const list = $("#jobList");
  if (!state.jobs.length) { list.innerHTML = '<p class="empty">No jobs yet.</p>'; return; }
  list.innerHTML = state.jobs.map((job) => {
    const canCancel = ["queued", "running"].includes(job.state);
    const detail = job.error || job.progress_message || job.state;
    return `<article class="job-card">
      <span class="job-state ${job.state}"></span>
      <div><div class="job-title"><strong>${escapeHtml(modeLabel(job.mode))}</strong><code>${job.id.slice(0, 10)}</code></div>
      <div class="job-prompt">${escapeHtml(job.prompt)}</div>
      <div class="job-meta">${escapeHtml(detail)} · seed ${job.seed} · ${formatTime(job.created_at)}${job.queue_position ? ` · position ${job.queue_position}` : ""}</div>
      <div class="progress-track"><div class="progress-fill" style="width:${Math.round((job.progress || 0) * 100)}%"></div></div></div>
      <div class="job-actions">${job.output_url ? `<a class="secondary-button" href="${job.output_url}" target="_blank">Open</a>` : ""}${canCancel ? `<button class="danger-button" data-cancel="${job.id}">Cancel</button>` : ""}</div>
    </article>`;
  }).join("");
  $$('[data-cancel]').forEach((button) => button.addEventListener("click", () => cancelJob(button.dataset.cancel)));
}

async function cancelJob(id) { try { await apiFetch(`/v1/jobs/${id}`, { method: "DELETE" }); await refreshJobs(); } catch (error) { showMessage(error.message, true); } }

function renderGallery() {
  const grid = $("#galleryGrid");
  if (!state.gallery.length) { grid.innerHTML = '<p class="empty">Completed work appears here from every LAN browser.</p>'; return; }
  grid.innerHTML = state.gallery.map((job) => {
    let media = '<div class="gallery-placeholder">No browser preview</div>';
    if ((job.media_type || "").startsWith("image/")) media = `<img src="${job.output_url}" loading="lazy" alt="${escapeHtml(job.prompt)}">`;
    if ((job.media_type || "").startsWith("video/")) media = `<video src="${job.output_url}" controls preload="metadata"></video>`;
    return `<article class="gallery-card"><div class="gallery-media">${media}</div><div class="gallery-body">
      <p>${escapeHtml(job.prompt)}</p><small>${escapeHtml(modeLabel(job.mode))} · seed ${job.seed} · ${formatTime(job.completed_at)}</small>
      <div class="gallery-links"><a href="${job.output_url}" target="_blank">Open</a><a href="${job.output_url}" download>Download</a><a href="/v1/jobs/${job.id}" target="_blank">Metadata</a></div>
    </div></article>`;
  }).join("");
}

async function refreshJobs() {
  try { const result = await apiFetch("/v1/jobs?limit=100"); state.jobs = result.data; renderJobs(); }
  catch (error) { if (key()) console.warn(error); }
}
async function refreshGallery() {
  try { const result = await apiFetch("/v1/gallery?limit=60"); state.gallery = result.data; renderGallery(); }
  catch (error) { if (key()) console.warn(error); }
}

async function refreshHealth() {
  try {
    const health = await fetch("/health").then((response) => response.json()); state.health = health;
    $("#statusText").textContent = health.status; $("#statusDot").className = `status-dot ${health.status === "ready" || health.status === "running" ? "ready" : "error"}`;
    $("#backendName").textContent = health.backend; $("#profileName").textContent = health.device_profile;
    $("#activeJob").textContent = health.current_job_id ? health.current_job_id.slice(0, 8) : "None";
    $("#metricQueued").textContent = health.queue.queued || 0; $("#metricCompleted").textContent = health.queue.completed || 0; $("#metricFailed").textContent = health.queue.failed || 0;
    $("#queuePill").textContent = (health.queue.queued || 0) + (health.queue.running || 0);
    $("#statusJson").textContent = JSON.stringify(health, null, 2);
  } catch (_) { $("#statusText").textContent = "Disconnected"; $("#statusDot").className = "status-dot error"; }
}

async function refreshConfig() { try { state.config = await apiFetch("/v1/config"); } catch (_) {} }
async function refreshAll() { await Promise.all([refreshHealth(), refreshConfig(), refreshJobs(), refreshGallery()]); }

function resetForm() {
  $("#generationForm").reset(); $("#structuredWrap").classList.add("hidden"); $("#structuredPrompt").value = "";
  updateMode(true); $("#seed").value = randomSeed(); showMessage("");
}

$("#generationForm").addEventListener("submit", enqueue);
$("#mode").addEventListener("change", () => updateMode(true));
$("#enhanceButton").addEventListener("click", enhanceNow);
$("#clearStructured").addEventListener("click", () => { $("#structuredPrompt").value = ""; $("#structuredWrap").classList.add("hidden"); $("#enhancePrompt").checked = true; });
$("#newSeed").addEventListener("click", () => $("#seed").value = randomSeed());
$("#resetButton").addEventListener("click", resetForm);
$("#saveKey").addEventListener("click", saveKey);
$("#refreshJobs").addEventListener("click", refreshJobs);
$("#refreshGallery").addEventListener("click", refreshGallery);
$("#statusButton").addEventListener("click", () => $("#statusDialog").showModal());
$("#closeDialog").addEventListener("click", () => $("#statusDialog").close());

restoreKey(); updateMode(true); $("#seed").value = randomSeed(); refreshAll();
setInterval(refreshHealth, 2000); setInterval(refreshJobs, 3000); setInterval(refreshGallery, 10000);
