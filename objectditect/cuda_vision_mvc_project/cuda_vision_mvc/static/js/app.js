const gpuInfo = document.getElementById("gpuInfo");
const toastContainer = document.getElementById("toastContainer");
const modelSelect = document.getElementById("modelSelect");
const threshInput = document.getElementById("threshInput");
const threshVal = document.getElementById("threshVal");
const sideInput = document.getElementById("sideInput");
const sideVal = document.getElementById("sideVal");
const strideInput = document.getElementById("strideInput");
const strideVal = document.getElementById("strideVal");
const maxFramesInput = document.getElementById("maxFramesInput");
const maxFramesVal = document.getElementById("maxFramesVal");
const targetLabelsInput = document.getElementById("targetLabelsInput");
const classChecks = document.getElementById("classChecks");
const humanAttrChk = document.getElementById("humanAttrChk");
const saveDbChk = document.getElementById("saveDbChk");
const saveDownloadChk = document.getElementById("saveDownloadChk");
const previewBox = document.getElementById("previewBox");
const statusMsg = document.getElementById("statusMsg");
const detList = document.getElementById("detList");
const notifList = document.getElementById("notifList");
const downloadsList = document.getElementById("downloadsList");
const dbResultsList = document.getElementById("dbResultsList");

let lastResultData = null;
let selectedImageFile = null;
let selectedVideoFile = null;
let videoEl = null;
let mediaStream = null;

threshInput.addEventListener("input", () => (threshVal.textContent = Number(threshInput.value).toFixed(2)));
sideInput.addEventListener("input", () => (sideVal.textContent = sideInput.value));
strideInput.addEventListener("input", () => (strideVal.textContent = strideInput.value));
maxFramesInput.addEventListener("input", () => (maxFramesVal.textContent = maxFramesInput.value));

function showToast(level, title, message) {
  const div = document.createElement("div");
  div.className = `toast ${level || "info"}`;
  div.innerHTML = `<strong>${title}</strong><div class="sub-txt" style="color:#dce5ff;">${message}</div>`;
  toastContainer.appendChild(div);
  setTimeout(() => div.remove(), 5000);
  if (window.Notification && Notification.permission === "granted" && (level === "warning" || level === "error")) {
    new Notification(title, { body: message });
  }
}

if (window.Notification && Notification.permission === "default") {
  Notification.requestPermission().catch(() => {});
}

function extractYouTubeId(url) {
  const m = (url || "").match(/(?:v=|\/embed\/|youtu\.be\/|\/shorts\/)([A-Za-z0-9_-]{6,})/);
  return m ? m[1] : null;
}

async function loadNotifications() {
  try {
    const res = await fetch("/api/notifications");
    const data = await res.json();
    const list = data.notifications || [];
    if (list.length === 0) {
      notifList.innerHTML = `<div class="row-item"><span class="sub-txt">কোনো নোটিফিকেশন নেই</span></div>`;
      return;
    }
    notifList.innerHTML = list
      .map(
        (n) => `
      <div class="row-item">
        <div>
          <strong>[${n.timestamp}] ${n.title} ${n.alert ? "🔔" : ""}</strong>
          <div class="sub-txt">${n.message}</div>
        </div>
        <span class="sub-txt">${n.level.toUpperCase()}</span>
      </div>
    `
      )
      .join("");
  } catch (_) {}
}

async function loadDownloads() {
  try {
    const res = await fetch("/api/downloads");
    const data = await res.json();
    const files = data.files || [];
    if (files.length === 0) {
      downloadsList.innerHTML = `<div class="row-item"><span class="sub-txt">./downloads ফোল্ডার খালি</span></div>`;
      return;
    }
    downloadsList.innerHTML = files
      .map(
        (f) => `
      <div class="row-item">
        <div>
          <div><strong>${f.filename}</strong></div>
          <div class="sub-txt">${f.size_kb} KB</div>
        </div>
        <a class="dl-link" href="${f.download_url}" download>⬇ Download</a>
      </div>
    `
      )
      .join("");
  } catch (_) {}
}

async function loadDbResults() {
  try {
    const res = await fetch("/api/db/results?limit=20");
    const data = await res.json();
    const rows = data.results || [];
    if (rows.length === 0) {
      dbResultsList.innerHTML = `<div class="row-item"><span class="sub-txt">SQLite ডেটাবেজে এখনো কোনো রেজাল্ট নেই</span></div>`;
      return;
    }
    dbResultsList.innerHTML = rows
      .map((r) => {
        const uniq = r.summary && r.summary.unique_labels ? r.summary.unique_labels.join(", ") : "-";
        return `
        <div class="row-item">
          <div>
            <div><strong>#${r.id} [${r.source_type.toUpperCase()}] ${r.model_name}</strong> • ${r.created_at}</div>
            <div class="sub-txt">Total: ${r.total_objects} | Unique (${r.unique_count}): ${uniq || "None"} | GPU: ${r.gpu_ms} ms</div>
          </div>
          <button class="btn-secondary btn-sm" onclick="viewDbRecord(${r.id})">View</button>
        </div>
      `;
      })
      .join("");
  } catch (_) {}
}

window.viewDbRecord = async function (id) {
  try {
    const res = await fetch(`/api/db/results/${id}`);
    const r = await res.json();
    if (!res.ok) throw new Error(r.detail || "Record load failed");

    if (r.preview_base64) {
      previewBox.innerHTML = `<img src="${r.preview_base64}" alt="DB Record #${r.id}" />`;
    }
    const sum = r.summary || {};
    applySummaryToUI({
      gpu_inference_ms: r.gpu_ms,
      h2d_time_ms: `DB #${r.id}`,
      peak_vram_mb: r.peak_vram_mb,
      total_objects: r.total_objects,
      unique_count: r.unique_count,
      counts_by_label: sum.counts_by_label || {},
      counts_by_classification: sum.counts_by_classification || {},
      counts_by_color: sum.counts_by_color || {},
      counts_by_gender: sum.counts_by_gender || {},
      detections: r.detections || [],
    });
    showToast("info", "SQLite Record Loaded", `Loaded record #${r.id} from database.`);
  } catch (err) {
    showToast("error", "DB Error", err.message);
  }
};

async function loadHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    gpuInfo.textContent = `${data.gpu} | VRAM: ${data.allocated_vram_mb}/${(data.total_vram_gb * 1024).toFixed(0)} MB | CV2:${data.cv2_available ? "ON" : "OFF"} | DF:${data.deepface_available ? "ON" : "OFF"}`;

    if (classChecks.children.length === 0 && data.available_classifications) {
      classChecks.innerHTML = data.available_classifications
        .map(
          (c) => `
        <label class="chk-item">
          <input type="checkbox" class="cls-chk" value="${c}" />
          <span>${c}</span>
        </label>
      `
        )
        .join("");
    }

    if (data.alert_config) {
      document.getElementById("alertOnDetChk").checked = !!data.alert_config.alert_on_detection;
      document.getElementById("alertNoDetVideoChk").checked = !!data.alert_config.alert_on_no_detection_video;
      document.getElementById("watchLabelsInput").value = (data.alert_config.watch_labels || []).join(", ");
      document.getElementById("webhookInput").value = data.alert_config.webhook_url || "";
    }
  } catch (_) {
    gpuInfo.textContent = "GPU স্ট্যাটাস পাওয়া যায়নি";
  }
}

function getSelectedClassesParam() {
  const checked = Array.from(document.querySelectorAll(".cls-chk:checked")).map((el) => el.value);
  return checked.length > 0 ? checked.join(",") : "";
}

document.getElementById("clearClassBtn").addEventListener("click", () => {
  document.querySelectorAll(".cls-chk").forEach((el) => {
    el.checked = false;
  });
});

document.getElementById("modeImageBtn").addEventListener("click", () => {
  document.getElementById("modeImageBtn").classList.add("active");
  document.getElementById("modeVideoBtn").classList.remove("active");
  document.getElementById("imageInputs").style.display = "block";
  document.getElementById("videoInputs").style.display = "none";
  document.getElementById("videoOptions").style.display = "none";
});

document.getElementById("modeVideoBtn").addEventListener("click", () => {
  document.getElementById("modeVideoBtn").classList.add("active");
  document.getElementById("modeImageBtn").classList.remove("active");
  document.getElementById("imageInputs").style.display = "none";
  document.getElementById("videoInputs").style.display = "block";
  document.getElementById("videoOptions").style.display = "block";
  stopCamera();
});

function stopCamera() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  document.getElementById("snapBtn").disabled = true;
  videoEl = null;
}

document.getElementById("camBtn").addEventListener("click", async () => {
  try {
    stopCamera();
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoEl = document.createElement("video");
    videoEl.srcObject = mediaStream;
    videoEl.autoplay = true;
    videoEl.playsInline = true;
    previewBox.innerHTML = "";
    previewBox.appendChild(videoEl);
    document.getElementById("snapBtn").disabled = false;
    showToast("info", "Webcam", "Camera started. Click Snap to capture.");
  } catch (err) {
    showToast("error", "Webcam Error", err.message);
  }
});

document.getElementById("snapBtn").addEventListener("click", () => {
  if (!videoEl) return;
  const canvas = document.createElement("canvas");
  canvas.width = videoEl.videoWidth || 640;
  canvas.height = videoEl.videoHeight || 480;
  canvas.getContext("2d").drawImage(videoEl, 0, 0, canvas.width, canvas.height);
  canvas.toBlob((blob) => {
    if (blob) {
      selectedImageFile = new File([blob], "snap.jpg", { type: "image/jpeg" });
      stopCamera();
      previewBox.innerHTML = `<img src="${URL.createObjectURL(selectedImageFile)}" alt="Snap" />`;
      showToast("success", "Captured", "Snapshot ready for detection.");
    }
  }, "image/jpeg", 0.9);
});

document.getElementById("fileInput").addEventListener("change", (e) => {
  if (e.target.files && e.target.files[0]) {
    selectedImageFile = e.target.files[0];
    stopCamera();
    previewBox.innerHTML = `<img src="${URL.createObjectURL(selectedImageFile)}" alt="Preview" />`;
    statusMsg.textContent = `ছবি নির্বাচিত: ${selectedImageFile.name}`;
  }
});

document.getElementById("videoInput").addEventListener("change", (e) => {
  if (e.target.files && e.target.files[0]) {
    selectedVideoFile = e.target.files[0];
    previewBox.innerHTML = `<video src="${URL.createObjectURL(selectedVideoFile)}" controls></video>`;
    statusMsg.textContent = `ভিডিও নির্বাচিত: ${selectedVideoFile.name}`;
  }
});

const playYtBtn = document.getElementById("playYtBtn");
if (playYtBtn) {
  playYtBtn.addEventListener("click", () => {
    const ytInput = document.getElementById("youtubeUrlInput");
    const rawUrl = ytInput ? ytInput.value.trim() : "";
    const vid = extractYouTubeId(rawUrl);
    if (!vid) {
      showToast("warning", "Invalid YouTube URL", "সঠিক YouTube লিংক দিন।");
      return;
    }
    selectedVideoFile = null;
    previewBox.innerHTML = `<iframe width="100%" height="380" src="https://www.youtube.com/embed/${vid}?autoplay=1" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>`;
    statusMsg.textContent = `YouTube ভিডিও প্লে হচ্ছে (ID: ${vid})। Live ডিটেকশনের জন্য Run Video চাপুন।`;
    showToast("info", "YouTube Player", "YouTube video loaded inside app.");
  });
}

function renderBadges(container, entries, withDot = false, colorLookup = {}, emptyText = "কোনো ডেটা নেই") {
  if (!entries || entries.length === 0) {
    container.innerHTML = `<span class="sub-txt">${emptyText}</span>`;
    return;
  }
  container.innerHTML = entries
    .map(([k, v]) => {
      const dot = withDot ? `<span class="color-dot" style="background:${colorLookup[k] || "#00e5ff"}"></span>` : "";
      return `<div class="obj-badge">${dot}${k}<span class="cnt">×${v}</span></div>`;
    })
    .join("");
}

function applySummaryToUI(data) {
  lastResultData = data;
  document.getElementById("gpuTime").textContent = `${data.gpu_inference_ms ?? data.avg_gpu_inference_ms ?? "-"} ms`;
  document.getElementById("h2dTime").textContent = data.h2d_time_ms !== undefined ? `${data.h2d_time_ms}` : "Video";
  document.getElementById("peakVram").textContent = `${data.peak_vram_mb ?? "-"} MB`;
  document.getElementById("objCount").textContent = data.total_objects ?? data.total_detections_across_frames ?? 0;
  document.getElementById("uniqueCount").textContent = data.unique_count ?? 0;

  renderBadges(document.getElementById("uniqueBadges"), Object.entries(data.counts_by_label || {}), false);
  renderBadges(document.getElementById("classBadges"), Object.entries(data.counts_by_classification || {}), false);

  const colorMap = {};
  (data.detections || []).forEach((d) => {
    if (d.color && d.color_hex && !colorMap[d.color]) colorMap[d.color] = d.color_hex;
  });
  renderBadges(document.getElementById("colorBadges"), Object.entries(data.counts_by_color || {}), true, colorMap);
  renderBadges(document.getElementById("genderBadges"), Object.entries(data.counts_by_gender || {}), false, {}, "কোনো Human ডেটা নেই");

  if (data.detections && data.detections.length > 0) {
    detList.innerHTML = data.detections
      .map((d, i) => {
        const hTxt =
          d.label.toLowerCase() === "person" && d.human_attributes
            ? `<br/>Human: <strong>${d.human_attributes.gender}</strong> • Age: <strong>${d.human_attributes.age_label}</strong>`
            : "";
        const fTxt = d.frame ? ` • Frame #${d.frame}` : "";
        return `
        <div class="row-item">
          <div>
            <strong>#${i + 1} ${d.label}</strong>
            <div class="sub-txt">Class: ${d.classification} • Color: ${d.color}${fTxt}${hTxt}</div>
          </div>
          <strong>${d.confidence}%</strong>
        </div>
      `;
      })
      .join("");
  } else {
    detList.innerHTML = `<div class="row-item"><span class="sub-txt">কোনো বিস্তারিত আইটেম নেই</span></div>`;
  }
}

function dispatchServerAlerts(alerts) {
  if (!Array.isArray(alerts)) return;
  alerts.forEach((a) => showToast(a.level || "warning", a.title, a.message));
}

document.getElementById("detectImageBtn").addEventListener("click", async () => {
  if (!selectedImageFile) {
    showToast("warning", "Input Required", "আগে একটি ছবি আপলোড বা ক্যাপচার করুন।");
    return;
  }
  const btn = document.getElementById("detectImageBtn");
  btn.disabled = true;
  statusMsg.textContent = "GTX 1650 জিপিইউতে ছবি বিশ্লেষণ চলছে...";

  const fd = new FormData();
  fd.append("file", selectedImageFile);

  const params = new URLSearchParams({
    threshold: threshInput.value,
    model_name: modelSelect.value,
    max_side: sideInput.value,
    enable_human_attr: humanAttrChk.checked,
    save_to_downloads: saveDownloadChk.checked,
    save_to_db: saveDbChk.checked,
  });
  const clsParam = getSelectedClassesParam();
  if (clsParam) params.set("allowed_classifications", clsParam);
  if (targetLabelsInput.value.trim()) params.set("target_labels", targetLabelsInput.value.trim());

  try {
    const res = await fetch(`/detect?${params.toString()}`, { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Detection failed");

    previewBox.innerHTML = `<img src="${data.annotated_image_base64}" alt="Result" />`;
    applySummaryToUI(data);
    dispatchServerAlerts(data.alerts_fired);
    statusMsg.textContent = `সম্পন্ন (DB #${data.db_record_id ?? "-"}) — ${data.total_objects} টি ডিটেকশন।`;
    await Promise.all([loadHealth(), loadNotifications(), loadDownloads(), loadDbResults()]);
  } catch (err) {
    statusMsg.textContent = "❌ " + err.message;
    showToast("error", "Detection Error", err.message);
  } finally {
    btn.disabled = false;
  }
});

document.getElementById("detectVideoBtn").addEventListener("click", async () => {
  const ytInput = document.getElementById("youtubeUrlInput");
  const ytUrl = ytInput ? ytInput.value.trim() : "";

  if (!selectedVideoFile && !ytUrl) {
    showToast("warning", "Video or YouTube URL Required", "আগে একটি ভিডিও ফাইল নির্বাচন করুন অথবা YouTube URL দিন।");
    return;
  }

  const btn = document.getElementById("detectVideoBtn");
  btn.disabled = true;

  const params = new URLSearchParams({
    threshold: threshInput.value,
    model_name: modelSelect.value,
    max_side: sideInput.value,
    frame_stride: strideInput.value,
    max_frames: maxFramesInput.value,
    enable_human_attr: humanAttrChk.checked,
    save_video_to_downloads: saveDownloadChk.checked,
    save_to_db: saveDbChk.checked,
  });
  const clsParam = getSelectedClassesParam();
  if (clsParam) params.set("allowed_classifications", clsParam);
  if (targetLabelsInput.value.trim()) params.set("target_labels", targetLabelsInput.value.trim());

  try {
    let res;
    if (selectedVideoFile) {
      statusMsg.textContent = "Running Video প্রসেসিং চলছে...";
      const fd = new FormData();
      fd.append("file", selectedVideoFile);
      res = await fetch(`/detect-video?${params.toString()}`, { method: "POST", body: fd });
    } else {
      statusMsg.textContent = "YouTube ভিডিও স্ট্রিম এনে Live Object Detection চলছে...";
      const fd = new FormData();
      fd.append("youtube_url", ytUrl);
      res = await fetch(`/detect-youtube?${params.toString()}`, { method: "POST", body: fd });
    }

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Video detection failed");

    if (data.sample_frame_base64) {
      previewBox.innerHTML = `<img src="${data.sample_frame_base64}" alt="Sample Frame" />`;
    }
    applySummaryToUI(data);
    dispatchServerAlerts(data.alerts_fired);
    statusMsg.textContent = `ভিডিও সম্পন্ন (DB #${data.db_record_id ?? "-"}) — ${data.processed_frames} ফ্রেম, No-Detect: ${data.no_detect_frames}।`;
    await Promise.all([loadHealth(), loadNotifications(), loadDownloads(), loadDbResults()]);
  } catch (err) {
    statusMsg.textContent = "❌ " + err.message;
    showToast("error", "Video Error", err.message);
  } finally {
    btn.disabled = false;
  }
});

document.getElementById("saveAlertCfgBtn").addEventListener("click", async () => {
  const payload = {
    alert_on_detection: document.getElementById("alertOnDetChk").checked,
    alert_on_no_detection_video: document.getElementById("alertNoDetVideoChk").checked,
    watch_labels_csv: document.getElementById("watchLabelsInput").value,
    webhook_url: document.getElementById("webhookInput").value,
  };
  const res = await fetch("/api/alerts/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (res.ok) {
    showToast("success", "Alert Rules Saved", "Alert & No-Detection Video settings updated.");
    await loadNotifications();
  }
});

async function triggerModelDownload(name) {
  showToast("info", "Model Download", `Preparing '${name}' in ./models/torch_hub...`);
  try {
    const res = await fetch("/api/models/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_name: name }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Download failed");
    showToast("success", "Model Ready", `Loaded: ${(data.loaded_models || []).join(", ")}`);
    await Promise.all([loadHealth(), loadNotifications()]);
  } catch (err) {
    showToast("error", "Model Error", err.message);
  }
}

document.getElementById("dlFasterBtn").addEventListener("click", () => triggerModelDownload("fasterrcnn"));
document.getElementById("dlSsdBtn").addEventListener("click", () => triggerModelDownload("ssdlite"));
document.getElementById("dlAllBtn").addEventListener("click", () => triggerModelDownload("all"));

document.getElementById("installPkgBtn").addEventListener("click", async () => {
  const pkg = document.getElementById("pkgSelect").value;
  showToast("info", "Installing Library", `Installing ${pkg}...`);
  try {
    const res = await fetch("/api/library/install", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ package: pkg }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Install failed");
    showToast("success", "Installed", `${pkg} is ready!`);
    await Promise.all([loadHealth(), loadNotifications()]);
  } catch (err) {
    showToast("error", "Install Error", err.message);
  }
});

document.getElementById("saveJsonBtn").addEventListener("click", async () => {
  if (!lastResultData) {
    showToast("warning", "No Report", "আগে একটি ডিটেকশন চালান।");
    return;
  }
  const cleanPayload = Object.assign({}, lastResultData);
  delete cleanPayload.annotated_image_base64;
  delete cleanPayload.sample_frame_base64;

  const res = await fetch("/api/save-report", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cleanPayload),
  });
  const data = await res.json();
  if (res.ok) {
    showToast("success", "JSON Saved", `Saved ${data.filename} in ./downloads`);
    await Promise.all([loadDownloads(), loadNotifications()]);
  }
});

document.getElementById("refreshDlBtn").addEventListener("click", loadDownloads);
document.getElementById("refreshDbBtn").addEventListener("click", loadDbResults);
document.getElementById("clearNotifBtn").addEventListener("click", async () => {
  await fetch("/api/notifications/clear", { method: "POST" });
  await loadNotifications();
});

loadHealth();
loadNotifications();
loadDownloads();
loadDbResults();