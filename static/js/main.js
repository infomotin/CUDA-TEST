let activeTab = 'vision';
let currentVisionMode = 'image';
let perfChart = null;
let lossChart = null;
let mathChart = null;
let threeScene, threeCamera, threeRenderer, threeControls, current3DMesh;

// Initialize
document.addEventListener("DOMContentLoaded", () => {
    initChart();
    initMathChart();
    initLossChart();
    initThreeJS();
    initGeometryCanvas();
    fetchSystemInfo();
    loadHistoryFromDB();

    // Auto-poll GPU temperature & utilization every 2.5 seconds
    setInterval(fetchSystemInfo, 2500);

    // Initial vision object detection run
    setTimeout(runObjectDetection, 400);
});

// Direct Math Formula Curve Chart
function initMathChart() {
    const ctx = document.getElementById("directMathChart");
    if (!ctx) return;
    mathChart = new Chart(ctx.getContext("2d"), {
        type: "line",
        data: {
            labels: [],
            datasets: [{
                label: "f(x) Function Curve",
                data: [],
                borderColor: "#34d399",
                backgroundColor: "rgba(16, 185, 129, 0.12)",
                fill: true,
                tension: 0.25,
                borderWidth: 2,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#9ca3af", font: { family: "'JetBrains Mono'" } }
                },
                x: {
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#9ca3af", font: { family: "'JetBrains Mono'" }, maxTicksLimit: 12 }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: (ctx) => ` f(x) = ${ctx.raw}` } }
            }
        }
    });
}

// Loss Chart for LLM Training
function initLossChart() {
    const ctx = document.getElementById("llmLossChart");
    if (!ctx) return;
    lossChart = new Chart(ctx.getContext("2d"), {
        type: "line",
        data: {
            labels: [1, 5, 10, 15, 20, 30, 40, 50],
            datasets: [{
                label: "Training Cross-Entropy Loss",
                data: [3.8, 3.2, 2.6, 2.1, 1.7, 1.3, 0.9, 0.6],
                borderColor: "#06b6d4",
                backgroundColor: "rgba(6, 182, 212, 0.15)",
                fill: true,
                tension: 0.35,
                borderWidth: 2,
                pointRadius: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#9ca3af", font: { family: "'JetBrains Mono'" } }
                },
                x: {
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#9ca3af", font: { family: "'JetBrains Mono'" } }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: (ctx) => ` Loss: ${ctx.raw}` } }
            }
        }
    });
}

// Chart.js initialization
function initChart() {
    const ctx = document.getElementById("performanceChart").getContext("2d");
    perfChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["CPU Time (ms)", "GPU Time (ms)"],
            datasets: [{
                label: "Execution Latency (lower is better)",
                data: [0, 0],
                backgroundColor: [
                    "rgba(245, 158, 11, 0.75)",
                    "rgba(16, 185, 129, 0.75)"
                ],
                borderColor: [
                    "#f59e0b",
                    "#10b981"
                ],
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#9ca3af", font: { family: "'JetBrains Mono'" } }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: "#f3f4f6", font: { family: "'Outfit'", weight: "bold" } }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: { label: (ctx) => ` ${ctx.raw} ms` }
                }
            }
        }
    });
}

// Three.js 3D WebGL Setup
function initThreeJS() {
    const container = document.getElementById("threejsViewport");
    if (!container) return;

    threeScene = new THREE.Scene();
    threeScene.background = new THREE.Color(0x0a0c12);

    threeCamera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
    threeCamera.position.set(25, 30, 45);

    threeRenderer = new THREE.WebGLRenderer({ antialias: true });
    threeRenderer.setSize(container.clientWidth, container.clientHeight);
    threeRenderer.setPixelRatio(window.devicePixelRatio);
    threeRenderer.shadowMap.enabled = true;
    container.appendChild(threeRenderer.domElement);

    // Orbit Controls
    if (typeof THREE.OrbitControls !== 'undefined') {
        threeControls = new THREE.OrbitControls(threeCamera, threeRenderer.domElement);
        threeControls.enableDamping = true;
        threeControls.dampingFactor = 0.05;
    }

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    threeScene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(30, 50, 30);
    dirLight.castShadow = true;
    threeScene.add(dirLight);

    // Grid helper
    const gridHelper = new THREE.GridHelper(60, 20, 0x06b6d4, 0x1f2937);
    threeScene.add(gridHelper);

    // Animation Loop
    function animate() {
        requestAnimationFrame(animate);
        if (threeControls) threeControls.update();
        if (current3DMesh) {
            current3DMesh.rotation.y += 0.002; // slow cinematic rotation
        }
        threeRenderer.render(threeScene, threeCamera);
    }
    animate();

    window.addEventListener("resize", () => {
        if (!container) return;
        threeCamera.aspect = container.clientWidth / container.clientHeight;
        threeCamera.updateProjectionMatrix();
        threeRenderer.setSize(container.clientWidth, container.clientHeight);
    });
}

// Render 3D Mesh in Three.js
function update3DMesh(data) {
    if (current3DMesh) {
        threeScene.remove(current3DMesh);
        if (current3DMesh.geometry) current3DMesh.geometry.dispose();
    }

    const geometry = new THREE.BufferGeometry();
    const vertices = new Float32Array(data.vertices);
    const colors = new Float32Array(data.colors);
    const indices = new Uint32Array(data.faces);

    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    geometry.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({
        vertexColors: true,
        roughness: 0.6,
        metalness: 0.1,
        flatShading: true,
        wireframe: false
    });

    current3DMesh = new THREE.Mesh(geometry, material);
    threeScene.add(current3DMesh);

    // Update stats
    document.getElementById("meshVertCount").innerText = data.vertex_count.toLocaleString();
    document.getElementById("meshFaceCount").innerText = data.face_count.toLocaleString();
    document.getElementById("meshDevice").innerText = data.device_used.toUpperCase();
}

// Equi-areal Geometry Canvas
function initGeometryCanvas() {
    runGeometryTransform();
}

async function runGeometryTransform() {
    const base = parseFloat(document.getElementById("geomBase").value);
    const height = parseFloat(document.getElementById("geomHeight").value);
    const shear = parseFloat(document.getElementById("geomShear").value);

    try {
        const res = await fetch("/api/geometry/area_transform", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ base, height, shear })
        });
        const data = await res.json();
        drawGeometry(data, base, height, shear);
    } catch (err) {
        console.error("Geometry calculation error:", err);
    }
}

function drawGeometry(data, base, height, shear) {
    const canvas = document.getElementById("geometryCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const targetArea = data.target_area;
    document.getElementById("legendTargetArea").innerText = targetArea.toFixed(1);
    document.getElementById("legendCurrentArea").innerText = targetArea.toFixed(1);

    // Draw grid background
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.lineWidth = 1;
    for (let x = 0; x < canvas.width; x += 40) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += 40) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
    }

    const startY = 220;

    // 1. Original Triangle (Left)
    const tX = 50;
    ctx.beginPath();
    ctx.moveTo(tX, startY);
    ctx.lineTo(tX + base, startY);
    ctx.lineTo(tX + base * 0.5, startY - height);
    ctx.closePath();
    ctx.fillStyle = "rgba(59, 130, 246, 0.25)";
    ctx.fill();
    ctx.strokeStyle = "#3b82f6";
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = "#93c5fd";
    ctx.font = "11px 'JetBrains Mono'";
    ctx.fillText("১. মূল ত্রিভুজ", tX, startY + 20);
    ctx.fillText(`A = ${targetArea} px²`, tX, startY + 35);

    // 2. Sheared Triangle with Det(M)=1 (Middle-Left)
    const sX = 220;
    ctx.beginPath();
    ctx.moveTo(sX, startY);
    ctx.lineTo(sX + base, startY);
    ctx.lineTo(sX + base * 0.5 + shear * height, startY - height);
    ctx.closePath();
    ctx.fillStyle = "rgba(16, 185, 129, 0.25)";
    ctx.fill();
    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = "#6ee7b7";
    ctx.fillText("২. শিয়ার ত্রিভুজ (Det=1)", sX, startY + 20);
    ctx.fillText(`A = ${targetArea} px² (স্থির)`, sX, startY + 35);

    // 3. Equi-areal Rectangle (Middle-Right)
    const rX = 390;
    const rectW = data.equiareal_rectangle.width;
    const rectH = data.equiareal_rectangle.height;
    ctx.beginPath();
    ctx.rect(rX, startY - rectH, rectW, rectH);
    ctx.fillStyle = "rgba(245, 158, 11, 0.25)";
    ctx.fill();
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = "#fcd34d";
    ctx.fillText("৩. সমান ক্ষেত্রফলের আয়ত", rX, startY + 20);
    ctx.fillText(`A = ${data.equiareal_rectangle.area} px²`, rX, startY + 35);

    // 4. Equi-areal Circle (Right)
    const cX = 580;
    const radius = data.equiareal_circle.radius;
    const centerY = startY - radius;
    ctx.beginPath();
    ctx.arc(cX, centerY, radius, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(139, 92, 246, 0.25)";
    ctx.fill();
    ctx.strokeStyle = "#8b5cf6";
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = "#c4b5fd";
    ctx.fillText("৪. সমান বৃত্ত", cX - 25, startY + 20);
    ctx.fillText(`A = ${data.equiareal_circle.area} px²`, cX - 25, startY + 35);
}

// Fetch System Hardware info with Temperature
async function fetchSystemInfo() {
    const badge = document.getElementById("cudaStatusBadge");

    try {
        const res = await fetch("/api/system_info");
        const data = await res.json();

        document.getElementById("specGpuName").innerText = data.gpu_name;
        document.getElementById("specTotalVram").innerText = data.total_vram_gb > 0 ? `${data.total_vram_gb} GB` : "N/A";
        document.getElementById("specTorch").innerText = `${data.torch_version} (${data.cuda_build ? 'CUDA ' + data.cuda_build : 'CPU'})`;

        // GPU Live Temperature & Util
        if (data.gpu_temp && data.gpu_temp !== "N/A") {
            document.getElementById("specGpuTemp").innerText = data.gpu_temp;
            document.getElementById("specGpuUtil").innerText = `${data.gpu_util}% (${data.gpu_clock_mhz || ''})`;
        } else {
            document.getElementById("specGpuTemp").innerText = "Normal";
            document.getElementById("specGpuUtil").innerText = "Active";
        }

        if (data.cuda_available) {
            badge.className = "status-badge cuda-on";
            badge.innerHTML = `🟢 CUDA ACTIVE: ${data.gpu_name}`;
        } else {
            badge.className = "status-badge cuda-off";
            badge.innerHTML = `🔴 CUDA UNAVAILABLE (CPU ONLY)`;
        }
    } catch (err) {
        // silent fail on fast polling
    }
}

function toggleCustomKernelInput(val) {
    const box = document.getElementById("customKernelBox");
    if (box) box.style.display = (val === "custom") ? "block" : "none";
}

function getCustomKernelFromInputs() {
    return [
        [parseFloat(document.getElementById("k00").value) || 0, parseFloat(document.getElementById("k01").value) || 0, parseFloat(document.getElementById("k02").value) || 0],
        [parseFloat(document.getElementById("k10").value) || 0, parseFloat(document.getElementById("k11").value) || 0, parseFloat(document.getElementById("k12").value) || 0],
        [parseFloat(document.getElementById("k20").value) || 0, parseFloat(document.getElementById("k21").value) || 0, parseFloat(document.getElementById("k22").value) || 0]
    ];
}

// Matrix Preset handler
function applyMatrixPreset(preset) {
    const presets = {
        sepia: [
            [0.39, 0.77, 0.19],
            [0.35, 0.69, 0.17],
            [0.27, 0.53, 0.13]
        ],
        gray: [
            [0.30, 0.59, 0.11],
            [0.30, 0.59, 0.11],
            [0.30, 0.59, 0.11]
        ],
        invert: [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0]
        ],
        boost: [
            [1.3, -0.15, -0.15],
            [-0.15, 1.3, -0.15],
            [-0.15, -0.15, 1.3]
        ]
    };

    const mat = presets[preset] || presets.sepia;
    document.getElementById("m00").value = mat[0][0];
    document.getElementById("m01").value = mat[0][1];
    document.getElementById("m02").value = mat[0][2];
    document.getElementById("m10").value = mat[1][0];
    document.getElementById("m11").value = mat[1][1];
    document.getElementById("m12").value = mat[1][2];
    document.getElementById("m20").value = mat[2][0];
    document.getElementById("m21").value = mat[2][1];
    document.getElementById("m22").value = mat[2][2];
}

function getMatrixFromInputs() {
    return [
        [parseFloat(document.getElementById("m00").value) || 0, parseFloat(document.getElementById("m01").value) || 0, parseFloat(document.getElementById("m02").value) || 0],
        [parseFloat(document.getElementById("m10").value) || 0, parseFloat(document.getElementById("m11").value) || 0, parseFloat(document.getElementById("m12").value) || 0],
        [parseFloat(document.getElementById("m20").value) || 0, parseFloat(document.getElementById("m21").value) || 0, parseFloat(document.getElementById("m22").value) || 0]
    ];
}

function renderPatchGrid(containerId, patch) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = "";
    if (!patch) return;
    for (let r = 0; r < patch.length; r++) {
        for (let c = 0; c < patch[r].length; c++) {
            const rgb = patch[r][c];
            const cell = document.createElement("div");
            cell.className = "patch-cell";
            cell.style.backgroundColor = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
            cell.title = `Row ${r}, Col ${c}: [R: ${rgb[0]}, G: ${rgb[1]}, B: ${rgb[2]}]`;
            cell.innerText = `${rgb[0]}`;
            container.appendChild(cell);
        }
    }
}

// =============================================================
// Vision, Object Detection & Running Video Analyzer
// =============================================================
let videoAnimId = null;
let prevFrameData = null;
let lastFrameTime = performance.now();
let frameCount = 0;
let currentFps = 0;
let offscreenCanvas = document.createElement("canvas");
let offscreenCtx = offscreenCanvas.getContext("2d", { willReadFrequently: true });
let isSyntheticVideoPlaying = false;
let syntheticAnimId = null;

function switchVisionMode(mode) {
    currentVisionMode = mode;
    const btnImg = document.getElementById("btnModeImg");
    const btnVid = document.getElementById("btnModeVid");
    const imgInBox = document.getElementById("visionImageInputBox");
    const vidInBox = document.getElementById("visionVideoInputBox");
    const imgView = document.getElementById("visionImageView");
    const vidView = document.getElementById("visionVideoView");

    if (mode === 'image') {
        if (btnImg) btnImg.classList.add("active");
        if (btnVid) btnVid.classList.remove("active");
        if (imgInBox) imgInBox.style.display = "block";
        if (vidInBox) vidInBox.style.display = "none";
        if (imgView) imgView.style.display = "block";
        if (vidView) vidView.style.display = "none";
        stopVideoAnalysis();
        runObjectDetection();
    } else {
        if (btnVid) btnVid.classList.add("active");
        if (btnImg) btnImg.classList.remove("active");
        if (vidInBox) vidInBox.style.display = "block";
        if (imgInBox) imgInBox.style.display = "none";
        if (vidView) vidView.style.display = "block";
        if (imgView) imgView.style.display = "none";
        startVideoAnalysis();
    }
}

async function runObjectDetection() {
    const fileInput = document.getElementById("visionImageFile");
    const formData = new FormData();
    if (fileInput && fileInput.files && fileInput.files[0]) {
        formData.append("image", fileInput.files[0]);
    }

    const badge = document.getElementById("visionBadge");
    if (badge) badge.innerText = "⏳ ডিটেকশন ও অবজেক্ট গণনা চলছে...";

    try {
        const res = await fetch("/api/vision/detect_objects", {
            method: "POST",
            body: formData
        });
        const result = await res.json();
        if (result.error) {
            alert(result.error);
            return;
        }

        const origEl = document.getElementById("visionOrigImg");
        const annoEl = document.getElementById("visionAnnotatedImg");
        if (origEl) origEl.src = "data:image/jpeg;base64," + result.original_image;
        if (annoEl) annoEl.src = "data:image/jpeg;base64," + result.annotated_image;
        if (badge) badge.innerText = `সনাক্তকৃত মোট অবজেক্ট: ${result.object_count} টি (${result.device_used})`;

        const listContainer = document.getElementById("visionObjectsList");
        if (listContainer) {
            if (result.objects.length === 0) {
                listContainer.innerHTML = '<span style="color: var(--text-muted);">কোনো আলাদা অবজেক্ট সনাক্ত করা যায়নি।</span>';
            } else {
                listContainer.innerHTML = result.objects.map(obj => 
                    `<div style="display: inline-block; background: rgba(255,255,255,0.06); padding: 4px 8px; border-radius: 4px; margin: 2px 4px; border-left: 3px solid #10b981;">
                        <strong>Object #${obj.id}:</strong> বক্স [${obj.bbox.join(", ")}] | ক্ষেত্রফল: ${obj.area.toLocaleString()} px² | কেন্দ্র: (${obj.center[0]}, ${obj.center[1]})
                    </div>`
                ).join("");
            }
        }

        renderResults({
            cpu_time_ms: null,
            gpu_time_ms: result.processing_time_ms,
            speedup: null
        }, "Vision Object Detection", `${result.object_count} Objects (${result.device_used})`);
        loadHistoryFromDB();
    } catch (err) {
        console.error("Object detection error:", err);
    }
}

function loadUploadedVideo(input) {
    if (!input.files || !input.files[0]) return;
    const file = input.files[0];
    const video = document.getElementById("visionVideoElement");
    if (!video) return;

    stopSyntheticVideo();
    const url = URL.createObjectURL(file);
    video.muted = true;
    video.defaultMuted = true;
    video.playsInline = true;
    video.loop = true;
    video.src = url;

    video.onloadeddata = () => {
        video.play().catch(e => {
            console.log("Autoplay gesture prompt:", e);
        });
        const resEl = document.getElementById("vidLiveRes");
        if (resEl) resEl.innerText = `${video.videoWidth || 640} x ${video.videoHeight || 360}`;
        startVideoAnalysis();
    };

    video.load();
}

// Synthetic moving object generator for immediate zero-upload live video testing
let syntheticParticles = [];
function initSyntheticParticles() {
    syntheticParticles = [];
    const colors = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"];
    for (let i = 0; i < 5; i++) {
        syntheticParticles.push({
            x: 60 + Math.random() * 400,
            y: 40 + Math.random() * 200,
            vx: (Math.random() > 0.5 ? 1 : -1) * (2.5 + Math.random() * 3),
            vy: (Math.random() > 0.5 ? 1 : -1) * (2.0 + Math.random() * 3),
            w: 50 + Math.random() * 25,
            h: 35 + Math.random() * 20,
            color: colors[i % colors.length]
        });
    }
}

function startSyntheticVideo(video, canvas) {
    if (isSyntheticVideoPlaying) return;
    isSyntheticVideoPlaying = true;
    initSyntheticParticles();

    const synthCanvas = document.createElement("canvas");
    synthCanvas.width = 640;
    synthCanvas.height = 360;
    const sCtx = synthCanvas.getContext("2d");

    try {
        const stream = synthCanvas.captureStream(30);
        video.muted = true;
        video.srcObject = stream;
        video.play().catch(() => {});
    } catch (e) {
        // Fallback if captureStream is restricted
    }

    const resEl = document.getElementById("vidLiveRes");
    if (resEl) resEl.innerText = "640 x 360 (Live Moving Objects Feed)";

    function updateSynthetic() {
        if (!isSyntheticVideoPlaying) return;

        // Render background on video/canvas
        sCtx.fillStyle = "#090d18";
        sCtx.fillRect(0, 0, synthCanvas.width, synthCanvas.height);

        // Background tech grid
        sCtx.strokeStyle = "rgba(255,255,255,0.06)";
        for (let x = 0; x < synthCanvas.width; x += 40) {
            sCtx.beginPath(); sCtx.moveTo(x, 0); sCtx.lineTo(x, synthCanvas.height); sCtx.stroke();
        }
        for (let y = 0; y < synthCanvas.height; y += 40) {
            sCtx.beginPath(); sCtx.moveTo(0, y); sCtx.lineTo(synthCanvas.width, y); sCtx.stroke();
        }

        // Move and draw each object
        syntheticParticles.forEach((p, idx) => {
            p.x += p.vx;
            p.y += p.vy;
            if (p.x <= 15 || p.x + p.w >= synthCanvas.width - 15) p.vx *= -1;
            if (p.y <= 15 || p.y + p.h >= synthCanvas.height - 15) p.vy *= -1;

            sCtx.fillStyle = p.color;
            sCtx.beginPath();
            if (sCtx.roundRect) {
                sCtx.roundRect(p.x, p.y, p.w, p.h, 6);
            } else {
                sCtx.rect(p.x, p.y, p.w, p.h);
            }
            sCtx.fill();
            sCtx.strokeStyle = "#ffffff";
            sCtx.lineWidth = 1.5;
            sCtx.stroke();

            // Label on particle
            sCtx.fillStyle = "#ffffff";
            sCtx.font = "bold 10px monospace";
            sCtx.fillText(`Obj #${idx+1}`, p.x + 4, p.y + 16);
        });

        syntheticAnimId = requestAnimationFrame(updateSynthetic);
    }
    updateSynthetic();
}

function stopSyntheticVideo() {
    isSyntheticVideoPlaying = false;
    if (syntheticAnimId) {
        cancelAnimationFrame(syntheticAnimId);
        syntheticAnimId = null;
    }
    const video = document.getElementById("visionVideoElement");
    if (video && video.srcObject) {
        video.srcObject = null;
    }
}

function startVideoAnalysis() {
    const video = document.getElementById("visionVideoElement");
    const canvas = document.getElementById("visionVideoCanvas");
    if (!video || !canvas) return;

    video.muted = true;
    if (!video.src && !video.srcObject) {
        startSyntheticVideo(video, canvas);
    }

    if (videoAnimId) cancelAnimationFrame(videoAnimId);

    const downW = 160;
    const downH = 90;
    offscreenCanvas.width = downW;
    offscreenCanvas.height = downH;
    prevFrameData = null;
    frameCount = 0;
    lastFrameTime = performance.now();

    function processFrame() {
        if (currentVisionMode !== 'video' || activeTab !== 'vision') return;

        // Ensure canvas matches video element display size
        const targetW = video.clientWidth || 640;
        const targetH = video.clientHeight || 360;
        if (canvas.width !== targetW || canvas.height !== targetH) {
            canvas.width = targetW;
            canvas.height = targetH;
        }

        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Compute live FPS
        frameCount++;
        const now = performance.now();
        if (now - lastFrameTime >= 400) {
            currentFps = Math.round((frameCount * 1000) / (now - lastFrameTime));
            frameCount = 0;
            lastFrameTime = now;
            const fpsEl = document.getElementById("vidLiveFps");
            if (fpsEl) fpsEl.innerText = `${currentFps} FPS`;
        }

        // Draw downscaled frame for high-speed difference calculation
        try {
            offscreenCtx.drawImage(video, 0, 0, downW, downH);
            const currentImgData = offscreenCtx.getImageData(0, 0, downW, downH);
            const currPixels = currentImgData.data;

            if (prevFrameData) {
                const prevPixels = prevFrameData.data;
                const motionGrid = new Uint8Array(downW * downH);

                // High-sensitivity difference threshold
                for (let i = 0; i < currPixels.length; i += 4) {
                    const diff = Math.abs(currPixels[i] - prevPixels[i]) +
                                 Math.abs(currPixels[i+1] - prevPixels[i+1]) +
                                 Math.abs(currPixels[i+2] - prevPixels[i+2]);
                    if (diff > 18) {
                        motionGrid[i / 4] = 1;
                    }
                }

                // Grid clustering
                const blockSize = 6;
                const gridCols = Math.floor(downW / blockSize);
                const gridRows = Math.floor(downH / blockSize);
                const activeBlocks = [];

                for (let gy = 0; gy < gridRows; gy++) {
                    for (let gx = 0; gx < gridCols; gx++) {
                        let activePix = 0;
                        for (let by = 0; by < blockSize; by++) {
                            for (let bx = 0; bx < blockSize; bx++) {
                                const px = gx * blockSize + bx;
                                const py = gy * blockSize + by;
                                if (motionGrid[py * downW + px]) activePix++;
                            }
                        }
                        if (activePix >= 2) {
                            activeBlocks.push({ gx, gy });
                        }
                    }
                }

                // Group adjacent active blocks into distinct bounding boxes
                const visited = new Set();
                const boxes = [];

                activeBlocks.forEach(b => {
                    const key = `${b.gx},${b.gy}`;
                    if (visited.has(key)) return;

                    let minX = b.gx, maxX = b.gx, minY = b.gy, maxY = b.gy;
                    const queue = [b];
                    visited.add(key);

                    while (queue.length > 0) {
                        const curr = queue.shift();
                        minX = Math.min(minX, curr.gx);
                        maxX = Math.max(maxX, curr.gx);
                        minY = Math.min(minY, curr.gy);
                        maxY = Math.max(maxY, curr.gy);

                        for (let dy = -1; dy <= 1; dy++) {
                            for (let dx = -1; dx <= 1; dx++) {
                                const nx = curr.gx + dx;
                                const ny = curr.gy + dy;
                                const nkey = `${nx},${ny}`;
                                if (nx >= 0 && nx < gridCols && ny >= 0 && ny < gridRows && !visited.has(nkey)) {
                                    if (activeBlocks.some(ab => ab.gx === nx && ab.gy === ny)) {
                                        visited.add(nkey);
                                        queue.push({ gx: nx, gy: ny });
                                    }
                                }
                            }
                        }
                    }

                    const scaleX = canvas.width / downW;
                    const scaleY = canvas.height / downH;
                    const bx1 = minX * blockSize * scaleX;
                    const by1 = minY * blockSize * scaleY;
                    const bw = (maxX - minX + 1) * blockSize * scaleX;
                    const bh = (maxY - minY + 1) * blockSize * scaleY;

                    if (bw >= 15 && bh >= 12) {
                        boxes.push({ x: bx1, y: by1, w: bw, h: bh });
                    }
                });

                // Render neon tracking boxes on video canvas overlay
                const palette = ["#06b6d4", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6", "#3b82f6"];
                boxes.forEach((box, idx) => {
                    const col = palette[idx % palette.length];
                    ctx.strokeStyle = col;
                    ctx.lineWidth = 2.5;
                    ctx.strokeRect(box.x, box.y, box.w, box.h);

                    // Corner brackets
                    const cLen = 10;
                    ctx.lineWidth = 3.5;
                    ctx.beginPath();
                    ctx.moveTo(box.x, box.y + cLen); ctx.lineTo(box.x, box.y); ctx.lineTo(box.x + cLen, box.y);
                    ctx.moveTo(box.x + box.w - cLen, box.y); ctx.lineTo(box.x + box.w, box.y); ctx.lineTo(box.x + box.w, box.y + cLen);
                    ctx.moveTo(box.x, box.y + box.h - cLen); ctx.lineTo(box.x, box.y + box.h); ctx.lineTo(box.x + cLen, box.y + box.h);
                    ctx.moveTo(box.x + box.w - cLen, box.y + box.h); ctx.lineTo(box.x + box.w, box.y + box.h); ctx.lineTo(box.x + box.w, box.y + box.h - cLen);
                    ctx.stroke();

                    // Tracking Label
                    const tag = `Track #${idx + 1}`;
                    ctx.fillStyle = col;
                    ctx.fillRect(box.x, Math.max(0, box.y - 18), 75, 18);
                    ctx.fillStyle = "#000000";
                    ctx.font = "bold 11px monospace";
                    ctx.fillText(tag, box.x + 4, Math.max(12, box.y - 4));
                });

                const countEl = document.getElementById("vidLiveObjCount");
                if (countEl) countEl.innerText = boxes.length;
            }

            prevFrameData = currentImgData;
        } catch (e) {
            // silent catch during frame render
        }

        videoAnimId = requestAnimationFrame(processFrame);
    }

    videoAnimId = requestAnimationFrame(processFrame);
}

function stopVideoAnalysis() {
    if (videoAnimId) {
        cancelAnimationFrame(videoAnimId);
        videoAnimId = null;
    }
    stopSyntheticVideo();
    const canvas = document.getElementById("visionVideoCanvas");
    if (canvas) {
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}

// =============================================================
// AI Book Reader & Hugging Face Lightweight Model Studio
// =============================================================
let currentBookPage = 0;
let bookTotalPages = 5;
let speechSynth = window.speechSynthesis;
let currentUtterance = null;
let isReadingAloud = false;

async function loadBookPage(pageIdx) {
    if (pageIdx < 0) pageIdx = 0;
    try {
        const res = await fetch(`/api/book/get_page?page=${pageIdx}`);
        const data = await res.json();
        if (data.status === "success") {
            currentBookPage = data.current_page - 1;
            bookTotalPages = data.total_pages;

            const curPageEl = document.getElementById("bookCurPage");
            if (curPageEl) curPageEl.innerText = data.current_page;
            const totalPageEl = document.getElementById("bookTotalPages");
            if (totalPageEl) totalPageEl.innerText = data.total_pages;
            const totalWordsEl = document.getElementById("bookTotalWords");
            if (totalWordsEl) totalWordsEl.innerText = data.total_words ? data.total_words.toLocaleString() : "--";
            const headerEl = document.getElementById("bookPageHeader");
            if (headerEl) headerEl.innerText = `📄 পৃষ্ঠা ${data.current_page} (${data.chapter})`;
            const contentEl = document.getElementById("bookPageContent");
            if (contentEl) contentEl.innerText = data.content;
            const titleEl = document.getElementById("bookReaderTitle");
            if (titleEl) titleEl.innerText = `📖 এআই বই রিডার: ${data.title}`;

            // Update chapter select if needed
            if (data.chapters && data.chapters.length > 0) {
                const chSelect = document.getElementById("bookChapterSelect");
                if (chSelect && chSelect.options.length !== data.chapters.length) {
                    chSelect.innerHTML = data.chapters.map(ch => 
                        `<option value="${ch.page}">${ch.title}</option>`
                    ).join("");
                }
            }

            if (isReadingAloud) {
                readPageAloud();
            }
        }
    } catch (err) {
        console.error("Failed to load book page:", err);
    }
}

function prevBookPage() {
    if (currentBookPage > 0) {
        loadBookPage(currentBookPage - 1);
    }
}

function nextBookPage() {
    if (currentBookPage < bookTotalPages - 1) {
        loadBookPage(currentBookPage + 1);
    }
}

function jumpToChapter(pageIdx) {
    loadBookPage(parseInt(pageIdx));
}

async function uploadBookFile(input) {
    if (!input.files || !input.files[0]) return;
    const file = input.files[0];
    const statusEl = document.getElementById("bookUploadStatus");
    if (statusEl) statusEl.innerText = `⏳ সম্পূর্ণ বই বিশ্লেষণ করা হচ্ছে: ${file.name}...`;

    const formData = new FormData();
    formData.append("book_file", file);

    try {
        const res = await fetch("/api/book/upload", {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        if (data.error) {
            if (statusEl) statusEl.innerText = `❌ এরর: ${data.error}`;
            return;
        }

        if (statusEl) {
            statusEl.innerHTML = `✅ সফল! <strong>${data.title}</strong> (${data.total_pages} পৃষ্ঠা, ${data.total_words} শব্দ) লোড করা হয়েছে।`;
        }

        const chSelect = document.getElementById("bookChapterSelect");
        if (chSelect && data.chapters) {
            chSelect.innerHTML = data.chapters.map(ch => 
                `<option value="${ch.page}">${ch.title}</option>`
            ).join("");
        }

        loadBookPage(0);
    } catch (err) {
        if (statusEl) statusEl.innerText = "Upload failed: " + err;
    }
}

async function downloadAndLoadHfModel() {
    const select = document.getElementById("bookHfModelSelect");
    const modelChoice = select ? select.value : "qwen";
    const statusEl = document.getElementById("bookHfStatus");
    if (statusEl) statusEl.innerText = `⏳ Hugging Face থেকে মডেল নামানো ও CUDA তে লোড হচ্ছে: ${modelChoice.toUpperCase()}...`;

    try {
        const res = await fetch("/api/book/download_model", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model: modelChoice })
        });
        const data = await res.json();
        if (data.error) {
            if (statusEl) statusEl.innerText = `❌ এরর: ${data.error}`;
            return;
        }

        if (statusEl) {
            statusEl.innerHTML = `✅ সফল! <strong>${data.model_name}</strong> অ্যাপের ভেতর (F:\\CUDA TEST\\models\\huggingface) সংরক্ষিত ও CUDA তে লোড হয়েছে (${data.params} Params, VRAM: ${data.vram_mb} MB, ${data.load_time_sec}s)।`;
        }
        const badgeEl = document.getElementById("bookActiveModelBadge");
        if (badgeEl) badgeEl.innerText = `Model: ${data.model_name} (Active)`;
        const displayEl = document.getElementById("bookModelNameDisplay");
        if (displayEl) displayEl.innerText = data.model_name;
    } catch (err) {
        if (statusEl) statusEl.innerText = "Model download/load failed: " + err;
    }
}

// Voice Audio Reading Aloud using Web Speech API
function readPageAloud() {
    if (!speechSynth) {
        alert("Speech synthesis is not supported in this browser.");
        return;
    }
    stopReadingAloud();

    const contentEl = document.getElementById("bookPageContent");
    const textToRead = contentEl ? contentEl.innerText : "";
    if (!textToRead.trim()) return;

    currentUtterance = new SpeechSynthesisUtterance(textToRead);
    currentUtterance.rate = 1.0;

    // Detect if Bengali text
    const isBengali = /[\u0980-\u09FF]/.test(textToRead);
    const voices = speechSynth.getVoices();
    if (isBengali) {
        const bnVoice = voices.find(v => v.lang.includes("bn") || v.name.includes("Bangla") || v.name.includes("Bengali"));
        if (bnVoice) currentUtterance.voice = bnVoice;
    }

    currentUtterance.onend = () => { isReadingAloud = false; };
    currentUtterance.onerror = () => { isReadingAloud = false; };

    isReadingAloud = true;
    speechSynth.speak(currentUtterance);
}

function stopReadingAloud() {
    isReadingAloud = false;
    if (speechSynth && speechSynth.speaking) {
        speechSynth.cancel();
    }
}

async function summarizeCurrentPage() {
    summarizeBookScope("page");
}

async function summarizeEntireBook() {
    summarizeBookScope("book");
}

async function summarizeBookScope(scope) {
    const chatHistory = document.getElementById("bookChatHistory");
    const statusEl = document.getElementById("aiChatStatus");
    if (statusEl) statusEl.innerText = "সারসংক্ষেপ তৈরি হচ্ছে...";

    try {
        const res = await fetch("/api/book/summarize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ scope })
        });
        const data = await res.json();
        if (data.error) {
            alert(data.error);
            return;
        }

        const bulletsHtml = data.summary_bullets.map(b => `<li style="margin-bottom: 4px;">${b}</li>`).join("");
        const msgDiv = document.createElement("div");
        msgDiv.style.cssText = "background: rgba(16, 185, 129, 0.12); border-left: 3px solid #10b981; padding: 10px; border-radius: 4px; color: #f3f4f6;";
        msgDiv.innerHTML = `<strong>📑 ${data.header}:</strong><ul style="margin: 6px 0 0 16px; padding: 0;">${bulletsHtml}</ul>`;
        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;

        if (statusEl) statusEl.innerText = "রেডি";
    } catch (err) {
        alert("Summarization failed: " + err);
    }
}

async function askBookAi() {
    const inputEl = document.getElementById("bookQuestionInput");
    const question = inputEl.value.trim();
    if (!question) return;

    inputEl.value = "";
    quickBookPrompt(question);
}

async function quickBookPrompt(question) {
    const chatHistory = document.getElementById("bookChatHistory");
    const statusEl = document.getElementById("aiChatStatus");

    // Add user question message
    const userMsg = document.createElement("div");
    userMsg.style.cssText = "background: rgba(59, 130, 246, 0.15); border-right: 3px solid var(--accent-blue); padding: 8px 10px; border-radius: 4px; color: #93c5fd; align-self: flex-end; max-width: 85%;";
    userMsg.innerHTML = `<strong>প্রশ্ন:</strong> ${question}`;
    chatHistory.appendChild(userMsg);
    chatHistory.scrollTop = chatHistory.scrollHeight;

    if (statusEl) statusEl.innerText = "হাগিংফেইস মডেল ভাবছে...";

    try {
        const res = await fetch("/api/book/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question })
        });
        const data = await res.json();
        if (data.error) {
            alert(data.error);
            if (statusEl) statusEl.innerText = "এরর";
            return;
        }

        // Add AI answer message with cited page badge
        const aiMsg = document.createElement("div");
        aiMsg.style.cssText = "background: rgba(255, 255, 255, 0.05); border-left: 3px solid #34d399; padding: 10px; border-radius: 4px; color: #f9fafb;";
        aiMsg.innerHTML = `
            <div style="display: flex; justify-content: space-between; font-size: 10px; color: var(--text-muted); margin-bottom: 4px;">
                <span>🤖 ${data.model_used} (${data.duration_ms} ms)</span>
                <span class="speedup-pill" style="font-size: 9px; padding: 1px 6px;">📌 সূত্র: পৃষ্ঠা ${data.cited_page}</span>
            </div>
            <div style="line-height: 1.6;">${data.answer}</div>
        `;
        chatHistory.appendChild(aiMsg);
        chatHistory.scrollTop = chatHistory.scrollHeight;

        if (statusEl) statusEl.innerText = "রেডি";
    } catch (err) {
        if (statusEl) statusEl.innerText = "ব্যর্থ";
        alert("Book Q&A error: " + err);
    }
}

// Tab Switching
function selectTab(tabId) {
    activeTab = tabId;
    document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

    if (window.event && window.event.currentTarget) {
        window.event.currentTarget.classList.add("active");
    }
    const tabEl = document.getElementById(`tab-${tabId}`);
    if (tabEl) tabEl.classList.add("active");

    // Toggle corresponding viewports
    const setDisplay = (id, show) => {
        const el = document.getElementById(id);
        if (el) el.style.display = show ? 'block' : 'none';
    };

    setDisplay("visionContainer", tabId === 'vision');
    setDisplay("directMathContainer", tabId === 'directmath');
    setDisplay("bookReaderContainer", tabId === 'bookreader');
    setDisplay("llmContainer", tabId === 'llm');
    setDisplay("terrain3dContainer", tabId === 'terrain3d');
    setDisplay("jpgMatrixContainer", tabId === 'jpgmatrix');
    setDisplay("geometryContainer", tabId === 'geometry');
    setDisplay("imageComparisonContainer", tabId === 'image');
    setDisplay("audioSpectrogramContainer", tabId === 'audio');
    setDisplay("fractalContainer", tabId === 'mandelbrot');

    if (tabId === 'vision') {
        if (currentVisionMode === 'image') {
            runObjectDetection();
        } else {
            startVideoAnalysis();
        }
    } else {
        stopVideoAnalysis();
        if (tabId === 'bookreader') {
            loadBookPage(currentBookPage);
        } else if (tabId === 'directmath') {
            runDirectMathCompute();
        } else if (tabId === 'terrain3d') {
            runTerrain3D();
        } else if (tabId === 'geometry') {
            runGeometryTransform();
        }
    }
}

// Sliders helpers
function updateMatrixLabel(val) {
    document.getElementById("matrixSizeLabel").innerText = val;
    document.getElementById("matrixSizeLabel2").innerText = val;
}

function previewInputImage(input) {
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById("imgOriginal").src = e.target.result;
        }
        reader.readAsDataURL(input.files[0]);
    }
}

// 3D Terrain Benchmark Call
async function runTerrain3D() {
    const grid_size = parseInt(document.getElementById("terrainGrid").value);
    const scale = parseFloat(document.getElementById("terrainScale").value);
    const octaves = parseInt(document.getElementById("terrainOctaves").value);

    try {
        const res = await fetch("/api/generate/3d_terrain", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ grid_size, scale, octaves })
        });
        const data = await res.json();
        update3DMesh(data);

        document.getElementById("speedupTitle").innerText = `3D ট্যারেন অবজেক্ট সফলভাবে সিন্থেসিস হয়েছে! 🏔️`;
        document.getElementById("speedupDesc").innerText = `ভার্টেক্স: ${data.vertex_count.toLocaleString()} | ট্রায়াঙ্গেল ফেস: ${data.face_count.toLocaleString()} (${data.computation_time_ms} ms)`;
        document.getElementById("gpuTimeVal").innerText = `${data.computation_time_ms} ms`;
        document.getElementById("speedupVal").innerText = "Real-time";
    } catch (err) {
        console.error("3D generation error:", err);
    }
}

// =============================================================
// LLM Studio: Pretrained Model Loading, LoRA Fine-Tuning & Ingestion
// =============================================================
let activePdfFilename = null;

async function loadSelectedLlmModel() {
    const select = document.getElementById("llmModelSelect");
    const modelName = select ? select.value : "gpt2";
    const badge = document.getElementById("modelLoadStatusBadge");
    if (badge) badge.innerText = `⏳ মডেল ডাউনলোড ও লোড হচ্ছে: ${modelName} (CUDA তে স্থানান্তর চলছে)...`;

    try {
        const res = await fetch("/api/llm/load_model", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model_name: modelName })
        });
        const data = await res.json();
        if (data.error) {
            if (badge) badge.innerText = `❌ এরর: ${data.error}`;
            return;
        }

        if (badge) badge.innerHTML = `✅ সফল! <strong>${data.model_name}</strong> অ্যাপের ডিরেক্টরিতে (F:\\CUDA TEST\\models\\) সেভ ও লোড হয়েছে (${data.params} প্যারামিটার, VRAM: ${data.vram_mb} MB)।`;
        const activeBadge = document.getElementById("activeModelBadge");
        if (activeBadge) activeBadge.innerText = `Model: ${data.model_name}`;
        const nameEl = document.getElementById("llmActiveModelName");
        if (nameEl) nameEl.innerText = data.model_name;
        const paramEl = document.getElementById("llmParamCount");
        if (paramEl) paramEl.innerText = data.params;
        const vramEl = document.getElementById("llmVram");
        if (vramEl) vramEl.innerText = `${data.vram_mb} MB`;
    } catch (err) {
        if (badge) badge.innerText = "Model loading failed: " + err;
    }
}

// Upload and extract text from PDF, TXT, MD, JSON, CSV documents
async function uploadAndExtractDoc(input) {
    if (!input.files || !input.files[0]) return;
    const file = input.files[0];
    const badge = document.getElementById("pdfStatusBadge");
    if (badge) badge.innerText = `⏳ ফাইল প্রসেসিং চলছে: ${file.name}...`;

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch("/api/llm/upload_file", {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        if (data.error) {
            if (badge) badge.innerText = `❌ এরর: ${data.error}`;
            return;
        }

        activePdfFilename = data.filename;
        const trainBox = document.getElementById("llmTrainText");
        if (trainBox) trainBox.value = data.extracted_text;
        if (badge) {
            badge.innerHTML = `✅ সফল! <strong>${data.filename}</strong> (${data.page_count} পৃষ্ঠা, ${data.word_count} শব্দ) এক্সট্র্যাক্ট করা হয়েছে। নিচে ফাইন-টিউন বাটনে চাপ দিন।`;
        }
    } catch (err) {
        if (badge) badge.innerText = "File extraction failed: " + err;
    }
}

// Backward-compatible alias
async function uploadAndExtractPdf(input) {
    return uploadAndExtractDoc(input);
}

async function runLlmTraining() {
    const text = document.getElementById("llmTrainText").value;
    const epochs = parseInt(document.getElementById("llmEpochs").value);
    const loraRankEl = document.getElementById("llmLoraRank");
    const lora_r = loraRankEl ? parseInt(loraRankEl.value) : 8;
    const lrEl = document.getElementById("llmLr");
    const lr = lrEl ? parseFloat(lrEl.value) : 0.0003;
    const select = document.getElementById("llmModelSelect");
    const modelChoice = select ? select.value : "gpt2";
    const docSource = activePdfFilename ? `File: ${activePdfFilename}` : "Custom Text";

    const badge = document.getElementById("modelLoadStatusBadge");
    if (badge) badge.innerText = `⚡ ${modelChoice.toUpperCase()} মডেলে LoRA ফাইন-টিউনিং চলছে (${epochs} Steps)...`;

    try {
        const res = await fetch("/api/llm/finetune", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text,
                epochs,
                lora_r,
                lr,
                model_choice: modelChoice,
                doc_source: docSource
            })
        });
        const result = await res.json();
        if (result.error) {
            alert(result.error);
            if (badge) badge.innerText = `❌ এরর: ${result.error}`;
            return;
        }

        // Update stats
        if (document.getElementById("llmActiveModelName")) {
            document.getElementById("llmActiveModelName").innerText = result.model_name;
        }
        if (document.getElementById("activeModelBadge")) {
            document.getElementById("activeModelBadge").innerText = `Model: ${result.model_name}`;
        }
        document.getElementById("llmParamCount").innerText = result.total_parameters;
        document.getElementById("llmThroughput").innerText = `${result.tokens_per_sec} Tok/s`;
        document.getElementById("llmVram").innerText = `${result.vram_used_mb} MB`;
        if (document.getElementById("llmDocSource")) {
            document.getElementById("llmDocSource").innerText = result.doc_source || docSource;
        }
        document.getElementById("llmOutputText").innerText = result.sample_generated;
        if (badge) {
            badge.innerHTML = `✅ ফাইন-টিউনিং সম্পন্ন! <strong>${result.model_name}</strong> (${result.tokens_per_sec} Tok/s, সময়: ${result.train_time_sec}s) — <a href="javascript:void(0)" onclick="convertFineTunedToGguf()" style="color: #38bdf8; font-weight: 700; text-decoration: underline;">👉 LLaMA.cpp GGUF-এ রূপান্তর করুন</a>`;
        }
        const llamaStatus = document.getElementById("llamaCppStatusText");
        if (llamaStatus) llamaStatus.innerHTML = `মডেল রেডি! নিচে <strong>'GGUF এ কনভার্ট করুন'</strong> বাটনে চাপ দিন।`;

        // Update Loss Curve
        if (lossChart && result.loss_history) {
            lossChart.data.labels = result.loss_history.map(h => `Step ${h.step}`);
            lossChart.data.datasets[0].data = result.loss_history.map(h => h.loss);
            lossChart.update();
        }

        document.getElementById("speedupTitle").innerText = `LLM LoRA ফাইন-টিউনিং সফলভাবে সম্পন্ন! 🧠`;
        document.getElementById("speedupDesc").innerText = `মডেল: ${result.model_name} | স্পিড: ${result.tokens_per_sec} Tok/s (${result.train_time_sec}s)`;
        document.getElementById("gpuTimeVal").innerText = `${(result.train_time_sec * 1000).toFixed(0)} ms`;
        document.getElementById("speedupVal").innerText = "Real-time";
        loadHistoryFromDB();
    } catch (err) {
        alert("Fine-tuning failed: " + err);
    }
}

async function generateFromLlm() {
    const prompt = document.getElementById("llmPrompt").value;
    const temp = parseFloat(document.getElementById("llmTemp").value);
    const outBox = document.getElementById("llmOutputText");
    outBox.innerText = "Generating text using active LoRA/Transformer weights on CUDA...";

    try {
        const res = await fetch("/api/llm/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt, temperature: temp, max_tokens: 100 })
        });
        const data = await res.json();
        if (data.error) {
            outBox.innerText = "Error: " + data.error;
            return;
        }
        outBox.innerText = data.generated_text;
    } catch (err) {
        outBox.innerText = "Generation failed: " + err;
    }
}

// =============================================================
// LLaMA.cpp & GGUF Runtime Client Handlers
// =============================================================
function setLlamaPrompt(text) {
    const input = document.getElementById("llamaPromptInput");
    if (input) input.value = text;
}

async function convertFineTunedToGguf() {
    const statusText = document.getElementById("llamaCppStatusText");
    const term = document.getElementById("llamaTerminalOutput");
    if (statusText) statusText.innerText = "⏳ সক্রিয় মডেলকে GGUF এ কনভার্ট করা হচ্ছে...";
    if (term) term.innerText = "GGUF কনভার্সন প্রক্রিয়া শুরু হচ্ছে...\n- হেডারের মেটাডেটা প্রস্তুত হচ্ছে\n- টেনসর ও ওয়েট ম্যাপিং চলছে (FP16/FP32)...\n- GGUF ফাইলে টোকেনাইজারের শব্দকোষ যুক্ত হচ্ছে...";

    try {
        const res = await fetch("/api/llm/export_gguf", {
            method: "POST",
            headers: { "Content-Type": "application/json" }
        });
        const data = await res.json();
        if (data.error) {
            if (statusText) statusText.innerText = "❌ কনভার্সন ব্যর্থ";
            if (term) term.innerText = "Error: " + data.error;
            return;
        }

        if (statusText) statusText.innerText = `✅ GGUF সম্পন্ন (${data.file_size_mb} MB)`;
        const infoCard = document.getElementById("llamaGgufInfoCard");
        if (infoCard) infoCard.style.display = "block";
        const archBadge = document.getElementById("llamaArchBadge");
        if (archBadge) archBadge.innerText = `Arch: ${data.architecture.toUpperCase()}`;
        const sizeEl = document.getElementById("llamaGgufSize");
        if (sizeEl) sizeEl.innerText = `${data.file_size_mb} MB`;
        const ctxEl = document.getElementById("llamaGgufCtx");
        if (ctxEl) ctxEl.innerText = data.context_length;
        const tensEl = document.getElementById("llamaGgufTensors");
        if (tensEl) tensEl.innerText = data.tensor_count;
        const quantEl = document.getElementById("llamaGgufQuant");
        if (quantEl) quantEl.innerText = data.quantization;

        const downloadBtn = document.getElementById("btnDownloadGguf");
        if (downloadBtn) {
            downloadBtn.style.display = "inline-block";
            downloadBtn.href = `/api/llm/download_gguf?file=${data.gguf_filename}`;
        }

        if (term) {
            term.innerText = `✨ ${data.message}\n\n[GGUF Header Info]\n- Model Name: ${data.model_name}\n- Architecture: ${data.architecture}\n- File: ${data.gguf_filename} (${data.file_size_mb} MB)\n- Tensors: ${data.tensor_count}\n- Context Length: ${data.context_length}\n\n👉 নিচে প্রম্পট লিখে 'LLaMA.cpp রান' বাটনে চাপ দিন এবং আউটপুট দেখুন!`;
        }
    } catch (err) {
        if (statusText) statusText.innerText = "ব্যর্থ";
        if (term) term.innerText = "GGUF conversion failed: " + err;
    }
}

async function runLlamaCppPrompt() {
    const input = document.getElementById("llamaPromptInput");
    const prompt = input ? input.value.trim() : "কৃত্রিম বুদ্ধিমত্তা";
    const statusText = document.getElementById("llamaCppStatusText");
    const term = document.getElementById("llamaTerminalOutput");

    if (statusText) statusText.innerText = "⚡ LLaMA.cpp ইনফ্যারেন্স চলছে (CUDA)...";
    if (term) term.innerText = `[LLaMA.cpp Engine Starting]\nPrompt: "${prompt}"\nEvaluating context tokens on NVIDIA GeForce GTX 1650 CUDA cores...\n\nGenerating response tokens: `;

    try {
        const res = await fetch("/api/llm/run_gguf", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prompt: prompt,
                max_tokens: 80,
                temperature: 0.7,
                top_p: 0.9
            })
        });
        const data = await res.json();
        if (data.error) {
            if (statusText) statusText.innerText = "❌ এরর";
            if (term) term.innerText = "LLaMA.cpp Error: " + data.error;
            return;
        }

        if (statusText) statusText.innerText = `✅ সফল (${data.tokens_per_sec} Tok/s)`;
        
        // Show terminal output
        if (term) {
            term.innerText = `[LLaMA.cpp Engine Output | ${data.architecture.toUpperCase()} | ${data.quantization}]\n\n` +
                             `Prompt: "${data.prompt}"\n\n` +
                             `Generated Text:\n${data.generated_text}\n\n` +
                             `--------------------------------------------------\n` +
                             `📊 Performance: ${data.tokens_per_sec} Tokens/sec | Latency: ${data.ms_per_token} ms/token\n` +
                             `Context Tokens: ${data.context_tokens} | New Generated: ${data.tokens_generated} tokens | Time: ${data.total_time_ms} ms`;
        }

        // Update metric badges
        const speedEl = document.getElementById("llamaSpeedVal");
        if (speedEl) speedEl.innerText = `${data.tokens_per_sec} Tok/s`;
        const latEl = document.getElementById("llamaLatencyVal");
        if (latEl) latEl.innerText = `${data.ms_per_token} ms/tok`;
        const tokEl = document.getElementById("llamaTokCount");
        if (tokEl) tokEl.innerText = `${data.tokens_generated} tokens`;
        const timeEl = document.getElementById("llamaTimeVal");
        if (timeEl) timeEl.innerText = `${data.total_time_ms} ms`;
        const devEl = document.getElementById("llamaDeviceVal");
        if (devEl) devEl.innerText = `${data.device} (${data.vram_mb} MB)`;

        // Also update main speedup title
        const sTitle = document.getElementById("speedupTitle");
        if (sTitle) sTitle.innerText = `LLaMA.cpp GGUF ইনফ্যারেন্স সম্পন্ন! 🦙`;
        const sDesc = document.getElementById("speedupDesc");
        if (sDesc) sDesc.innerText = `স্পিড: ${data.tokens_per_sec} Tok/s | মোট সময়: ${data.total_time_ms} ms (${data.engine})`;

        loadHistoryFromDB();
    } catch (err) {
        if (statusText) statusText.innerText = "ব্যর্থ";
        if (term) term.innerText = "Execution failed: " + err;
    }
}

function setMathPreset(expr) {
    document.getElementById("directMathExpr").value = expr;
    runDirectMathCompute();
}

async function runDirectMathCompute() {
    const expr = document.getElementById("directMathExpr").value.trim();
    const xMin = parseFloat(document.getElementById("mathXMin").value) || -10;
    const xMax = parseFloat(document.getElementById("mathXMax").value) || 10;
    const points = parseInt(document.getElementById("mathPoints").value) || 2000000;

    document.getElementById("mathActiveExpr").innerText = `f(x) = ${expr}`;

    try {
        const res = await fetch("/api/math/direct_compute", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ expression: expr, x_min: xMin, x_max: xMax, points: points })
        });
        const result = await res.json();
        if (result.error) {
            alert(result.error);
            return;
        }

        // Update stats
        document.getElementById("mathIntegral").innerText = result.stats.integral.toFixed(4);
        document.getElementById("mathMax").innerText = result.stats.max.toFixed(4);
        document.getElementById("mathMin").innerText = result.stats.min.toFixed(4);
        document.getElementById("mathMean").innerText = result.stats.mean.toFixed(4);

        // Update 2D Curve Plot
        if (mathChart && result.plot) {
            mathChart.data.labels = result.plot.x;
            mathChart.data.datasets[0].data = result.plot.y;
            mathChart.data.datasets[0].label = `f(x) = ${expr.length > 25 ? expr.substring(0, 25) + '...' : expr}`;
            mathChart.update();
        }

        renderResults(result, `Math: f(x)=${expr.substring(0, 18)}`, `${points.toLocaleString()} Points`);
        loadHistoryFromDB();
    } catch (err) {
        alert("Math computation failed: " + err);
    }
}

// Run current selected benchmark
async function runCurrentBenchmark() {
    const btn = document.getElementById("btnRun");
    const spinner = document.getElementById("btnSpinner");
    btn.disabled = true;
    spinner.style.display = "inline-block";

    let testName = "";
    let paramStr = "";

    try {
        if (activeTab === "vision") {
            if (currentVisionMode === "image") {
                await runObjectDetection();
            } else {
                startVideoAnalysis();
            }
        } else if (activeTab === "directmath") {
            await runDirectMathCompute();
        } else if (activeTab === "bookreader") {
            await summarizeCurrentPage();
        } else if (activeTab === "llm") {
            await runLlmTraining();
        } else if (activeTab === "terrain3d") {
            await runTerrain3D();
        } else if (activeTab === "jpgmatrix") {
            const formData = new FormData();
            const fileInput = document.getElementById("jpgMatrixInput");
            if (fileInput.files[0]) formData.append("image", fileInput.files[0]);
            
            const matrixObj = getMatrixFromInputs();
            formData.append("matrix", JSON.stringify(matrixObj));

            testName = "JPG Matrix Dot Product";
            paramStr = "3x3 Color Transformation Matrix";

            const res = await fetch("/api/matrix/image_transform", { method: "POST", body: formData });
            const result = await res.json();

            if (result.transformed_image) {
                document.getElementById("jpgMatOriginalImg").src = "data:image/jpeg;base64," + result.original_image;
                document.getElementById("jpgMatTransformedImg").src = "data:image/jpeg;base64," + result.transformed_image;
                
                // Render numerical 8x8 matrix inspection grids
                renderPatchGrid("origMatrixPatchGrid", result.original_patch_matrix);
                renderPatchGrid("transformedMatrixPatchGrid", result.transformed_patch_matrix);

                document.getElementById("jpgMatrixContainer").style.display = "block";
            }
            renderResults(result, testName, result.resolution || paramStr);
            loadHistoryFromDB();

        } else if (activeTab === "geometry") {
            await runGeometryTransform();
            document.getElementById("speedupTitle").innerText = "জ্যামিতিক ক্ষেত্রফল ধ্রুবক যাচাই সম্পন্ন! 📐";
            document.getElementById("speedupDesc").innerText = "ত্রিভুজ, শিয়ার ত্রিভুজ, আয়ত ও বৃত্তের ক্ষেত্রফল হুবহু সমান সংরক্ষিত হয়েছে।";
        } else if (activeTab === "image") {
            const formData = new FormData();
            const fileInput = document.getElementById("imageInput");
            if (fileInput.files[0]) formData.append("image", fileInput.files[0]);
            const filterType = document.getElementById("imageFilter").value;
            formData.append("filter_type", filterType);

            if (filterType === "custom") {
                const customK = getCustomKernelFromInputs();
                formData.append("custom_kernel", JSON.stringify(customK));
            }

            testName = `Image Filter (${filterType.toUpperCase()})`;
            paramStr = `GPGPU Filter: ${filterType}`;

            const res = await fetch("/api/process/image", { method: "POST", body: formData });
            const result = await res.json();
            
            if (result.processed_image) {
                document.getElementById("imgOriginal").src = "data:image/jpeg;base64," + result.original_image;
                document.getElementById("imgProcessed").src = "data:image/jpeg;base64," + result.processed_image;
                document.getElementById("imageComparisonContainer").style.display = "block";
            }
            renderResults(result, testName, result.resolution || paramStr);

        } else if (activeTab === "audio") {
            const formData = new FormData();
            const audioInput = document.getElementById("audioInput");
            if (audioInput.files[0]) formData.append("audio", audioInput.files[0]);
            testName = "Audio cuFFT & Spectrogram";
            paramStr = "cuFFT / STFT Frequency Heatmap";

            const res = await fetch("/api/process/audio", { method: "POST", body: formData });
            const result = await res.json();

            if (result.spectrogram_image) {
                document.getElementById("audioSpectrogramImg").src = "data:image/jpeg;base64," + result.spectrogram_image;
                document.getElementById("audioSpectrogramContainer").style.display = "block";
            }
            renderResults(result, testName, result.audio_samples || paramStr);

        } else if (activeTab === "matrix") {
            const size = parseInt(document.getElementById("matrixSize").value);
            const precision = document.getElementById("matrixPrecision").value;
            testName = "Matrix Multiplication";
            paramStr = `${size}x${size} (${precision})`;

            const res = await fetch("/api/benchmark/matrix", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ size, precision })
            });
            const result = await res.json();
            renderResults(result, testName, paramStr);

        } else if (activeTab === "mandelbrot") {
            const resVal = document.getElementById("mandelRes").value.split("x");
            const width = parseInt(resVal[0]);
            const height = parseInt(resVal[1]);
            const max_iter = parseInt(document.getElementById("mandelIter").value);
            testName = "Mandelbrot Fractal Math";
            paramStr = `${width}x${height} (${max_iter} iters)`;

            const res = await fetch("/api/benchmark/mandelbrot", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ width, height, max_iter })
            });
            const result = await res.json();
            if (result.image_base64) {
                document.getElementById("mandelbrotImage").src = "data:image/jpeg;base64," + result.image_base64;
                document.getElementById("fractalContainer").style.display = "block";
            }
            renderResults(result, testName, paramStr);
        }
    } catch (err) {
        alert("Execution failed: " + err);
    } finally {
        btn.disabled = false;
        spinner.style.display = "none";
    }
}

// Display results
function renderResults(result, testName, paramStr) {
    const cpuTime = result.cpu_time_ms;
    const gpuTime = result.gpu_time_ms;
    const speedup = result.speedup;

    document.getElementById("cpuTimeVal").innerText = cpuTime ? `${cpuTime.toFixed(1)} ms` : "N/A";
    document.getElementById("gpuTimeVal").innerText = gpuTime ? `${gpuTime.toFixed(1)} ms` : "N/A";

    if (speedup) {
        document.getElementById("speedupVal").innerText = `${speedup}x`;
        document.getElementById("speedupTitle").innerText = `জিপিইউ ${speedup} গুণ দ্রুত ফলাফল সম্পন্ন করেছে! 🚀`;
        document.getElementById("speedupDesc").innerText = `CPU সময়: ${cpuTime}ms | GPU সময়: ${gpuTime}ms (${testName})`;
    } else {
        document.getElementById("speedupVal").innerText = "--";
        document.getElementById("speedupTitle").innerText = "টেস্ট সম্পন্ন হয়েছে";
        document.getElementById("speedupDesc").innerText = `প্রসেসিং সময়: ${gpuTime || cpuTime} ms`;
    }

    if (perfChart) {
        perfChart.data.datasets[0].data = [cpuTime || 0, gpuTime || 0];
        perfChart.update();
    }

    addHistoryRecord(testName, paramStr, cpuTime, gpuTime, speedup);
}

// =============================================================
// SQLite3 History Persistence with Pagination [5 items per page]
// =============================================================
let historyCurrentPage = 1;
let historyPageLimit = 5;
let historyTotalPages = 1;
let historyTotalRecords = 0;

// Add history record and refresh current page view
function addHistoryRecord(name, param, cpu, gpu, speedup) {
    // Reload page 1 to show the latest record within the 5 items limit
    loadHistoryFromDB(1);
}

// Load saved history from SQLite database with pagination
async function loadHistoryFromDB(page = historyCurrentPage) {
    historyCurrentPage = page;
    try {
        const res = await fetch(`/api/history?page=${historyCurrentPage}&limit=${historyPageLimit}`);
        const data = await res.json();
        const tbody = document.getElementById("historyBody");
        if (!tbody) return;

        // Support both paginated object and legacy array format
        const rows = data.records !== undefined ? data.records : (Array.isArray(data) ? data : []);
        const total = data.total_records !== undefined ? data.total_records : rows.length;
        const totalPages = data.total_pages !== undefined ? data.total_pages : Math.max(1, Math.ceil(total / historyPageLimit));

        historyTotalPages = totalPages;
        historyTotalRecords = total;

        if (!rows || rows.length === 0) {
            tbody.innerHTML = '<tr class="empty-row"><td colspan="6">এখনো কোনো টেস্ট হিস্ট্রি ডাটাবেসে নেই।</td></tr>';
            updateHistoryPaginationUI(0, 0, 0, 1, 1);
            return;
        }

        tbody.innerHTML = "";
        rows.forEach(r => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>${r.formatted_time || ''}</td>
                <td><strong>${r.test_name}</strong></td>
                <td>${r.parameter || '-'}</td>
                <td style="color: #f59e0b;">${r.cpu_time_ms ? r.cpu_time_ms.toFixed(1) + ' ms' : '-'}</td>
                <td style="color: #10b981;">${r.gpu_time_ms ? r.gpu_time_ms.toFixed(1) + ' ms' : '-'}</td>
                <td>${r.speedup ? `<span class="speedup-pill">${r.speedup}x Faster</span>` : '<span style="color: #9ca3af;">Saved</span>'}</td>
            `;
            tbody.appendChild(tr);
        });

        const startIdx = (historyCurrentPage - 1) * historyPageLimit + 1;
        const endIdx = Math.min(startIdx + rows.length - 1, total);
        updateHistoryPaginationUI(startIdx, endIdx, total, historyCurrentPage, totalPages);
    } catch (err) {
        console.error("Failed to load history from SQLite DB:", err);
    }
}

function updateHistoryPaginationUI(start, end, total, page, totalPages) {
    const startEl = document.getElementById("histStartIdx");
    const endEl = document.getElementById("histEndIdx");
    const totalEl = document.getElementById("histTotalRecords");
    const pageEl = document.getElementById("histCurrentPage");
    const totalPageEl = document.getElementById("histTotalPages");
    const btnPrev = document.getElementById("btnHistPrev");
    const btnNext = document.getElementById("btnHistNext");

    if (startEl) startEl.innerText = total > 0 ? start : 0;
    if (endEl) endEl.innerText = total > 0 ? end : 0;
    if (totalEl) totalEl.innerText = total;
    if (pageEl) pageEl.innerText = page;
    if (totalPageEl) totalPageEl.innerText = totalPages;

    if (btnPrev) {
        btnPrev.disabled = (page <= 1);
        btnPrev.style.opacity = (page <= 1) ? "0.4" : "1";
        btnPrev.style.cursor = (page <= 1) ? "not-allowed" : "pointer";
    }
    if (btnNext) {
        btnNext.disabled = (page >= totalPages);
        btnNext.style.opacity = (page >= totalPages) ? "0.4" : "1";
        btnNext.style.cursor = (page >= totalPages) ? "not-allowed" : "pointer";
    }
}

function navHistoryPage(direction) {
    const targetPage = historyCurrentPage + direction;
    if (targetPage >= 1 && targetPage <= historyTotalPages) {
        loadHistoryFromDB(targetPage);
    }
}

function changeHistoryLimit(newLimit) {
    historyPageLimit = parseInt(newLimit) || 5;
    historyCurrentPage = 1;
    loadHistoryFromDB(1);
}

// Clear history from both UI and SQLite database
async function clearHistory() {
    if (!confirm("আপনি কি SQLite ডাটাবেস থেকে পূর্ববর্তী সকল ফলাফল মুছে ফেলতে চান?")) return;
    try {
        await fetch("/api/history/clear", { method: "POST" });
        const tbody = document.getElementById("historyBody");
        if (tbody) {
            tbody.innerHTML = '<tr class="empty-row"><td colspan="6">হিস্ট্রি ক্লিয়ার করা হয়েছে (SQLite DB Reset)।</td></tr>';
        }
        historyCurrentPage = 1;
        updateHistoryPaginationUI(0, 0, 0, 1, 1);
    } catch (err) {
        alert("Clear history failed: " + err);
    }
}
