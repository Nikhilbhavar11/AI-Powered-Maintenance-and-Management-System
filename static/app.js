/**
 * PredictiveMaintAI — Dashboard Application
 *
 * Consumes ONLY backend REST APIs and WebSocket.
 * No direct Firebase or LLM access from the frontend.
 *
 * Modules:
 *   1. State management
 *   2. API client
 *   3. WebSocket manager
 *   4. Chart.js setup
 *   5. UI updaters
 *   6. Chat interface
 *   7. Initialization
 */

"use strict";

// ═══════════════════════════════════════════════════════════
// 1. STATE
// ═══════════════════════════════════════════════════════════

const state = {
    selectedDevice: null,
    devices: [],
    ws: null,
    chartSeconds: 60,
    sensorChart: null,
    healthChart: null,
    chartRefreshTimer: null,
};

const API_BASE = "";  // Same origin

// ═══════════════════════════════════════════════════════════
// 2. API CLIENT
// ═══════════════════════════════════════════════════════════

async function apiGet(path) {
    try {
        const resp = await fetch(`${API_BASE}${path}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return await resp.json();
    } catch (err) {
        console.error(`API GET ${path} failed:`, err);
        return null;
    }
}

async function apiPost(path, body) {
    try {
        const resp = await fetch(`${API_BASE}${path}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return await resp.json();
    } catch (err) {
        console.error(`API POST ${path} failed:`, err);
        return null;
    }
}

async function fetchDevices() {
    const data = await apiGet("/api/devices");
    if (data && data.devices) {
        state.devices = data.devices;
        populateDeviceSelector(data.devices);
    }
}

async function fetchStatus(deviceId) {
    const data = await apiGet(`/api/status?device_id=${encodeURIComponent(deviceId)}`);
    if (data) updateStatusCards(data);
    return data;
}

async function fetchChartData(deviceId, seconds) {
    const data = await apiGet(
        `/api/chart-data?device_id=${encodeURIComponent(deviceId)}&seconds=${seconds}`
    );
    if (data) updateCharts(data);
    return data;
}

// ═══════════════════════════════════════════════════════════
// 3. WEBSOCKET MANAGER
// ═══════════════════════════════════════════════════════════

function connectWebSocket(deviceId) {
    // Close existing connection
    if (state.ws) {
        state.ws.close();
        state.ws = null;
    }

    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${location.host}/ws/live?device_id=${encodeURIComponent(deviceId)}`;

    console.log(`WebSocket connecting: ${url}`);
    const ws = new WebSocket(url);

    ws.onopen = () => {
        console.log("WebSocket connected");
        setConnectionStatus(true);
    };

    ws.onmessage = (event) => {
        try {
            const payload = JSON.parse(event.data);
            handleWsMessage(payload);
        } catch (e) {
            console.error("WS message parse error:", e);
        }
    };

    ws.onclose = () => {
        console.log("WebSocket disconnected");
        setConnectionStatus(false);
        // Reconnect after 3 seconds
        setTimeout(() => {
            if (state.selectedDevice === deviceId) {
                connectWebSocket(deviceId);
            }
        }, 3000);
    };

    ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        setConnectionStatus(false);
    };

    state.ws = ws;
}

function handleWsMessage(payload) {
    // Update status cards from WebSocket data
    if (payload.prediction) {
        updateStatusFromPrediction(payload.prediction);
    }
    if (payload.live_data) {
        updateGauges(payload.live_data, payload.prediction);
    }
    // Update footer timestamp
    document.getElementById("footer-timestamp").textContent =
        `Last update: ${new Date().toLocaleTimeString()}`;
}

function setConnectionStatus(connected) {
    const badge = document.getElementById("connection-status");
    const text = badge.querySelector(".conn-text");
    if (connected) {
        badge.className = "conn-badge connected";
        text.textContent = "Live";
    } else {
        badge.className = "conn-badge disconnected";
        text.textContent = "Offline";
    }
}

// ═══════════════════════════════════════════════════════════
// 4. CHART.JS SETUP
// ═══════════════════════════════════════════════════════════

const CHART_COLORS = {
    current: { line: "#6366f1", fill: "rgba(99,102,241,0.1)" },
    temperature: { line: "#f43f5e", fill: "rgba(244,63,94,0.1)" },
    vibration: { line: "#22d3ee", fill: "rgba(34,211,238,0.1)" },
    health: { line: "#34d399", fill: "rgba(52,211,153,0.15)" },
};

const CHART_DEFAULTS = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
        legend: {
            labels: {
                color: "#94a3b8",
                font: { family: "'Inter', sans-serif", size: 11 },
                usePointStyle: true,
                pointStyle: "circle",
                padding: 16,
            },
        },
        tooltip: {
            backgroundColor: "rgba(17,24,39,0.95)",
            borderColor: "rgba(255,255,255,0.1)",
            borderWidth: 1,
            titleFont: { family: "'Inter', sans-serif" },
            bodyFont: { family: "'Inter', sans-serif" },
            padding: 10,
            cornerRadius: 8,
        },
    },
    scales: {
        x: {
            grid: { color: "rgba(255,255,255,0.04)" },
            ticks: {
                color: "#64748b",
                font: { family: "'Inter', sans-serif", size: 10 },
                maxTicksLimit: 10,
            },
        },
        y: {
            grid: { color: "rgba(255,255,255,0.04)" },
            ticks: {
                color: "#64748b",
                font: { family: "'Inter', sans-serif", size: 10 },
            },
        },
    },
    elements: {
        line: { tension: 0.35, borderWidth: 2 },
        point: { radius: 0, hoverRadius: 4 },
    },
};

function initCharts() {
    // Sensor chart
    const sensorCtx = document.getElementById("sensor-chart").getContext("2d");
    state.sensorChart = new Chart(sensorCtx, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Current (A)",
                    data: [],
                    borderColor: CHART_COLORS.current.line,
                    backgroundColor: CHART_COLORS.current.fill,
                    fill: true,
                },
                {
                    label: "Temperature (°C)",
                    data: [],
                    borderColor: CHART_COLORS.temperature.line,
                    backgroundColor: CHART_COLORS.temperature.fill,
                    fill: true,
                },
                {
                    label: "Vibration (g)",
                    data: [],
                    borderColor: CHART_COLORS.vibration.line,
                    backgroundColor: CHART_COLORS.vibration.fill,
                    fill: true,
                },
            ],
        },
        options: JSON.parse(JSON.stringify(CHART_DEFAULTS)),
    });

    // Health chart
    const healthCtx = document.getElementById("health-chart").getContext("2d");
    state.healthChart = new Chart(healthCtx, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Health Score",
                    data: [],
                    borderColor: CHART_COLORS.health.line,
                    backgroundColor: CHART_COLORS.health.fill,
                    fill: true,
                },
            ],
        },
        options: {
            ...JSON.parse(JSON.stringify(CHART_DEFAULTS)),
            scales: {
                ...JSON.parse(JSON.stringify(CHART_DEFAULTS.scales)),
                y: {
                    ...JSON.parse(JSON.stringify(CHART_DEFAULTS.scales.y)),
                    min: 0,
                    max: 100,
                },
            },
        },
    });
}

function updateCharts(data) {
    if (!data) return;

    // Sensor chart
    const sd = data.sensor_data;
    if (sd && state.sensorChart) {
        const labels = sd.timestamps.map(formatTimestamp);
        state.sensorChart.data.labels = labels;
        state.sensorChart.data.datasets[0].data = sd.current;
        state.sensorChart.data.datasets[1].data = sd.temperature;
        state.sensorChart.data.datasets[2].data = sd.vibration;
        state.sensorChart.update("none");
    }

    // Health chart
    const pd = data.prediction_data;
    if (pd && state.healthChart) {
        const labels = pd.timestamps.map(formatTimestamp);
        state.healthChart.data.labels = labels;
        state.healthChart.data.datasets[0].data = pd.health_score;
        state.healthChart.update("none");
    }
}

function formatTimestamp(ts) {
    if (!ts) return "";
    try {
        const d = new Date(ts);
        if (isNaN(d.getTime())) {
            // Might be a Firebase key — show last 8 chars
            return ts.slice(-8);
        }
        return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    } catch {
        return ts.slice(-8);
    }
}

// ═══════════════════════════════════════════════════════════
// 5. UI UPDATERS
// ═══════════════════════════════════════════════════════════

function populateDeviceSelector(devices) {
    const select = document.getElementById("device-select");
    select.innerHTML = "";

    if (devices.length === 0) {
        select.innerHTML = '<option value="">No devices</option>';
        return;
    }

    devices.forEach((d) => {
        const opt = document.createElement("option");
        opt.value = d.device_id;
        opt.textContent = d.device_id;
        select.appendChild(opt);
    });

    // Auto-select first device
    if (!state.selectedDevice && devices.length > 0) {
        selectDevice(devices[0].device_id);
    }
}

function selectDevice(deviceId) {
    state.selectedDevice = deviceId;
    document.getElementById("device-select").value = deviceId;

    // Fetch initial data
    fetchStatus(deviceId);
    fetchChartData(deviceId, state.chartSeconds);

    // Connect WebSocket
    connectWebSocket(deviceId);

    // Start chart refresh interval
    if (state.chartRefreshTimer) clearInterval(state.chartRefreshTimer);
    state.chartRefreshTimer = setInterval(() => {
        fetchChartData(deviceId, state.chartSeconds);
    }, 5000);
}

function updateStatusCards(data) {
    if (data.prediction) {
        updateStatusFromPrediction(data.prediction);
    }
    if (data.live_data) {
        updateGauges(data.live_data, data.prediction);
    }
}

function updateStatusFromPrediction(pred) {
    // Health score
    const healthVal = document.getElementById("val-health");
    healthVal.textContent = pred.health_score != null
        ? `${pred.health_score}`
        : "—";
    document.getElementById("sub-health").textContent =
        pred.model_type === "ml" ? "ML Model" : "Rule-based";

    // Risk level
    const riskVal = document.getElementById("val-risk");
    riskVal.textContent = pred.risk_level || "—";
    const riskCard = document.getElementById("card-risk");
    riskCard.className = `card risk-card risk-${(pred.risk_level || "").toLowerCase()}`;
    document.getElementById("sub-risk").textContent =
        pred.timestamp ? `Updated: ${formatTimestamp(pred.timestamp)}` : "";

    // Maintenance
    const maintVal = document.getElementById("val-maint");
    if (pred.maintenance_required) {
        maintVal.textContent = "REQUIRED";
        maintVal.style.color = "#f43f5e";
    } else {
        maintVal.textContent = "Not Needed";
        maintVal.style.color = "#34d399";
    }
    document.getElementById("sub-maint").textContent = "";

    // Stress index
    const stressVal = document.getElementById("val-stress");
    const fs = pred.features_summary;
    if (fs && fs.stress_index != null) {
        stressVal.textContent = `${fs.stress_index}`;
        document.getElementById("sub-stress").textContent = "/ 100";
    }

    // Failure reason
    document.getElementById("failure-reason").textContent =
        pred.failure_reason || "No issues detected";
}

function updateGauges(liveData, prediction) {
    if (!liveData) return;

    // Current
    const curEl = document.getElementById("gauge-current");
    curEl.textContent = liveData.current != null
        ? parseFloat(liveData.current).toFixed(2)
        : "—";

    // Temperature
    const tempEl = document.getElementById("gauge-temp");
    tempEl.textContent = liveData.temperature != null
        ? parseFloat(liveData.temperature).toFixed(1)
        : "—";

    // Vibration
    const vibEl = document.getElementById("gauge-vib");
    vibEl.textContent = liveData.vibration != null
        ? parseFloat(liveData.vibration).toFixed(2)
        : "—";

    // Trends (from prediction features_summary)
    if (prediction && prediction.features_summary) {
        const fs = prediction.features_summary;
        setTrend("trend-current", fs.current_trend);
        setTrend("trend-temp", fs.temperature_trend);
        setTrend("trend-vib", fs.vibration_trend);
    }
}

function setTrend(elementId, trend) {
    const el = document.getElementById(elementId);
    if (!el || !trend) return;

    const arrows = { RISING: "↑ RISING", FALLING: "↓ FALLING", STABLE: "→ STABLE" };
    const classes = { RISING: "trend-rising", FALLING: "trend-falling", STABLE: "trend-stable" };

    el.textContent = arrows[trend] || trend;
    el.className = `gauge-trend ${classes[trend] || ""}`;
}

// ═══════════════════════════════════════════════════════════
// 6. CHAT INTERFACE
// ═══════════════════════════════════════════════════════════

function addChatMessage(role, content) {
    const container = document.getElementById("chat-messages");
    const msgDiv = document.createElement("div");
    msgDiv.className = `chat-msg ${role}`;

    const contentDiv = document.createElement("div");
    contentDiv.className = "msg-content";
    contentDiv.textContent = content;

    msgDiv.appendChild(contentDiv);
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
}

function addTypingIndicator() {
    const container = document.getElementById("chat-messages");
    const msgDiv = document.createElement("div");
    msgDiv.className = "chat-msg assistant";
    msgDiv.id = "typing-indicator";

    const dotsDiv = document.createElement("div");
    dotsDiv.className = "typing-indicator";
    dotsDiv.innerHTML = "<span></span><span></span><span></span>";

    msgDiv.appendChild(dotsDiv);
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
}

function removeTypingIndicator() {
    const el = document.getElementById("typing-indicator");
    if (el) el.remove();
}

async function sendChatMessage(message, questionId = null) {
    if (!state.selectedDevice) {
        addChatMessage("assistant", "Please select a device first.");
        return;
    }

    if (!message.trim()) return;

    // Show user message
    addChatMessage("user", message);

    // Show typing indicator
    addTypingIndicator();

    // Disable input
    const input = document.getElementById("chat-input");
    const sendBtn = document.getElementById("chat-send");
    input.disabled = true;
    sendBtn.disabled = true;

    // Call API
    const body = {
        device_id: state.selectedDevice,
        message: message,
    };
    if (questionId) body.question_id = questionId;

    const result = await apiPost("/api/chat", body);

    // Remove typing indicator
    removeTypingIndicator();

    // Show response
    if (result && result.response) {
        addChatMessage("assistant", result.response);
        // Update provider badge
        if (result.provider) {
            document.getElementById("chat-provider").textContent =
                result.provider === "groq" ? "Llama 3 (Groq)" : "Llama 3.2 (Local)";
        }
    } else {
        addChatMessage("assistant", "Sorry, I couldn't get a response. Please check the server logs.");
    }

    // Re-enable input
    input.disabled = false;
    sendBtn.disabled = false;
    input.value = "";
    input.focus();
}

// ═══════════════════════════════════════════════════════════
// 7. INITIALIZATION & EVENT LISTENERS
// ═══════════════════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", async () => {
    // Initialize charts
    initCharts();

    // Fetch devices
    await fetchDevices();

    // Device selector change
    document.getElementById("device-select").addEventListener("change", (e) => {
        if (e.target.value) {
            selectDevice(e.target.value);
        }
    });

    // Chart time range buttons
    document.querySelectorAll(".btn-time").forEach((btn) => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".btn-time").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            state.chartSeconds = parseInt(btn.dataset.seconds, 10);
            if (state.selectedDevice) {
                fetchChartData(state.selectedDevice, state.chartSeconds);
            }
        });
    });

    // Chat send button
    document.getElementById("chat-send").addEventListener("click", () => {
        const input = document.getElementById("chat-input");
        sendChatMessage(input.value);
    });

    // Chat enter key
    document.getElementById("chat-input").addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage(e.target.value);
        }
    });

    // Quick action buttons
    document.querySelectorAll(".quick-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const qid = btn.dataset.qid;
            const message = btn.textContent;
            sendChatMessage(message, qid);
        });
    });

    // Footer timestamp
    document.getElementById("footer-timestamp").textContent =
        `Started: ${new Date().toLocaleTimeString()}`;

    // Refresh device list every 30 seconds
    setInterval(fetchDevices, 30000);
});
