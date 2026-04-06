document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const countL1 = document.getElementById("count-l1");
    const countL2 = document.getElementById("count-l2");
    
    const densityBarL1 = document.getElementById("density-bar-l1");
    const densityValL1 = document.getElementById("density-val-l1");
    
    const densityBarL2 = document.getElementById("density-bar-l2");
    const densityValL2 = document.getElementById("density-val-l2");
    
    const dataStream = document.getElementById("data-stream");

    // Initialize WebSocket
    const wsUrl = `ws://${window.location.host}/ws`;
    let ws;

    function connectDataStream() {
        ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
            console.log("Connected to backend loop");
            dataStream.innerHTML = `<p style="color: #00ffcc;">[SYS] STREAM CONNECTED</p>`;
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        };

        ws.onclose = () => {
            console.log("Disconnected. Reconnecting...");
            dataStream.innerHTML = `<p style="color: #ff3333;">[SYS] CONNECTION LOST. RETRYING...</p>`;
            setTimeout(connectDataStream, 2000);
        };
    }

    function laneCardId(laneId) {
        return laneId === "l1" ? "lane1-card" : "lane2-card";
    }

    function updateLights(laneId, color) {
        const tl = document.getElementById(`tl-${laneId}`);
        if (!tl) return;

        const redEl = tl.querySelector(".light.red");
        const yellowEl = tl.querySelector(".light.yellow");
        const greenEl = tl.querySelector(".light.green");
        if (!redEl || !yellowEl || !greenEl) return;

        const valid = ["red", "yellow", "green"];
        const c = valid.includes(color) ? color : "red";

        redEl.classList.remove("active");
        yellowEl.classList.remove("active");
        greenEl.classList.remove("active");
        const bulb = tl.querySelector(`.light.${c}`);
        if (bulb) bulb.classList.add("active");

        const card = document.getElementById(laneCardId(laneId));
        if (!card) return;

        if (c === "green") {
            card.style.boxShadow = "0 0 20px rgba(0, 255, 51, 0.1)";
            card.style.borderColor = "rgba(0, 255, 51, 0.3)";
        } else if (c === "red") {
            card.style.boxShadow = "0 0 20px rgba(255, 51, 51, 0.05)";
            card.style.borderColor = "rgba(255, 255, 255, 0.08)";
        } else {
            card.style.boxShadow = "0 0 20px rgba(255, 204, 0, 0.1)";
            card.style.borderColor = "rgba(255, 204, 0, 0.3)";
        }
    }

    function updateDensityColor(barElement, value) {
        if (value < 30) {
            barElement.style.background = 'linear-gradient(90deg, #00ffcc, #00b3ff)';
        } else if (value < 70) {
            barElement.style.background = 'linear-gradient(90deg, #ffcc00, #ff9900)';
        } else {
            barElement.style.background = 'linear-gradient(90deg, #ff3333, #ff0000)';
        }
    }

    function updateDashboard(data) {
        // Update Counts
        countL1.innerText = data.l1_count;
        countL2.innerText = data.l2_count;
        
        // Update Densities
        densityBarL1.style.width = `${data.l1_density}%`;
        densityValL1.innerText = `${Math.round(data.l1_density)}%`;
        updateDensityColor(densityBarL1, data.l1_density);
        
        densityBarL2.style.width = `${data.l2_density}%`;
        densityValL2.innerText = `${Math.round(data.l2_density)}%`;
        updateDensityColor(densityBarL2, data.l2_density);
        
        // Update Lights
        if (data.signals) {
            const c1 = data.signals.l1 && data.signals.l1.color;
            const c2 = data.signals.l2 && data.signals.l2.color;
            if (c1) updateLights("l1", c1);
            if (c2) updateLights("l2", c2);
        }

        const card1 = document.getElementById("lane1-card");
        const card2 = document.getElementById("lane2-card");
        const ambSim = data.mode === "sim";

        // Highlight card for any emergency vehicle (ambulance or firetruck)
        if (card1) {
            card1.classList.toggle(
                "lane-ambulance-priority",
                ambSim && (Boolean(data.ambulance_l1) || Boolean(data.firetruck_l1))
            );
        }
        if (card2) {
            card2.classList.toggle(
                "lane-ambulance-priority",
                ambSim && (Boolean(data.ambulance_l2) || Boolean(data.firetruck_l2))
            );
        }
        
        // Update Data Stream
        const timestamp = new Date().toISOString().split('T')[1].slice(0, 8);
        let serialLine = "";
        if (data.serial) {
            const s = data.serial;
            const label = s.arduino_connected ? "HARDWARE" : "MOCK (no USB serial)";
            serialLine = `<p style="color:${s.arduino_connected ? '#00ffcc' : '#ffaa00'}">[${timestamp}] Arduino: ${label} — ${s.port}</p>`;
        }
        let sig = "?";
        if (data.signals && data.signals.l1 && data.signals.l2) {
            sig = `${String(data.signals.l1.color).toUpperCase()}_${String(data.signals.l2.color).toUpperCase()}`;
        }
        const reason =
            data.signals && data.signals.last_signal_change_reason
                ? String(data.signals.last_signal_change_reason)
                : "";
        const amb =
            ambSim && (data.ambulance_l1 || data.ambulance_l2)
                ? `<p style="color:#ff4499">[${timestamp}] 🚑 AMB: L1×${data.ambulance_count_l1 ?? 0} L2×${data.ambulance_count_l2 ?? 0}</p>`
                : "";
        const ft =
            ambSim && (data.firetruck_l1 || data.firetruck_l2)
                ? `<p style="color:#ff6600">[${timestamp}] 🚒 FT:  L1×${data.firetruck_count_l1 ?? 0} L2×${data.firetruck_count_l2 ?? 0}</p>`
                : "";
        const reasonLine =
            ambSim && reason
                ? `<p style="color:#88eeff">[${timestamp}] LAST_SIGNAL: ${reason}</p>`
                : "";

        // Update pedestrian panel
        const pedPanel = document.getElementById("ped-panel");
        const pedBar   = document.getElementById("ped-bar");
        const pedLabel = document.getElementById("ped-count-label");
        const pedStatus = document.getElementById("ped-status");
        if (pedPanel && data.mode === "sim") {
            pedPanel.style.display = "block";
            const pct = data.pedestrian_threshold > 0
                ? Math.round((data.pedestrian_count / data.pedestrian_threshold) * 100)
                : 0;
            pedBar.style.width = `${pct}%`;
            pedBar.style.background = data.pedestrian_phase_active
                ? "linear-gradient(90deg,#ff3333,#ff0000)"
                : pct >= 100
                ? "linear-gradient(90deg,#ffcc00,#ff6600)"
                : "linear-gradient(90deg,#00ffcc,#00b3ff)";
            pedLabel.innerText = `${data.pedestrian_count} / ${data.pedestrian_threshold}`;
            pedStatus.innerText = data.pedestrian_phase_active
                ? "🚦 CROSSING IN PROGRESS — both lanes RED"
                : pct >= 100
                ? "⚠️ Threshold reached — crossing queued"
                : `Auto-accumulating every 3s`;
            pedStatus.style.color = data.pedestrian_phase_active ? "#ff3333"
                : pct >= 100 ? "#ffcc00" : "var(--text-muted)";
        } else if (pedPanel) {
            pedPanel.style.display = "none";
        }
        let logBlock = "";
        if (data.logs && data.logs.length > 0) {
            logBlock = data.logs.slice(-8).map((log) => `<p class="log-line">${log}</p>`).join("");
        }
        dataStream.innerHTML = `
            ${serialLine}
            <p>[${timestamp}] L1_V: ${data.l1_count} | L2_V: ${data.l2_count}</p>
            <p>[${timestamp}] C_ST: ${sig}</p>
            ${amb}
            ${ft}
            ${reasonLine}
            ${logBlock}
        `;

        const banner = document.getElementById("priority-banner");
        if (banner) {
            if (ambSim && reason) {
                banner.hidden = false;
                if (reason.startsWith("ambulance_lane")) {
                    const lane = reason.endsWith("1") ? "Lane 1" : "Lane 2";
                    banner.className = "priority-banner emergency";
                    banner.textContent =
                        "Signal updated for emergency: " +
                        lane +
                        " was given GREEN (preemption — bypasses normal wait).";
                } else if (reason === "traffic_density") {
                    banner.className = "priority-banner normal";
                    banner.textContent =
                        "Last change: normal adaptive timing (vehicle counts / max green).";
                } else {
                    banner.className = "priority-banner normal";
                    banner.textContent = "Last change: " + reason.replace(/_/g, " ");
                }
            } else {
                banner.hidden = true;
                banner.textContent = "";
            }
        }
        
        // Update Mode Badge
        if (data.mode) {
             const modeBadge = document.getElementById("mode-badge");
             const btn = document.getElementById("toggle-mode-btn");
             if (data.mode === 'sim') {
                 modeBadge.innerText = "PIXEL SIMULATOR";
                 modeBadge.style.color = "#00ffcc";
                 btn.innerText = "SWITCH TO REAL VIDEO";
             } else {
                 modeBadge.innerText = "REAL VIDEO";
                 modeBadge.style.color = "#ff3333";
                 btn.innerText = "SWITCH TO PIXEL SIMULATOR";
             }
        }

        dataStream.scrollTop = dataStream.scrollHeight;
    }

    // Toggle button handler
    document.getElementById("toggle-mode-btn").addEventListener("click", () => {
        const text = document.getElementById("toggle-mode-btn").innerText;
        const targetMode = text.includes("VIDEO") ? "video" : "sim";
        fetch(`/api/toggle_mode?mode=${targetMode}`)
            .then(res => res.json())
            .then(data => console.log("Switched mode"))
            .catch(err => console.error(err));
    });

    // Start simulation clock
    connectDataStream();
});

// Expose control functions to global scope
window.updateIntensity = function(lane, value) {
    document.getElementById(`intensity-val-${lane}`).innerText = value;
    fetch('/api/sim_controls', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lane: lane, intensity: value })
    }).catch(err => console.error("Sim control error:", err));
};

window.spawnAmbulance = function(lane) {
    fetch('/api/sim_controls', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lane: lane, spawn_ambulance: true })
    }).catch(err => console.error("Sim control error:", err));
};

window.spawnFiretruck = function(lane) {
    fetch('/api/sim_controls', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lane: lane, spawn_firetruck: true })
    }).catch(err => console.error("Sim control error:", err));
};

window.pedAdd = function() {
    fetch('/api/pedestrian_crossing', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'add' })
    }).catch(err => console.error("Pedestrian add error:", err));
};

window.pedTrigger = function() {
    fetch('/api/pedestrian_crossing', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'trigger' })
    }).catch(err => console.error("Pedestrian trigger error:", err));
};

