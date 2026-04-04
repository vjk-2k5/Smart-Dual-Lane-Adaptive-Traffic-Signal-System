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
        
        // Reset all
        tl.querySelector('.red').classList.remove('active');
        tl.querySelector('.yellow').classList.remove('active');
        tl.querySelector('.green').classList.remove('active');
        
        // Set active
        if (color) {
            tl.querySelector(`.${color}`).classList.add('active');
            
            // Add a subtle glow to the lane card based on the active light
            const card = document.getElementById(laneCardId(laneId));
            if (color === 'green') {
                card.style.boxShadow = '0 0 20px rgba(0, 255, 51, 0.1)';
                card.style.borderColor = 'rgba(0, 255, 51, 0.3)';
            } else if (color === 'red') {
                card.style.boxShadow = '0 0 20px rgba(255, 51, 51, 0.05)';
                card.style.borderColor = 'rgba(255, 255, 255, 0.08)';
            } else {
                card.style.boxShadow = '0 0 20px rgba(255, 204, 0, 0.1)';
                card.style.borderColor = 'rgba(255, 204, 0, 0.3)';
            }
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
            updateLights('l1', data.signals.l1.color);
            updateLights('l2', data.signals.l2.color);
        }
        
        // Update Data Stream (telemetry + optional serial / command log)
        const timestamp = new Date().toISOString().split('T')[1].slice(0, 8);
        let serialLine = "";
        if (data.serial) {
            const s = data.serial;
            const label = s.arduino_connected ? "HARDWARE" : "MOCK (no USB serial)";
            serialLine = `<p style="color:${s.arduino_connected ? '#00ffcc' : '#ffaa00'}">[${timestamp}] Arduino: ${label} — ${s.port}</p>`;
        }
        const sig = data.signals
            ? `${data.signals.l1.color.toUpperCase()}_${data.signals.l2.color.toUpperCase()}`
            : "?";
        let logBlock = "";
        if (data.logs && data.logs.length > 0) {
            logBlock = data.logs.slice(-8).map((log) => `<p class="log-line">${log}</p>`).join("");
        }
        dataStream.innerHTML = `
            ${serialLine}
            <p>[${timestamp}] L1_V: ${data.l1_count} | L2_V: ${data.l2_count}</p>
            <p>[${timestamp}] C_ST: ${sig}</p>
            ${logBlock}
        `;
        
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
