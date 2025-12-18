
window.addEventListener('DOMContentLoaded', () => {
  const sidebar = document.getElementById('sidebar');
  const mainContent = document.getElementById('main-content');
  const toggleBtn = document.getElementById('toggle-btn');
  const logo = document.querySelector('.sidebar-logo');
  const infoIcon = document.querySelector('.info-icon');
  const sheet = document.getElementById('prerequisite-sheet');
  const closeSheetBtn = document.getElementById('close-sheet-btn');

  // Sidebar items
  const predictionItem = document.querySelector('.fa-palette').closest('.item-header');
  const modelItem = document.querySelector('.fa-laptop-code').closest('.item-header');
  const instructionItem = document.querySelector('.fa-person-chalkboard').closest('.item-header');
  const settingsItem = document.querySelector('.fa-gear').closest('.item-header');

  // Page sections
  const homeSection = document.getElementById('home-section');
  const predictionSection = document.getElementById('prediction-section');
  const modelSection = document.getElementById('model-section');
  const instructionSection = document.getElementById('instruction-section');
  const settingsSection = document.getElementById('settings-section');

  // Initial rotation
  toggleBtn.style.transform = sidebar.classList.contains('collapsed')
    ? 'rotate(180deg)'
    : 'rotate(0deg)';

  // Sidebar toggle only when clicking empty space
  sidebar.addEventListener('click', (e) => {
    if (
      e.target.closest('button') ||
      e.target.closest('.sidebar-item') ||
      e.target.closest('.sidebar-logo')
    ) {
      return;
    }
    toggleSidebar();
  });

  // Toggle button click
  toggleBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    toggleSidebar();
  });

  // Logo → go home
  logo.addEventListener('click', (e) => {
    e.stopPropagation();
    showSection('home');
  });

  // Sidebar navigation
  predictionItem.addEventListener('click', (e) => {
    e.stopPropagation();
    showSection('prediction');
  });

  modelItem.addEventListener('click', (e) => {
    e.stopPropagation();
    showSection('model');
  });

  instructionItem.addEventListener('click', (e) => {
    e.stopPropagation();
    showSection('instruction');
  });

  settingsItem.addEventListener('click', (e) => {
    e.stopPropagation();
    showSection('settings');
  });

  function toggleSidebar() {
    sidebar.classList.toggle('collapsed');
    mainContent.classList.toggle('collapsed');

    toggleBtn.style.transform = sidebar.classList.contains('collapsed')
      ? 'rotate(180deg)'
      : 'rotate(0deg)';
  }

  function showSection(section) {
    // Hide all sections
    [homeSection, predictionSection, modelSection, instructionSection, settingsSection].forEach(
      sec => (sec.style.display = 'none')
    );

    // Show selected one
    if (section === 'home') homeSection.style.display = 'block';
    if (section === 'prediction') predictionSection.style.display = 'block';
    if (section === 'model') modelSection.style.display = 'block';
    if (section === 'instruction') instructionSection.style.display = 'block';
    if (section === 'settings') settingsSection.style.display = 'block';
  }

  // Default: show home page
  showSection('home');

  // Prerequisite sheet toggle
  infoIcon.addEventListener('click', (e) => {
  e.stopPropagation();
  sheet.classList.add('active');
});

  closeSheetBtn.addEventListener('click', () => {
    sheet.classList.remove('active');
  });

  sheet.addEventListener('click', (e) => {
  if (e.target === sheet) sheet.classList.remove('active');
});
});


// Check prerequisites + Start Collecting data

const imuOutput = document.getElementById('imu-output');
const imuCheckBtn = document.getElementById('imu-check-btn');
const imuStartBtn = document.getElementById('imu-start-btn');
const imuPauseBtn = document.getElementById('imu-pause-btn');
const imuStopBtn = document.getElementById('imu-stop-btn');
const toggleCameraBtn = document.getElementById("toggleCameraBtn");
const cameraFeed = document.getElementById("cameraFeed");
const cameraMessage = document.getElementById('cameraMessage');
const messageTerminals = document.getElementById('cameraTerminals');
const modal = document.getElementById("imuModal");
const img = document.getElementById("imu-thumb");
const modalImg = document.getElementById("imuFull");
const closeBtn = document.getElementsByClassName("close")[0];

let imuPrerequisitesChecked = false;
let imuIsRecording = false;
let imuIsPaused = false;
let eventSource = null;
let readyCheckActive = true;
let cameraVisible = false;

img.onclick = function() {
  modal.style.display = "block";
  modalImg.src = this.src;
};

closeBtn.onclick = function() {
  modal.style.display = "none";
};

window.onclick = function(event) {
  if (event.target == modal) {
    modal.style.display = "none";
  }
};

function formatLog(line) {

    const importantWords = [
        "Detected", "Missing", "Expected", "IMU", "UDP",
        "Camera", "Batteries", "%", "connected", "Connected",
        "not received", "Ready"
    ];

    const forceNewLine = [
        "Connected to IMUs",
        "IMUs Batteries",
        "Ready to start collecting data"
    ];

    let formatted = line;

    if (importantWords.some(w => line.includes(w))) {
        formatted = "• " + formatted;
    }

    if (forceNewLine.some(w => line.includes(w))) {
        formatted = "\n" + formatted;
    }

    return formatted;
}

function startRealtimeStream() {
    if (eventSource) eventSource.close();

    eventSource = new EventSource("http://127.0.0.1:5000/stream");

    eventSource.onmessage = (event) => {
        const raw = event.data.trim();
        if (!raw) return;

        if (raw === "STREAM_CLOSED") {
          eventSource.close();
          eventSource = null;
          return;
        }

        if (imuIsRecording) {
        const recordingIgnorePatterns = [
            "Camera stream started",
            "Recording started",
            "Press SPACE to pause",
            "Programme running",
            "ctrl + C to stop",
            "Clean Folder",
            "Using New Camera"
        ];

        if (recordingIgnorePatterns.some(p => raw.includes(p))) {
            return; 
        }
      }

        const ignorePatterns = [
        "Traceback (most recent call last):",
        "cStarting",
        "Temporary_Data",
        "Clearing",
        "Checking Wifi",
        "Checking IMU",
        "Checking camera",
        "Waiting for battery updates",
        "Programme Ready",
        "Starting in",
        ];

        if (ignorePatterns.some(p => raw.includes(p))) {
        return; 
        }

        let line = raw;
        if (line.includes("Ready. Press Enter to start recording")) {
        line = "Ready to start collecting data\n";
        }

        let formatted = formatLog(line);

        imuOutput.textContent += "\n" + formatted;
        imuOutput.scrollTop = imuOutput.scrollHeight;

        // Detect "Ready"
        if (line.includes("Ready to start collecting data")) {
            imuPrerequisitesChecked = true;
            imuStartBtn.disabled = false;
            imuPauseBtn.disabled = true;
            imuStopBtn.disabled = true;
        }
    };

    eventSource.onerror = () => {
        console.warn("Stream stopped.");
        eventSource.close();
    };
}

imuCheckBtn.addEventListener('click', async () => {

    imuOutput.classList.remove("hidden");
    imuOutput.textContent = "";

    try {
        const response = await fetch("http://127.0.0.1:5000/check");
        const data = await response.json();

        imuOutput.textContent += "• " + data.output + "\n";

        startRealtimeStream();

    } catch (error) {
        imuOutput.textContent += "\nError connecting backend.";
        console.error(error);
    }
});

imuStartBtn.addEventListener('click', async () => {

    readyCheckActive = false;

    if (!imuPrerequisitesChecked) {
        imuOutput.textContent += "\nPlease check prerequisites before collecting data.\n";
        return;
    }

    if (imuIsRecording) {
        imuOutput.textContent += "\nRecording already in progress.\n";
        return;
    }

    imuIsRecording = true;
    imuIsPaused = false;

    imuStartBtn.disabled = true;
    imuPauseBtn.disabled = false;
    imuStopBtn.disabled = false;

    try {
        await fetch("http://127.0.0.1:5000/start_recording");

        imuOutput.textContent += "\n----- Starting recording -----\n";

    } catch (err) {
        imuOutput.textContent += "\nError: Cannot continue recording.\n";
        console.error(err);
    }

    imuOutput.scrollTop = imuOutput.scrollHeight;
});

imuPauseBtn.addEventListener('click', async () => {

    if (!imuIsRecording) return;

    imuIsPaused = !imuIsPaused;
    imuPauseBtn.textContent = imuIsPaused ? "Resume" : "Pause";

    try {
        await fetch("http://127.0.0.1:5000/pause", { method: "POST" });
        imuOutput.textContent += imuIsPaused
            ? "\n• Data collection paused.\n"
            : "\n• Data collection resumed.\n";

    } catch (err) {
        imuOutput.textContent += "\nError trying to pause/resume.\n";
        console.error(err);
    }

    imuOutput.scrollTop = imuOutput.scrollHeight;
});

imuStopBtn.addEventListener('click', async () => {

    if (!imuIsRecording) return;

    imuIsRecording = false;
    imuIsPaused = false;

    imuStartBtn.disabled = true;
    imuPauseBtn.disabled = true;
    imuStopBtn.disabled = true;

    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }

    try {
        await fetch("http://127.0.0.1:5000/stop", { method: "POST" });
        imuOutput.textContent += "\n----- Data collection stopped -----\n";
    } catch (err) {
        imuOutput.textContent += "\nError stopping process.\n";
    }

    imuOutput.textContent += "Please check prerequisites again to continue.\n";
    imuPrerequisitesChecked = false;
});

const checkForReadyMessage = () => {

  if (!readyCheckActive) return;

  if (imuOutput.textContent.includes("Ready to start collecting data")) {
    imuPrerequisitesChecked = true;
    imuIsRecording = false;
    imuIsPaused = false;

    imuStartBtn.disabled = false;
    imuPauseBtn.disabled = true;
    imuStopBtn.disabled = true;
  }
};

const observer = new MutationObserver(checkForReadyMessage);
observer.observe(imuOutput, { childList: true, subtree: true, characterData: true });

setInterval(checkForReadyMessage, 500);


toggleCameraBtn.addEventListener("click", () => {
    cameraVisible = !cameraVisible;
    

    if (cameraVisible) {
        cameraFeed.src = "http://127.0.0.1:5000/camera_feed";
        cameraFeed.style.display = "block";
        cameraMessage.style.display = "none";
        toggleCameraBtn.innerHTML = `<i class="fa-solid fa-camera"></i> Hide camera`;
    } else {
        cameraFeed.src = "";
        cameraFeed.style.display = "none";
        cameraMessage.style.display = "block";
        toggleCameraBtn.innerHTML = `<i class="fa-solid fa-camera"></i> Display the camera's screen`;
    }
    if (cameraVisible === false) {
        cameraMessage.textContent = "Camera Stream in real-time";
        toggleCameraBtn.innerHTML = `<i class="fa-solid fa-camera"></i> Display the camera's screen`;
    }
});

// End Check prerequisites + Start Collecting data

// Model section 

let activeModels = {};
let streams = {};

document.querySelectorAll(".model-btn").forEach(btn => {
    btn.addEventListener("click", () => toggleModel(btn));
});


async function toggleModel(btn) {
    const model = btn.dataset.model;

    // If already active → stop model
    if (btn.classList.contains("active")) {
        stopModel(model);
        btn.classList.remove("active");
        return;
    }

    // Otherwise → start model
    btn.classList.add("active");
    await startModel(model);
}


/* -------------------------
   START MODEL
----------------------------*/
async function startModel(model) {
    const res = await fetch("http://127.0.0.1:5000/run_model_stream", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ models: [model] })
    });

    const data = await res.json();
    if (!data.started.includes(model)) return;

    activeModels[model] = true;
    

    createTerminal(model);
    updateTerminalMessage();
    startModelStream(model);
}

/* -------------------------
   STOP MODEL
----------------------------*/
function stopModel(model) {
    if (streams[model]) {
        streams[model].close();
        delete streams[model];
    }
    fetch(`http://127.0.0.1:5000/stop_model/${model}`, { method: "POST" });

    const terminal = document.getElementById(`terminal-${model}`);
    if (terminal) terminal.remove();
    updateTerminalMessage();

    delete activeModels[model];
}

/* -------------------------
   CREATE TERMINAL BOX
----------------------------*/
function createTerminal(model) {
    const container = document.getElementById("modelTerminals");

    const div = document.createElement("div");
    div.classList.add("terminal-box");
    div.id = `terminal-${model}`;

    div.innerHTML = `
    <h3>${model}</h3>

    <table class="action-table">
        <thead>
            <tr>
                <th>Hand</th>
                <th>Action</th>
                <th>Tool</th>
                <th>Time (s)</th>
            </tr>
        </thead>
        <tbody id="output-${model}">
        </tbody>
    </table>
`;

    container.appendChild(div);
}

function parseActionLine(line) {
    const regex = /Sample_\d+\s*:\s*(Unimanual|Bimanual)_(\w+)\s*\|\s*Tool:\s*(\w+)\s*at\s*([\d.]+)/;

    const match = line.match(regex);
    if (!match) return null;

    const [, handType, actionType, toolType, time] = match;

    // Hand → icônes
    const handIcon =
        handType === "Unimanual"
            ? `<i class="fa-solid fa-hand" title="using one hand"></i>`
            : `<i class="fa-solid fa-hand"></i><i class="fa-solid fa-hand" title="using two hands"></i>`;

    // Action → icône
    const icons = {
    "Up": `<i class="fa-solid fa-arrow-up-long" title="from the bottom up"></i>`,
    "Down": `<i class="fa-solid fa-arrow-down-long" title="from the top down"></i>`,
    "Prepare": `<i class="fa-solid fa-palette" title="in preparation"></i>`,
    "Left": `<i class="fa-solid fa-arrow-right-long" title="from left to right"></i>`,
    "Right": `<i class="fa-solid fa-arrow-left-long" title="from right to left"></i>`
};


    const actionIcon = icons[actionType] || actionType;

    // Tool → icône
    const toolIcons = {
        "none": "",
        "brush": `<i class="fa-solid fa-brush" title="using a brush"></i>`,
        "shortroller": `<i class="fa-solid fa-paint-roller" title="using a short roller"></i>`,
        "longroller": `Long <i class="fa-solid fa-paint-roller" title="using a long roller"></i>`
    };

    const toolIcon = toolIcons[toolType.toLowerCase()] || toolType;

    return { handIcon, actionIcon, toolIcon, time };
}




/* -------------------------
   EVENTSOURCE STREAM
----------------------------*/
function startModelStream(model) {
    const eventSource = new EventSource(`http://127.0.0.1:5000/model_stream/${model}`);
    streams[model] = eventSource;

    const output = document.getElementById(`output-${model}`);

    const ignorePatterns = [
        "Starting...",
        "Recording data? (Y/N):",
        "Loading Fusion model:",
        "Loading YOLO model:",
        "Using",
        "Programme running   ctrl + C to stop",
        "Loading"    
    ];

    let buffer = "";     // Text waiting to be flushed to DOM
    let flushing = false; // Prevent multiple timers

    function flushBuffer() {
        if (!buffer) return;
        output.textContent += buffer;
        buffer = "";
        output.scrollTop = output.scrollHeight;
        flushing = false;
    }


    eventSource.onmessage = (event) => {
        let line = event.data.trim();

        if (line.startsWith("\x1bc")) return;

        if (ignorePatterns.some(pattern => line.includes(pattern))) return;

        const cleanedLine = line.replace(/^\x1bc\s*/, "");

        if (ignorePatterns.some(pattern => cleanedLine.includes(pattern))) return;

        const parsed = parseActionLine(line);

        if (parsed) {
            const tbody = document.getElementById(`output-${model}`);
            const row = document.createElement("tr");

            row.innerHTML = `
                <td>${parsed.handIcon}</td>
                <td>${parsed.actionIcon}</td>
                <td>${parsed.toolIcon}</td>
                <td>${parsed.time}</td>
            `;



            tbody.appendChild(row);
            return; // Ne pas enregistrer dans buffer
        }

        if (!flushing) {
            flushing = true;
            setTimeout(flushBuffer, 50);
        }
    };

    eventSource.onerror = () => {
        console.log("Stream ended for", model);
        eventSource.close();
        flushBuffer();
    };
}

/* -------------------------
   STOP ALL MODELS
----------------------------*/
document.getElementById("stopAll").addEventListener("click", () => {
    Object.keys(activeModels).forEach(model => stopModel(model));
    document.querySelectorAll(".model-btn.active").forEach(btn => btn.classList.remove("active"));
});

/* -------------------------
   Terminal message
----------------------------*/

function updateTerminalMessage() {
    const msg = document.getElementById("messageTerminals");
    const container = document.getElementById("modelTerminals");

    const terminalCount = container.querySelectorAll(".terminal-box").length;

    msg.style.display = terminalCount === 0 ? "block" : "none";
}



// End Model section 


