# Network Sensory Monitor — Advanced

A real-time network intelligence and visualization tool built using **Python + Tkinter** that monitors network stability and simulates core computer networking concepts in an interactive GUI.

This project combines **real diagnostics** (latency, jitter, packet loss, bandwidth via Speedtest) with **educational simulations** (routing, congestion control, QoS, packet scheduling).

---

## Features

### Real-Time Network Monitoring

* Live ping monitoring
* Latency, jitter, and packet loss tracking
* Network Stability Index (NSI)
* QoS classification (EF / AF / BE)

### Real Bandwidth Testing

* Integrated **Ookla Speedtest CLI** support
* Real download/upload speeds
* Background testing (non-blocking UI)

### Networking Simulations

* Routing algorithm visualization (distance vector model)
* TCP congestion window simulator
* Packet scheduling (WFQ / RR / Priority queues)
* QoS classification visualization

### Network Awareness

* ARP table monitoring (basic spoof detection)
* DNS resolution timing
* Gateway MAC tracking

### GUI Highlights

* Smooth animated network status indicator
* Interactive popups for each simulation module
* Live dashboard metrics

---

## Tech Stack

* Python 3.x
* Tkinter (GUI)
* Ookla Speedtest CLI
* Threading + Subprocess modules

---

## 📁 Project Structure

```
Network Monitor/
│
├── network_sensory_monitor_advanced.py   # Main application
├── speedtest.exe                        # Ookla Speedtest CLI (Windows)
└── README.md
```

---

## Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/network-sensory-monitor.git
cd network-sensory-monitor
```

---

### 2️⃣ Install Python Requirements

This project uses mostly built-in libraries.

Optional (for sound support on non-Windows):

```bash
pip install simpleaudio numpy
```

---

### 3️⃣ Setup Speedtest CLI

#### Option A — Windows (Already included)

* `speedtest.exe` is bundled in this repo
* No extra setup needed

#### Option B — Mac/Linux

Install Ookla Speedtest CLI:

```bash
# Ubuntu/Debian
sudo apt install speedtest-cli

# Or install official Ookla CLI from:
https://www.speedtest.net/apps/cli
```

Make sure `speedtest` is available in PATH.

---

## ▶️ How to Run

```bash
python network_sensory_monitor_advanced.py
```

Then:

1. Click **Start Monitoring**
2. Watch real-time network metrics
3. Use buttons to explore simulations:

   * Routing View
   * Congestion View
   * Queues View
   * ARP / DNS
   * Run Speedtest Now

---

## What You Can Learn From This Project

This project demonstrates practical understanding of:

* Computer Networks fundamentals
* TCP congestion control
* Routing algorithms
* QoS and traffic prioritization
* Network diagnostics tools
* GUI system design
* Multithreading in Python

---

##  Demo Ideas (Optional)

* Add screenshots of GUI here
* Add GIF of live monitoring

---

## 💡 Future Improvements

* Dark mode / cyberpunk UI
* Export network logs
* Web dashboard version
* AI-based network anomaly detection

---

## 🧑Author

Built by **H M Akshay** — Computer Science Undergraduate focused on systems, networking, and high-performance engineering.

---

## If you found this useful

Give the repo a star and feel free to fork
