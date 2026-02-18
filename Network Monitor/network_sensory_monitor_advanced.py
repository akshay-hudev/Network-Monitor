"""
Network Sensory Monitor — Advanced (Tkinter) — Speedtest-integrated
- Uses real Ookla Speedtest CLI for download/upload bandwidth
- Keeps previous advanced CN features (routing sim, congestion sim, ARP, DNS, queues)
- Runs speedtest in background thread to avoid UI blocking
"""

import tkinter as t
from tkinter import ttk, messagebox
import threading
import subprocess
import platform
import re
import time
import statistics
import collections
import math
import socket
import random
import json

# Optional audio (winsound on Windows, simpleaudio on others)
HAS_WINSOUND = False
HAS_SIMPLEAUDIO = False
try:
    import winsound
    HAS_WINSOUND = True
except Exception:
    HAS_WINSOUND = False

try:
    import simpleaudio as sa
    import numpy as np
    HAS_SIMPLEAUDIO = True
except Exception:
    HAS_SIMPLEAUDIO = False

# ------------------------
# Config
# ------------------------
PING_HOST = "8.8.8.8"
PING_INTERVAL = 1.0
ROLLING_WINDOW = 20
TIMEOUT_SECONDS = 1
BANDWIDTH_PROBES = 3  # fallback probes
ROUTER_COUNT = 5
SPEEDTEST_INTERVAL = 300  # seconds between automatic full speedtests (5 minutes)

# State color mapping
STATE_MAP = {
    "excellent": {"color": (0, 200, 100), "label": "Excellent", "freq": 880, "dur": 150},
    "moderate":  {"color": (255, 200, 0), "label": "Moderate",  "freq": 660, "dur": 150},
    "congested": {"color": (220, 40, 40), "label": "Congested", "freq": 440, "dur": 250},
    "loss":      {"color": (160, 30, 160),"label": "High Loss", "freq": 300, "dur": 300},
    "down":      {"color": (60, 120, 220), "label": "Disconnected", "freq": 200, "dur": 500},
}

# ------------------------
# Utility: ping once cross-platform
# ------------------------
def ping_once(host, timeout=1):
    plat = platform.system().lower()
    try:
        if plat == "windows":
            cmd = ["ping", "-n", "1", "-w", str(int(timeout*1000)), host]
        else:
            cmd = ["ping", "-c", "1", host]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout+1)
        out = p.stdout
        m = re.search(r"time[=<]\s*([0-9]+(?:\.[0-9]+)?)\s*ms", out)
        if not m:
            m2 = re.search(r"Average = ([0-9]+)ms", out)
            if m2:
                return float(m2.group(1))
            return None
        return float(m.group(1))
    except Exception:
        return None

# ------------------------
# Audio: play tone on state change
# ------------------------
def play_tone(freq_hz=440, duration_ms=200):
    if HAS_WINSOUND:
        try:
            winsound.Beep(int(freq_hz), int(duration_ms))
            return
        except Exception:
            pass
    if HAS_SIMPLEAUDIO:
        fs = 44100
        t = np.linspace(0, duration_ms/1000.0, int(fs * (duration_ms/1000.0)), False)
        note = np.sin(freq_hz * t * 2 * np.pi)
        audio = note * (2**15 - 1) / np.max(np.abs(note))
        audio = audio.astype(np.int16)
        try:
            sa.play_buffer(audio, 1, 2, fs)
        except Exception:
            pass
    else:
        # no audio available
        pass

# ------------------------
# Real Ookla Speedtest CLI integration
# ------------------------
def run_real_speedtest_blocking(timeout=120):
    """
    Run Ookla speedtest CLI synchronously (blocking). Returns (ping_ms, dl_mbps, ul_mbps) or None on failure.
    Ensure 'speedtest' is in PATH. Uses --format=json output.
    """
    try:
        proc = subprocess.run(
            ["speedtest", "--format=json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout
        )
        if proc.returncode != 0:
            # try speedtest.exe (windows explicit)
            try:
                proc = subprocess.run(["speedtest.exe", "--format=json"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
            except Exception:
                return None
        data = json.loads(proc.stdout)
        ping = float(data.get("ping", {}).get("latency", 0.0))
        # download.upload bandwidth fields are in bytes/sec in current speedtest CLI versions
        dl = float(data.get("download", {}).get("bandwidth", 0.0)) * 8 / 1e6
        ul = float(data.get("upload", {}).get("bandwidth", 0.0)) * 8 / 1e6
        return ping, dl, ul
    except Exception as e:
        # debug print
        # print("Speedtest error:", e)
        return None

def run_real_speedtest_async(callback):
    """
    Run speedtest in background thread and call callback(result) when done.
    result is (ping, dl, ul) or None.
    """
    def worker():
        res = run_real_speedtest_blocking()
        try:
            callback(res)
        except Exception:
            pass
    threading.Thread(target=worker, daemon=True).start()

# ------------------------
# Fallback bandwidth estimator (old, noisy) - kept as fallback
# ------------------------
def estimate_bandwidth(host, probes=3):
    samples = []
    for _ in range(probes):
        t1 = ping_once(host, timeout=1)
        t2 = ping_once(host, timeout=1)
        if t1 is None or t2 is None:
            continue
        delta_ms = abs(t2 - t1)
        if delta_ms > 0:
            bw_mbps = (1500*8) / (delta_ms/1000.0) / 1e6
            samples.append(bw_mbps)
    if samples:
        return statistics.median(samples)
    return None

# ------------------------
# ARP table read (best-effort)
# ------------------------
def read_arp_table():
    plat = platform.system().lower()
    try:
        if plat == "windows":
            proc = subprocess.run(["arp", "-a"], stdout=subprocess.PIPE, text=True)
            out = proc.stdout
            entries = {}
            for line in out.splitlines():
                m = re.search(r"(\d+\.\d+\.\d+\.\d+)\s+([0-9a-fA-F:-]{17})", line)
                if m:
                    ip, mac = m.group(1), m.group(2).replace('-', ':').lower()
                    entries[ip] = mac
            return entries
        else:
            proc = subprocess.run(["arp", "-a"], stdout=subprocess.PIPE, text=True)
            out = proc.stdout
            entries = {}
            for line in out.splitlines():
                m = re.search(r"\((\d+\.\d+\.\d+\.\d+)\)\s+at\s+([0-9a-fA-F:]{17})", line)
                if m:
                    ip, mac = m.group(1), m.group(2).lower()
                    entries[ip] = mac
            return entries
    except Exception:
        return {}

# ------------------------
# Routing simulation: distance vector among virtual routers
# ------------------------
class VirtualRouterNetwork:
    def __init__(self, n=5):
        self.n = n
        self.cost = [[math.inf]*n for _ in range(n)]
        for i in range(n):
            self.cost[i][i] = 0
        for i in range(n):
            for j in range(i+1, n):
                c = random.choice([1,2,5,10])
                self.cost[i][j] = self.cost[j][i] = c
        self.dist = [[self.cost[i][j] for j in range(n)] for i in range(n)]
        self.next_hop = [[j if self.cost[i][j] < math.inf else None for j in range(n)] for i in range(n)]
        self._distance_vector_converge()

    def _distance_vector_converge(self, iterations=10):
        n = self.n
        for _ in range(iterations):
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        if self.cost[i][k] < math.inf and self.dist[k][j] < math.inf:
                            nd = self.cost[i][k] + self.dist[k][j]
                            if nd < self.dist[i][j]:
                                self.dist[i][j] = nd
                                self.next_hop[i][j] = self.next_hop[i][k]

    def perturb_random_link(self):
        i = random.randrange(0, self.n)
        j = random.randrange(0, self.n)
        if i == j: return
        newc = random.choice([1,2,5,8,12,20])
        self.cost[i][j] = self.cost[j][i] = newc
        self.dist = [[self.cost[i][j] for j in range(self.n)] for i in range(self.n)]
        self._distance_vector_converge()

    def get_routing_table(self):
        table = []
        for dest in range(self.n):
            nh = self.next_hop[0][dest]
            cost = self.dist[0][dest]
            table.append((dest, nh, cost))
        return table

# ------------------------
# Packet scheduler emulator (3 queues)
# ------------------------
class PacketScheduler:
    def __init__(self):
        self.queues = { "EF": collections.deque(), "AF": collections.deque(), "BE": collections.deque() }
        self.counter = 0

    def generate_packets(self, ef_rate=0.3, af_rate=0.5, be_rate=0.2, burst=False):
        for _ in range(random.randint(1, 4) if burst else random.randint(0,2)):
            r = random.random()
            self.counter += 1
            if r < ef_rate:
                self.queues["EF"].append((self.counter, random.randint(100, 800), time.time()))
            elif r < ef_rate + af_rate:
                self.queues["AF"].append((self.counter, random.randint(200, 1500), time.time()))
            else:
                self.queues["BE"].append((self.counter, random.randint(400, 2000), time.time()))

    def schedule(self, algorithm="WFQ", wfq_weights=None):
        if algorithm == "PRIORITY":
            for q in ["EF","AF","BE"]:
                if self.queues[q]:
                    return q, self.queues[q].popleft()
            return None, None
        elif algorithm == "RR":
            for q in ["EF","AF","BE"]:
                if self.queues[q]:
                    return q, self.queues[q].popleft()
            return None, None
        else:
            wfq_weights = wfq_weights or {"EF":3, "AF":2, "BE":1}
            scores = {}
            for q in ["EF","AF","BE"]:
                scores[q] = (len(self.queues[q]) + 1) / wfq_weights[q]
            qpick = min(scores, key=scores.get)
            if self.queues[qpick]:
                return qpick, self.queues[qpick].popleft()
            for q in ["EF","AF","BE"]:
                if self.queues[q]:
                    return q, self.queues[q].popleft()
            return None, None

    def pending_counts(self):
        return {k: len(v) for k,v in self.queues.items()}

# ------------------------
# Congestion control model (simple cwnd)
# ------------------------
class CongestionSimulator:
    def __init__(self):
        self.cwnd = 1.0
        self.ssthresh = 64.0
        self.state = "slow-start"
        self.history = []

    def on_ack(self):
        if self.cwnd < self.ssthresh:
            self.cwnd += 1.0
            self.state = "slow-start"
        else:
            self.cwnd += 1.0/self.cwnd
            self.state = "congestion-avoidance"
        self.history.append((time.time(), self.cwnd))

    def on_loss(self):
        self.ssthresh = max(2.0, self.cwnd / 2.0)
        self.cwnd = max(1.0, self.cwnd / 2.0)
        self.state = "fast-recovery"
        self.history.append((time.time(), self.cwnd))

    def tick_with_network(self, loss_prob):
        if random.random() < loss_prob:
            self.on_loss()
        else:
            for _ in range(max(1, int(self.cwnd))):
                self.on_ack()

# ------------------------
# DNS resolver check
# ------------------------
def dns_lookup_time(domain="www.google.com"):
    try:
        t0 = time.time()
        socket.gethostbyname(domain)
        return (time.time() - t0) * 1000.0
    except Exception:
        return None

# ------------------------
# NSI: combine jitter/loss/variance/bw into 0..1 score (1=stable)
# ------------------------
def compute_nsi(jitter_ms, loss_percent, var_latency, bandwidth_mbps):
    jscore = max(0.0, 1.0 - min(jitter_ms/100.0, 1.0))
    lscore = max(0.0, 1.0 - min(loss_percent/100.0, 1.0))
    vscore = max(0.0, 1.0 - min(var_latency/500.0, 1.0))
    if bandwidth_mbps is None:
        bwscore = 0.2
    else:
        bwscore = min(bandwidth_mbps / 200.0, 1.0)
    score = (0.35*lscore + 0.25*jscore + 0.2*vscore + 0.2*bwscore)
    return max(0.0, min(1.0, score))

# ------------------------
# GUI Application
# ------------------------
class NetworkSensoryApp:
    def __init__(self, root):
        self.root = root
        root.title("Network Sensory Monitor — Advanced")
        self.main = ttk.Frame(root, padding=12)
        self.main.grid()
        self.canvas_size = 320
        self.canvas = tk.Canvas(self.main, width=self.canvas_size, height=self.canvas_size, highlightthickness=0)
        self.canvas.grid(row=0, column=0, columnspan=5, pady=(0,6))
        self.bulb = self.canvas.create_oval(30,30,self.canvas_size-30,self.canvas_size-30, fill="#111111", outline="")
        # info
        self.status_var = tk.StringVar(value="Initializing...")
        ttk.Label(self.main, textvariable=self.status_var, font=("Helvetica",14)).grid(row=1,column=0,columnspan=5)
        self.ping_var = tk.StringVar(value="Ping: - ms")
        self.loss_var = tk.StringVar(value="Loss: - %")
        self.jitter_var = tk.StringVar(value="Jitter: - ms")
        self.bw_var = tk.StringVar(value="Bandwidth: - Mbps")
        self.dl_var = tk.StringVar(value="DL: - Mbps")
        self.ul_var = tk.StringVar(value="UL: - Mbps")
        self.qos_var = tk.StringVar(value="QoS: -")
        self.nsi_var = tk.StringVar(value="NSI: -")
        ttk.Label(self.main, textvariable=self.ping_var).grid(row=2,column=0,sticky="w")
        ttk.Label(self.main, textvariable=self.loss_var).grid(row=2,column=1,sticky="w")
        ttk.Label(self.main, textvariable=self.jitter_var).grid(row=2,column=2,sticky="w")
        ttk.Label(self.main, textvariable=self.bw_var).grid(row=2,column=3,sticky="w")
        ttk.Label(self.main, textvariable=self.qos_var).grid(row=3,column=0,sticky="w",pady=(6,0))
        ttk.Label(self.main, textvariable=self.nsi_var).grid(row=3,column=1,sticky="w",pady=(6,0))
        ttk.Label(self.main, textvariable=self.dl_var).grid(row=3,column=2,sticky="w",pady=(6,0))
        ttk.Label(self.main, textvariable=self.ul_var).grid(row=3,column=3,sticky="w",pady=(6,0))
        # controls
        ttk.Label(self.main, text="Host:").grid(row=4,column=0,sticky="e",pady=(6,0))
        self.host_entry = ttk.Entry(self.main, width=18); self.host_entry.insert(0,PING_HOST); self.host_entry.grid(row=4,column=1,sticky="w")
        self.sound_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.main, text="Sound ON", variable=self.sound_var).grid(row=4,column=2,sticky="w")
        self.interval_spin = ttk.Spinbox(self.main, from_=1,to=10,width=5); self.interval_spin.set(str(int(PING_INTERVAL))); self.interval_spin.grid(row=4,column=3,sticky="w")
        # buttons for popups + speedtest
        ttk.Button(self.main, text="Routing View", command=self.show_routing).grid(row=5,column=0,pady=(8,0))
        ttk.Button(self.main, text="Congestion View", command=self.show_congestion).grid(row=5,column=1,pady=(8,0))
        ttk.Button(self.main, text="Queues View", command=self.show_queues).grid(row=5,column=2,pady=(8,0))
        ttk.Button(self.main, text="ARP / DNS", command=self.show_arp_dns).grid(row=5,column=3,pady=(8,0))
        ttk.Button(self.main, text="Run Speedtest Now", command=self.run_speedtest_now).grid(row=5,column=4,pady=(8,0))
        # start/stop
        self.running = False
        self.start_btn = ttk.Button(self.main, text="Start Monitoring", command=self.start_stop); self.start_btn.grid(row=6,column=0,columnspan=5,pady=(10,0))
        # internal state buffers
        self.history = collections.deque(maxlen=ROLLING_WINDOW)
        self.current_color = (20,20,20)
        self.target_color = self.current_color
        self.fade_speed = 0.12
        # advanced modules
        self.router_net = VirtualRouterNetwork(n=ROUTER_COUNT)
        self.scheduler = PacketScheduler()
        self.cong = CongestionSimulator()
        self.prev_arp = read_arp_table()
        self.prev_gateway_mac = self._guess_default_gateway_mac()
        self.dns_last = None
        # IMPORTANT: bandwidth variables updated by real speedtest async
        self.bandwidth_est = None  # fallback numeric Mbps (old estimator)
        self.real_dl = None
        self.real_ul = None
        self.real_ping = None
        self.last_speedtest_time = 0
        self.loss_percent = 0.0
        self.jitter = 0.0
        self.nsi = 1.0
        self.qos_class = "BE"
        # start animation
        self._animate()

    def _animate(self):
        r = int(self.current_color[0] + (self.target_color[0]-self.current_color[0])*self.fade_speed)
        g = int(self.current_color[1] + (self.target_color[1]-self.current_color[1])*self.fade_speed)
        b = int(self.current_color[2] + (self.target_color[2]-self.current_color[2])*self.fade_speed)
        self.current_color = (r,g,b)
        self.canvas.delete("glow")
        steps = 6
        for i in range(steps):
            factor = 1 - i/(steps+1)
            rr = int(r + (255-r)*(1-factor)*0.18)
            gg = int(g + (255-g)*(1-factor)*0.18)
            bb = int(b + (255-b)*(1-factor)*0.18)
            pad = 30 - i*4
            color = f"#{rr:02x}{gg:02x}{bb:02x}"
            self.canvas.create_oval(pad,pad,self.canvas_size-pad,self.canvas_size-pad, fill=color, outline="", tags="glow")
        self.canvas.tag_lower("glow")
        self.root.after(50, self._animate)

    def _guess_default_gateway_mac(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            parts = local_ip.split(".")
            parts[-1] = "1"
            gw = ".".join(parts)
            table = read_arp_table()
            return table.get(gw)
        except Exception:
            return None

    def start_stop(self):
        if not self.running:
            self.running = True
            self.start_btn.config(text="Stop Monitoring")
            self.history.clear()
            try:
                self.interval = max(0.2, float(self.interval_spin.get()))
            except:
                self.interval = PING_INTERVAL
            self.host = self.host_entry.get().strip() or PING_HOST
            self.run_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.run_thread.start()
            self.sched_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.sched_thread.start()
            # start periodic speedtest scheduler thread (doesn't block)
            self.speedtest_scheduler_thread = threading.Thread(target=self._speedtest_scheduler_loop, daemon=True)
            self.speedtest_scheduler_thread.start()
        else:
            self.running = False
            self.start_btn.config(text="Start Monitoring")
            self.status_var.set("Stopped")

    def _monitor_loop(self):
        prev_state = None
        route_instability_counter = 0
        while self.running:
            latency = ping_once(self.host, timeout=TIMEOUT_SECONDS)
            now = time.time()
            self.history.append((now, latency))
            values = [v for (_, v) in self.history]
            num_none = sum(1 for v in values if v is None)
            loss_pct = (num_none / max(1, len(values))) * 100
            lat_vals = [v for v in values if v is not None]
            mean_ping = statistics.mean(lat_vals) if lat_vals else None
            jitter = statistics.pstdev(lat_vals) if len(lat_vals) >= 2 else 0.0
            var_latency = statistics.pvariance(lat_vals) if len(lat_vals) >=2 else 0.0
            # occasionally run fallback estimator (kept only as fallback)
            if random.random() < 0.1:
                bw = estimate_bandwidth(self.host, probes=BANDWIDTH_PROBES)
                if bw: self.bandwidth_est = bw
            # DNS check occasionally
            if random.random() < 0.15:
                dns_t = dns_lookup_time()
                self.dns_last = dns_t
            # ARP check
            arp = read_arp_table()
            gw_mac = self._guess_default_gateway_mac()
            arp_attack = False
            if self.prev_gateway_mac and gw_mac and gw_mac != self.prev_gateway_mac:
                arp_attack = True
            self.prev_gateway_mac = gw_mac
            # Use real speedtest values if available, else fallback to estimator
            dl = self.real_dl if self.real_dl is not None else self.bandwidth_est
            ul = self.real_ul if self.real_ul is not None else None
            # NSI
            self.nsi = compute_nsi(jitter, loss_pct, var_latency, dl)
            # QoS classification
            self.qos_class = self._classify_qos(mean_ping, jitter, loss_pct)
            # congestion sim
            loss_prob = min(0.9, loss_pct/100.0)
            self.cong.tick_with_network(loss_prob)
            # routing perturbation when instability detected
            if loss_pct > 15 or jitter > 100:
                route_instability_counter += 1
                if route_instability_counter > 3 and random.random() < 0.4:
                    self.router_net.perturb_random_link()
                    route_instability_counter = 0
            else:
                route_instability_counter = 0
            # scheduler generation
            burst = loss_pct > 10
            self.scheduler.generate_packets(ef_rate=0.2, af_rate=0.5, be_rate=0.3, burst=burst)
            # classify state
            state = self._classify_state(mean_ping, loss_pct)
            if state != prev_state or arp_attack:
                prev_state = state
                if self.sound_var.get():
                    cfg = STATE_MAP.get(state, STATE_MAP["down"])
                    threading.Thread(target=play_tone, args=(cfg["freq"], cfg["dur"]), daemon=True).start()
                if arp_attack and self.sound_var.get():
                    threading.Thread(target=play_tone, args=(120, 600), daemon=True).start()
            # update UI
            self.root.after(0, self._update_ui, mean_ping, loss_pct, jitter, dl, ul, self.qos_class, self.nsi)
            time.sleep(self.interval)

    def _scheduler_loop(self):
        sched_alg = "WFQ"
        while True:
            if not self.running:
                time.sleep(0.2)
                continue
            if random.random() < 0.01:
                sched_alg = random.choice(["WFQ", "RR", "PRIORITY"])
            q, pkt = self.scheduler.schedule(algorithm=sched_alg)
            if pkt:
                time.sleep(min(0.05, pkt[1]/1500.0*0.01))
            else:
                time.sleep(0.05)

    def _speedtest_scheduler_loop(self):
        """
        Periodic automatic speedtest runner. Runs speedtest in background every SPEEDTEST_INTERVAL seconds.
        """
        while True:
            if self.running:
                # if last speedtest older than interval, schedule a new one
                if time.time() - self.last_speedtest_time > SPEEDTEST_INTERVAL:
                    self.last_speedtest_time = time.time()
                    # start async speedtest; callback updates real_dl/real_ul
                    run_real_speedtest_async(self._on_speedtest_result)
            time.sleep(5)

    def run_speedtest_now(self):
        # Manual run triggered by button: shows popup and runs
        w = tk.Toplevel(self.root); w.title("Running Speedtest")
        txt = tk.Text(w, width=60, height=12)
        txt.pack(padx=8, pady=8)
        txt.insert("end", "Starting speedtest (this may take 30-60s)...\n")
        def cb(res):
            txt.delete("1.0","end")
            if not res:
                txt.insert("end", "Speedtest failed. Ensure 'speedtest' CLI is installed and reachable.\n")
                return
            ping, dl, ul = res
            txt.insert("end", f"Ping: {ping:.1f} ms\nDownload: {dl:.2f} Mbps\nUpload: {ul:.2f} Mbps\n")
            # update main UI values
            self.real_ping = ping
            self.real_dl = dl
            self.real_ul = ul
            self.last_speedtest_time = time.time()
        # run async and pass callback
        run_real_speedtest_async(cb)

    def _on_speedtest_result(self, res):
        # callback from background speedtest
        if not res:
            return
        ping, dl, ul = res
        self.real_ping = ping
        self.real_dl = dl
        self.real_ul = ul
        # update some UI labels on main thread
        def upd():
            self.dl_var.set(f"DL: {dl:.2f} Mbps")
            self.ul_var.set(f"UL: {ul:.2f} Mbps")
            # if ping label not set, set it
            if self.ping_var.get().startswith("Ping: timeout"):
                self.ping_var.set(f"Ping: {ping:.1f} ms")
        self.root.after(0, upd)

    def _classify_state(self, mean_ping, loss_pct):
        if mean_ping is None and loss_pct > 50:
            return "down"
        if loss_pct > 20:
            return "loss"
        if mean_ping is None:
            return "down"
        if mean_ping < 50:
            return "excellent"
        if mean_ping < 150:
            return "moderate"
        return "congested"

    def _classify_qos(self, mean_ping, jitter, loss_pct):
        if loss_pct < 1 and (mean_ping is not None and mean_ping < 40) and jitter < 20:
            return "EF"
        if loss_pct < 5 and (mean_ping is not None and mean_ping < 100):
            return "AF"
        return "BE"

    def _update_ui(self, mean_ping, loss_pct, jitter, bw, ul, qos, nsi):
        self.ping_var.set(f"Ping: {'timeout' if mean_ping is None else f'{mean_ping:.1f} ms'}")
        self.loss_var.set(f"Loss: {loss_pct:.0f} %")
        self.jitter_var.set(f"Jitter: {jitter:.1f} ms")
        # Use real_dl if available (most reliable)
        display_bw = self.real_dl if self.real_dl is not None else (bw if bw is not None else 0)
        self.bw_var.set(f"Bandwidth: {display_bw:.2f} Mbps" if display_bw else "Bandwidth: - Mbps")
        self.dl_var.set(f"DL: {self.real_dl:.2f} Mbps" if self.real_dl else "DL: - Mbps")
        self.ul_var.set(f"UL: {self.real_ul:.2f} Mbps" if self.real_ul else "UL: - Mbps")
        self.qos_var.set(f"QoS: {qos}")
        self.nsi_var.set(f"NSI: {nsi:.2f}")
        base = {"EF": (0,220,120), "AF": (255,200,0), "BE": (220,40,40)}.get(qos, (100,100,200))
        brightness = 0.4 + 0.6 * nsi
        color = tuple(int(c * brightness) for c in base)
        self.target_color = color
        self.status_var.set(f"Status: {STATE_MAP[self._classify_state(mean_ping, loss_pct)]['label']}")

    # ------------------ Popups ------------------
    def show_routing(self):
        w = tk.Toplevel(self.root); w.title("Routing Simulation (node 0 view)")
        table = self.router_net.get_routing_table()
        txt = tk.Text(w, width=50, height=12); txt.pack(padx=8,pady=8)
        txt.insert("end", "Dest | NextHop | Cost\n"); txt.insert("end", "-"*30 + "\n")
        for dest, nh, cost in table:
            txt.insert("end", f"{dest:4} | {nh!s:7} | {cost}\n")
        ttk.Button(w, text="Perturb Link", command=lambda: (self.router_net.perturb_random_link(), messagebox.showinfo("Perturb","Link costs changed")) ).pack(pady=(0,8))

    def show_congestion(self):
        w = tk.Toplevel(self.root); w.title("Congestion Window (cwnd) Simulator")
        canvas = tk.Canvas(w, width=500, height=200, bg="#111"); canvas.pack(padx=8,pady=8)
        def draw():
            canvas.delete("all")
            hist = self.cong.history[-100:]
            if not hist:
                canvas.create_text(250,100, text="No data yet", fill="white")
            else:
                times = [t for (t,_) in hist]
                vals = [v for (_,v) in hist]
                if times:
                    tmin, tmax = min(times), max(times)
                else:
                    tmin, tmax = 0,1
                if tmax==tmin: tmax = tmin+1
                vmax = max(vals)*1.2 if vals else 1
                for i in range(len(hist)-1):
                    x1 = 10 + (times[i]-tmin)/(tmax-tmin)*480
                    y1 = 180 - (vals[i]/vmax)*160
                    x2 = 10 + (times[i+1]-tmin)/(tmax-tmin)*480
                    y2 = 180 - (vals[i+1]/vmax)*160
                    canvas.create_line(x1,y1,x2,y2, fill="lime", width=2)
                if vals:
                    canvas.create_text(80,12, text=f"cwnd {vals[-1]:.2f}", fill="white")
            w.after(500, draw)
        draw()

    def show_queues(self):
        w = tk.Toplevel(self.root); w.title("Packet Queues & Scheduler")
        lbl = ttk.Label(w, text="Scheduler: (internal WFQ/RR/PRIORITY switching demo)"); lbl.pack()
        tree = tk.Text(w, width=50, height=12); tree.pack(padx=8,pady=8)
        def refresh():
            tree.delete("1.0","end")
            counts = self.scheduler.pending_counts()
            tree.insert("end", f"Queue counts: EF={counts['EF']} AF={counts['AF']} BE={counts['BE']}\n\n")
            tree.insert("end", "Sample pending packet heads:\n")
            for q in ["EF","AF","BE"]:
                dq = list(self.scheduler.queues[q])[:5]
                tree.insert("end", f"{q}: {dq}\n")
            w.after(400, refresh)
        refresh()

    def show_arp_dns(self):
        w = tk.Toplevel(self.root); w.title("ARP & DNS Monitor")
        txt = tk.Text(w, width=60, height=20); txt.pack(padx=8,pady=8)
        def refresh():
            txt.delete("1.0","end")
            txt.insert("end", "ARP table snapshot (best-effort):\n")
            table = read_arp_table()
            for ip, mac in table.items():
                txt.insert("end", f"{ip} -> {mac}\n")
            txt.insert("end", "\nEstimated gateway MAC (heuristic):\n")
            txt.insert("end", f"{self.prev_gateway_mac}\n")
            txt.insert("end", "\nDNS last lookup time (ms):\n")
            txt.insert("end", f"{'timeout' if self.dns_last is None else f'{self.dns_last:.1f} ms'}\n")
            w.after(1000, refresh)
        refresh()

# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except:
        pass
    app = NetworkSensoryApp(root)
    root.mainloop()
