import argparse
import collections
import os
import platform
import queue
import threading
import time
import traceback

import cv2
import numpy as np

# Model paths and camera configuration
WEIGHTS_PATH = "with9.pt"
DATA_YAML = "data.yaml"
CAM_INDEX = 0

# Logging
VERBOSE = True

def vlog(msg):
    if VERBOSE:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

# Class names
def load_names(yaml_path):
    names = {}
    in_names = False
    try:
        with open(yaml_path) as f:
            for raw in f:
                line = raw.split("#", 1)[0].rstrip()
                if not line.strip():
                    continue
                if line.startswith("names:"):
                    in_names = True
                    continue
                if in_names:
                    if not line[0].isspace():
                        break
                    key, _, val = line.strip().partition(":")
                    if key.isdigit() and val.strip():
                        names[int(key)] = val.strip().strip("'\"")
    except OSError:
        pass
    return names

# Device selection
def pick_device(requested):
    if requested != "auto":
        return requested
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# FPS tracking and shared state
class FPSCounter:

    def __init__(self, window=30):
        self._t = collections.deque(maxlen=window)
        self._lock = threading.Lock()
        self.total = 0
        self.start = time.perf_counter()

    def tick(self):
        with self._lock:
            self._t.append(time.perf_counter())
            self.total += 1

    @property
    def fps(self):
        with self._lock:
            if len(self._t) < 2:
                return 0.0
            return (len(self._t) - 1) / (self._t[-1] - self._t[0] + 1e-9)

    @property
    def avg_fps(self):
        return self.total / max(time.perf_counter() - self.start, 1e-9)

class Shared:

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._dets = []
        self._latency = 0.0

    def set_frame(self, frame):
        with self._lock:
            self._frame = frame

    def get_frame(self):
        with self._lock:
            return self._frame

    def set_dets(self, dets, latency):
        with self._lock:
            self._dets = dets
            self._latency = latency

    def get_dets(self):
        with self._lock:
            return self._dets, self._latency

def put_latest(q, item):
    dropped = False
    while True:
        try:
            q.put_nowait(item)
            return dropped
        except queue.Full:
            try:
                q.get_nowait()
                dropped = True
            except queue.Empty:
                pass

# Detection drawing and label helpers
GREEN, RED, WHITE, GRAY = (0, 200, 0), (0, 0, 230), (255, 255, 255), (170, 170, 170)
ALERT_TXT = (90, 90, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

def norm(label):
    return label.lower().replace("_", "-")

def is_violation(label):
    k = norm(label)
    return k.startswith("without") or "cockroach" in k

def is_unused(label):
    return norm(label).startswith("unused")

def warn_text(label):
    k = norm(label)
    if k.startswith("without-"):
        return f"No {k[len('without-'):]}"
    return f"{k.replace('-', ' ').capitalize()} detected"

def pretty(label):
    return norm(label).replace("-", " ").capitalize()

def draw_dets(frame, dets):
    for x, y, bw, bh, label, score in dets:
        color = RED if is_violation(label) else GREEN
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(
            frame, f"{label} {score:.2f}", (x, max(y - 5, 12)),
            FONT, 0.5, color, 1,
        )

# Dashboard
class Dashboard:

    def __init__(self, width=320, hold=1.5, log_size=8):
        self.width = width
        self.hold = hold
        self.counts = {}
        self.last_seen = {}
        self.log = collections.deque(maxlen=log_size)
        self.total_alerts = 0

    def update(self, dets, now):
        counts = collections.Counter(
            norm(d[4]) for d in dets if not is_unused(d[4])
        )
        for key, n in counts.items():
            if not is_violation(key):
                continue
            prev = self.last_seen.get(key)
            if prev is None or now - prev[0] > self.hold:
                text = warn_text(key)
                self.log.appendleft((time.strftime("%H:%M:%S"), text))
                self.total_alerts += 1
                vlog(f"[ALERT] {text}")
            self.last_seen[key] = (now, n)
        self.counts = counts

    def active_violations(self, now):
        return {k: v[1] for k, v in self.last_seen.items() if now - v[0] <= self.hold}

    def render(self, height, now, stats):
        w = self.width
        panel = np.full((height, w, 3), 32, np.uint8)
        violations = self.active_violations(now)
        footer_top = height - 52

        cv2.putText(panel, "DETECTION DASHBOARD", (12, 24), FONT, 0.6, WHITE, 2, cv2.LINE_AA)
        cv2.line(panel, (0, 36), (w, 36), (80, 80, 80), 1)

        if violations:
            color = RED if int(now * 2) % 2 == 0 else (0, 0, 140)
            msg = f"WARNING: {len(violations)} violation(s)"
        else:
            color, msg = (0, 150, 0), "ALL CLEAR"
        cv2.rectangle(panel, (10, 46), (w - 10, 88), color, -1)
        cv2.putText(panel, msg, (20, 73), FONT, 0.6, WHITE, 2, cv2.LINE_AA)

        y = 112
        cv2.putText(panel, "DETECTIONS", (12, y), FONT, 0.5, GRAY, 1, cv2.LINE_AA)
        rows = dict(self.counts)
        for key, n in violations.items():
            rows.setdefault(key, n)
        if not rows:
            y += 22
            cv2.putText(panel, "nothing detected", (12, y), FONT, 0.5, GRAY, 1, cv2.LINE_AA)
        for key in sorted(rows, key=lambda k: (not is_violation(k), k)):
            y += 22
            if y > footer_top - 60:
                break
            bad = is_violation(key)
            cv2.rectangle(panel, (12, y - 11), (22, y - 1), RED if bad else GREEN, -1)
            txt_color = ALERT_TXT if bad else WHITE
            cv2.putText(panel, warn_text(key) if bad else pretty(key), (32, y),
                        FONT, 0.55, txt_color, 1, cv2.LINE_AA)
            cv2.putText(panel, f"x{rows[key]}", (w - 52, y),
                        FONT, 0.55, txt_color, 2, cv2.LINE_AA)

        y += 30
        if y < footer_top:
            cv2.putText(panel, f"RECENT ALERTS ({self.total_alerts})", (12, y),
                        FONT, 0.5, GRAY, 1, cv2.LINE_AA)
        if not self.log:
            y += 20
            if y < footer_top:
                cv2.putText(panel, "no alerts yet", (12, y), FONT, 0.45, GRAY, 1, cv2.LINE_AA)
        for stamp, text in self.log:
            y += 19
            if y > footer_top - 4:
                break
            cv2.putText(panel, f"{stamp}  {text}", (12, y), FONT, 0.45, ALERT_TXT, 1, cv2.LINE_AA)

        cam, inf, disp, latency = stats
        cv2.line(panel, (0, footer_top), (w, footer_top), (80, 80, 80), 1)
        cv2.putText(panel, f"Cam {cam:.1f} | Infer {inf:.1f} FPS", (12, height - 30),
                    FONT, 0.5, GRAY, 1, cv2.LINE_AA)
        cv2.putText(panel, f"Display {disp:.1f} FPS | {latency:.0f} ms", (12, height - 10),
                    FONT, 0.5, GRAY, 1, cv2.LINE_AA)
        return panel

# YOLO model loading and inference
class Detector:
    def __init__(self, weights, names_override, device, imgsz, half):
        from ultralytics import YOLO

        vlog(f"[model] loading {weights}")
        self.model = YOLO(weights)
        self.device = pick_device(device)
        self.imgsz = imgsz

        self.half = bool(half) and self.device.startswith("cuda")

        self.names = dict(self.model.names)
        if names_override:
            self.names.update(names_override)
        vlog(f"[model] device={self.device} imgsz={self.imgsz} half={self.half} "
             f"classes={len(self.names)}")

        dummy = np.zeros((imgsz, imgsz, 3), np.uint8)
        for _ in range(2):
            self.predict(dummy, 0.25, 0.45)
        vlog("[model] ready")

    def predict(self, frame, conf, iou):
        res = self.model.predict(
            frame, imgsz=self.imgsz, conf=conf, iou=iou,
            device=self.device, half=self.half, verbose=False,
        )[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()

        dets = []
        for (x1, y1, x2, y2), c, s in zip(xyxy, cls, scores):
            label = self.names.get(int(c), str(c))
            dets.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), label, float(s)))
        return dets

# Capture and inference threads
def capture_loop(cap, q_out, shared, stop, cam_fps, is_file, src_fps):
    n, dropped = 0, 0
    vlog("[capture] started")
    try:
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok:
                vlog("[capture] no more frames" if is_file else "[capture] camera read failed")
                break
            cam_fps.tick()
            n += 1
            shared.set_frame(frame)
            if put_latest(q_out, (frame, time.perf_counter())):
                dropped += 1
            if is_file:
                time.sleep(1.0 / src_fps)
            if n % 100 == 0:
                vlog(f"[capture] {n} frames read, {dropped} not inferred "
                     f"(inference slower than camera)")
    except Exception:
        traceback.print_exc()
    finally:
        put_latest(q_out, None)
        stop.set()
        vlog(f"[capture] stopped: {n} frames read, {dropped} not inferred")

def infer_loop(det, q_in, shared, stop, cam_fps, inf_fps, args):
    n = 0
    vlog("[infer] started")
    try:
        while not stop.is_set():
            try:
                item = q_in.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            frame, t_cap = item

            t0 = time.perf_counter()
            dets = det.predict(frame, args.conf, args.iou)
            t_inf = (time.perf_counter() - t0) * 1000

            inf_fps.tick()
            n += 1
            latency = (time.perf_counter() - t_cap) * 1000
            shared.set_dets(dets, latency)

            if VERBOSE and n % args.log_every == 0:
                vlog(
                    f"[infer] frame {n:5d} | model {t_inf:6.1f} ms | "
                    f"dets {len(dets):3d} | latency {latency:6.1f} ms | "
                    f"cam {cam_fps.fps:5.1f} fps | infer {inf_fps.fps:5.1f} fps"
                )
    except Exception:
        traceback.print_exc()
        stop.set()
    finally:
        vlog(f"[infer] stopped after {n} frames")

# Video or webcam source
def open_source(args):
    if args.source:
        if not os.path.exists(args.source):
            raise SystemExit(f"Video file not found: {args.source}")
        cap = cv2.VideoCapture(args.source)
        vlog(f"[source] video file {args.source}")
        is_file = True
    else:

        system = platform.system()
        if system == "Windows":
            cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
        elif system == "Darwin":
            cap = cv2.VideoCapture(args.cam, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
        vlog(f"[source] webcam index {args.cam}")

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.cam_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        is_file = False

    if not cap.isOpened():
        raise SystemExit(f"Could not open {'video' if is_file else 'camera'}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    vlog(f"[source] opened: {W}x{H} @ {fps:.1f} fps (as reported by driver)")
    return cap, is_file, fps

# Main program
def main():
    global VERBOSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=WEIGHTS_PATH, help="path to the .pt model")
    ap.add_argument("--data", default=DATA_YAML,
                    help="optional data.yaml; its class names override the ones in the .pt")
    ap.add_argument("--source", default=None, help="video file (default: webcam)")
    ap.add_argument("--cam", type=int, default=CAM_INDEX, help="webcam index")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--cam-fps", type=int, default=30)
    ap.add_argument("--display-fps", type=float, default=None,
                    help="rate the preview window is refreshed at (defaults to source FPS)")
    ap.add_argument("--device", default="auto", help="auto, cuda, cuda:0, mps or cpu")
    ap.add_argument("--imgsz", type=int, default=640, help="inference size")
    ap.add_argument("--no-half", action="store_true", help="disable FP16 on CUDA")
    ap.add_argument("--warn-hold", type=float, default=1.5,
                    help="seconds a warning stays up after the violation was last seen")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--no-show", action="store_true", help="no preview window")
    ap.add_argument("--quiet", action="store_true", help="turn verbose logging off")
    ap.add_argument("--log-every", type=int, default=10,
                    help="print per-frame log every N frames")

    args, _unknown = ap.parse_known_args()
    VERBOSE = not args.quiet
    show = not args.no_show

    if not os.path.exists(args.weights):
        raise SystemExit(f"Weights not found: {args.weights} (use --weights path/to/best.pt)")

    names_override = load_names(args.data)
    if names_override:
        vlog(f"[names] loaded {len(names_override)} classes from {args.data}")
    else:
        vlog("[names] no data.yaml found, using the names stored in the .pt")

    det = Detector(args.weights, names_override, args.device, args.imgsz, not args.no_half)
    cap, is_file, src_fps = open_source(args)

    q_cap = queue.Queue(maxsize=1)
    shared = Shared()
    stop = threading.Event()
    cam_fps, inf_fps, disp_fps = FPSCounter(), FPSCounter(), FPSCounter()

    threads = [
        threading.Thread(target=capture_loop, name="capture", daemon=True,
                         args=(cap, q_cap, shared, stop, cam_fps, is_file, src_fps)),
        threading.Thread(target=infer_loop, name="infer", daemon=True,
                         args=(det, q_cap, shared, stop, cam_fps, inf_fps, args)),
    ]
    for t in threads:
        t.start()

    period = 1.0 / max(args.display_fps if args.display_fps is not None else src_fps, 1.0)
    dash = Dashboard(hold=args.warn_hold)
    next_t = time.perf_counter()
    last_stats = next_t
    try:
        while not stop.is_set():
            frame = shared.get_frame()
            if frame is not None:
                disp = frame.copy()
                dets, latency = shared.get_dets()
                now_t = time.perf_counter()
                dash.update(dets, now_t)
                draw_dets(disp, dets)
                disp_fps.tick()
                if show:
                    panel = dash.render(
                        disp.shape[0], now_t,
                        (cam_fps.fps, inf_fps.fps, disp_fps.fps, latency),
                    )
                    cv2.imshow("yolo11", np.hstack((disp, panel)))

            now = time.perf_counter()
            if VERBOSE and now - last_stats >= 1.0:
                vlog(f"[stats] camera {cam_fps.fps:5.1f} fps | "
                     f"infer {inf_fps.fps:5.1f} fps | "
                     f"display {disp_fps.fps:5.1f} fps")
                last_stats = now

            next_t += period
            delay = next_t - time.perf_counter()
            if delay < 0:
                next_t = time.perf_counter()
                delay = 0.0
            if show:
                if cv2.waitKey(max(1, int(delay * 1000))) & 0xFF in (ord("q"), 27):
                    vlog("[main] quit key pressed")
                    break
            else:
                time.sleep(delay)
    except KeyboardInterrupt:
        vlog("[main] Ctrl+C")
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=2)
        cap.release()
        cv2.destroyAllWindows()
        for _ in range(4):
            cv2.waitKey(1)

    print(
        f"\nSummary: camera {cam_fps.total} frames ({cam_fps.avg_fps:.1f} FPS avg) | "
        f"inference {inf_fps.total} frames ({inf_fps.avg_fps:.1f} FPS avg) | "
        f"display {disp_fps.total} frames ({disp_fps.avg_fps:.1f} FPS avg)"
    )

if __name__ == "__main__":
    main()
