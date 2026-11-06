"""Run a YOLO11n TensorRT engine (built with trtexec) on a webcam.

Pipeline: every stage is its own thread, so the stages overlap instead of
running one after another.

    capture -> preprocess -> tensorrt -> postprocess  (detections)
        \\                                                |
         `--------> display (main thread, paced at 30 FPS) <--'

The display is decoupled from inference: it shows the newest camera frame at
a steady rate (default 30 FPS) and draws the most recent detections on it.
If inference is slower than the display, boxes lag by a frame or two but the
video stays smooth.

A dashboard panel on the right of the video shows a warning banner, live
counts per class and a log of recent alerts. Violations (shown in red) are the
"without-..." classes (missing mask / gloves / hairnet) and cockroaches.

Usage:
    python3 yolo11_trt_video_fast.py                    # webcam 0, 30 FPS display
    python3 yolo11_trt_video_fast.py --cam 1            # different camera
    python3 yolo11_trt_video_fast.py --display-fps 30   # display rate
    python3 yolo11_trt_video_fast.py --warn-hold 2      # keep a warning up 2 s after last seen
    python3 yolo11_trt_video_fast.py --quiet            # less logging
    python3 yolo11_trt_video_fast.py --no-show          # headless
Press 'q' or ESC in the window to quit.
"""
import argparse
import collections
import ctypes
import queue
import threading
import time
import traceback

import cv2
import numpy as np
import tensorrt as trt

# ---- EDIT THESE ----
ENGINE_PATH = "model_fp16.engine"
DATA_YAML = "data.yaml"
CAM_INDEX = 0
DISPLAY_FPS = 30
# --------------------

# Fallback class names (mirrors data.yaml) used if the yaml file can't be read.
DEFAULT_NAMES = {
    0: "cockroach",
    1: "without_hairnet",
    2: "gloves",
    3: "hairnet",
    4: "mask",
    5: "without-gloves",
    6: "without_mask",
    7: "mask",
    8: "without-mask",
}


VERBOSE = True


def load_names(yaml_path):
    """Read the `names:` block from data.yaml -> {id: name}.
    Tiny parser so PyYAML isn't required on the Jetson."""
    names = {}
    in_names = False
    try:
        with open(yaml_path) as f:
            for raw in f:
                line = raw.split("#", 1)[0].rstrip()   # drop comments
                if not line.strip():
                    continue
                if line.startswith("names:"):
                    in_names = True
                    continue
                if in_names:
                    if not line[0].isspace():          # next top-level key
                        break
                    key, _, val = line.strip().partition(":")
                    if key.isdigit() and val.strip():
                        names[int(key)] = val.strip().strip("'\"")
    except OSError:
        pass
    return names


def vlog(msg):
    if VERBOSE:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# ---- Minimal CUDA runtime wrapper (ships with JetPack, no pycuda) ----
_cudart = None
for _name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
    try:
        _cudart = ctypes.CDLL(_name)
        break
    except OSError:
        continue
if _cudart is None:
    raise SystemExit(
        "libcudart not found. Try: export LD_LIBRARY_PATH=/usr/local/cuda/lib64"
    )

_cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_cudart.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
_cudart.cudaMemcpyAsync.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p,
]
H2D, D2H = 1, 2


def _check(err):
    if err != 0:
        raise RuntimeError(f"CUDA runtime error code {err}")


def cuda_malloc(nbytes):
    ptr = ctypes.c_void_p()
    _check(_cudart.cudaMalloc(ctypes.byref(ptr), nbytes))
    return ptr


def cuda_stream():
    s = ctypes.c_void_p()
    _check(_cudart.cudaStreamCreate(ctypes.byref(s)))
    return s


def memcpy(dst, src, nbytes, kind, stream):
    dst = getattr(dst, "value", dst)
    src = getattr(src, "value", src)
    _check(_cudart.cudaMemcpyAsync(dst, src, nbytes, kind, stream))


def sync(stream):
    _check(_cudart.cudaStreamSynchronize(stream))


# ---- Helpers ----
class FPSCounter:
    """Thread-safe sliding-window FPS counter."""

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
    """Latest camera frame and latest detections, shared with the display loop.

    Frames are never modified after capture (the display draws on a copy), so
    the pipeline and the display can use the same frame safely.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._dets = []          # [(x, y, w, h, label, score), ...]
        self._latency = 0.0      # ms, camera -> detections ready

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
    """Put item, discarding the oldest entry if the queue is full, so latency
    stays low on a live camera. Returns True if something was dropped."""
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


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh))
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


GREEN, RED, WHITE, GRAY = (0, 200, 0), (0, 0, 230), (255, 255, 255), (170, 170, 170)
ALERT_TXT = (90, 90, 255)   # light red, readable on the dark panel
FONT = cv2.FONT_HERSHEY_SIMPLEX


def norm(label):
    """'without_mask' and 'without-mask' are the same thing."""
    return label.lower().replace("_", "-")


def is_violation(label):
    """Classes that raise a warning: missing PPE and pests."""
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


# ---- Dashboard (side panel) ----
class Dashboard:
    """Warning banner, live counts per class and a log of recent alerts."""

    def __init__(self, width=320, hold=1.5, log_size=8):
        self.width = width
        self.hold = hold               # seconds a warning stays up after last seen
        self.counts = {}               # class key -> count in the latest detections
        self.last_seen = {}            # violation key -> (time, count)
        self.log = collections.deque(maxlen=log_size)   # (time string, text), newest first
        self.total_alerts = 0

    def update(self, dets, now):
        counts = collections.Counter(
            norm(d[4]) for d in dets if not is_unused(d[4])
        )
        for key, n in counts.items():
            if not is_violation(key):
                continue
            prev = self.last_seen.get(key)
            if prev is None or now - prev[0] > self.hold:   # new alert, not a flicker
                text = warn_text(key)
                self.log.appendleft((time.strftime("%H:%M:%S"), text))
                self.total_alerts += 1
                vlog(f"[ALERT] {text}")
            self.last_seen[key] = (now, n)
        self.counts = counts

    def active_violations(self, now):
        return {k: v[1] for k, v in self.last_seen.items() if now - v[0] <= self.hold}

    def render(self, height, now, stats):
        """Returns a (height, width, 3) image. stats = (cam, infer, display, latency_ms)."""
        w = self.width
        panel = np.full((height, w, 3), 32, np.uint8)
        violations = self.active_violations(now)
        footer_top = height - 52

        # title
        cv2.putText(panel, "DETECTION DASHBOARD", (12, 24), FONT, 0.6, WHITE, 2, cv2.LINE_AA)
        cv2.line(panel, (0, 36), (w, 36), (80, 80, 80), 1)

        # status banner (blinks while a warning is active)
        if violations:
            color = RED if int(now * 2) % 2 == 0 else (0, 0, 140)
            msg = f"WARNING: {len(violations)} violation(s)"
        else:
            color, msg = (0, 150, 0), "ALL CLEAR"
        cv2.rectangle(panel, (10, 46), (w - 10, 88), color, -1)
        cv2.putText(panel, msg, (20, 73), FONT, 0.6, WHITE, 2, cv2.LINE_AA)

        # live counts per class (violations first, red)
        y = 112
        cv2.putText(panel, "DETECTIONS", (12, y), FONT, 0.5, GRAY, 1, cv2.LINE_AA)
        rows = dict(self.counts)
        for key, n in violations.items():      # keep a warning's row while it is held
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

        # recent alerts
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

        # footer: speeds
        cam, inf, disp, latency = stats
        cv2.line(panel, (0, footer_top), (w, footer_top), (80, 80, 80), 1)
        cv2.putText(panel, f"Cam {cam:.1f} | Infer {inf:.1f} FPS", (12, height - 30),
                    FONT, 0.5, GRAY, 1, cv2.LINE_AA)
        cv2.putText(panel, f"Display {disp:.1f} FPS | {latency:.0f} ms", (12, height - 10),
                    FONT, 0.5, GRAY, 1, cv2.LINE_AA)
        return panel


# ---- TensorRT detector (split into three steps so they can run in 3 threads) ----
class Detector:
    def __init__(self, engine_path, names=None):
        self.names = names or DEFAULT_NAMES
        level = trt.Logger.INFO if VERBOSE else trt.Logger.WARNING
        logger = trt.Logger(level)
        vlog(f"[engine] loading {engine_path}")
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise SystemExit("Failed to deserialize engine")
        self.ctx = self.engine.create_execution_context()

        inp = out = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            vlog(f"[engine] tensor {i}: {name} {mode.name} {shape} {dtype}")
            if mode == trt.TensorIOMode.INPUT:
                inp = name
            else:
                out = name
        self.inp, self.out = inp, out

        in_shape = tuple(self.engine.get_tensor_shape(inp))    # (1, 3, 640, 640)
        out_shape = tuple(self.engine.get_tensor_shape(out))   # (1, 4+nc, 8400)
        self.size = in_shape[-1]
        self.in_dtype = trt.nptype(self.engine.get_tensor_dtype(inp))
        out_dtype = trt.nptype(self.engine.get_tensor_dtype(out))

        self.in_nbytes = int(np.prod(in_shape)) * np.dtype(self.in_dtype).itemsize
        self.h_out = np.zeros(out_shape, out_dtype)
        self.d_in = cuda_malloc(self.in_nbytes)
        self.d_out = cuda_malloc(self.h_out.nbytes)
        self.ctx.set_tensor_address(inp, self.d_in.value)
        self.ctx.set_tensor_address(out, self.d_out.value)
        self.stream = cuda_stream()
        vlog(f"[engine] ready: input {in_shape} {self.in_dtype.__name__}, "
             f"output {out_shape} {out_dtype.__name__}")

    # Stage 1 (CPU): letterbox + normalise. blobFromImage returns float32 NCHW in
    # one native call; the old `/255.0` created a float64 array, which is slow
    # on the Jetson's ARM cores.
    def preprocess(self, frame):
        img, r, dx, dy = letterbox(frame, self.size)
        blob = cv2.dnn.blobFromImage(img, 1.0 / 255.0, swapRB=True)
        if blob.dtype != self.in_dtype:
            blob = blob.astype(self.in_dtype)
        return np.ascontiguousarray(blob), r, dx, dy

    # Stage 2 (GPU): copy in, run, copy out.
    def run(self, blob):
        memcpy(self.d_in, blob.ctypes.data, self.in_nbytes, H2D, self.stream)
        self.ctx.execute_async_v3(self.stream.value)
        memcpy(self.h_out.ctypes.data, self.d_out, self.h_out.nbytes, D2H, self.stream)
        sync(self.stream)
        return self.h_out.copy()   # copy so the next frame can reuse h_out

    # Stage 3 (CPU): filter + NMS. Returns [(x, y, w, h, label, score), ...]
    # in original-frame pixels. Drawing is done by the display loop.
    def postprocess(self, out, r, dx, dy, conf, iou):
        p = out[0]                                   # (4+nc, 8400)
        scores = p[4:].max(0)
        keep = np.flatnonzero(scores > conf)         # filter before transposing
        if keep.size == 0:
            return []
        pred = p[:, keep].T.astype(np.float32)       # (n, 4+nc)
        scores = scores[keep].astype(np.float32)
        ids = pred[:, 4:].argmax(1)

        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        x1 = (cx - w / 2 - dx) / r
        y1 = (cy - h / 2 - dy) / r
        boxes = np.stack([x1, y1, w / r, h / r], 1)
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf, iou)

        dets = []
        for i in np.array(idx).flatten():
            x, y, bw, bh = boxes[i].astype(int)
            label = self.names.get(int(ids[i]), str(ids[i]))
            dets.append((int(x), int(y), int(bw), int(bh), label, float(scores[i])))
        return dets


# ---- Thread bodies ----
def capture_loop(cap, q_out, shared, stop, cam_fps):
    n, dropped = 0, 0
    vlog("[capture] started")
    try:
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok:
                vlog("[capture] camera read failed")
                break
            cam_fps.tick()
            n += 1
            shared.set_frame(frame)                  # for the display loop
            if put_latest(q_out, (frame, time.perf_counter())):
                dropped += 1                         # for the inference pipeline
            if n % 100 == 0:
                vlog(f"[capture] {n} frames read, {dropped} not inferred "
                     f"(inference slower than camera)")
    except Exception:
        traceback.print_exc()
    finally:
        put_latest(q_out, None)  # sentinel
        stop.set()               # camera gone -> shut everything down
        vlog(f"[capture] stopped: {n} frames read, {dropped} not inferred")


def pre_loop(det, q_in, q_out, stop):
    vlog("[pre] started")
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
            blob, r, dx, dy = det.preprocess(frame)
            t_pre = (time.perf_counter() - t0) * 1000
            put_latest(q_out, (blob, r, dx, dy, t_cap, t_pre))
    except Exception:
        traceback.print_exc()
        stop.set()
    finally:
        put_latest(q_out, None)
        vlog("[pre] stopped")


def trt_loop(det, q_in, q_out, stop):
    vlog("[trt] started")
    try:
        while not stop.is_set():
            try:
                item = q_in.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            blob, r, dx, dy, t_cap, t_pre = item
            t0 = time.perf_counter()
            out = det.run(blob)
            t_trt = (time.perf_counter() - t0) * 1000
            put_latest(q_out, (out, r, dx, dy, t_cap, t_pre, t_trt))
    except Exception:
        traceback.print_exc()
        stop.set()
    finally:
        put_latest(q_out, None)
        vlog("[trt] stopped")


def post_loop(det, q_in, shared, stop, cam_fps, inf_fps, args):
    n = 0
    vlog("[post] started")
    try:
        while not stop.is_set():
            try:
                item = q_in.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            out, r, dx, dy, t_cap, t_pre, t_trt = item

            t0 = time.perf_counter()
            dets = det.postprocess(out, r, dx, dy, args.conf, args.iou)
            t_post = (time.perf_counter() - t0) * 1000

            inf_fps.tick()
            n += 1
            latency = (time.perf_counter() - t_cap) * 1000
            shared.set_dets(dets, latency)

            if VERBOSE and n % args.log_every == 0:
                vlog(
                    f"[post] frame {n:5d} | pre {t_pre:5.1f} ms | "
                    f"trt {t_trt:5.1f} ms | post {t_post:5.1f} ms | "
                    f"dets {len(dets):3d} | latency {latency:6.1f} ms | "
                    f"cam {cam_fps.fps:5.1f} fps | infer {inf_fps.fps:5.1f} fps"
                )
    except Exception:
        traceback.print_exc()
        stop.set()
    finally:
        vlog(f"[post] stopped after {n} frames")


def open_webcam(args):
    cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
    vlog(f"[source] webcam index {args.cam}")
    # MJPG gives much higher FPS than raw YUYV on most USB webcams
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.cam_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.cam}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    vlog(f"[source] opened: {W}x{H} @ {fps:.1f} fps (as reported by driver)")
    return cap


def main():
    global VERBOSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=ENGINE_PATH)
    ap.add_argument("--data", default=DATA_YAML, help="data.yaml with class names")
    ap.add_argument("--cam", type=int, default=CAM_INDEX, help="webcam index")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--cam-fps", type=int, default=30)
    ap.add_argument("--display-fps", type=float, default=DISPLAY_FPS,
                    help="rate the preview window is refreshed at")
    ap.add_argument("--warn-hold", type=float, default=1.5,
                    help="seconds a warning stays up after the violation was last seen")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--no-show", action="store_true", help="no preview window")
    ap.add_argument("--quiet", action="store_true", help="turn verbose logging off")
    ap.add_argument("--log-every", type=int, default=10,
                    help="print per-frame log every N frames")
    args = ap.parse_args()
    VERBOSE = not args.quiet
    show = not args.no_show

    names = load_names(args.data)
    if names:
        vlog(f"[names] loaded {len(names)} classes from {args.data}")
    else:
        vlog(f"[names] could not read {args.data}, using built-in defaults")
    det = Detector(args.engine, names)
    cap = open_webcam(args)

    # Small queues = low latency; the oldest frame is dropped when a queue is full.
    q_cap = queue.Queue(maxsize=1)    # capture     -> preprocess
    q_pre = queue.Queue(maxsize=2)    # preprocess  -> tensorrt
    q_trt = queue.Queue(maxsize=2)    # tensorrt    -> postprocess
    shared = Shared()                 # camera frame + detections -> display
    stop = threading.Event()
    cam_fps, inf_fps, disp_fps = FPSCounter(), FPSCounter(), FPSCounter()

    threads = [
        threading.Thread(target=capture_loop, name="capture", daemon=True,
                         args=(cap, q_cap, shared, stop, cam_fps)),
        threading.Thread(target=pre_loop, name="pre", daemon=True,
                         args=(det, q_cap, q_pre, stop)),
        threading.Thread(target=trt_loop, name="trt", daemon=True,
                         args=(det, q_pre, q_trt, stop)),
        threading.Thread(target=post_loop, name="post", daemon=True,
                         args=(det, q_trt, shared, stop, cam_fps, inf_fps, args)),
    ]
    for t in threads:
        t.start()

    # ---- Display loop (main thread), paced at --display-fps ----
    period = 1.0 / max(args.display_fps, 1.0)
    dash = Dashboard(hold=args.warn_hold)
    next_t = time.perf_counter()
    last_stats = next_t
    try:
        while not stop.is_set():
            frame = shared.get_frame()
            if frame is not None:
                disp = frame.copy()                  # never draw on the pipeline's frame
                dets, latency = shared.get_dets()
                now_t = time.perf_counter()
                dash.update(dets, now_t)             # also prints [ALERT] lines when headless
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

            # Sleep until the next display tick (waitKey also services the window).
            next_t += period
            delay = next_t - time.perf_counter()
            if delay < 0:                            # fell behind: don't try to catch up
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

    print(
        f"\nSummary: camera {cam_fps.total} frames ({cam_fps.avg_fps:.1f} FPS avg) | "
        f"inference {inf_fps.total} frames ({inf_fps.avg_fps:.1f} FPS avg) | "
        f"display {disp_fps.total} frames ({disp_fps.avg_fps:.1f} FPS avg)"
    )


if __name__ == "__main__":
    main()
