# Latency Benchmark Analysis

This document explains the latency benchmarking process for the Android emulator RL environment, the knowledge gained through iterative optimization, and how to interpret the results.

## Background

Our RL agent interacts with a game running on an Android emulator on a headless Linux workstation. Each RL "step" requires:

1. **Action**: Send a touch event to the emulator
2. **Observation**: Capture the screen to see the result

The round-trip latency of this cycle directly determines the maximum training speed (steps/sec). We benchmarked two observation methods to find the optimal pipeline.

## Architecture

```
┌─────────────┐     scrcpy control socket      ┌──────────────────┐
│  Python RL  │ ──── tap() (5ms) ────────────►  │                  │
│   Agent     │                                 │  Android Emulator│
│             │ ◄─── observation ────────────── │  (headless)      │
└─────────────┘    (Method A or B)              └──────────────────┘
```

**Method A — H.264 Stream (scrcpy video)**
```
Android renders → MediaCodec H.264 encode → TCP → PyAV decode → get_frame()
```

**Method B — ADB Screencap**
```
Android renders → adb exec-out screencap → raw framebuffer via pipe → numpy array
```

## Benchmark Setup

- **Emulator**: Pixel 5 AVD, API 31, `gpu=swiftshader_indirect` (CPU-based software rendering)
- **VM display**: `adb shell wm size 540x1170` (reduced from native 1080×2340)
- **Test app**: [Tap Counter](https://samvlu.github.io/web-04-tap-counter/) running in Chrome
- **Method**: Tap the counter → poll until the counter display region changes → measure elapsed time
- **Touch injection**: scrcpy control socket binary protocol (~5ms per tap)

### Coordinate Systems

Three different resolutions coexist in this system:

| Component | Resolution | Purpose |
|---|---|---|
| VM display (`wm size`) | 540×1170 | Emulator's logical framebuffer |
| screencap output | 540×1170 | Always matches VM display (raw RGBA, 2.5MB) |
| scrcpy H.264 stream | 268×584 | Downscaled by `max_size=585` for encoding |
| Web server view | 220×480 | Downscaled by `max_size=480` for developer viewing |

Coordinates measured from the web server (220×480) are auto-scaled at runtime to match whichever resolution the benchmark is using.

## Results

### Final Benchmark: H.264 Stream (900 cycles)

```
======================================================================
  H.264 BENCHMARK RESULTS  (900 successful, 0 timeouts)
  scrcpy max_size=585  poll_interval=50ms  threshold=2.0
======================================================================

  Round-trip (tap → H.264 frame shows change):
    mean =   104.4 ms
    std  =    13.7 ms
    min  =    56.5 ms
    max  =   128.0 ms
    p50  =   108.1 ms
    p95  =   108.7 ms

  Polls until change:
    mean =     2.9
    min  =       2
    max  =       3
    avg poll time = 50.2 ms (sleep 50ms + get_frame 0.2ms + cmp 0.02ms)

  get_frame() time:     mean = 0.178 ms
  crop+compare time:    mean = 0.018 ms
  tap() time:           mean = 5.3 ms  (includes 5ms sleep)

  Estimated RL step rate: ~9.1 steps/sec
  Video resolution: 268x584 (max_size=585)
```

### Comparison: screencap vs H.264

Both measured with `wm size 540x1170`, same tap counter app, same tap coordinates.

| Metric | screencap (540×1170) | H.264 stream (268×584) |
|---|---|---|
| **Mean round-trip** | 271 ms | **104 ms** |
| **Std dev** | 24 ms | **14 ms** |
| **p50** | 270 ms | **108 ms** |
| **p95** | 313 ms | **109 ms** |
| **RL steps/sec** | ~3.6 | **~9.1** |
| Observation resolution | 540×1170 | 268×584 |
| Frame data per call | 2.5 MB raw RGBA | ~0.2 ms mutex copy |
| Polls per cycle | always 1 | 2-3 |

### Full Optimization History

| Iteration | Method | VM Resolution | Obs Resolution | Mean RT | Steps/sec |
|---|---|---|---|---|---|
| v1 | H.264 + 5ms poll | 1080×2340 | 944×2048 | 804 ms | ~1.2 |
| v2 | screencap | 1080×2340 | 1080×2340 | 616 ms | ~1.6 |
| v3 | screencap | 540×1170 | 540×1170 | 271 ms | ~3.6 |
| **v4** | **H.264 + 50ms poll** | **540×1170** | **268×584** | **104 ms** | **~9.1** |

**Total improvement: 8× faster** (804ms → 104ms), from ~1.2 to ~9.1 steps/sec.

## How to Read the Benchmark Output

### Round-trip time
The full pipeline latency: `tap() → Android processes touch → app renders new frame → encoder/screencap captures → Python detects change`. This is the metric that determines RL training speed.

### Polls until change
How many times `get_frame()` was called before the new frame appeared. With H.264 at 50ms poll intervals:
- **2 polls** = change appeared in 50-100ms (fast)
- **3 polls** = change appeared in 100-150ms (typical)
- **1 poll** = change was already present (previous frame boundary aligned)

With screencap, polls = 1 always because each screencap call takes ~270ms — long enough that the change has already happened.

### get_frame() time
Time to copy the latest decoded frame from the background decoder thread (mutex lock + numpy copy). This is NOT the frame's age — the frame may have been decoded moments ago or milliseconds ago. At ~0.2ms, this is negligible.

### tap() time
Time to send touch DOWN + sleep 5ms + send touch UP via scrcpy's binary control protocol. At ~5ms, this is negligible.

### crop+compare time
Time to slice the numpy array and compute mean absolute pixel difference. At ~0.02ms, this is negligible.

## Key Insights

### 1. H.264 streaming beats screencap for RL

The H.264 stream is **continuously decoded in the background**. `get_frame()` just copies the latest frame from memory (~0.2ms). The only overhead is the encoding pipeline delay (~50-100ms at this resolution).

Screencap must read the full framebuffer and pipe it through ADB **on every call**. Each call transfers 2.5MB of raw data, making it inherently slow (~270ms).

### 2. Resolution reduction provides compounding benefits

Reducing VM resolution from 1080×2340 to 540×1170:
- For screencap: 4× fewer pixels → ~2× faster (602ms → 271ms)
- For H.264: smaller frames encode faster, less pipeline delay

Using scrcpy's `max_size` further downscales the H.264 stream without changing the VM:
- `max_size=585` → 268×584 video (sufficient for RL, which resizes to 84×84 anyway)

### 3. Poll interval determines the measurement floor

With 50ms polling, the round-trip is quantized to multiples of ~50ms:
- Best case: change arrives before first poll → ~57ms
- Typical: change arrives during 2nd or 3rd poll → ~108ms
- The p50 (108ms) and p95 (109ms) being nearly identical shows the pipeline is very consistent

Reducing the poll interval (e.g., to 10ms) would lower latency further but increase CPU usage. For RL training, the current ~9 steps/sec is sufficient.

### 4. SwiftShader is the ultimate bottleneck

The emulator uses CPU-based software rendering (`swiftshader_indirect`) because the workstation has no GPU passthrough. This means:
- Frame rendering is slow (~16-50ms per frame vs <1ms with real GPU)
- H.264 encoding uses software MediaCodec
- There's no way to accelerate this without GPU hardware

The optimizations above work *around* SwiftShader by minimizing data transfer and using efficient encoding.

## Recommended Configuration for RL Training

```python
# In your RL environment:
from scrcpy_client import ScrcpyClient

client = ScrcpyClient(max_size=585)  # 268×584 H.264 stream
client.start()

# Each step:
client.tap(x, y)          # 5ms  — scrcpy control socket
time.sleep(0.05)          # 50ms — let Android process + encode
obs = client.get_frame()  # 0.2ms — latest decoded frame
obs = cv2.resize(obs, (84, 84))  # resize for neural network
```

**Expected throughput: ~9 steps/sec** (~104ms per step)

## Scripts

| Script | Purpose |
|---|---|
| `env_scripts/scrcpy_client.py` | Core client: scrcpy connection, `get_frame()`, `tap()`, `fast_screenshot()` |
| `env_scripts/web_server.py` | Flask web UI for manual interaction (developer tool) |
| `env_scripts/benchmark_h264.py` | H.264 stream latency benchmark (recommended method) |
