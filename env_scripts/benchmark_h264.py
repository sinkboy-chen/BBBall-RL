#!/usr/bin/env python3
"""
benchmark_h264.py — Measure scrcpy H.264 stream round-trip latency
===================================================================
Uses the H.264 video stream (get_frame) instead of adb screencap.
Polls every 50ms until the crop region changes.

Prerequisites:
  - Emulator running with wm size 540x1170 (or as desired)
  - Navigate to https://samvlu.github.io/web-04-tap-counter/
  - Close web_server.py (shares scrcpy port 27183)
  - Then run this script.
"""

import os
import sys
import time
import signal
import numpy as np
import cv2

from scrcpy_client import ScrcpyClient

# Coordinates measured from web server view (220x480 with max_size=480)
# Auto-scaled at runtime to match the benchmark's scrcpy resolution.
WEB_VIEW_W, WEB_VIEW_H = 220, 480
WEB_CROP_X1, WEB_CROP_Y1 = 30, 186
WEB_CROP_X2, WEB_CROP_Y2 = 200, 340
WEB_TAP_X, WEB_TAP_Y = 106, 411

NUM_CYCLES = 900
SAVE_EVERY = 100
POLL_SLEEP_S = 0.050  # 50ms between polls
POLL_TIMEOUT_S = 10.0
CHANGE_THRESHOLD = 2.0  # mean abs pixel diff to count as "changed"

SCRCPY_MAX_SIZE = 585  # → 270x585 with wm size 540x1170

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_h264")


def frames_differ(a, b):
    """Check if two crops differ meaningfully (above H.264 compression noise)."""
    if a is None or b is None:
        return True
    diff = np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16)))
    return diff > CHANGE_THRESHOLD


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    client = ScrcpyClient(max_size=SCRCPY_MAX_SIZE)

    def shutdown(sig, frame):
        client.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown)

    try:
        client.start()
    except Exception as e:
        print(f"Failed to start: {e}")
        sys.exit(1)

    # Wait for stable frame
    print("\nWaiting for stable frame...")
    for _ in range(200):
        f = client.get_frame()
        if f is not None:
            break
        time.sleep(0.05)
    else:
        print("ERROR: No frame received")
        client.stop()
        sys.exit(1)

    print(f"scrcpy video: {client.video_width}x{client.video_height}")

    # ── Auto-scale coordinates ────────────────────────────────────────────
    sx = client.video_width / WEB_VIEW_W
    sy = client.video_height / WEB_VIEW_H
    CROP_X1 = int(WEB_CROP_X1 * sx)
    CROP_Y1 = int(WEB_CROP_Y1 * sy)
    CROP_X2 = int(WEB_CROP_X2 * sx)
    CROP_Y2 = int(WEB_CROP_Y2 * sy)
    TAP_X = int(WEB_TAP_X * sx)
    TAP_Y = int(WEB_TAP_Y * sy)

    print(f"  Web view:     {WEB_VIEW_W}x{WEB_VIEW_H}")
    print(f"  Crop region:  ({CROP_X1},{CROP_Y1})-({CROP_X2},{CROP_Y2}) [scaled]")
    print(f"  Tap target:   ({TAP_X},{TAP_Y}) [scaled]")
    print(f"  Poll interval: {POLL_SLEEP_S*1000:.0f}ms")
    print(f"  Change threshold: {CHANGE_THRESHOLD}")
    print(f"  Cycles: {NUM_CYCLES}, save every {SAVE_EVERY}")

    # ── Initial state ────────────────────────────────────────────────────
    frame = client.get_frame()
    prev_crop = frame[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]
    cv2.imwrite(os.path.join(OUT_DIR, "initial.png"), frame)
    print(f"\nSaved initial screenshot ({frame.shape}).")

    # Send first tap to kick things off
    print("Sending initial tap...")
    t_initial_tap = time.time()
    client.tap(TAP_X, TAP_Y)
    last_tap_time = t_initial_tap

    # ── Measurement loop ─────────────────────────────────────────────────
    results = []

    for cycle in range(NUM_CYCLES):
        polls = 0
        poll_start = time.time()
        t_frame_get = 0
        t_crop_compare = 0
        changed = False

        while time.time() - poll_start < POLL_TIMEOUT_S:
            t0 = time.time()
            frame = client.get_frame()
            t1 = time.time()

            if frame is None:
                polls += 1
                time.sleep(POLL_SLEEP_S)
                continue

            crop = frame[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]
            t2 = time.time()

            changed = frames_differ(crop, prev_crop)
            polls += 1
            t_frame_get = (t1 - t0) * 1000
            t_crop_compare = (t2 - t1) * 1000

            if changed:
                break
            time.sleep(POLL_SLEEP_S)

        t_change_detected = time.time()
        round_trip_ms = (t_change_detected - last_tap_time) * 1000

        if not changed:
            print(f"  [Cycle {cycle:3d}] TIMEOUT ({polls} polls)")
            results.append({
                "cycle": cycle, "round_trip_ms": None, "polls": polls,
                "frame_get_ms": t_frame_get, "crop_compare_ms": t_crop_compare,
                "tap_ms": None, "save_ms": None, "changed": False, "timeout": True,
            })
            client.tap(TAP_X, TAP_Y)
            last_tap_time = time.time()
            f = client.get_frame()
            if f is not None:
                prev_crop = f[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]
            continue

        # Change detected → send next tap
        prev_crop = crop.copy()

        t_tap_start = time.time()
        client.tap(TAP_X, TAP_Y)
        t_tap_end = time.time()
        tap_ms = (t_tap_end - t_tap_start) * 1000
        last_tap_time = t_tap_start

        # Save screenshot every SAVE_EVERY cycles
        save_ms = None
        if cycle % SAVE_EVERY == 0:
            t_save_start = time.time()
            cv2.imwrite(os.path.join(OUT_DIR, f"cycle_{cycle:03d}.png"), frame)
            save_ms = (time.time() - t_save_start) * 1000

        results.append({
            "cycle": cycle,
            "round_trip_ms": round_trip_ms,
            "polls": polls,
            "frame_get_ms": t_frame_get,
            "crop_compare_ms": t_crop_compare,
            "tap_ms": tap_ms,
            "save_ms": save_ms,
            "changed": True,
            "timeout": False,
        })

        flag = " [SAVED]" if save_ms is not None else ""
        print(f"  [Cycle {cycle:3d}] RT={round_trip_ms:7.1f}ms  polls={polls:3d}  "
              f"frame_get={t_frame_get:.2f}ms  crop_cmp={t_crop_compare:.2f}ms  "
              f"tap={tap_ms:.1f}ms{flag}")

    # ── Save final screenshot ────────────────────────────────────────────
    final_frame = client.get_frame()
    if final_frame is not None:
        cv2.imwrite(os.path.join(OUT_DIR, "final.png"), final_frame)

    client.stop()

    # ── Print summary ────────────────────────────────────────────────────
    valid = [r for r in results if r["changed"] and not r["timeout"]]
    timeouts = [r for r in results if r["timeout"]]

    print("\n" + "=" * 70)
    print(f"  H.264 BENCHMARK RESULTS  ({len(valid)} successful, {len(timeouts)} timeouts)")
    print(f"  scrcpy max_size={SCRCPY_MAX_SIZE}  poll_interval={POLL_SLEEP_S*1000:.0f}ms  threshold={CHANGE_THRESHOLD}")
    print("=" * 70)

    if valid:
        rts = [r["round_trip_ms"] for r in valid]
        polls_list = [r["polls"] for r in valid]
        fg = [r["frame_get_ms"] for r in valid]
        cc = [r["crop_compare_ms"] for r in valid]
        taps = [r["tap_ms"] for r in valid]
        saves = [r["save_ms"] for r in valid if r["save_ms"] is not None]

        print(f"\n  Round-trip (tap → H.264 frame shows change):")
        print(f"    mean = {np.mean(rts):7.1f} ms")
        print(f"    std  = {np.std(rts):7.1f} ms")
        print(f"    min  = {np.min(rts):7.1f} ms")
        print(f"    max  = {np.max(rts):7.1f} ms")
        print(f"    p50  = {np.percentile(rts, 50):7.1f} ms")
        print(f"    p95  = {np.percentile(rts, 95):7.1f} ms")

        print(f"\n  Polls until change:")
        print(f"    mean = {np.mean(polls_list):7.1f}")
        print(f"    min  = {np.min(polls_list):7d}")
        print(f"    max  = {np.max(polls_list):7d}")
        print(f"    avg poll time = {POLL_SLEEP_S*1000 + np.mean(fg) + np.mean(cc):.1f} ms "
              f"(sleep {POLL_SLEEP_S*1000:.0f}ms + get_frame {np.mean(fg):.1f}ms + cmp {np.mean(cc):.2f}ms)")

        print(f"\n  get_frame() time:     mean = {np.mean(fg):.3f} ms")
        print(f"  crop+compare time:    mean = {np.mean(cc):.3f} ms")
        print(f"  tap() time:           mean = {np.mean(taps):.1f} ms  (includes 5ms sleep)")
        if saves:
            print(f"  screenshot save time: mean = {np.mean(saves):.1f} ms")

        print(f"\n  What each measurement means:")
        print(f"    round_trip  = tap() → Android → render → H.264 encode → TCP → decode → detected")
        print(f"    get_frame() = mutex lock + numpy copy of latest decoded frame")
        print(f"    tap()       = sendall(DOWN) + 5ms sleep + sendall(UP)")
        print(f"    crop+cmp    = numpy slice + mean absolute difference > {CHANGE_THRESHOLD}")

        # Estimate RL step rate
        avg_step_ms = np.mean(rts) + np.mean(taps)
        print(f"\n  Estimated RL step rate: ~{1000/avg_step_ms:.1f} steps/sec "
              f"(RT {np.mean(rts):.0f}ms + tap {np.mean(taps):.0f}ms)")
        print(f"  Video resolution: {client.video_width}x{client.video_height} "
              f"(max_size={SCRCPY_MAX_SIZE})")
    else:
        print("\n  No successful measurements!")

    print()


if __name__ == "__main__":
    main()
