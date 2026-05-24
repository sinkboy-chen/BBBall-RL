#!/usr/bin/env python3
import os
import subprocess
import time
import threading
import sys
import signal

USER = os.environ.get("USER")
if not USER:
    print("[Error] USER environment variable not found. Please set it or run within standard environment.")
    sys.exit(1)

ANDROID_HOME = f"/tmp2/{USER}/DRL_final_workspace/android-sdk"
ENV_VARS = {
    "ANDROID_HOME": ANDROID_HOME,
    "ANDROID_SDK_ROOT": ANDROID_HOME,
    "ANDROID_USER_HOME": f"/tmp2/{USER}/DRL_final_workspace/.android",
    "ANDROID_AVD_HOME": f"/tmp2/{USER}/DRL_final_workspace/.android/avd",
    "ANDROID_EMULATOR_HOME": f"/tmp2/{USER}/DRL_final_workspace/.android",
    # Note: Do NOT limit OMP/MKL/BLAS threads here — emulators need them for SwiftShader rendering
}

# Update PATH
env = os.environ.copy()
env.update(ENV_VARS)
env["PATH"] = f"{env.get('PATH', '')}:{ANDROID_HOME}/cmdline-tools/latest/bin:{ANDROID_HOME}/platform-tools:{ANDROID_HOME}/emulator"

class EmulatorInstance:
    def __init__(self, port, avd_name="pixel5_api31", snapshot="game_ready"):
        self.port = port
        self.avd_name = avd_name
        self.snapshot = snapshot
        self.serial = f"emulator-{port}"
        self.process = None
        self._stop_requested = False
        self.thread = None

    def start(self):
        self._stop_requested = False
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop_requested = True
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        # Also send console kill command as fallback
        subprocess.run(["adb", "-s", self.serial, "emu", "kill"], env=env, capture_output=True)

    def _run_loop(self):
        while not self._stop_requested:
            # Create a unique temp directory for this instance to avoid temp file collisions
            tmp_dir = f"/tmp2/{USER}/DRL_final_workspace/tmp/emu_{self.port}"
            os.makedirs(tmp_dir, exist_ok=True)
            
            print(f"[*] Starting emulator {self.serial} on port {self.port}...")
            instance_env = env.copy()
            instance_env["ANDROID_TMP"] = tmp_dir
            cmd = [
                "emulator",
                "-avd", self.avd_name,
                "-port", str(self.port),
                "-no-window",
                "-no-audio",
                "-no-boot-anim",
                "-gpu", "swiftshader_indirect",
                "-no-metrics",
                "-read-only",
                "-snapshot", self.snapshot
            ]
            self.process = subprocess.Popen(cmd, env=instance_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Wait for the process to exit
            self.process.wait()
            
            if self._stop_requested:
                break
                
            print(f"[!] Warning: Emulator {self.serial} died unexpectedly! Restarting in 5 seconds...")
            time.sleep(5)

def main():
    # Capped at N=3 emulators per user to prevent triggering thread watchdog
    ports = [5554, 5556, 5558]
    emulators = [EmulatorInstance(p) for p in ports]
    
    def signal_handler(sig, frame):
        print("\n[!] Ctrl+C detected. Shutting down all emulators...")
        for emu in emulators:
            emu.stop()
        print("[*] All emulators shut down. Exiting.")
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start emulators sequentially with staggering of 8 seconds to avoid disk and CPU spikes
    for emu in emulators:
        emu.start()
        time.sleep(8) # 8 seconds launch stagger
        
    print("[*] All emulators spawned. Monitoring... (Press Ctrl+C to terminate)")
    while True:
        try:
            time.sleep(10)
            statuses = []
            for emu in emulators:
                alive = emu.process and emu.process.poll() is None
                statuses.append(f"{emu.serial}: {'ALIVE' if alive else 'DEAD'}")
            print(f"[Status] {', '.join(statuses)}")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
