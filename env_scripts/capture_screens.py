#!/usr/bin/env python3
import os
import time
import urllib.request
from datetime import datetime

# Configuration
WEB_SERVER_URL = "http://localhost:5000/api/screenshot"
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
INTERVAL_SECONDS = 0.5

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Starting screenshot capture every {INTERVAL_SECONDS}s...")
    print(f"Saving to: {SAVE_DIR}")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"screen_{timestamp}.png"
            filepath = os.path.join(SAVE_DIR, filename)

            try:
                urllib.request.urlretrieve(WEB_SERVER_URL, filepath)
                print(f"Saved: {filename}")
            except Exception as e:
                print(f"Failed to capture screenshot: {e}")

            # Sleep to maintain the desired interval
            elapsed = time.time() - start_time
            sleep_time = max(0, INTERVAL_SECONDS - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopped screenshot capture.")

if __name__ == "__main__":
    main()
