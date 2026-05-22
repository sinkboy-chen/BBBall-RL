#!/usr/bin/env python3
import os
import re
from datetime import datetime
import matplotlib.pyplot as plt

LOG_FILE = os.path.join(os.path.dirname(__file__), "web_events.log")
PLOT_FILE = os.path.join(os.path.dirname(__file__), "touch_duration_distribution.png")

def parse_logs():
    if not os.path.exists(LOG_FILE):
        print(f"Log file {LOG_FILE} not found.")
        return []

    # Regex to match the timestamp and the event type
    # Expected format: [2026-05-22 17:35:10.123] TOUCH_DOWN   | {'x': 120, 'y': 345}
    pattern = re.compile(r'\[(.*?)\]\s+(TOUCH_DOWN|TOUCH_UP)\s+\|')
    
    durations = []
    last_down_time = None

    with open(LOG_FILE, "r") as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue
                
            time_str = match.group(1)
            event_type = match.group(2)
            
            try:
                # Try parsing with milliseconds first
                if '.' in time_str:
                    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
                else:
                    # Ignore old logs without milliseconds because they are not accurate enough for touch duration
                    continue
            except ValueError:
                continue

            if event_type == "TOUCH_DOWN":
                last_down_time = dt
            elif event_type == "TOUCH_UP" and last_down_time is not None:
                duration_ms = (dt - last_down_time).total_seconds() * 1000
                
                # Ignore abnormally long holds (e.g. user left tab open) > 5 seconds
                # or negative holds (out of order)
                if 0 < duration_ms < 5000:
                    durations.append(duration_ms)
                last_down_time = None

    return durations

def main():
    durations = parse_logs()
    
    if not durations:
        print("No valid touch durations found. Make sure you have played the game in the web server and generated TOUCH_DOWN/TOUCH_UP events with millisecond timestamps.")
        return
        
    print(f"Found {len(durations)} touch events.")
    print(f"Average duration: {sum(durations)/len(durations):.1f} ms")
    print(f"Min: {min(durations):.1f} ms | Max: {max(durations):.1f} ms")

    plt.figure(figsize=(10, 6))
    
    # Create histogram
    plt.hist(durations, bins=30, color='#4ecdc4', edgecolor='black', alpha=0.7)
    
    plt.title("Touch Duration Distribution", fontsize=16)
    plt.xlabel("Duration (ms)", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.savefig(PLOT_FILE)
    print(f"\nPlot saved to: {PLOT_FILE}")

if __name__ == "__main__":
    main()
