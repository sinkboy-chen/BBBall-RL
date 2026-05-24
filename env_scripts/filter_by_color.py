#!/usr/bin/env python3
import os
import glob
import cv2
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser(description="Move screenshots based on average color in a bounding box.")
    parser.add_argument("--src-dir", default="/home/student/12/b12902131/Desktop/BBBall-RL/env_scripts/screenshots", help="Directory containing screenshots")
    parser.add_argument("--dest-dir", default="/home/student/12/b12902131/Desktop/BBBall-RL/env_scripts/screenshots/filtered_green", help="Destination directory")
    
    # Bounding box defaults
    parser.add_argument("--x1", type=int, default=240, help="Top-left X")
    parser.add_argument("--y1", type=int, default=201, help="Top-left Y")
    parser.add_argument("--x2", type=int, default=342, help="Bottom-right X")
    parser.add_argument("--y2", type=int, default=203, help="Bottom-right Y")
    
    # RGB Ranges
    parser.add_argument("--r-min", type=float, default=2.5, help="Minimum Red")
    parser.add_argument("--r-max", type=float, default=3.0, help="Maximum Red")
    parser.add_argument("--g-min", type=float, default=230.0, help="Minimum Green")
    parser.add_argument("--g-max", type=float, default=235.0, help="Maximum Green")
    parser.add_argument("--b-min", type=float, default=2.0, help="Minimum Blue")
    parser.add_argument("--b-max", type=float, default=3.0, help="Maximum Blue")
    
    args = parser.parse_args()

    # Ensure destination directory exists
    os.makedirs(args.dest_dir, exist_ok=True)

    png_files = glob.glob(os.path.join(args.src_dir, "*.png"))
    
    # Do not process files already in the destination directory
    png_files = [f for f in png_files if os.path.dirname(f) != args.dest_dir]
    
    if not png_files:
        print(f"No .png files found in {args.src_dir}")
        return

    print(f"Found {len(png_files)} images. Analyzing...")
    print(f"Target RGB Range -> R: [{args.r_min}-{args.r_max}], G: [{args.g_min}-{args.g_max}], B: [{args.b_min}-{args.b_max}]")

    moved_count = 0

    for path in png_files:
        img = cv2.imread(path)
        if img is None:
            continue
        
        h, w = img.shape[:2]
        x1 = max(0, min(args.x1, args.x2))
        x2 = min(w, max(args.x1, args.x2))
        y1 = max(0, min(args.y1, args.y2))
        y2 = min(h, max(args.y1, args.y2))

        if x1 >= x2 or y1 >= y2:
            continue

        crop = img[y1:y2, x1:x2]
        
        # cv2.mean returns (B, G, R, Alpha)
        mean_bgr = cv2.mean(crop)[:3]
        mean_rgb = (mean_bgr[2], mean_bgr[1], mean_bgr[0])
        
        r, g, b = mean_rgb
        
        # Check if color is within the specified range
        if (args.r_min <= r <= args.r_max) and \
           (args.g_min <= g <= args.g_max) and \
           (args.b_min <= b <= args.b_max):
            
            filename = os.path.basename(path)
            dest_path = os.path.join(args.dest_dir, filename)
            
            # Move the file
            shutil.move(path, dest_path)
            print(f"Moved {filename} (R:{r:.1f} G:{g:.1f} B:{b:.1f})")
            moved_count += 1

    print(f"\nFinished! Moved {moved_count} matching images to {args.dest_dir}")

if __name__ == "__main__":
    main()
