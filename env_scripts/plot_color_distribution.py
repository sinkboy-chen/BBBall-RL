#!/usr/bin/env python3
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot color distribution of screenshots.")
    parser.add_argument("--dir", default="/home/student/12/b12902131/Desktop/BBBall-RL/env_scripts/screenshots", help="Directory containing screenshots")
    parser.add_argument("--x1", type=int, default=257, help="Top-left X")
    parser.add_argument("--y1", type=int, default=159, help="Top-left Y")
    parser.add_argument("--x2", type=int, default=324, help="Bottom-right X")
    parser.add_argument("--y2", type=int, default=163, help="Bottom-right Y")
    args = parser.parse_args()

    png_files = glob.glob(os.path.join(args.dir, "*.png"))
    if not png_files:
        print(f"No .png files found in {args.dir}")
        return

    print(f"Found {len(png_files)} images. Processing...")

    colors_rgb = []

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
        mean_bgr = cv2.mean(crop)[:3]
        mean_rgb = (mean_bgr[2], mean_bgr[1], mean_bgr[0])
        colors_rgb.append(mean_rgb)

    if not colors_rgb:
        print("No valid colors could be extracted.")
        return

    # Convert to numpy array for easier slicing
    colors_arr = np.array(colors_rgb)
    
    r_vals = colors_arr[:, 0]
    g_vals = colors_arr[:, 1]
    b_vals = colors_arr[:, 2]
    
    # Normalize colors for matplotlib to display the actual point colors
    facecolors = colors_arr / 255.0

    print("Plotting distribution...")
    
    fig = plt.figure(figsize=(12, 6))

    # 1. 3D Scatter Plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(r_vals, g_vals, b_vals, c=facecolors, s=50, alpha=0.7, edgecolors='none')
    ax1.set_title("3D Color Distribution (Actual Colors)")
    ax1.set_xlabel("Red")
    ax1.set_ylabel("Green")
    ax1.set_zlabel("Blue")
    ax1.set_xlim([0, 255])
    ax1.set_ylim([0, 255])
    ax1.set_zlim([0, 255])

    # 2. Histograms for R, G, B
    ax2 = fig.add_subplot(122)
    ax2.hist(r_vals, bins=30, color='red', alpha=0.5, label='Red')
    ax2.hist(g_vals, bins=30, color='green', alpha=0.5, label='Green')
    ax2.hist(b_vals, bins=30, color='blue', alpha=0.5, label='Blue')
    ax2.set_title("RGB Channel Histograms")
    ax2.set_xlabel("Pixel Value (0-255)")
    ax2.set_ylabel("Frequency")
    ax2.legend()

    plt.tight_layout()
    
    # Save the plot instead of just showing it, in case running on a headless server
    save_path = os.path.join(args.dir, "color_distribution_plot.png")
    plt.savefig(save_path)
    print(f"Saved plot to: {save_path}")
    
    # Try to show it as well (might fail on headless SSH)
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
