#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Calculate average color in a bounding box.")
    parser.add_argument("image_path", help="Path to the screenshot image")
    parser.add_argument("--x1", type=int, default=257, help="Top-left X coordinate")
    parser.add_argument("--y1", type=int, default=159, help="Top-left Y coordinate")
    parser.add_argument("--x2", type=int, default=324, help="Bottom-right X coordinate")
    parser.add_argument("--y2", type=int, default=163, help="Bottom-right Y coordinate")
    
    args = parser.parse_args()

    # Read the image
    img = cv2.imread(args.image_path)
    if img is None:
        print(f"Error: Could not read image at {args.image_path}")
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"Image loaded successfully. Resolution: {w}x{h}")

    # Ensure coordinates are within bounds
    x1 = max(0, min(args.x1, args.x2))
    x2 = min(w, max(args.x1, args.x2))
    y1 = max(0, min(args.y1, args.y2))
    y2 = min(h, max(args.y1, args.y2))

    if x1 >= x2 or y1 >= y2:
        print(f"Error: Invalid bounding box ({x1}, {y1}) to ({x2}, {y2}) for image size {w}x{h}")
        sys.exit(1)

    print(f"Analyzing Region: ({x1}, {y1}) to ({x2}, {y2})")

    # Crop the region
    # Note: OpenCV uses numpy arrays where format is img[y1:y2, x1:x2]
    crop = img[y1:y2, x1:x2]

    # Calculate mean color
    # cv2.mean returns (B, G, R, Alpha)
    mean_color_bgr = cv2.mean(crop)[:3]
    
    # Convert BGR to RGB for easier reading
    mean_color_rgb = (mean_color_bgr[2], mean_color_bgr[1], mean_color_bgr[0])

    print("\nResults:")
    print(f"Average Color (BGR): (B={mean_color_bgr[0]:.1f}, G={mean_color_bgr[1]:.1f}, R={mean_color_bgr[2]:.1f})")
    print(f"Average Color (RGB): (R={mean_color_rgb[0]:.1f}, G={mean_color_rgb[1]:.1f}, B={mean_color_rgb[2]:.1f})")
    print(f"Hex Code: #{int(mean_color_rgb[0]):02X}{int(mean_color_rgb[1]):02X}{int(mean_color_rgb[2]):02X}")

    # Optional: Display the crop if you run this on a machine with a display
    # cv2.imshow("Crop", crop)
    # cv2.waitKey(0)

if __name__ == "__main__":
    main()
