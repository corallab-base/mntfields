import cv2
import numpy as np

# Load your map
img = cv2.imread('map.pgm', cv2.IMREAD_GRAYSCALE)
# Convert to 0.0 - 1.0 probability scale (255 is white/free, 0 is black/occupied)
# ROS standard: probability = (255 - pixel_value) / 255.0
probs = (255 - img) / 255.0

# Identify 'seen' pixels (Free or Occupied)
seen_mask = (probs > 0.65) | (probs < 0.196)
num_seen_pixels = np.sum(seen_mask)

# Calculate area
area = num_seen_pixels * (0.05**2)
print(f"Mapped Area: {area} square meters")