import cv2
import numpy as np
import math

def calculate_finger_lengths(image_path):
    # Load the image
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define green color range based on R:0, G:128, B:0
    lower_green = np.array([55, 100, 50])
    upper_green = np.array([65, 255, 200])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    cv2.imshow('Green Sticker Mask', mask)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

    # Find contours in the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter for round shapes (assume reasonable radius range for stickers)
    centers = []
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if 15 < radius < 35:
            centers.append((int(x), int(y)))

    # Sort the centers by their y-coordinates (top to bottom)
    centers = sorted(centers, key=lambda c: c[1])

    if len(centers) == 3:
        # Identify points
        tip = centers[0]
        dip = centers[1]
        pip = centers[2]

        # Calculate distances
        tip_to_dip = math.sqrt((dip[0] - tip[0]) ** 2 + (dip[1] - tip[1]) ** 2)
        dip_to_pip = math.sqrt((pip[0] - dip[0]) ** 2 + (pip[1] - dip[1]) ** 2)

        return tip_to_dip, dip_to_pip
    else:
        raise ValueError("Could not detect three stickers in the image.")

# Example usage:
image_path = 'IMG_2434.jpg'
tip_to_dip, dip_to_pip = calculate_finger_lengths(image_path)
print("Length from tip to DIP joint:", tip_to_dip)
print("Length from DIP to PIP joint:", dip_to_pip)

