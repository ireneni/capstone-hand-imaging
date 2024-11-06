import cv2
import numpy as np
import math

def calculate_finger_lengths(image_path):
    # Load the image
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define blue color range for line detection of ruler
    lower_blue = np.array([110, 100, 100])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours of the blue lines
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_centers = []
    
    for cnt in contours:
        # Compute the center of each contour using minEnclosingCircle
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        blue_centers.append((int(x), int(y)))
    
    # Sort centers by their y-coordinates (top to bottom) and calculate the pixel distance
    if len(blue_centers) >= 2:
        blue_centers = sorted(blue_centers, key=lambda c: c[1])
        line1, line2 = blue_centers[:2]
        pixel_distance = math.sqrt((line2[0] - line1[0]) ** 2 + (line2[1] - line1[1]) ** 2)
        
        # Calculate scale in cm per pixel
        scale_cm_per_pixel = 1.5 / pixel_distance

        # Define green color range for sticker detection
        lower_green = np.array([55, 100, 50])
        upper_green = np.array([65, 255, 200])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # cv2.imshow('Green Sticker Mask', mask)
        # cv2.waitKey(0)  # Wait for a key press to close the window
        # cv2.destroyAllWindows()

        # Find contours for green stickers
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_centers = []
        
        for cnt in contours:
            # Compute the center of each sticker using minEnclosingCircle
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if 15 < radius < 35:
                green_centers.append((int(x), int(y)))

        # Sort the centers by their y-coordinates (top to bottom)
        green_centers = sorted(green_centers, key=lambda c: c[1])

        if len(green_centers) == 3:
            # Identify points
            tip = green_centers[0]
            dip = green_centers[1]
            pip = green_centers[2]

            # Measure distances in pixels
            tip_to_dip_px = math.sqrt((dip[0] - tip[0]) ** 2 + (dip[1] - tip[1]) ** 2)
            dip_to_pip_px = math.sqrt((pip[0] - dip[0]) ** 2 + (pip[1] - dip[1]) ** 2)

            # Convert pixel distances to centimeters
            tip_to_dip_cm = tip_to_dip_px * scale_cm_per_pixel
            dip_to_pip_cm = dip_to_pip_px * scale_cm_per_pixel

            return tip_to_dip_cm, dip_to_pip_cm

        else:
            raise ValueError("Could not detect three stickers in the image.")
    else:
        raise ValueError("Could not detect both blue reference lines in the image.")

# Example usage:
image_path = 'IMG_2434.jpg'
tip_to_dip_cm, dip_to_pip_cm = calculate_finger_lengths(image_path)
print("Length from tip to DIP joint (cm):", tip_to_dip_cm)
print("Length from DIP to PIP joint (cm):", dip_to_pip_cm)

