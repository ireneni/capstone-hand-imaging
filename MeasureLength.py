import cv2
import numpy as np
import math

def extract_finger_area(image):
    # Load and convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Find contours on the processed edges
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a blank mask to draw the largest contour (assumed to be the finger region)
    mask = np.zeros_like(thresh)

    if contours:
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Draw the largest contour on the mask and fill it
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    # Display the filled contour mask for verification
    cv2.imshow("Finger Area Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return largest_contour

def extract_scale(hsv):
    # Define blue color range for circle detection
    lower_blue = np.array([110, 100, 100])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours of the blue circle
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour assuming it is the blue circle
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Compute the enclosing circle around the largest contour
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        
        # Calculate the pixel distance as the diameter of the circle
        pixel_distance = 2 * radius

        # Calculate scale in cm per pixel (assuming the circle diameter is known to be 1.5 cm)
        scale_cm_per_pixel = 1.9 / pixel_distance

        return scale_cm_per_pixel

    return False

def extract_stickers(hsv):
    # Define green color range for sticker detection
    lower_green = np.array([55, 100, 50])
    upper_green = np.array([65, 255, 200])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # cv2.imshow('Green Sticker Mask', green_mask)
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

    return green_centers

def longest_continuous_segment(tuples):
    if not tuples:
        return []

    longest_start = 0
    longest_end = 0
    current_start = 0

    for i in range(1, len(tuples)):
        # Check if the current tuple continues the sequence
        if tuples[i][0] != tuples[i - 1][0] + 1:
            # If a discontinuity is found, check if the current segment is the longest
            if i - current_start > longest_end - longest_start:
                longest_start = current_start
                longest_end = i - 1
            # Start a new segment
            current_start = i

    # Check the last segment
    if len(tuples) - current_start > longest_end - longest_start:
        longest_start = current_start
        longest_end = len(tuples) - 1

    # Return the longest continuous segment
    return tuples[longest_start:longest_end + 1]

def calculate_finger_dimensions(image_path):
    # Load the image
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    finger_contour = extract_finger_area(image)
    scale_cm_per_pixel = extract_scale(hsv) or ValueError("Could not detect both blue reference lines in the image.")
    stickers = extract_stickers(hsv) 

    if len(stickers) == 3:
        tip = stickers[0]
        dip = stickers[1]
        pip = stickers[2]

        # Measure distances in pixels
        tip_to_dip_px = math.sqrt((dip[0] - tip[0]) ** 2 + (dip[1] - tip[1]) ** 2)
        dip_to_pip_px = math.sqrt((pip[0] - dip[0]) ** 2 + (pip[1] - dip[1]) ** 2)

        # Convert pixel distances to centimeters
        tip_to_dip_cm = tip_to_dip_px * scale_cm_per_pixel
        dip_to_pip_cm = dip_to_pip_px * scale_cm_per_pixel

        # Calculate width at DIP and PIP joints using largest contour
        def get_width_at_joint(joint_center):
            # Define a 2cm horizontal line at the y-coordinate of the joint center
            x_i = joint_center[0]
            y = joint_center[1]
            intersecting_points = []
            left_bound = int(x_i - 1/scale_cm_per_pixel)
            right_bound = int(x_i + 1/scale_cm_per_pixel)

            # Find points along the line that intersect with the contour
            for x in range(left_bound, right_bound):
                if cv2.pointPolygonTest(finger_contour, (x, y), False) >=  0:
                    intersecting_points.append((x, y))

            if len(intersecting_points) >= 2:
                trimmed = longest_continuous_segment(intersecting_points)
                left_point = trimmed[0]
                right_point = trimmed[-1]
                width_px = math.sqrt((right_point[0] - left_point[0]) ** 2)

                return width_px * scale_cm_per_pixel
            return None
        
        dip_width_cm = get_width_at_joint(dip)
        pip_width_cm = get_width_at_joint(pip)

        return {
            "tip_to_dip_cm": tip_to_dip_cm,
            "dip_to_pip_cm": dip_to_pip_cm,
            "dip_width_cm": dip_width_cm,
            "pip_width_cm": pip_width_cm
        }

    else:
        raise ValueError("Could not detect exactly three stickers in the image.")


# Import image
image_path = 'IMG_2510.jpg'

# Calculate finger dimensions:
measurements = calculate_finger_dimensions(image_path)
print("Length from tip to DIP joint (cm):", measurements["tip_to_dip_cm"])
print("Length from DIP to PIP joint (cm):", measurements["dip_to_pip_cm"])
print("Width at DIP joint (cm):", measurements["dip_width_cm"])
print("Width at PIP joint (cm):", measurements["pip_width_cm"])
