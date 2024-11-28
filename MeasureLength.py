import os
import cv2
import numpy as np
import math
from rembg import remove
from PIL import Image

def remove_background(image_path):
    file_name = os.path.basename(image_path)
    name_ext_removed = os.path.splitext(file_name)[0]
    output_path = f'hand_pics/background_removed/{name_ext_removed}.png'
    input = Image.open(image_path)
    output = remove(input)
    output.save(output_path)

    return output_path

def extract_finger_area(image):
    # Load and convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)[1]

    # Display the thresholded image
    cv2.imshow("Thresholded Image", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

def extract_finger_area_bg_removed(image):
    """
    Extract the largest contour (assumed to be the finger region) from a pre-processed PNG image 
    with a transparent background.

    Args:
        image (np.array): Input PNG image with transparent background.

    Returns:
        largest_contour (np.array): The largest contour, representing the finger area.
    """
     # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a binary mask of non-black areas
    mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Optional: Visualize the mask of the largest contour
        largest_contour_mask = np.zeros_like(mask)
        cv2.drawContours(largest_contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        cv2.imshow("Finger Area Mask", largest_contour_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return largest_contour

    # Return None if no contours are found
    return None

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

    return None

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

def calculate_euclidean_angle(tip_to_corner_cm, tip_to_dip_cm):
    # Calculate angle from image
    sine_ratio = tip_to_corner_cm/tip_to_dip_cm
    angle_radians = math.asin(sine_ratio)
    angle_degrees = math.degrees(angle_radians)
    # print(f"The angle in radians: {angle_radians}")
    # print(f"The angle in degrees: {angle_degrees}")

    # subtract angle from 180 
    bend_angle = 180 - angle_degrees

    return bend_angle

def calculate_angle_between_lines(tip, dip, pip):
    """
    Calculate the angle between the line segments pip-dip and dip-tip.

    Args:
        pip (tuple): Coordinates of the PIP joint (x, y).
        dip (tuple): Coordinates of the DIP joint (x, y).
        tip (tuple): Coordinates of the fingertip (x, y).

    Returns:
        float: Angle in degrees between the two line segments.
    """
    # Convert points to numpy arrays for vector calculations
    tip = np.array(tip)
    dip = np.array(dip)
    pip = np.array(pip)

    # Create vectors
    vec1 = dip - pip  # Vector from pip to dip
    vec2 = tip - dip  # Vector from dip to tip

    # Calculate the dot product and magnitudes
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        raise ValueError("One of the line segments has zero length.")

    # Calculate the angle in radians
    cos_theta = dot_product / (magnitude1 * magnitude2)
    angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid numerical errors

    # Convert to degrees
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

def calculate_finger_dimensions(image_path):
    # Load the image
    base_image = cv2.imread(image_path)
    image_background_removed = remove_background(image_path)
    bg_removed_image = cv2.imread(image_background_removed)
    hsv = cv2.cvtColor(base_image, cv2.COLOR_BGR2HSV)

    # finger_contour = extract_finger_area(image)
    finger_contour = extract_finger_area_bg_removed(bg_removed_image)
    if finger_contour is None:
        raise ValueError("No contours found.")
    
    scale_cm_per_pixel = extract_scale(hsv) or ValueError("Could not detect both blue reference lines in the image.")
    stickers = extract_stickers(hsv) 

    if len(stickers) == 3:
        tip = stickers[0]
        dip = stickers[1]
        pip = stickers[2]

        # Calculate corner for angle measurement
        # corner = [dip[0], tip[1]]
        # tip_to_corner = (tip[0] - corner[0])

        # Measure distances in pixels
        tip_to_dip_px = math.sqrt((dip[0] - tip[0]) ** 2 + (dip[1] - tip[1]) ** 2)
        dip_to_pip_px = math.sqrt((pip[0] - dip[0]) ** 2 + (pip[1] - dip[1]) ** 2)

        # Convert pixel distances to centimeters
        tip_to_dip_cm = tip_to_dip_px * scale_cm_per_pixel
        dip_to_pip_cm = dip_to_pip_px * scale_cm_per_pixel
        # tip_to_corner_cm = tip_to_corner * scale_cm_per_pixel

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
        # bend_angle_degrees = calculate_euclidean_angle(tip_to_corner_cm, tip_to_dip_cm)
        bend_angle_degrees = calculate_angle_between_lines(tip, dip, pip)

        return {
            "tip_to_dip_cm": tip_to_dip_cm,
            "dip_to_pip_cm": dip_to_pip_cm,
            "dip_width_cm": dip_width_cm,
            "pip_width_cm": pip_width_cm,
            "bend_angle_degrees": bend_angle_degrees
        }

    else:
        raise ValueError("Could not detect exactly three stickers in the image.")


# Import image
image_path = 'hand_pics/IMG_2510.jpg'

# Calculate finger dimensions:
measurements = calculate_finger_dimensions(image_path)
print("Length from tip to DIP joint (cm): {:0.2f}".format(measurements["tip_to_dip_cm"]))
print("Length from DIP to PIP joint (cm): {:0.2f}".format(measurements["dip_to_pip_cm"]))
print("Width at DIP joint (cm): {:0.2f}".format(measurements["dip_width_cm"]))
print("Width at PIP joint (cm): {:0.2f}".format(measurements["pip_width_cm"]))
print("The bend angle in degrees: {:0.2f}".format(measurements["bend_angle_degrees"]))
