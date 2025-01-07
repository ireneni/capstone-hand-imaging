import os
import cv2
import numpy as np
import math
import json
from rembg import remove
from PIL import Image

def remove_background(image_path):
    file_name = os.path.basename(image_path)
    name_ext_removed = os.path.splitext(file_name)[0]
    output_path = f'hand_pics/background_removed/{name_ext_removed}.png'

     # Check if the output file already exists
    if not os.path.exists(output_path):
        input_image = Image.open(image_path)
        output_image = remove(input_image)
        output_image.save(output_path)
        print(f"Background removed and saved to {output_path}")
    else:
        print(f"File already exists at {output_path}. Skipping background removal.")

    return output_path

def extract_finger_area(image):
    # Load and convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)[1]

    # Display the thresholded image
    # cv2.imshow("Thresholded Image", thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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
    # cv2.imshow("Finger Area Mask", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
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

    # Filter out contours that intersect the top of the image (assumed to be the stand)
    valid_contours = [
        cnt for cnt in contours if not any(point[0][1] == 0 for point in cnt)
    ]

    if valid_contours:
        # Find the largest valid contour by area
        largest_contour = max(valid_contours, key=cv2.contourArea)

        # Optional: Visualize the mask of the largest contour
        largest_contour_mask = np.zeros_like(mask)
        cv2.drawContours(largest_contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        cv2.imshow("Finger Area Mask", largest_contour_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return largest_contour

    # Return None if no valid contours are found
    return None

def extract_scale(hsv):
    lower_yellow = np.array([25, 150, 200])  
    upper_yellow = np.array([35, 255, 255])  
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Find contours of the yellow circle
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour assuming it is the yellow circle
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Compute the enclosing circle around the largest contour
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        
        # Calculate the pixel distance as the diameter of the circle
        pixel_distance = 2 * radius

        # Calculate scale in mm per pixel
        scale_mm_per_pixel = 19.05 / pixel_distance

        return scale_mm_per_pixel

    return None

def extract_stickers(hsv):
    # Define green color range for sticker detection
    lower_green = np.array([55, 90, 150])  
    upper_green = np.array([80, 180, 255]) 
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
        if 45 < radius < 75:
            green_centers.append([int(x), int(y), radius])

    # Sort the centers by their y-coordinates (top to bottom)
    green_centers = sorted(green_centers, key=lambda c: c[1])
    green_centers[0][1] -= green_centers[0][2] # Adjust the tip sticker to use the top edge 
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

def calculate_euclidean_angle(tip_to_corner_mm, tip_to_dip_mm):
    # Calculate angle from image
    sine_ratio = tip_to_corner_mm/tip_to_dip_mm
    angle_radians = math.asin(sine_ratio)
    angle_degrees = math.degrees(angle_radians)
    # print(f"The angle in radians: {angle_radians}")
    # print(f"The angle in degrees: {angle_degrees}")

    # subtract angle from 180 
    bend_angle = 180 - angle_degrees

    return bend_angle

def calculate_angle_with_direction(pip, dip, tip):
    """
    Calculate the angle between the line segments pip-dip and dip-tip,
    indicating if the rotation is clockwise or counterclockwise.

    Args:
        pip (tuple): Coordinates of the PIP joint (x, y).
        dip (tuple): Coordinates of the DIP joint (x, y).
        tip (tuple): Coordinates of the fingertip (x, y).

    Returns:
        tuple: Angle in degrees and direction ('clockwise' or 'counterclockwise').
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

    # Calculate the cross product (2D version gives a scalar)
    cross_product = vec1[0] * vec2[1] - vec1[1] * vec2[0]

    # Calculate the angle in radians
    cos_theta = dot_product / (magnitude1 * magnitude2)
    angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid numerical errors

    # Convert to degrees
    angle_degrees = np.degrees(angle_radians)

    # Determine the rotation direction
    direction = "counterclockwise" if cross_product > 0 else "clockwise"

    return angle_degrees, direction

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
    
    scale_mm_per_pixel = extract_scale(hsv) or ValueError("Could not detect scale in the image.")
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
        tip_to_dip_mm = tip_to_dip_px * scale_mm_per_pixel
        dip_to_pip_mm = dip_to_pip_px * scale_mm_per_pixel
        # tip_to_corner_mm = tip_to_corner * scale_mm_per_pixel

        # Calculate width at DIP and PIP joints using largest contour
        def get_width_at_joint(joint_center):
            # Define a 30mm horizontal line extending left and right of the joint center
            x_i = joint_center[0]
            y = joint_center[1]
            intersecting_points = []
            left_bound = int(x_i - 15/scale_mm_per_pixel)
            right_bound = int(x_i + 15/scale_mm_per_pixel)

            # Find points along the line that intersect with the contour
            for x in range(left_bound, right_bound):
                if cv2.pointPolygonTest(finger_contour, (x, y), False) >=  0:
                    intersecting_points.append((x, y))

            if len(intersecting_points) >= 2:
                trimmed = longest_continuous_segment(intersecting_points)
                left_point = trimmed[0]
                right_point = trimmed[-1]
                width_px = math.sqrt((right_point[0] - left_point[0]) ** 2)

                return width_px * scale_mm_per_pixel
            return None
        
        dip_width_mm = get_width_at_joint(dip)
        pip_width_mm = get_width_at_joint(pip)
        # bend_angle_degrees = calculate_euclidean_angle(tip_to_corner_mm, tip_to_dip_mm)
        bend_angle_degrees, bend_angle_direction = calculate_angle_with_direction(tip, dip, pip)

        return {
            "tip_to_dip_mm": tip_to_dip_mm,
            "dip_to_pip_mm": dip_to_pip_mm,
            "dip_width_mm": dip_width_mm,
            "pip_width_mm": pip_width_mm,
            "bend_angle_degrees": bend_angle_degrees,
            "bend_angle_direction": bend_angle_direction
        }

    else:
        raise ValueError("Could not detect exactly three stickers in the image.")


# Import image
image_path = 'hand_pics/IMG_2603.png'

# Calculate finger dimensions:
measurements = calculate_finger_dimensions(image_path)
print("Length from tip to DIP joint (mm): {:0.2f}".format(measurements["tip_to_dip_mm"]))
print("Length from DIP to PIP joint (mm): {:0.2f}".format(measurements["dip_to_pip_mm"]))
print("Width at DIP joint (mm): {:0.2f}".format(measurements["dip_width_mm"]))
print("Width at PIP joint (mm): {:0.2f}".format(measurements["pip_width_mm"]))
print("The bend angle in degrees: {:0.2f} {}".format(measurements["bend_angle_degrees"], measurements["bend_angle_direction"]))

directory_path = "C:\\Users\\Amanda\\Documents\\Capstone_code\\capstone-hand-imaging\\hand_measurements\\JSON"

# Create the directory if it doesn't exist
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

# Extract the base name of the image file (without extension)
image_base_name = os.path.splitext(os.path.basename(image_path))[0]

# Construct the JSON file name
file_name = f"final_measurements_{image_base_name}.json"
file_path = os.path.join(directory_path, file_name)

with open(file_path, "w") as f:
    json.dump(measurements, f)
