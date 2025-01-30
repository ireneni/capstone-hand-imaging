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
    # Use lower bound of 20 to exclude darkest shadows
    mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)[1]

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
    green_stickers = []
    
    for cnt in contours:
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (major_axis, minor_axis), angle = ellipse
            if 45 < max(major_axis, minor_axis) / 2 < 75:
                green_stickers.append({
                    "center": (int(x), int(y)),
                    "axes": (major_axis / 2, minor_axis / 2),
                    "angle": angle
                })

    green_stickers = sorted(green_stickers, key=lambda c: c["center"][1])
    return green_stickers

def ellipse_pixel_length(axes, angle, vector):
    """
    Calculate the diameter of the ellipse along the given vector.
    This implementation uses the formula:
        d = 2 * sqrt(a^2 * cos^2(phi) + b^2 * sin^2(phi))
    where:
        - a, b are the semi-major and semi-minor axes of the ellipse,
        - phi is the angle of the vector with respect to the ellipse's major axis.
    """
    # Semi-major and semi-minor axes
    a, b = axes

    # Angle of the vector with respect to the ellipse's major axis
    phi = np.arctan2(vector[1], vector[0]) - np.deg2rad(angle)

    # Compute cos^2(phi) and sin^2(phi)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Compute the diameter using the given formula
    pixel_distance = 2 * np.sqrt((a ** 2) * (cos_phi ** 2) + (b ** 2) * (sin_phi ** 2))

    return pixel_distance

def get_orthogonal_vector(vector):
    # Rotate vector by 90 degrees counter-clockwise
    return np.array([-vector[1], vector[0]])

def extract_scale_from_stickers(tip, dip, pip, tip_to_dip_vector, dip_to_pip_vector):
    scales = []

    # Calculate pixel distances along the ellipses (Length scale)
    tip_length_pixel_distance = ellipse_pixel_length(tip["axes"], tip["angle"], tip_to_dip_vector)
    dip_length_pixel_distance = ellipse_pixel_length(dip["axes"], dip["angle"], dip_to_pip_vector)
    pip_length_pixel_distance = ellipse_pixel_length(pip["axes"], pip["angle"], dip_to_pip_vector)

    # Calculate pixel distances along the orthogonal vectors (Width scale)
    dip_to_pip_orthogonal = get_orthogonal_vector(dip_to_pip_vector)
    dip_width_pixel_distance = ellipse_pixel_length(dip["axes"], dip["angle"], get_orthogonal_vector(dip_to_pip_orthogonal))
    pip_width_pixel_distance = ellipse_pixel_length(pip["axes"], pip["angle"], get_orthogonal_vector(dip_to_pip_orthogonal))

    # Calculate scales (mm/pixel)
    scales.append(6.35 / tip_length_pixel_distance)
    scales.append(6.35 / dip_length_pixel_distance)
    scales.append(6.35 / pip_length_pixel_distance)

    scales.append(6.35 / dip_width_pixel_distance)
    scales.append(6.35 / pip_width_pixel_distance)

    return scales

def longest_continuous_segment(points, max_gap=2):
    """
    Finds the longest continuous segment in a list of points, 
    allowing for any slope rather than just horizontal continuity.

    Args:
        points (list of tuples): List of (x, y) coordinates.
        max_gap (int): Maximum Euclidean distance between consecutive points 
                       to be considered continuous.

    Returns:
        list of tuples: Longest continuous segment of points.
    """
    if not points:
        return []

    longest_start = 0
    longest_end = 0
    current_start = 0

    for i in range(1, len(points)):
        # Compute Euclidean distance between consecutive points
        dist = np.linalg.norm(np.array(points[i]) - np.array(points[i - 1]))

        if dist > max_gap:
            # If discontinuity is found, check if current segment is the longest
            if i - current_start > longest_end - longest_start:
                longest_start = current_start
                longest_end = i - 1
            # Start a new segment
            current_start = i

    # Check the last segment
    if len(points) - current_start > longest_end - longest_start:
        longest_start = current_start
        longest_end = len(points) - 1

    return points[longest_start:longest_end + 1]

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

    finger_contour = extract_finger_area(bg_removed_image)
    if finger_contour is None:
        raise ValueError("No contours found.")

    stickers = extract_stickers(hsv)
    if len(stickers) == 3:
        tip = stickers[0]
        dip = stickers[1]
        pip = stickers[2]

        tip_center, dip_center, pip_center = np.array(tip["center"]), np.array(dip["center"]), np.array(pip["center"])

        # Vectors for scale calculations
        tip_to_dip_vector = dip_center - tip_center
        dip_to_pip_vector = pip_center - dip_center

        scale_tip_length, scale_dip_length, scale_pip_length, scale_dip_width, scale_pip_width = extract_scale_from_stickers(tip, dip, pip, tip_to_dip_vector, dip_to_pip_vector)

        # Measure distances in pixels
        dip_to_tip_px = math.sqrt((dip_center[0] - tip_center[0]) ** 2 + (dip_center[1] - tip_center[1]) ** 2)
        pip_to_dip_px = math.sqrt((pip_center[0] - dip_center[0]) ** 2 + (pip_center[1] - dip_center[1]) ** 2)

        # Convert pixel distances to centimeters
        # Add 2 for tip-to-dip to measure from top edge of tip sticker
        dip_to_tip_mm = dip_to_tip_px * ((scale_tip_length + scale_dip_length) / 2) + 2
        pip_to_dip_mm = pip_to_dip_px * ((scale_dip_length + scale_pip_length) / 2)

        pip_to_dip_unit = dip_to_pip_vector / np.linalg.norm(dip_to_pip_vector)
        perp_vector = np.array([pip_to_dip_unit[1], -pip_to_dip_unit[0]])

        # Calculate width at DIP and PIP joints using largest contour
        def get_width_at_joint(joint_center, scale, perp_vector):
            x_i, y_i = joint_center
            intersecting_points = []

            search_distance = 15 / scale  # Convert mm to pixels
            num_samples = int(search_distance * 2)  # Adjust resolution
            for i in range(-num_samples // 2, num_samples // 2 + 1):
                x = int(x_i + i * perp_vector[0])
                y = int(y_i + i * perp_vector[1])

                if cv2.pointPolygonTest(finger_contour, (x, y), False) >= 0:
                    intersecting_points.append((x, y))

            if len(intersecting_points) >= 2:
                # Extract longest continuous segment to remove noise
                trimmed = longest_continuous_segment(intersecting_points)
                if len(trimmed) >= 2:
                    left_point = trimmed[0]
                    right_point = trimmed[-1]
                    width_px = math.sqrt((right_point[0] - left_point[0]) ** 2 + (right_point[1] - left_point[1]) ** 2)

                    return width_px * scale

            return None

        dip_width_mm = get_width_at_joint(dip["center"], scale_dip_width, perp_vector)
        pip_width_mm = get_width_at_joint(pip["center"], scale_pip_width, perp_vector)
        bend_angle_degrees, bend_angle_direction = calculate_angle_with_direction(tip["center"], dip["center"], pip["center"])

        return {
            "tip_to_dip_mm": dip_to_tip_mm,
            "dip_to_pip_mm": pip_to_dip_mm,
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
