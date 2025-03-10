import os
import cv2
import numpy as np
import math
from rembg import remove
from PIL import Image

# Define constants
STICKER_WIDTH_MM = 6.35

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

    def sticker_within_range(major_axis, minor_axis):
        if 45 < major_axis / 2 < 135:
            if 35 < minor_axis / 2 < 135:
                return True
        return False

    for cnt in contours:
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (major_axis, minor_axis), angle = ellipse
            if sticker_within_range(major_axis, minor_axis):
                green_stickers.append({
                    "center": (int(x), int(y)),
                    "axes": (major_axis / 2, minor_axis / 2),
                    "angle": angle
                })
            else:
                print("Contour outside of sticker size range.")

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

def extract_scale_from_stickers(sticker, vector):
    # Calculate pixel distance along ellipse length
    length_pixel_distance = ellipse_pixel_length(sticker["axes"], sticker["angle"], vector)

    # Calculate pixel distance along ellipse width
    orthogonal = get_orthogonal_vector(vector)
    width_pixel_distance = ellipse_pixel_length(sticker["axes"], sticker["angle"], orthogonal)

    # Calculate scale (mm/pixel)
    return {"length": STICKER_WIDTH_MM / length_pixel_distance, "width": STICKER_WIDTH_MM / width_pixel_distance}

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

def get_finger_width(joint_center, scale, perp_vector, finger_contour):
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

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def midpoint(point1, point2):
    return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)

def average(x, y):
    return (x + y) / 2

def extract_measurement_features(image_path):
    # Load the image
    base_image = cv2.imread(image_path)
    rembg_image = remove_background(image_path)
    bg_removed_image = cv2.imread(rembg_image)
    hsv = cv2.cvtColor(base_image, cv2.COLOR_BGR2HSV)

    finger_contour = extract_finger_area(bg_removed_image)
    if finger_contour is None:
        raise ValueError("No contours found.")

    stickers = extract_stickers(hsv)
    return finger_contour, stickers

def calculate_major_axis_finger_dimensions(image_path):
    finger_contour, stickers = extract_measurement_features(image_path)

    if len(stickers) == 3:
        tip = stickers[0]
        dip = stickers[1]
        pip = stickers[2]

        tip_center, dip_center, pip_center = np.array(tip["center"]), np.array(dip["center"]), np.array(pip["center"])

        # Vectors for scale calculations
        tip_dip_vector = dip_center - tip_center
        dip_pip_vector = pip_center - dip_center

        # Extract scales
        scale_tip = extract_scale_from_stickers(tip, tip_dip_vector)
        scale_dip = extract_scale_from_stickers(dip, dip_pip_vector)
        scale_pip = extract_scale_from_stickers(pip, dip_pip_vector)

        # Measure distances in pixels
        dip_tip_px = euclidean_distance(dip_center, tip_center)
        pip_dip_px = euclidean_distance(pip_center, dip_center)

        # Convert pixel distances to centimeters
        # Add 2 mm for tip-to-dip to measure from top edge of tip sticker
        dip_tip_mm = dip_tip_px * ((scale_tip["length"]+ scale_dip["length"]) / 2) + 2
        pip_dip_mm = pip_dip_px * ((scale_dip["length"] + scale_pip["length"]) / 2)

        # Find perpendicular width vectors
        dip_tip_unit = tip_dip_vector / np.linalg.norm(tip_dip_vector)
        pip_dip_unit = dip_pip_vector / np.linalg.norm(dip_pip_vector)
        dip_tip_perp_vector = np.array([dip_tip_unit[1], -dip_tip_unit[0]])
        pip_dip_perp_vector = np.array([pip_dip_unit[1], -pip_dip_unit[0]])

        # Find midpoints of length vectors
        mid_dip_tip = midpoint(dip["center"], tip["center"])
        mid_pip_dip = midpoint(pip["center"], dip["center"])

        # Calculate width at joint
        dip_width_major_axis = get_finger_width(dip["center"], scale_dip["width"], pip_dip_perp_vector, finger_contour)
        # Calculate width at midpoints
        dip_tip_major_axis = get_finger_width(mid_dip_tip, average(scale_tip["width"], scale_dip["width"]), dip_tip_perp_vector, finger_contour)
        pip_dip_major_axis = get_finger_width(mid_pip_dip, average(scale_dip["width"], scale_pip["width"]), pip_dip_perp_vector, finger_contour)

        # Angle measure used for model development
        bend_angle_degrees, bend_angle_direction = calculate_angle_with_direction(tip["center"], dip["center"], pip["center"])
        # tip to DIP x-offset for parameterized model
        x_offset = (tip["center"][0] - dip["center"][0]) * average(scale_tip["width"], scale_dip["width"])


        return {
            "tip_dip_mm": dip_tip_mm,
            "dip_pip_mm": pip_dip_mm,
            "dip_tip_major_axis": dip_tip_major_axis,
            "dip_width_major_axis": dip_width_major_axis,
            "pip_dip_major_axis": pip_dip_major_axis,
            "dist_dip_tip_midpoint_mm": dip_tip_mm / 2,
            "dist_pip_dip_midpoint_mm": pip_dip_mm / 2,
            "bend_angle_degrees": bend_angle_degrees,
            "bend_angle_direction": bend_angle_direction,
            "x_offset": x_offset
        }

    else:
        raise ValueError("Could not detect exactly three stickers in the image.")

def calculate_minor_axis_finger_dimensions(image_path, major_measurements):
    finger_contour, stickers = extract_measurement_features(image_path)

    if len(stickers) == 1:
        dip = stickers[0]
        x_dip, y_dip = dip["center"]

        # Define unit basis vectors for sticker orientation
        y_vector = np.array([0, 1])
        x_vector = get_orthogonal_vector(y_vector)
        scale_dip = extract_scale_from_stickers(dip, y_vector)

        # Calculate midpoints using y-offset from major axis measurements,
        # converted from mm to pixel units using the DIP scale factor.
        mid_dip_tip = np.array([x_dip, (y_dip - (major_measurements["dist_dip_tip_midpoint_mm"] / scale_dip["length"]))])
        mid_pip_dip = np.array([x_dip, (y_dip + (major_measurements["dist_pip_dip_midpoint_mm"] / scale_dip["length"]))])

        # Calculate width at joint
        dip_width_minor_axis = get_finger_width(dip["center"], scale_dip["width"], x_vector, finger_contour)
        # Calculate width at midpoints
        dip_tip_minor_axis = get_finger_width(mid_dip_tip, scale_dip["width"], x_vector, finger_contour)
        pip_dip_minor_axis = get_finger_width(mid_pip_dip, scale_dip["width"], x_vector, finger_contour)

        return {
            "dip_tip_minor_axis": dip_tip_minor_axis,
            "dip_width_minor_axis": dip_width_minor_axis,
            "pip_dip_minor_axis": pip_dip_minor_axis,
        }

    else:
        raise ValueError("Could not detect exactly one sticker in the image.")

# Import image
major_image_path = 'hand_pics/' + 'index.jpg'
minor_image_path = 'hand_pics/' + 'index_side.jpg'

# Scale width measurements to account for warp due to curvature
def scale_down_five_percent(measurement):
    return ("{:0.2f}".format(measurement*0.95))

# Calculate finger dimensions:
major_measurements = calculate_major_axis_finger_dimensions(major_image_path)
minor_measurements = calculate_minor_axis_finger_dimensions(minor_image_path, major_measurements)

print("----------------Length measurements----------------")
print("Length from tip to DIP joint (mm): {:0.2f}".format(major_measurements["tip_dip_mm"]))
print("Length from DIP to PIP joint (mm): {:0.2f}".format(major_measurements["dip_pip_mm"]))
print("---------------------------------------------------")
print("Length of 1/2 tip to DIP joint (mm): {:0.2f}".format(major_measurements["tip_dip_mm"] / 2))
print("Length of 1/2 DIP to PIP joint (mm): {:0.2f}".format(major_measurements["dip_pip_mm"] / 2))
print("----------------Width measurements-----------------")
print("Width at 1/2 DIP to tip for distal major axis (mm): ", scale_down_five_percent(major_measurements["dip_tip_major_axis"]))
print("Width at DIP major axis (mm): ", scale_down_five_percent(major_measurements["dip_width_major_axis"]))
print("Width at 1/2 PIP to DIP for proximal major axis (mm): ", scale_down_five_percent(major_measurements["pip_dip_major_axis"]))
print("---------------------------------------------------")
print("Width at 1/2 DIP to tip for distal minor axis (mm): ", scale_down_five_percent(minor_measurements["dip_tip_minor_axis"]))
print("Width at DIP minor axis (mm): ", scale_down_five_percent(minor_measurements["dip_width_minor_axis"]))
print("Width at 1/2 PIP to DIP for proximal minor axis (mm): ", scale_down_five_percent(minor_measurements["pip_dip_minor_axis"]))
print("------------------------Angle----------------------")
print("The bend angle in degrees: {:0.2f} {}".format(major_measurements["bend_angle_degrees"], major_measurements["bend_angle_direction"]))
print("The tip to DIP x-offset (mm, positive x to the right): {:0.2f}".format(major_measurements["x_offset"]))