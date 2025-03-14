
import os
import json
import csv
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Function to load the final measurements from a selected file
def load_measurements():

    relative_folder = "hand_measurements"
    initial_folder = os.path.join(os.getcwd(), relative_folder)
    Tk().withdraw() 
    file_path = askopenfilename(
        title="Select the JSON file with hand measurements",
        initialdir=initial_folder,
        filetypes=[("JSON files", "*.json")]
    )
    if not file_path:
        raise FileNotFoundError("No file was selected.")
    
    filename = os.path.basename(file_path)  # Save the JSON filename as a variable
    with open(file_path, "r") as f:
        return json.load(f), filename

# Load the final measurements
final_measurements, filename = load_measurements()
minor_measurements = final_measurements["minor_measurements"]
major_measurements = final_measurements["major_measurements"]

# Calculations for parameters of distal and proximal rings 
def distal_minor_axis(distal_depth) -> float:
    return (distal_depth / 2) - 0.8 

def proximal_minor_axis(proximal_depth) -> float:
    return (proximal_depth / 2) - 0.8

def dip_minor_axis(dip_depth) -> float:
    return dip_depth / 2

def distal_major_axis(distal_width) -> float:
    return (distal_width / 2) - 0.4 

def proximal_major_axis(proximal_width) -> float:
    return (proximal_width / 2) - 0.4

def dip_major_axis(dip_width) -> float:
    return (dip_width / 2) - 0.4

# Write the results into a CSV file
def write_to_csv(filename, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    csv_filename = os.path.splitext(filename)[0] + ".csv"
    output_path = os.path.join(output_dir, csv_filename)
    
    # Prepare the data for the CSV
    data = [
        ("Distal Minor Axis", distal_minor_axis(minor_measurements["dip_tip_minor_axis"])),
        ("Proximal Minor Axis", proximal_minor_axis(minor_measurements["pip_dip_minor_axis"])),
        ("DIP Minor Axis", dip_minor_axis(minor_measurements["dip_width_minor_axis"])),
        ("Distal Major Axis", distal_major_axis(major_measurements["dip_tip_major_axis"])),
        ("Proximal Major Axis", proximal_major_axis(major_measurements["pip_dip_major_axis"])),
        ("DIP Major Axis", dip_major_axis(major_measurements["dip_width_major_axis"])),
        ("Dist Plane Offset", major_measurements["dist_dip_tip_midpoint_mm"]),
        ("Prox Plane Offset", major_measurements["dist_pip_dip_midpoint_mm"]),
        ("Radial-Ulnar Offset", major_measurements["x_offset"])
    ]
    
    # Write the data to the CSV file (without header)
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)  # Write only the data rows

output_directory = "C:\\Users\\Amanda\\Documents\\Capstone_code\\capstone-hand-imaging\\hand_measurements\\CSV"
write_to_csv(filename, output_directory)