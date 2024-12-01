from rembg import remove
from PIL import Image
import cv2
import numpy as np

# Test background removal
# input_path = 'hand_pics/IMG_2510.jpg'
# output_path = 'output.png'

# input = Image.open(input_path)
# output = remove(input)
# output.save(output_path)

# Convert rgb to hsv
# GREEN
# rgb_color = np.uint8([[[145, 249, 159]]]) 
# hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)
# print("HSV Value for OpenCV:", hsv_color[0][0])

# rgb_color = np.uint8([[[137, 238, 141]]])  
# hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)
# print("HSV Value for OpenCV:", hsv_color[0][0])

# rgb_color = np.uint8([[[153, 251, 166]]])  
# hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)
# print("HSV Value for OpenCV:", hsv_color[0][0])

# rgb_color = np.uint8([[[122, 239, 129]]])  
# hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)
# print("HSV Value for OpenCV:", hsv_color[0][0])

# rgb_color = np.uint8([[[108, 221, 158]]])  
# hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)
# print("HSV Value for OpenCV:", hsv_color[0][0])

# rgb_color = np.uint8([[[124, 236, 175]]])  
# hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)
# print("HSV Value for OpenCV:", hsv_color[0][0])

# rgb_color = np.uint8([[[151, 250, 200]]])  
# hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)
# print("HSV Value for OpenCV:", hsv_color[0][0])

# rgb_color = np.uint8([[[85, 208, 128]]])  
# hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)
# print("HSV Value for OpenCV:", hsv_color[0][0])

# YELLOW
rgb_color = np.uint8([[[250, 255, 39]]])  
hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)
print("HSV Value for OpenCV:", hsv_color[0][0])

rgb_color = np.uint8([[[250, 249, 32]]])  
hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)
print("HSV Value for OpenCV:", hsv_color[0][0])

rgb_color = np.uint8([[[252, 255, 35]]])  
hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)
print("HSV Value for OpenCV:", hsv_color[0][0])

rgb_color = np.uint8([[[250, 254, 76]]])  
hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)
print("HSV Value for OpenCV:", hsv_color[0][0])