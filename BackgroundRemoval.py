from rembg import remove
from PIL import Image

input_path = 'hand_pics/IMG_8314.png'
output_path = 'output.png'

input = Image.open(input_path)
output = remove(input)
output.save(output_path)