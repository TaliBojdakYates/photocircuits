import pytesseract
from PIL import Image

# Open image using PIL
image = Image.open('divider.png')

# Convert image to string using Pytesseract
text = pytesseract.image_to_string(image)

# Print extracted text
print(text)