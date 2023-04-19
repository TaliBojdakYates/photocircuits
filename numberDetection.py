import pytesseract
import cv2


def detect_numbers(image_path):
    # Load the image and convert it to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to the image to convert it to black and white
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Run Tesseract 5 on the image to detect handwriting and get the word coordinates
    config = '--psm 11 -c tessedit_char_whitelist=0123456789kKloOsSkKvVQ --oem 1'
    data = pytesseract.image_to_data(threshold, output_type=pytesseract.Output.DICT, config=config)

 

    values = []
    # Loop over the detected words and print their coordinates
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 0:  # Only consider words with a confidence score above 50
            x = int((data['left'][i] + data['left'][i] + data['width'][i]) / 2)
            y = int((data['top'][i] + data['top'][i] + data['height'][i]) / 2)
            center = [x,y]
            number = data["text"][i]
            number = number.replace('l','1')
            number = number.replace('o','0')
            number = number.replace('O','0')
            number = number.replace('q','9')
            number = number.replace('s','5')
            number = number.replace('S','5')
            number = number.replace('Q','')
            
            
                
            values.append([number,center])
    return values


