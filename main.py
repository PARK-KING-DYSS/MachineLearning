import cv2
import easyocr
import matplotlib.pyplot as plt

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Load the image
image = cv2.imread("1.jpg")

# Perform ANPR
results = reader.readtext(image)

# Specify the text you want to find (e.g., license plate text)
target_text = "YOUR_TARGET_TEXT"  # Replace with the actual text you want to find

# Find the bounding box for the target text
for result in results:
    bounding_box, text, _ = result
    if text == target_text:
        x, y, w, h = bounding_box
        break
else:
    # Handle the case where the target text is not found
    x, y, w, h = 0, 0, 0, 0

# Check if the license plate region has valid dimensions
if w > 0 and h > 0:
    # Crop the license plate region
    license_plate_roi = image[y:y + h, x:x + w]

    # Display the cropped license plate region using OpenCV
    cv2.imshow("Zoomed License Plate (OpenCV)", license_plate_roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Display the cropped license plate region using Matplotlib
    plt.imshow(cv2.cvtColor(license_plate_roi, cv2.COLOR_BGR2RGB))
    plt.title("Zoomed License Plate (Matplotlib)")
    plt.axis("off")
    plt.show()

    # Print the detected license plate text
    print("License Plate:", target_text)
else:
    print("License plate not found.")
