# **JUPYTER (.IPYNB FILES):**
---

## 1) CAR_IMAGE.ipynb:

### 1. Installation of Required Libraries
```
pip install roboflow

```
This command installs the roboflow library, which is a Python package for computer vision tasks such as object detection and image classification.

### 2. License Plate Detection using Roboflow
```
from roboflow import Roboflow
rf = Roboflow(api_key="omuIheE81rZpDLFZJcNC")
project = rf.workspace().project("license-plate-recognition-rxg4e")
model = project.version(4).model
```
This code initializes the Roboflow client with the provided API key, accesses a specific project within the workspace, and selects a particular version of the model for license plate recognition.

### 3. Infer on a Local Image
```
print(model.predict("/content/1.jpg", confidence=40, overlap=30).json())

```
This line of code performs inference on a local image (1.jpg) using the selected model. It specifies the confidence threshold and overlap parameters for the predictions and prints the results in JSON format.

### 4. Visualize the Prediction
```
model.predict("/content/1.jpg", confidence=40, overlap=30).save("prediction.jpg")
```
This code saves the predicted bounding boxes on the input image (1.jpg) as a new image file named prediction.jpg.

### 5. Character Recognition using EasyOCR
```
import easyocr

# Initialize the EasyOCR Reader
reader = easyocr.Reader(['en'])

# Load the image
image_path = "1.jpg"

# Perform character recognition
results = reader.readtext(image_path)

# Print the recognized text
if results:
    for (text, _, _) in results:
        print("Recognized Text:", text)
else:
    print("No text detected in the image.")

```
This code utilizes the EasyOCR library for optical character recognition (OCR). It initializes an OCR reader for English language, reads text from an image (1.jpg), and prints the recognized text.

### 6. Explanation of Warning Message
```
WARNING:easyocr.easyocr:Neither CUDA nor MPS are available - defaulting to CPU. Note: This module is much faster with a GPU.
```
This warning message indicates that the EasyOCR library is running on the CPU instead of a GPU, which might affect its performance.

### 7. Additional Installation of EasyOCR
```
pip install easyocr
```
This command installs the EasyOCR library, which is used for optical character recognition tasks.

## 2)Licence_Plate_Detection_YOLO_V8 (1).ipynb:

### 1. 
```
```python
!pip install opencv-python-headless
```
This command installs the opencv-python-headless package, which is a headless version of the OpenCV library. The headless version omits the GUI-related functionality, making it more suitable for server-side or non-interactive environments.


```
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
```
These lines import the necessary libraries for image processing. cv2 is the OpenCV library, numpy is used for numerical operations, and cv2_imshow is a utility function for displaying images in Colab.

 
```
image_path = '/content/1.jpg'
```
This line defines the path to the image file (1.jpg) that will be processed.

### 2.
```
# Read the image
img = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```
These lines read the image from the specified path using cv2.imread() and convert it to grayscale using cv2.cvtColor(). Grayscale images are commonly used for processing as they contain only intensity information, making them simpler to analyze.


```
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
<ipython-input-4-0ccb82b15c44> in <cell line: 5>()
      3 
      4 # Convert the image to grayscale
----> 5 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

error: OpenCV(4.8.0) /io/opencv/modules/imgproc/src/color.cpp:182: error: (-215:Assertion failed) !_src.empty() in function 'cvtColor'
```
This error message indicates that there was an issue with converting the image to grayscale. Specifically, the error message (-215:Assertion failed) !_src.empty() suggests that the input image (img) is empty or invalid.


```
plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 30))
```
These lines use a pre-trained Cascade Classifier (plate_cascade) to detect potential regions containing license plates in the grayscale image (gray). The parameters such as scaleFactor, minNeighbors, and minSize control the sensitivity and accuracy of the detection algorithm.


```
# Draw rectangles around detected license plates
for (x, y, w, h) in plates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the image with detected license plates
cv2_imshow(img)
```
These lines draw rectangles around the detected license plates on the original color image (img). The cv2.rectangle() function is used to draw rectangles, and cv2_imshow() displays the image with the rectangles drawn.


```
```python
# Import necessary libraries
import cv2
from google.colab.patches import cv2_imshow
from google.colab import files
# Load the pre-trained Cascade Classifier for license plate detection
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

```
These lines import the necessary libraries for image processing in a Google Colab environment. cv2 is the OpenCV library, cv2_imshow is a utility function for displaying images in Colab, and files is used for uploading files. This line loads a pre-trained Cascade Classifier for license plate detection. The file path cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml' points to the XML file containing the trained model for detecting Russian license plates.

### 3. 
```
# Function to detect number plates in an image
def detect_number_plate(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect license plates in the image
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 30))

    # Draw rectangles around detected license plates
    for (x, y, w, h) in plates:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the image with detected license plates
    cv2_imshow(img)
```
This function detect_number_plate() takes an image path as input, reads the image, converts it to grayscale, detects license plates using the loaded Cascade Classifier, draws rectangles around detected license plates, and finally displays the image with the detected license plates using cv2_imshow.


```
# Function to upload an image and detect number plates
def detect_number_plate_from_upload():
    uploaded = files.upload()
    for filename in uploaded.keys():
        print('Uploaded file:', filename)
        detect_number_plate(filename)
# Upload an image and detect number plates
detect_number_plate_from_upload()

```
This function detect_number_plate_from_upload() allows users to upload an image file. It uses the files.upload() method to prompt the user to upload files. Once the file is uploaded, it calls the detect_number_plate() function to detect license plates in the uploaded image. This line invokes the detect_number_plate_from_upload() function to upload an image and detect number plates in it.

### 4 
```
```python
# Import necessary libraries
import cv2
from google.colab.patches import cv2_imshow
from google.colab import files
import easyocr
```
These lines import the necessary libraries for image processing, displaying images in Colab, uploading files, and performing optical character recognition (OCR) using EasyOCR.

```
# Load the pre-trained Cascade Classifier for license plate detection
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
```
This line loads a pre-trained Cascade Classifier for license plate detection. The file path cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml' points to the XML file containing the trained model for detecting Russian license plates.

```
# Function to detect number plates in an image
def detect_number_plate(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect license plates in the image
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 30))

    # Create an EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Draw rectangles around detected license plates
    for (x, y, w, h) in plates:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Crop the detected license plate region
        plate_img = gray[y:y+h, x:x+w]

        # Use EasyOCR to perform text detection on the license plate region
        result = reader.readtext(plate_img)

        # Print the detected text with highest accuracy
        if result:
            text = max(result, key=lambda x: x[2])  # Get the text with highest confidence score
            print('Detected Text:', text[1])

    # Display the image with detected license plates
    cv2_imshow(img)
```
This function detect_number_plate() takes an image path as input, reads the image, converts it to grayscale, detects license plates using the loaded Cascade Classifier, creates an EasyOCR reader, draws rectangles around detected license plates, crops the detected license plate region, performs text detection on the cropped region using EasyOCR, prints the detected text with the highest accuracy, and finally displays the image with detected license plates using cv2_imshow.

```
# Function to upload an image and detect number plates
def detect_number_plate_from_upload():
    uploaded = files.upload()
    for filename in uploaded.keys():
        print('Uploaded file:', filename)
        detect_number_plate(filename)
```
This function detect_number_plate_from_upload() allows users to upload an image file. It uses the files.upload() method to prompt the user to upload files. Once the file is uploaded, it calls the detect_number_plate() function to detect license plates in the uploaded image.
```
# Upload an image and detect number plates
detect_number_plate_from_upload()
```
This line invokes the detect_number_plate_from_upload() function to upload an image and detect number plates in it.


### 5. 
```
```python
# Install Tesseract OCR and its dependencies
!apt-get install tesseract-ocr
!apt-get install libtesseract-dev
```
These commands use apt-get to install Tesseract OCR and its development libraries (libtesseract-dev), which are required for running Tesseract OCR and its associated functionalities.

```
import pytesseract

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
```
This code configures the path to the Tesseract executable (tesseract_cmd) so that Python can locate and utilize the Tesseract OCR engine.

```
```python
# Install EasyOCR and its dependencies
!pip install easyocr
```
This command uses pip to install EasyOCR, a Python package for optical character recognition (OCR) tasks. EasyOCR simplifies the process of extracting text from images by providing an easy-to-use interface.

```
import easyocr
```




---

# **PYTHON FILES(.PY FILES):**

---
## 1) main.py :

## Automatic Number Plate Recognition (ANPR) using EasyOCR, OpenCV, and Matplotlib

This repository contains a Python script for performing Automatic Number Plate Recognition (ANPR) using EasyOCR for text recognition, OpenCV for image processing, and Matplotlib for visualization.

## Requirements

- Python 3.x
- EasyOCR
- OpenCV (cv2)
- Matplotlib
- GPU (optional, for faster processing with EasyOCR)

You can install the required Python packages using pip:

```bash
pip install easyocr opencv-python matplotlib
```

# **Pre-Trained Files(.pt)**

1)  best.pt : 
```
- The term "best.pt" typically refers to a file named "best.pt" that is used in machine learning, particularly in the
context of deep learning models trained using frameworks like PyTorch or TensorFlow. This file usually contains the
 parameters (weights and biases) of a trained neural network model
that achieved the best performance on a specific task or dataset during the training process.

- The ".pt" extension commonly indicates that the file is in PyTorch's native format for saving model checkpoints
or state dictionaries. These files can be used to restore the trained model's state for further evaluation,
inference, or fine-tuning without having to retrain the model from scratch.

- In summary, "best.pt" refers to a file containing the parameters of a neural network model trained using
 PyTorch (or another framework that uses the ".pt" extension) that achieved the best performance during
training.

```








