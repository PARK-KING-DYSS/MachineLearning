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
This line imports the easyocr module, allowing you to utilize EasyOCR functionalities in your Python code.

```
reader = easyocr.Reader(['en'])
```
This code initializes an EasyOCR reader object with English ('en') as the language parameter. The reader will be used to perform text detection and recognition on images.

### 6. 
```
```python
import cv2
import pandas as pd
from google.colab.patches import cv2_imshow
from google.colab import files
import easyocr
from datetime import datetime
```
These lines import necessary libraries for image processing (cv2), data manipulation (pandas), displaying images in Colab (cv2_imshow), uploading files (files), performing optical character recognition (easyocr), and working with date and time (datetime).

```
These lines import necessary libraries for image processing (cv2), data manipulation (pandas), displaying images in Colab (cv2_imshow), uploading files (files), performing optical character recognition (easyocr), and working with date and time (datetime).
```
This code loads a pre-trained cascade classifier for detecting license plates in images.

```
# Create an EasyOCR reader
reader = easyocr.Reader(['en'])
```
Here, an EasyOCR reader object is created with English as the chosen language for text recognition.

```
# Function to detect number plates in an image
def detect_number_plate(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect license plates in the image
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 30))

    # Prompt user for information
    name = input("Enter your name: ")
    phone_number = input("Enter your phone number: ")

    # Get current date and time
    current_time = datetime.now()
    time_in = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # Create a list to store the data
    data = []

    # Process each detected license plate
    for (x, y, w, h) in plates:
        # Crop the detected license plate region
        plate_img = gray[y:y+h, x:x+w]

        # Use EasyOCR to perform text detection on the license plate region
        result = reader.readtext(plate_img)

        # Print the detected text with highest accuracy
        if result:
            text = max(result, key=lambda x: x[2])[1]  # Get the text with highest confidence score

            # Prompt user for time out
            time_out = input("Enter the time out (HH:MM:SS) for the detected plate text (or leave blank for current time): ")

            # Use current time if time_out is blank
            if not time_out:
                time_out = current_time.strftime("%H:%M:%S")

            # Append data to the list
            data.append({'Name': name, 'Phone number': phone_number, 'Number plate detected text': text, 'Time in': time_in, 'Time out': time_out})

    # Display the image with detected license plates
    cv2_imshow(img)

    return data
```
This function detects number plates in an image, prompts the user for their name and phone number, captures the current date and time, detects text from the license plate using EasyOCR, and collects relevant data such as the name, phone number, detected text, and timestamps.

```
# Function to upload an image and detect number plates
def detect_number_plate_from_upload():
    uploaded = files.upload()
    all_data = []
    for filename in uploaded.keys():
        print('Uploaded file:', filename)
        data = detect_number_plate(filename)
        all_data.extend(data)
    return all_data
```
This function allows users to upload an image and detects number plates in the uploaded image. It returns a list containing all the collected data.

```
# Upload an image and detect number plates
all_data = detect_number_plate_from_upload()

# Create a DataFrame from the collected data
df_final = pd.DataFrame(all_data)

# Print the final DataFrame with formatted output
print("\nFinal DataFrame:")
print(df_final)
```
This part of the code executes the process of uploading an image, detecting number plates, collecting data, creating a DataFrame from the collected data, and printing the final DataFrame with the detected information.

```
# Create a DataFrame from the collected data
df_final = pd.DataFrame(all_data)
```
The DataFrame df_final will have columns named 'Name', 'Phone number', 'Number plate detected text', 'Time in', and 'Time out', based on the keys in the dictionaries stored in all_data.


```
import cv2
import pandas as pd
from google.colab.patches import cv2_imshow
from google.colab import files
import easyocr
from datetime import datetime

# Load the pre-trained Cascade Classifier for license plate detection
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

# Create an EasyOCR reader
reader = easyocr.Reader(['en'])

# Function to detect number plates in an image
def detect_number_plate(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect license plates in the image
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 30))

    # Prompt user for information
    name = input("Enter your name: ")
    phone_number = input("Enter your phone number: ")

    # Get current date and time
    current_time = datetime.now()
    time_in = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # Create a list to store the data
    data = []

    # Process each detected license plate
    for (x, y, w, h) in plates:
        # Crop the detected license plate region
        plate_img = gray[y:y+h, x:x+w]

        # Use EasyOCR to perform text detection on the license plate region
        result = reader.readtext(plate_img)

        # Print the detected text with highest accuracy
        if result:
            text = max(result, key=lambda x: x[2])[1]  # Get the text with highest confidence score
            print('Detected Text:', text)

            # Prompt user for time out
            time_out = input("Enter the time out (HH:MM:SS) for the detected plate text (or leave blank for current time): ")

            # Use current time if time_out is blank
            if not time_out:
                time_out = current_time.strftime("%H:%M:%S")

            # Append data to the list
            data.append({'Name': name, 'Phone number': phone_number, 'Number plate detected text': text, 'Time in': time_in, 'Time out': time_out})

    # Display the image with detected license plates
    cv2_imshow(img)

    return data

# Function to upload an image and detect number plates
def detect_number_plate_from_upload():
    uploaded = files.upload()
    all_data = []
    for filename in uploaded.keys():
        print('Uploaded file:', filename)
        data = detect_number_plate(filename)
        all_data.extend(data)
    return all_data

# Load existing data from CSV (if available)
try:
    df_final = pd.read_csv('license_plate_data.csv')
except FileNotFoundError:
    df_final = pd.DataFrame(columns=['Name', 'Phone number', 'Number plate detected text', 'Time in', 'Time out'])

# Append new data
new_data = detect_number_plate_from_upload()
df_final = df_final.append(new_data, ignore_index=True)

# Save updated DataFrame to CSV
df_final.to_csv('license_plate_data.csv', index=False)

# Print the final DataFrame with formatted output
print("\nFinal DataFrame:")
print(df_final)
```
-Importing Libraries
import cv2: Imports OpenCV library for image processing.
import pandas as pd: Imports Pandas library for data manipulation and analysis.
from google.colab.patches import cv2_imshow: Imports cv2_imshow function from Google Colab patches for displaying images.
from google.colab import files: Imports files module from Google Colab for uploading files.
import easyocr: Imports EasyOCR library for optical character recognition.
from datetime import datetime: Imports datetime class from the datetime module for working with dates and times.

-Loading Pre-trained Models
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'): Loads the pre-trained Cascade Classifier for license plate detection from the OpenCV data directory.
reader = easyocr.Reader(['en']): Creates an EasyOCR reader configured to recognize English text.

-Function to Detect Number Plates
detect_number_plate(image_path): This function takes the path of an image as input, detects license plates in the image, prompts the user for additional information (name, phone number, time out), and returns a list of dictionaries containing the collected data.

-Function to Upload Image and Detect Number Plates
detect_number_plate_from_upload(): This function allows the user to upload an image, detects number plates in the uploaded image(s), and returns a list of dictionaries containing the collected data.

-Loading Existing Data from CSV (if available)
The code tries to read existing data from a CSV file named 'license_plate_data.csv' into a DataFrame. If the file does not exist, an empty DataFrame is created.

-Appending New Data and Saving to CSV
After detecting number plates in the uploaded image(s), the new data is appended to the existing DataFrame.
The updated DataFrame is then saved to the CSV file 'license_plate_data.csv'.
 
-Printing the Final DataFrame
Finally, the code prints the final DataFrame showing the collected data.

```
import cv2
import pandas as pd
from google.colab.patches import cv2_imshow
from google.colab import files
import easyocr
from datetime import datetime

# Load the pre-trained Cascade Classifier for license plate detection
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

# Create an EasyOCR reader
reader = easyocr.Reader(['en'])

# Function to detect number plates in an image
def detect_number_plate(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect license plates in the image
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 30))

    # Prompt user for information
    name = input("Enter your name: ")
    phone_number = input("Enter your phone number: ")

    # Get current date and time
    current_time = datetime.now()
    time_in = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # Create a list to store the data
    data = []

    # Process each detected license plate
    for (x, y, w, h) in plates:
        # Crop the detected license plate region
        plate_img = gray[y:y+h, x:x+w]

        # Use EasyOCR to perform text detection on the license plate region
        result = reader.readtext(plate_img)

        # Print the detected text with highest accuracy
        if result:
            text = max(result, key=lambda x: x[2])[1]  # Get the text with highest confidence score
            print('Detected Text:', text)

            # Prompt user for time out
            time_out = input("Enter the time out (HH:MM:SS) for the detected plate text (or leave blank for current time): ")

            # Use current time if time_out is blank
            if not time_out:
                time_out = current_time.strftime("%H:%M:%S")

            # Append data to the list
            data.append({'Name': name, 'Phone number': phone_number, 'Number plate detected text': text, 'Time in': time_in, 'Time out': time_out})

    # Display the image with detected license plates
    cv2_imshow(img)

    return data

# Function to upload an image and detect number plates
def detect_number_plate_from_upload():
    uploaded = files.upload()
    all_data = []
    for filename in uploaded.keys():
        print('Uploaded file:', filename)
        data = detect_number_plate(filename)
        all_data.extend(data)
    return all_data

# Load existing data from CSV (if available)
try:
    df_final = pd.read_csv('license_plate_data.csv')
except FileNotFoundError:
    df_final = pd.DataFrame(columns=['Name', 'Phone number', 'Number plate detected text', 'Time in', 'Time out'])

# Append new data
new_data = detect_number_plate_from_upload()
df_final = df_final.append(new_data, ignore_index=True)

# Save updated DataFrame to CSV
df_final.to_csv('license_plate_data.csv', index=False)

# Print the final DataFrame with formatted output
print("\nFinal DataFrame:")
print(df_final)
```

-Importing Libraries:
cv2, pd: Imports OpenCV and Pandas libraries for image processing and data manipulation.
cv2_imshow: Imports the cv2_imshow function for displaying images in Google Colab.
files: Imports the files module from Google Colab for file upload functionality.
easyocr: Imports EasyOCR library for optical character recognition.
datetime: Imports the datetime class from the datetime module for working with dates and times.

-Loading Pre-trained Models:
plate_cascade: Loads the pre-trained Cascade Classifier for license plate detection from the OpenCV data directory.
reader: Creates an EasyOCR reader configured to recognize English text.

-Function to Detect Number Plates:
detect_number_plate(image_path): Takes the path of an image as input, detects license plates in the image, prompts the user for name and phone number, and returns a list of dictionaries containing the collected data.

-Function to Upload Image and Detect Number Plates:
detect_number_plate_from_upload(): Allows the user to upload an image, detects number plates in the uploaded image(s), and returns a list of dictionaries containing the collected data.

-Loading Existing Data from CSV:
Tries to read existing data from a CSV file named 'license_plate_data.csv' into a DataFrame. If the file does not exist, an empty DataFrame is created.

-Appending New Data and Saving to CSV:
After detecting number plates in the uploaded image(s), the new data is appended to the existing DataFrame.
The updated DataFrame is then saved to the CSV file 'license_plate_data.csv'.

-Printing the Final DataFrame:
Prints the final DataFrame showing the collected data.

-Output:
The script processes an uploaded image, detects the license plate number, and collects user information (name, phone number, time out).
It then displays the image with the detected license plate and prints the final DataFrame containing the collected data.

```
df_final

     
Name	Phone number	Number plate detected text	Time in	Time out
0	Krupa	1111111111	KA 64 N 00991	2024-02-23 12:00:21	7:20 PM
1	Darshan	123456789	'NHZODV2366	2024-02-23 12:01:14	7:30 Pm
2	Sankya	6666666666	TYI2hh5766	2024-02-23 12:02:06	9:20 PM
```
DataFrame Display:

The DataFrame is displayed in tabular format, showing the collected data.
Each row represents a unique instance of a detected license plate along with associated user information.
Columns represent different attributes associated with the detected license plates and the user information.
Columns:

Name: Represents the name of the person associated with the detected license plate.
Phone number: Represents the phone number of the person associated with the detected license plate.
Number plate detected text: Represents the text detected from the license plate.
Time in: Represents the time when the image was processed and the license plate was detected.
Time out: Represents the time provided by the user or the current time if left blank during data collection.
Rows:

Each row corresponds to a unique instance of a detected license plate along with the associated user information.
Explanation:

Row 1:

Name: Krupa
Phone number: 1111111111
Number plate detected text: KA 64 N 00991
Time in: 2024-02-23 12:00:21
Time out: 7:20 PM
Row 2:

Name: Darshan
Phone number: 123456789
Number plate detected text: NHZODV2366
Time in: 2024-02-23 12:01:14
Time out: 7:30 PM
Row 3:

Name: Sankya
Phone number: 6666666666
Number plate detected text: TYI2hh5766
Time in: 2024-02-23 12:02:06
Time out: 9:20 PM
Time Format:

The time values in the Time in and Time out columns are formatted as YYYY-MM-DD HH:MM:SS.
The time out is provided in HH:MM AM/PM format as input by the user.

## 3) Scratch_CAR.ipynb
```

Open In Colab

pip install easyocr

     
Collecting easyocr
  Downloading easyocr-1.7.1-py3-none-any.whl (2.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0.0/2.9 MB ? eta -:--:--
     ╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0.1/2.9 MB 1.7 MB/s eta 0:00:02
     ━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━ 0.9/2.9 MB 13.4 MB/s eta 0:00:01
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╸ 2.9/2.9 MB 32.5 MB/s eta 0:00:01
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.9/2.9 MB 23.0 MB/s eta 0:00:00
Requirement already satisfied: torch in /usr/local/lib/python3.10/dist-packages (from easyocr) (2.1.0+cu121)
Requirement already satisfied: torchvision>=0.5 in /usr/local/lib/python3.10/dist-packages (from easyocr) (0.16.0+cu121)
Requirement already satisfied: opencv-python-headless in /usr/local/lib/python3.10/dist-packages (from easyocr) (4.8.0.74)
Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from easyocr) (1.11.4)
Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from easyocr) (1.25.2)
Requirement already satisfied: Pillow in /usr/local/lib/python3.10/dist-packages (from easyocr) (9.4.0)
Requirement already satisfied: scikit-image in /usr/local/lib/python3.10/dist-packages (from easyocr) (0.19.3)
Collecting python-bidi (from easyocr)
  Downloading python_bidi-0.4.2-py2.py3-none-any.whl (30 kB)
Requirement already satisfied: PyYAML in /usr/local/lib/python3.10/dist-packages (from easyocr) (6.0.1)
Requirement already satisfied: Shapely in /usr/local/lib/python3.10/dist-packages (from easyocr) (2.0.2)
Collecting pyclipper (from easyocr)
  Downloading pyclipper-1.3.0.post5-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (908 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 908.3/908.3 kB 45.6 MB/s eta 0:00:00
Collecting ninja (from easyocr)
  Downloading ninja-1.11.1.1-py2.py3-none-manylinux1_x86_64.manylinux_2_5_x86_64.whl (307 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 307.2/307.2 kB 30.6 MB/s eta 0:00:00
Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from torchvision>=0.5->easyocr) (2.31.0)
Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch->easyocr) (3.13.1)
Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from torch->easyocr) (4.9.0)
Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch->easyocr) (1.12)
Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch->easyocr) (3.2.1)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch->easyocr) (3.1.3)
Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch->easyocr) (2023.6.0)
Requirement already satisfied: triton==2.1.0 in /usr/local/lib/python3.10/dist-packages (from torch->easyocr) (2.1.0)
Requirement already satisfied: six in /usr/local/lib/python3.10/dist-packages (from python-bidi->easyocr) (1.16.0)
Requirement already satisfied: imageio>=2.4.1 in /usr/local/lib/python3.10/dist-packages (from scikit-image->easyocr) (2.31.6)
Requirement already satisfied: tifffile>=2019.7.26 in /usr/local/lib/python3.10/dist-packages (from scikit-image->easyocr) (2024.1.30)
Requirement already satisfied: PyWavelets>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from scikit-image->easyocr) (1.5.0)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from scikit-image->easyocr) (23.2)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch->easyocr) (2.1.5)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision>=0.5->easyocr) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision>=0.5->easyocr) (2.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision>=0.5->easyocr) (2.0.7)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision>=0.5->easyocr) (2023.7.22)
Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch->easyocr) (1.3.0)
Installing collected packages: pyclipper, ninja, python-bidi, easyocr
Successfully installed easyocr-1.7.1 ninja-1.11.1.1 pyclipper-1.3.0.post5 python-bidi-0.4.2

# Install required libraries
!pip install opencv-python-headless

# Import necessary libraries
from google.colab import files
import cv2
from google.colab.patches import cv2_imshow

# Load the pre-trained Cascade Classifier for license plate detection
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

# Function to detect license plates in an image
def detect_license_plate(image_path):
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

# Function to handle image upload and license plate detection
def upload_and_detect_license_plate():
    uploaded = files.upload()
    for filename in uploaded.keys():
        print('Uploaded file:', filename)
        detect_license_plate(filename)

# Upload an image and detect license plates
upload_and_detect_license_plate()

     
Requirement already satisfied: opencv-python-headless in /usr/local/lib/python3.10/dist-packages (4.8.0.74)
Requirement already satisfied: numpy>=1.21.2 in /usr/local/lib/python3.10/dist-packages (from opencv-python-headless) (1.25.2)
Upload widget is only available when the cell has been executed in the current browser session. Please rerun this cell to enable.
Saving Krupa.jpg to Krupa (2).jpg
Uploaded file: Krupa (2).jpg


# Install required libraries
!pip install opencv-python-headless

# Import necessary libraries
from google.colab import files
import cv2
from google.colab.patches import cv2_imshow

# Load the pre-trained Cascade Classifier for license plate detection
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

# Function to detect license plates in an image
def detect_license_plate(image_path):
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

# Function to handle image upload and license plate detection
def upload_and_detect_license_plate():
    uploaded = files.upload()
    for filename in uploaded.keys():
        print('Uploaded file:', filename)
        detect_license_plate(filename)

# Upload an image and detect license plates
upload_and_detect_license_plate()

     
Requirement already satisfied: opencv-python-headless in /usr/local/lib/python3.10/dist-packages (4.8.0.74)
Requirement already satisfied: numpy>=1.21.2 in /usr/local/lib/python3.10/dist-packages (from opencv-python-headless) (1.25.2)
Upload widget is only available when the cell has been executed in the current browser session. Please rerun this cell to enable.
Saving 700.jpg to 700.jpg
Uploaded file: 700.jpg


# Install required libraries
!pip install opencv-python-headless

# Import necessary libraries
from google.colab import files
import cv2
from google.colab.patches import cv2_imshow

# Load the pre-trained Cascade Classifier for license plate detection
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

# Function to detect license plates in an image
def detect_license_plate(image_path):
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

# Function to handle image upload and license plate detection
def upload_and_detect_license_plate():
    uploaded = files.upload()
    for filename in uploaded.keys():
        print('Uploaded file:', filename)
        detect_license_plate(filename)

# Upload an image and detect license plates
upload_and_detect_license_plate()

     
Requirement already satisfied: opencv-python-headless in /usr/local/lib/python3.10/dist-packages (4.8.0.74)
Requirement already satisfied: numpy>=1.21.2 in /usr/local/lib/python3.10/dist-packages (from opencv-python-headless) (1.25.2)
Upload widget is only available when the cell has been executed in the current browser session. Please rerun this cell to enable.
Saving lalu.jpg to lalu.jpg
Uploaded file: lalu.jpg


# Install required libraries
!pip install opencv-python-headless

# Import necessary libraries
from google.colab import files
import cv2
from google.colab.patches import cv2_imshow

# Load the pre-trained Cascade Classifier for license plate detection
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

# Function to detect license plates in an image
def detect_license_plate(image_path):
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

# Function to handle image upload and license plate detection
def upload_and_detect_license_plate():
    uploaded = files.upload()
    for filename in uploaded.keys():
        print('Uploaded file:', filename)
        detect_license_plate(filename)

# Upload an image and detect license plates
upload_and_detect_license_plate()

     
Requirement already satisfied: opencv-python-headless in /usr/local/lib/python3.10/dist-packages (4.8.0.74)
Requirement already satisfied: numpy>=1.21.2 in /usr/local/lib/python3.10/dist-packages (from opencv-python-headless) (1.25.2)
Upload widget is only available when the cell has been executed in the current browser session. Please rerun this cell to enable.
Saving landrover.jpg to landrover.jpg
Uploaded file: landrover.jpg


# Install required libraries
!pip install opencv-python-headless

# Import necessary libraries
from google.colab import files
import cv2
from google.colab.patches import cv2_imshow

# Load the pre-trained Cascade Classifier for license plate detection
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

# Function to detect license plates in an image
def detect_license_plate(image_path):
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

# Function to handle image upload and license plate detection
def upload_and_detect_license_plate():
    uploaded = files.upload()
    for filename in uploaded.keys():
        print('Uploaded file:', filename)
        detect_license_plate(filename)

# Upload an image and detect license plates
upload_and_detect_license_plate()

     
Requirement already satisfied: opencv-python-headless in /usr/local/lib/python3.10/dist-packages (4.8.0.74)
Requirement already satisfied: numpy>=1.21.2 in /usr/local/lib/python3.10/dist-packages (from opencv-python-headless) (1.25.2)
Upload widget is only available when the cell has been executed in the current browser session. Please rerun this cell to enable.
Saving rover.jpg to rover.jpg
Uploaded file: rover.jpg
```
1. **Installation of Required Libraries**:
   - The code starts with installing the required library `opencv-python-headless` for image processing tasks.

2. **Importing Libraries**:
   - Necessary libraries are imported, including `files` from `google.colab` for file uploading, `cv2` for image processing, and `cv2_imshow` from `google.colab.patches` for displaying images.

3. **Loading Pre-trained Cascade Classifier**:
   - The pre-trained Cascade Classifier for license plate detection (`haarcascade_russian_plate_number.xml`) is loaded using OpenCV's `CascadeClassifier` class.

4. **Function to Detect License Plates** (`detect_license_plate`):
   - This function takes the path of an image as input.
   - It reads the image, converts it to grayscale, and detects license plates using the loaded Cascade Classifier.
   - Detected license plates are outlined with rectangles using OpenCV's `rectangle` function.
   - The image with detected license plates is displayed using `cv2_imshow`.

5. **Function to Upload and Detect License Plates** (`upload_and_detect_license_plate`):
   - This function allows users to upload an image and calls the `detect_license_plate` function to detect license plates in the uploaded image.
   - It utilizes `files.upload()` to prompt the user to upload an image file.
   - For each uploaded file, the function calls `detect_license_plate` to process and display the image.

6. **Image Upload and License Plate Detection**:
   - The `upload_and_detect_license_plate` function is called multiple times to demonstrate license plate detection in different images.
   - Each time the function is called, the user is prompted to upload an image, and the detected license plates are displayed.

7. **Explanation**:
   - The code is designed to be executed in a Jupyter Notebook environment, such as Google Colab.
   - It allows users to upload images containing vehicles with visible license plates and then detects and displays the license plates within those images.
   - The license plate detection is performed using the pre-trained Cascade Classifier, which detects rectangular regions resembling license plates in the images.
   - Detected license plates are outlined with rectangles to highlight them in the displayed images.

8. **Output**:
   - The output of the code execution is not included in the provided snippet but would consist of the uploaded images with detected license plates outlined by rectangles.
  





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








