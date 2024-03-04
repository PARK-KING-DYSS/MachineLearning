from flask import Flask, render_template, request
import cv2
import numpy as np
import pandas as pd
import easyocr
from datetime import datetime

app = Flask(__name__)

# Load the pre-trained Cascade Classifier for license plate detection
plate_cascade = cv2.CascadeClassifier('Darshan_imrpoved_casscader_model.xml')  # Assuming the model file is in the same directory as your app.py

# Create an EasyOCR reader
reader = easyocr.Reader(['en'])

# Function to detect number plates in an image
def detect_number_plate(img, name, phone_number):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect license plates in the image
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 30))

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

            # Append data to the list
            data.append({'Name': name, 'Phone number': phone_number, 'Number plate detected text': text, 'Time in': time_in})

    return data

@app.route('/')
def index():
    try:
        df_final = pd.read_csv('license_plate_data.csv')
    except FileNotFoundError:
        df_final = pd.DataFrame(columns=['Name', 'Phone number', 'Number plate detected text', 'Time in'])
    except pd.errors.EmptyDataError:
        df_final = pd.DataFrame(columns=['Name', 'Phone number', 'Number plate detected text', 'Time in'])
        
    return render_template('index.html', df=df_final.to_html())

@app.route('/upload', methods=['POST'])
def upload():
    name = request.form['name']
    phone_number = request.form['phone_number']

    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    data = detect_number_plate(img, name, phone_number)

    # Append new data to DataFrame
    try:
        df_final = pd.read_csv('license_plate_data.csv')
    except FileNotFoundError:
        df_final = pd.DataFrame(columns=['Name', 'Phone number', 'Number plate detected text', 'Time in'])
    except pd.errors.EmptyDataError:
        df_final = pd.DataFrame(columns=['Name', 'Phone number', 'Number plate detected text', 'Time in'])
        
    new_data_df = pd.DataFrame(data)
    df_final = pd.concat([df_final, new_data_df], ignore_index=True)

    # Save updated DataFrame to CSV
    df_final.to_csv('license_plate_data.csv', index=False)

    return render_template('index.html', df=df_final.to_html())

if __name__ == '__main__':
    app.run(debug=True)
