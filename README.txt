# Gaze Detection Application

This application detects gaze using a webcam feed and displays whether the user is "Focused" or "Not Focused".

## Setup

1. Clone this repository.
2. Install the required Python packages: 
    
pip install flask opencv-python dlib numpy
   
3. Download `shape_predictor_68_face_landmarks.dat` and place it in the project directory :

https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2

4. Run the Flask application:
 
    python final.py

## Usage

- Open a web browser and navigate to `http://127.0.0.1:5000/` to view the application.
