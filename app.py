from flask import Flask, render_template, request, jsonify
import cv2
import dlib
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)

# Load the pre-trained face detector
detector = dlib.get_frontal_face_detector()

# Load the pre-trained shape predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def detect_gaze(landmarks, gray, frame):
    left_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                (landmarks.part(37).x, landmarks.part(37).y),
                                (landmarks.part(38).x, landmarks.part(38).y),
                                (landmarks.part(39).x, landmarks.part(39).y),
                                (landmarks.part(40).x, landmarks.part(40).y),
                                (landmarks.part(41).x, landmarks.part(41).y)], np.int32)
    right_eye_region = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                 (landmarks.part(43).x, landmarks.part(43).y),
                                 (landmarks.part(44).x, landmarks.part(44).y),
                                 (landmarks.part(45).x, landmarks.part(45).y),
                                 (landmarks.part(46).x, landmarks.part(46).y),
                                 (landmarks.part(47).x, landmarks.part(47).y)], np.int32)

    height, width = gray.shape
    mask = np.zeros((height, width), np.uint8)

    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    left_eye = cv2.bitwise_and(gray, gray, mask=mask)

    mask.fill(0)  # Clear the mask for the right eye
    cv2.polylines(mask, [right_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [right_eye_region], 255)
    right_eye = cv2.bitwise_and(gray, gray, mask=mask)

    left_eye_center = (landmarks.part(36).x + landmarks.part(39).x) // 2, (landmarks.part(36).y + landmarks.part(39).y) // 2
    right_eye_center = (landmarks.part(42).x + landmarks.part(45).x) // 2, (landmarks.part(42).y + landmarks.part(45).y) // 2

    if left_eye_center[0] < width // 2:
        return "Focused"
    else:
        return "Not Focused"

def process_frame(image_data):
    image_data = base64.b64decode(image_data.split(',')[1])
    image = Image.open(BytesIO(image_data))
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        gaze = detect_gaze(landmarks, gray, frame)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 64, 100), -1)

        cv2.putText(frame, gaze, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    data = request.json
    image_data = data['image']
    frame = process_frame(image_data)
    return jsonify({'image': frame})

@app.route('/')
def index():
    return render_template('index.html')

# This block is optional for Google Cloud Run but useful for local development.
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
