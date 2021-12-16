import streamlit as st
from PIL import Image
import cv2
import numpy as np

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Defining a function that will do the detections
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    return frame

    

st.title("OpenCV Haar Cascades Demo")
st.sidebar.title('Configuration')
method = st.sidebar.radio('Select Input', options=['Webcam', 'Image'])
camera = cv2.VideoCapture(0)
FRAME = st.image([])

if method == 'Image':
    img_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    if img_file is not None:
        img = Image.open(img_file)
        img_arr = np.array(img.convert('RGB'))
        gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        canvas = detect(gray, img_arr)
        st.image(canvas, width=640)

while method == 'Webcam':
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    canvas = detect(gray, frame)
    FRAME.image(canvas, width=640)
