#!/usr/bin/env python
# coding: utf-8

# ``` Oscar Lares - ELEE 6280 - Machine Learning for Computer Vision```

import requests
import json
import vlc
import cv2
import time
import imutils
import numpy as np

# personal API key
api_key = "use your own API key here - obtain from GDOT 511 website"

# Set up the API endpoint URL and query parameters
url = "https://511ga.org/api/v2/get/cameras"

params = {"key": api_key, "format":'json'}

# Make the API call and retrieve the response data
response = requests.get(url, params=params)
data = json.loads(response.text)

# Loop through the camera data and print the video URL for each camera, and sort through to find camera to use for analysis
updated_data = []

for camera in data:
    video_url = camera['VideoUrl']
    if video_url is not None:
        updated_data.append(camera)

for index, cam in enumerate(updated_data):
    if updated_data[index]['VideoUrl'] == 'https://vss5live.dot.ga.gov/lo/bibb-cam-014.stream/playlist.m3u8':
        print(cam)


# https://vss5live.dot.ga.gov/lo/bibb-cam-014.stream/playlist.m3u8

#Load Haar Cascade Classifier
car_cascade = cv2.CascadeClassifier('cars.xml')

#Load YOLO model
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

url = 'https://vss5live.dot.ga.gov/lo/bibb-cam-014.stream/playlist.m3u8'
cap = cv2.VideoCapture(url)

# Define the output video file name and codec
output_haar = 'Haar-output.avi'
output_yolo = 'YOLO-output.avi'
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

# Get the video dimensions and create the video writer object
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out_haar = cv2.VideoWriter(output_haar, fourcc, fps, (width, height))
out_yolo = cv2.VideoWriter(output_yolo, fourcc, fps, (width, height))

#HAAR Loop

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Resize the frame for faster processing
    frame = imutils.resize(frame, width=500)

    # Convert the frame to grayscale for object detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars in the frame using the pre-trained model
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=4, minSize=(30, 30))

    # Draw a rectangle around each detected car
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Write the frame with the detected cars to the output video
    out_haar.write(frame)

    # Display the frame with the detected cars
    cv2.imshow('frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object, video writer object, and close all windows
cap.release()
out_haar.release()
cv2.destroyAllWindows()

#YOLO Loop

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (608, 608), swapRB=True, crop=False)  # Change input size to 608x608
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    # Process detections
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter for cars (class ID 2 in the COCO dataset) and a confidence threshold
            if class_id in (2, 7) and confidence > 0.5:
                
                # Bounding box coordinates
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, classes[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("Car Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    # Break the loop when 'q' is pressed
    if key == ord("q"):
        break

cap.release()
out_yolo.release()
cv2.destroyAllWindows()

