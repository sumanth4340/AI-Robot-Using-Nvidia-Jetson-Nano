import cv2 
import torch 
import numpy as np 
from jetson_inference import detectNet 
from jetson_utils import videoSource, videoOutput 
import RPi.GPIO as GPIO 
 
# Initialize Object Detection Model (YOLOv5) 
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5_model.pt') 
 
# Initialize Stereo Camera Input 
camera = videoSource("csi://0")  
 
def detect_objects(frame): 
    results = model(frame) 
    return results.pandas().xyxy[0]  # Extract detection results 
 
def control_motors(direction): 
    if direction == "left": 
        GPIO.output(left_motor, GPIO.HIGH) 
    elif direction == "right": 
        GPIO.output(right_motor, GPIO.HIGH) 
    else: 
        GPIO.output(left_motor, GPIO.LOW) 
        GPIO.output(right_motor, GPIO.LOW) 
 
while True: 
    frame = camera.Capture() 
    detections = detect_objects(frame) 
     
    for obj in detections.iterrows(): 
        label = obj[1]['name'] 
        x_center = (obj[1]['xmin'] + obj[1]['xmax']) / 2 
         
        if x_center < frame.shape[1] // 3: 
            control_motors("left") 

        elif x_center > (frame.shape[1] // 3) * 2: 
            control_motors("right") 
        else: 
            control_motors("stop") 
 
    cv2.imshow("Object Detection", frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
