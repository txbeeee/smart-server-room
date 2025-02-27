import serial 

import cv2 

import time 

import datetime 

import torch 

import os 

from ultralytics import YOLO 

from plyer import notification 

 

SERIAL_PORT = "/dev/ttyACM0"   

BAUD_RATE = 115200 

VIDEO_FOLDER = "detect/vid_detect"   

LOG_FOLDER = "detect/logs_detect"   

FRAME_RATE = 15   

notification_shown = False 

 

CLASS_NAMES = { 

    0: 'person',1: 'bicycle',2: 'car',3: 'motorcycle',4: 'airplane',5: 'bus',6: 'train',7: 'truck',8: 'boat',9: 'traffic light',10: 'fire hydrant',11: 'stop sign',12: 'parking meter',13: 'bench',14: 'bird',15: 'cat',16: 'dog',17: 'horse',18: 'sheep',19: 'cow',20: 'elephant',21: 'bear',22: 'zebra',23: 'giraffe',24: 'backpack',25: 'umbrella',26: 'handbag',27: 'tie',28: 'suitcase',29: 'frisbee',30: 'skis',31: 'snowboard',32: 'sports ball',33: 'kite',34: 'baseball bat',35: 'baseball glove',36: 'skateboard',37: 'surfboard',38: 'tennis racket',39: 'bottle',40: 'wine glass',41: 'cup',42: 'fork',43: 'knife',44: 'spoon',45: 'bowl',46: 'banana',47: 'apple',48: 'sandwich',49: 'orange',50: 'broccoli',51: 'carrot',52: 'hot dog',53: 'pizza',54: 'donut',55: 'cake',56: 'chair',57: 'couch',58: 'potted plant',59: 'bed',60: 'dining table',61: 'toilet',62: 'tv',63: 'laptop',64: 'mouse',65: 'remote',66: 'keyboard',67: 'cell phone',68: 'microwave',69: 'oven',70: 'toaster',71: 'sink',72: 'refrigerator',73: 'book',74: 'clock',75: 'vase',76: 'scissors',77: 'teddy bear',78: 'hair drier',79: 'toothbrush'} 

os.makedirs(VIDEO_FOLDER, exist_ok=True) 

os.makedirs(LOG_FOLDER, exist_ok=True) 

 

ser = serial.Serial(SERIAL_PORT, baudrate=BAUD_RATE, timeout=1) 

 

 

model = YOLO("yolov8l.pt")   

 

def log_detection(log_filename, classes, frame_time): 

    try: 

        with open(log_filename, "a") as log_file: 

            log_file.write(f"{frame_time} - Classes detected: {', '.join(classes)}\n") 

    except Exception as e: 

        print(f"Error writing log: {e}")   

 

try: 

    while True: 

        if ser.in_waiting > 0: 

            data = ser.readline().decode('utf-8').strip() 

            if data == "MOTION_DETECTED": 

                print("Motion detected! Starting video capture and detection...") 

 

                cap = cv2.VideoCapture(0) 

                if not cap.isOpened(): 

                    print("Error: Could not open camera.") 

                    continue 

 

                cap.set(cv2.CAP_PROP_FPS, FRAME_RATE) 

 

                timestamp = datetime.datetime.now().strftime("%d.%m.%Y_%H:%M:%S") 

                video_filename = os.path.join(VIDEO_FOLDER, f"detected_video_{timestamp}.avi") 

 

                fourcc = cv2.VideoWriter_fourcc(*"XVID") 

                out = cv2.VideoWriter(video_filename, fourcc, FRAME_RATE, (640, 480)) 

 

                log_filename = os.path.join(LOG_FOLDER, f"detection_log_{timestamp}.txt") 

 

                start_time = time.time() 

                last_log_time = start_time 

                notification_shown = False  # Сброс флага уведомления 

 

                while time.time() - start_time < 20: 

                    ret, frame = cap.read() 

                    if not ret: 

                        print("Error: Failed to capture frame.") 

                        break 

 

                    results = model(frame, conf=0.65) 

                    detected_classes = set() 

 

                    for box in results[0].boxes: 

                        cls = int(box.cls[0]) 

                        if cls in CLASS_NAMES: 

                            detected_classes.add(CLASS_NAMES[cls]) 

 

                    current_time = time.time() 

                    if detected_classes and current_time - last_log_time >= 0.5: 

                        frame_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") 

                        log_detection(log_filename, list(detected_classes), frame_time) 

                        last_log_time = current_time 

 

                    if "person" in detected_classes and not notification_shown: 

                        notification.notify( 

                            title="Person detected!", 

                            message="An object of class 'person' was detected in the frame", 

                            timeout=5 

                        ) 

                        notification_shown = True 

 

                    annotated_frame = results[0].plot() 

                    cv2.imshow("Detections", annotated_frame) 

 

                    out.write(annotated_frame) 

 

                    if cv2.waitKey(1) & 0xFF == ord('q'): 

                        break 

 

                cap.release() 

                out.release() 

                cv2.destroyAllWindows() 

                print(f"Video saved to {video_filename}") 

                print(f"Log saved to {log_filename}") 

                print("Video capture and detection completed.") 

 

except KeyboardInterrupt: 

    print("Script interrupted by user.") 

finally: 

    if 'cap' in locals() and cap.isOpened(): 

        cap.release() 

    if 'out' in locals(): 

        out.release() 

    cv2.destroyAllWindows() 

    ser.close() 

    print("Resources released and script terminated.") 
