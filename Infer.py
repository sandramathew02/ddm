import tkinter as tk
from tkinter import scrolledtext
import cv2
import torch
import numpy as np
import mediapipe as mp
import pathlib
from PIL import Image, ImageTk
import datetime

# Temporarily change the PosixPath to WindowsPath
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load YOLOv5 model
model = torch.hub.load("yolov5", 'custom', r"X:\Project\final\best (2).pt", source='local', force_reload=True)

# Initialize MediaPipe Hand Tracker
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Function to check if a point (x, y) is inside a bounding box (x_min, y_min, x_max, y_max)
def is_inside_bbox(point, bbox):
    x, y = point
    x_min, y_min, x_max, y_max = bbox
    return x_min <= x <= x_max and y_min <= y <= y_max

# Function to log detection time of class 3
def log_detection_time():
    detection_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_text = f"Driver Sleeping  Detected - Time: {detection_time}\n"
    log_text_widget.insert(tk.END, log_text)
    log_text_widget.see(tk.END)  # Auto-scroll to the bottom of the text widget

# Function to detect hands and overlay text on frame
def detect_and_overlay_hands(frame, class_2_index, bboxes):
    # Convert frame to RGB for hand detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Run hand detection using MediaPipe Hand Tracker
    results_hands = hands.process(frame_rgb)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            # Check if any finger's tip is inside the bounding box of class 3
            for landmark in hand_landmarks.landmark:
                # Get the coordinates of the finger tip
                finger_tip = (landmark.x * frame.shape[1], landmark.y * frame.shape[0])
                # Check if finger tip is inside the bounding box of class 3
                for bbox in bboxes:
                    x_min, y_min, x_max, y_max = bbox
                    if is_inside_bbox(finger_tip, (x_min, y_min, x_max, y_max)):
                        # Display "Hand On Steering Wheel: Safe" text near the center of class 3 bounding box
                        cv2.putText(frame, 'Hand On Steering Wheel: Safe', (int((x_min + x_max) / 2), int((y_min + y_max) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        break  # No need to check other finger tips if one is inside the bounding box
                    else:
                        # Display "Hand On Steering Wheel: Unsafe" text near the center of class 3 bounding box
                        cv2.putText(frame, 'Hand On Steering Wheel: Unsafe', (int((x_min + x_max) / 2), int((y_min + y_max) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        break
    return frame

# Function to update the video frame
def update_frame():
    ret, frame = cap.read()

    # Convert frame to RGB for YOLOv5 model
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run object detection using YOLOv5 model
    results_yolo = model(frame_rgb)

    # Extract bounding boxes, classes, and scores from YOLOv5 results
    pred = results_yolo.pred[0]  # Extract predictions
    bboxes = pred[:, :4].cpu().numpy()  # Extract bounding boxes
    classes = pred[:, 5].cpu().numpy().astype(int)  # Extract classes
    scores = pred[:, 4].cpu().numpy()  # Extract scores

    # Find the index of class 2 in the classes array
    class_1_index = np.where(classes == 1)[0]
    class_2_index = np.where(classes == 2)[0]

    # Iterate over detected objects
    for idx, (bbox, cls, score) in enumerate(zip(bboxes, classes, scores)):
        x_min, y_min, x_max, y_max = bbox
        label = f"{cls}: {score:.2f}"

        # Draw bounding box and label
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x_min), int(y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Check if class 3 is detected
        if idx in class_1_index:
            log_detection_time()  # Log the detection time

    # Detect hands and overlay text on frame
    frame = detect_and_overlay_hands(frame, class_1_index, bboxes)

    # Convert the frame to display in the GUI
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame = ImageTk.PhotoImage(frame)

    # Update the video frame
    label_video.imgtk = frame
    label_video.configure(image=frame)
    label_video.after(10, update_frame)  # Update after 10 milliseconds

# Create the main window
root = tk.Tk()
root.title("Object Detection")

# Create a label to display the video frame
label_video = tk.Label(root)
label_video.pack()

# Create a scrolled text widget for logging detections
log_text_widget = scrolledtext.ScrolledText(root, width=50, height=10)
log_text_widget.pack()

# Open the video file
cap = cv2.VideoCapture(r"X:\Project\final\video (3).mp4")

# Start updating the video frame
update_frame()

# Start the GUI main loop
root.mainloop()

# Release resources
cap.release()
cv2.destroyAllWindows()
