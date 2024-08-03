import torch
import cv2
import numpy as np
import mediapipe as mp
import pathlib

# Temporarily change the PosixPath to WindowsPath
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load YOLOv5 model
model = torch.hub.load("yolov5", 'custom', r"X:\Project\final\best (1).pt", source='local', force_reload=True)

# Initialize MediaPipe Hand Tracker
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Function to check if a point (x, y) is inside a bounding box (x_min, y_min, x_max, y_max)
def is_inside_bbox(point, bbox):
    x, y = point
    x_min, y_min, x_max, y_max = bbox
    return x_min <= x <= x_max and y_min <= y <= y_max

cap = cv2.VideoCapture(r"X:\Project\final\video (4).mp4")

while True:
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

    # Find the index of class 3 in the classes array
    class_3_index = np.where(classes == 2)[0]

    # Iterate over detected objects
    for idx, (bbox, cls, score) in enumerate(zip(bboxes, classes, scores)):
        x_min, y_min, x_max, y_max = bbox
        label = f"{cls}: {score:.2f}"

        # Draw bounding box and label
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x_min), int(y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Run hand detection using MediaPipe Hand Tracker
        results_hands = hands.process(frame_rgb)
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                # Check if any finger's tip is inside the bounding box of class 3
                for landmark in hand_landmarks.landmark:
                    # Get the coordinates of the finger tip
                    finger_tip = (landmark.x * frame.shape[1], landmark.y * frame.shape[0])
                    # Check if finger tip is inside the bounding box of class 3
                    if idx in class_3_index and is_inside_bbox(finger_tip, (x_min, y_min, x_max, y_max)):
                        # Display "Overlap" text near the center of class 3 bounding box
                        cv2.putText(frame, 'Hand On Steering Wheel: Safe', (int((x_min + x_max) / 2), int((y_min + y_max) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        break  # No need to check other finger tips if one is inside the bounding box
                    else:
                        cv2.putText(frame, 'Hand On Steering Wheel: Unsafe',(int((x_min + x_max) / 2), int((y_min + y_max) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 2)
                        break

    # Display frame
    cv2.imshow("frame", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
