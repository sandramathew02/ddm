import tkinter as tk
from tkinter import scrolledtext
import cv2
import torch
import numpy as np
import mediapipe as mp
import pathlib
from pygame import mixer
from PIL import Image, ImageTk
import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import geocoder
import threading  # Import threading module for asynchronous execution


# Initialize geocoder
g = geocoder.ip('me')
latitude, longitude = g.latlng

# Temporarily change the PosixPath to WindowsPath
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Initialize pygame mixer
mixer.init()

# Load audio files
audio_files = {
    "drowsy": r"X:\Project\final\alarm.mp3",
    "safe": r"X:\Project\final\safe.mp3"
}

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
                        # Display "Hand On Steering Wheel: Safe" text at the top of the screen
                        safe_label.place(x=20, y=20)
                        #pause()
                        unsafe_label.place_forget()  # Hide "Hand On Steering Wheel: Unsafe" text
                        break  # No need to check other finger tips if one is inside the bounding box
                    else:
                        # Display "Hand On Steering Wheel: Unsafe" text at the top of the screen
                        unsafe_label.place(x=20, y=20)
                        #play()
                        safe_label.place_forget()  # Hide "Hand On Steering Wheel: Safe" text
                        break
    else:
        # No hand landmarks detected, display "Hand On Steering Wheel: Unsafe" text
        unsafe_label.place(x=20, y=20)
        safe_label.place_forget()  # Hide "Hand On Steering Wheel: Safe" text
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
    class_0_index = np.where(classes == 0)[0]
    class_3_index = np.where(classes == 3)[0]

    # Count the number of bounding boxes of class 3
    num_class_3_boxes = len(class_3_index)

    # Display "Overload" text if more than 4 bounding boxes of class 3 are detected
    if num_class_3_boxes > 4:
        cv2.putText(frame, "Overload", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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

    # Call Driver_Sleepy function
    Driver_Sleepy(class_1_index,class_0_index)

    # Convert the frame to display in the GUI
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame = ImageTk.PhotoImage(frame)

    # Update the video frame
    label_video.imgtk = frame
    label_video.configure(image=frame)
    label_video.after(10, update_frame)  # Update after 10 milliseconds

# Function to send email
def send_email():
    # Gmail SMTP server settings
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    smtp_username = 'animelover2002tg@gmail.com'  # Replace with your Gmail email address
    smtp_password = 'moixhxcysdfhdalo'  # Replace with your Gmail password

    sender_email = 'animelover2002tg@gmail.com'  # Replace with your Gmail email address
    receiver_email = 'sandramathewmaria@gmail.com'  # Replace with the recipient email address
    subject = 'Test Email'
    message = (f"""THIS IS MY LOCATION , PLEASE SEND HELP Latitude: 
{latitude}, Longitude: {longitude}""")
    # Create a multipart message and set headers
    msg = MIMEMultipart()
    msg['From'] = "animelover2002tg@gmail.com"
    msg['To'] = "Sahilmarch20@gmail.com"
    msg['Subject'] = "Send help please "

    # Attach the message to the email
    msg.attach(MIMEText(message, 'plain'))

    # Create a secure connection to the SMTP server
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        # Log in to your Gmail account
        server.login(smtp_username, smtp_password)
        # Send the email
        server.send_message(msg)

    print('Email sent successfully.')

# Function to send email when button is clicked
def send_email_button():
    # Disable the button after it's clicked to prevent multiple clicks
    send_button.config(state=tk.DISABLED)
    # Call the send_email function in a separate thread
    threading.Thread(target=send_email).start()

# Function to play audio for drowsy detection
def Driver_Sleepy(class_1_index, class_0_index):
    # Check if Driver_Drowsy is detected
    temp1 = 0
    if len(class_1_index) > 0:
        # Play the audio for drowsy detection
        mixer.music.load(audio_files["drowsy"])
        mixer.music.play(-1)
        temp1 = temp1+1
        print(temp1)
        # Call the send_email function in a separate thread
        if(temp1>10):
            threading.Thread(target=send_email).start()
    elif len(class_0_index) > 0:
        # Stop playing the audio if no drowsiness is detected
        temp1 = 0
        mixer.music.stop()

# Create the main window
root = tk.Tk()
root.title("Object Detection")

# Create a label to display the video frame
label_video = tk.Label(root)
label_video.pack()

# Create a scrolled text widget for logging detections
log_text_widget = scrolledtext.ScrolledText(root, width=50, height=10)
log_text_widget.pack()

# Create a label for "Hand On Steering Wheel: Safe" text
safe_label = tk.Label(root, text="Hand On Steering Wheel: Safe", fg="blue", font=("Helvetica", 14, "bold"))
safe_label.pack()
safe_label.place_forget()  # Initially hide the label

# Create a label for "Hand On Steering Wheel: Unsafe" text
unsafe_label = tk.Label(root, text="Hand On Steering Wheel: Safe", fg="blue", font=("Helvetica", 14, "bold"))
unsafe_label.pack()
unsafe_label.place(x=20, y=20)  # Initially show the label

# Create a button to send email
send_button = tk.Button(root, text="Send Email", command=send_email_button, bg="red", fg="green", activeforeground="white", relief=tk.RAISED, font=("Comic Sans MS", 15, "bold"), width=9, height=3)
send_button.place(x=0, y=500)

# Open the video file
cap = cv2.VideoCapture(r"X:\Project\final\video (2).mp4")

# Start updating the video frame
update_frame()

# Start the GUI main loop
root.mainloop()

# Release resources
cap.release()
cv2.destroyAllWindows()
