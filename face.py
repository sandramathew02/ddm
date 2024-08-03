import cv2
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
import threading
import playsound
from pygame import mixer
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import geocoder 


mixer.init()
mixer.music.load("/Users/sandra/Downloads/drowsy/mixkit-classic-alarm-995.wav")


def pause():
    mixer.music.pause()
def play():
    mixer.music.play(-1)
# Load YOLO
weights_path = '/Users/sandra/Downloads/drowsy/yolov3_training_last (1).weights'  # add exact path location here
cfgpath = '/Users/sandra/Downloads/drowsy/yolov3_testing (1).cfg'  # add exact cfg file location
net = cv2.dnn.readNet(weights_path, cfgpath)
classes = ["close", "open"]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def close_window():
    window.quit()
    window.destroy()
    os._exit(0)

# Open webcam
cap = cv2.VideoCapture(0)
temp=0
temp1=0
# Create Tkinter window
window = tk.Tk()
window.title("Face Cam")
window.geometry("1440x900")
window.iconbitmap("/Users/sandra/Downloads/drowsy/eye_logo.ico")

# Load and resize the background image
background_image = Image.open("/Users/sandra/Downloads/drowsy/pexels-rovenimagescom-949587.jpeg")

background_image = ImageTk.PhotoImage(background_image)

# Create a Label widget to hold the background image
background_label = tk.Label(window, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
# Create a canvas to display the video stream
canvas = tk.Canvas(window, width=635, height=475)
canvas.place(x=400, y=0)

# Create a button to stop the program
stop_button = tk.Button(window, text="Stop", command=close_window, bg="red", fg="green", activeforeground="blue", relief=tk.RAISED, font=("Comic Sans MS", 15, "bold"),width=10, height=3)
stop_button.place(x=500, y=480)

# Location
g = geocoder.ip('me')
latitude, longitude = g.latlng
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

def send_email_button():
    # Disable the button after it's clicked to prevent multiple clicks
    send_button.config(state=tk.DISABLED)
    # Call the send_email function
    send_email()

send_button = tk.Button(window, text="Send Email", command=send_email_button, bg="red", fg="green", activeforeground="white", relief=tk.RAISED, font=("Comic Sans MS", 15, "bold"),width=10, height=3)
send_button.place(x=800, y=480)

def play_sound():
    playsound.playsound("/Users/sandra/Downloads/drowsy/mixkit-classic-alarm-995.wav")
 
def update_video_stream():
    global temp
    global temp1
    
    while True:
        # Capture frame from webcam
        ret, frame = cap.read()

        # Resize frame
        frame = cv2.resize(frame, None, fx=0.8, fy=0.8)
        height, width, channels = frame.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                    

                if confidence > 0.3:
                    # Object detected
                    if class_id==0:
                        temp=temp+1
                        temp1=temp1+1
                    else:
                        temp=0
                        temp1=0
                    print(class_id)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * int(width))
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(indexes)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y + 30), font, 3, color, 2)
        if temp>=10:
            play()
        elif temp==0:
            pause()
        if temp1>=20:
            send_email()

           
            

        # Convert the OpenCV frame to ImageTk format
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img)

        # Update the canvas with the new image
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.image = img_tk

        # Break the loop on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Create a thread for running the video stream update
thread = threading.Thread(target=update_video_stream)
thread.start()



# Configure the window close button to call the function
window.protocol("WM_DELETE_WINDOW", close_window)
# Start the Tkinter event loop
window.mainloop()

