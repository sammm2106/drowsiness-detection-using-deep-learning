# Import necessary libraries
from scipy.spatial import distance as dist  # For calculating Euclidean distance
from imutils.video import VideoStream       # For accessing video stream
from imutils import face_utils              # Utilities for facial landmarks
from threading import Thread                # For handling alarm sound in a separate thread
import numpy as np                          # For numerical operations
import argparse                             # For parsing command-line arguments
import imutils                              # For image processing utilities
import time                                 # For time-related functions
import cv2                                  # For OpenCV image processing
import dlib                                 # For detecting and predicting facial landmarks
import playsound                            # For playing sound
import os                                   # For operating system path utilities

# Function to play an alarm sound
def sound_alarm(path):
    global alarm_status
    global alarm_status2
    global saying

    while alarm_status:  # Check if the first alarm condition is true
        print('call')
        playsound.playsound(path)  # Play the alarm sound

    if alarm_status2:  # Check if the second alarm condition is true
        print('call')
        saying = True
        playsound.playsound(path)  # Play the alarm sound
        saying = False

# Function to calculate Eye Aspect Ratio (EAR) for drowsiness detection
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  # Distance between vertical eye landmarks
    B = dist.euclidean(eye[2], eye[4])  # Distance between another pair of vertical eye landmarks
    C = dist.euclidean(eye[0], eye[3])  # Distance between horizontal eye landmarks
    ear = (A + B) / (2.0 * C)           # Compute the EAR
    return ear

# Function to calculate EAR for both eyes and return the average EAR
def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]  # Indices for the left eye
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"] # Indices for the right eye

    leftEye = shape[lStart:lEnd]         # Extract left eye landmarks
    rightEye = shape[rStart:rEnd]        # Extract right eye landmarks

    leftEAR = eye_aspect_ratio(leftEye)  # Calculate EAR for the left eye
    rightEAR = eye_aspect_ratio(rightEye) # Calculate EAR for the right eye

    ear = (leftEAR + rightEAR) / 2.0     # Average EAR of both eyes
    return (ear, leftEye, rightEye)

# Function to calculate lip distance for yawning detection
def lip_distance(shape):
    top_lip = shape[50:53]                   # Top lip landmarks
    top_lip = np.concatenate((top_lip, shape[61:64]))  # Additional top lip landmarks

    low_lip = shape[56:59]                   # Bottom lip landmarks
    low_lip = np.concatenate((low_lip, shape[65:68]))  # Additional bottom lip landmarks

    top_mean = np.mean(top_lip, axis=0)      # Mean of top lip coordinates
    low_mean = np.mean(low_lip, axis=0)      # Mean of bottom lip coordinates

    distance = abs(top_mean[1] - low_mean[1])  # Calculate vertical lip distance
    return distance

# Parse command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
ap.add_argument("-a", "--alarm", type=str, default=r'C:\Users\Lenovo\OneDrive\Desktop\samm minor project\Alert.wav', 
                help="path alarm .WAV file")
args = vars(ap.parse_args())

# Define thresholds and initial states
EYE_AR_THRESH = 0.3  # EAR threshold for detecting drowsiness
EYE_AR_CONSEC_FRAMES = 30  # Number of consecutive frames for drowsiness
YAWN_THRESH = 30  # Threshold for detecting a yawn
alarm_status = False  # Status of drowsiness alarm
alarm_status2 = False # Status of yawning alarm
saying = False  # To prevent multiple alarms
COUNTER = 0  # Frame counter for EAR

# Load Haarcascade detector and Dlib predictor
print("-> Loading the predictor and detector...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    # Faster but less accurate face detector
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Predicts 68 facial landmarks

# Start the video stream
print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()  # Start video stream
time.sleep(1.0)  # Allow camera to warm up

# Main loop for processing video frames
while True:
    frame = vs.read()                          # Read frame from video stream
    frame = imutils.resize(frame, width=450)   # Resize frame for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale

    # Detect faces in the grayscale frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:  # Iterate through detected faces
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))  # Convert to Dlib's rectangle

        shape = predictor(gray, rect)          # Predict facial landmarks
        shape = face_utils.shape_to_np(shape)  # Convert landmarks to NumPy array

        eye = final_ear(shape)                 # Calculate EAR
        ear = eye[0]
        leftEye = eye[1]
        rightEye = eye[2]

        distance = lip_distance(shape)         # Calculate lip distance

        # Draw contours around eyes and lips
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        # Check for drowsiness
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if alarm_status == False:
                    alarm_status = True
                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm, args=(args["alarm"],))
                        t.daemon = True
                        t.start()

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            alarm_status = False

        # Check for yawning
        if (distance > YAWN_THRESH):
            cv2.putText(frame, "Yawn Alert", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if alarm_status2 == False and saying == False:
                alarm_status2 = True
                if args["alarm"] != "":
                    t = Thread(target=sound_alarm, args=(args["alarm"],))
                    t.daemon = True
                    t.start()
        else:
            alarm_status2 = False

        # Display EAR and lip distance on the frame
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the processed video frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Exit loop when 'q' key is pressed
    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()  # Close all OpenCV windows
vs.stop()                # Stop video stream
 