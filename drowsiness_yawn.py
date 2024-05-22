# python drowniness_yawn.py --webcam webcam_index

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
import requests
import random


# while True:
#     time.sleep(10)
#     s = 'espeak "'+"I hope you are doing good"+'"'
#     os.system(s)

def sendEyeClosureAPI(eye_closed_duration):
    api_url = 'http://localhost:5000/api/details/addeyeclosure'
    headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}
    data = {'duration': eye_closed_duration}
    response = requests.put(api_url, json=data, headers=headers)
    if response.status_code == 200:
        print("Eye closure duration added successfully.")
    else:
        print("Failed to add eye closure duration:", response.text)

def sendYawnAPI(yawn_duration):
    api_url = 'http://localhost:5000/api/details/addyawn'
    headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}
    data = {'duration': yawn_duration}
    response = requests.put(api_url, json=data, headers=headers)
    if response.status_code == 200:
        print("Yawn duration added successfully.")
    else:
        print("Failed to add yawn duration:", response.text)
        
def sendMotorActivitiesAPI(speed, elapsed_time):
    api_url = 'http://localhost:5000/api/details/addmotoractivities'
    headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}
    data = {'speed': speed, 'time': elapsed_time}
    response = requests.put(api_url, json=data, headers=headers)
    if response.status_code == 200:
        print("Motor activities added successfully.")
    else:
        print("Failed to add motor activities:", response.text)

def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying

    while alarm_status:
        print('call')
        s = 'espeak "'+msg+'"'
        os.system(s)

    if alarm_status2:
        print('call')
        saying = True
        s = 'espeak "' + msg + '"'
        os.system(s)
        saying = False


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear


def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)


def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance


ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 100
YAWN_THRESH = 22
YAWN_THRES_FRAMES = 50
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0
YAWN_COUNTER=0
already_detected=False
yawn_already_detected=False

frame_rate = 30

# Token for authentication
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyaWQiOiI2NjQ4NTAzMDBhODA3ZTgzMWU3OTFmYzciLCJpYXQiOjE3MTYwMTUxNjYsImV4cCI6MTcxODYwNzE2Nn0.XUkelJlkJPi4o7pSUuubKy8yI__Hrcw2GgshKTF_Ofg"


print("-> Loading the predictor and detector...")
#detector = dlib.get_frontal_face_detector()
detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml")  # Faster but less accurate
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
# vs= VideoStream(usePiCamera=True).start()       //For Raspberry Pi
time.sleep(1.0)

# Function to get speed (placeholder)
def get_speed():
    # Placeholder: Replace this with actual speed retrieval logic
    return random.uniform(0,150)

# Buffer for speed values
speed_buffer = []
buffer_size = 20

# Function to get smoothed speed
def get_smoothed_speed(new_speed):
    if len(speed_buffer) >= buffer_size:
        speed_buffer.pop(0)
    speed_buffer.append(new_speed)
    return sum(speed_buffer) / len(speed_buffer)


# Get start time
last_speed_change_time = time.time()

previous_speed = 0
last_elapsed_time = 0

while True:
    speed = get_speed()
    smoothed_speed = get_smoothed_speed(speed)
    
    # Adjust thresholds based on speed
    if smoothed_speed <= 40:
        if previous_speed > 40:
            elapsed_time = time.time() - last_speed_change_time
            sendMotorActivitiesAPI(previous_speed,elapsed_time)
            last_speed_change_time = time.time()
        EYE_AR_CONSEC_FRAMES = 150
        YAWN_THRES_FRAMES = 70
        
    elif 40 < smoothed_speed <= 80:
        if previous_speed <= 40 or previous_speed > 80:
            elapsed_time = time.time() - last_speed_change_time
            sendMotorActivitiesAPI(previous_speed,elapsed_time)
            last_speed_change_time = time.time()
        EYE_AR_CONSEC_FRAMES = 80
        YAWN_THRES_FRAMES = 50
    elif 80 < smoothed_speed < 150:
        if previous_speed <= 80:
            elapsed_time = time.time() - last_speed_change_time
            sendMotorActivitiesAPI(previous_speed,elapsed_time)
            last_speed_change_time = time.time()
        EYE_AR_CONSEC_FRAMES = 40
        YAWN_THRES_FRAMES = 30
    
    previous_speed = smoothed_speed

    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #rects = detector(gray, 0)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)

    # for rect in rects:
    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye[1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)
        
        
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if alarm_status == False:
                    alarm_status = True
                    t = Thread(target=alarm, args=('wake up sir',))
                    t.deamon = True
                    t.start()
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                already_detected=True

        else:
            if(already_detected):
                duration_seconds = COUNTER / frame_rate
                sendEyeClosureAPI(duration_seconds)
                print("Duration: {:.2f} seconds".format(duration_seconds))
                already_detected=False
            COUNTER = 0
            alarm_status = False
            

        if (distance > YAWN_THRESH):
            YAWN_COUNTER+=1
            if YAWN_COUNTER >= YAWN_THRES_FRAMES:
                cv2.putText(frame, "Yawn Alert", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                yawn_already_detected=True
                if alarm_status2 == False and saying == False:
                    alarm_status2 = True
                    t = Thread(target=alarm, args=('take some fresh air sir',))
                    t.deamon = True
                    t.start()
                
        else:
            if(yawn_already_detected):
                duration_seconds = YAWN_COUNTER / frame_rate
                sendYawnAPI(duration_seconds)
                print("Duration: {:.2f} seconds".format(duration_seconds))
                yawn_already_detected=False
            YAWN_COUNTER = 0
            alarm_status2 = False
            
        cv2.putText(frame, "Ear: {:.2f}".format(ear), (200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame, "Eye Frames: {:.2f}".format(EYE_AR_CONSEC_FRAMES), (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, "EAR: {:.2f}".format(ear), (200, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (200, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame, "Yawn Frames: {:.2f}".format(YAWN_THRES_FRAMES), (200, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame, "SPEED: {:.2f}".format(smoothed_speed), (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        
        

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
