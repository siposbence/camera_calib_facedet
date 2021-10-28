import cv2
import mediapipe as mp
import numpy as np
import pyfakewebcam

# sudo modprobe v4l2loopback devices=2 exclusive_caps=1,1 video_nr=5,6 card_label="WebVideo","Video4FaceDetection"

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

DIM=(744, 600)
K=np.array([[328.66960044713676, 0.0, 461.9005635813701], [0.0, 329.7390185115684, 296.8864597194912], [0.0, 0.0, 1.0]])
D=np.array([[-0.03254196233072562], [0.058705976062386735], [-0.08592867131653371], [0.04283255216208141]])


camera = pyfakewebcam.FakeWebcam('/dev/video5', 744, 600) #600, 744
camera_alt = pyfakewebcam.FakeWebcam('/dev/video6', 744, 600)

def undistort(img):
    x,y,_ = img.shape
    w, h = [500, 300]
    img = img[int(x/2)-h:int(x/2)+h, int(y/2)-w:int(x/2)+w,:]
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
width = 2048
height = 1536
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue
    
    image = undistort(image)
    #image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #results = face_detection.process(image)

    #image.flags.writeable = True
    #if results.detections:
    #  for detection in results.detections:
    #    mp_drawing.draw_detection(image, detection)
    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    #print(image.shape)
    camera.schedule_frame(cv2.flip(image, 1))
    camera_alt.schedule_frame(cv2.flip(image, 1))
    if cv2.waitKey(1) & 0xFF == 27:
      break
cap.release()

