import time
import pyfakewebcam
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
camera = pyfakewebcam.FakeWebcam('/dev/video6', 640, 480)

while cap.isOpened():
    success, image = cap.read()
    if success:
        #print(np.array(image).shape)
        camera.schedule_frame(np.array(image).astype(np.uint8))#np.array(image))
    time.sleep(1/30.0)
        
        
        
"""
import time
import pyfakewebcam
import numpy as np

blue = np.zeros((480,640,3), dtype=np.uint8)
blue[:,:,2] = 255

red = np.zeros((480,640,3), dtype=np.uint8)
red[:,:,0] = 255

camera = pyfakewebcam.FakeWebcam('/dev/video6', 640, 480)

while True:

    camera.schedule_frame(red)
    time.sleep(1/30.0)

    camera.schedule_frame(blue)
    time.sleep(1/30.0)
    """