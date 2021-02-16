import numpy as np
import cv2
import imutils
import mahotas

cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()

    modified = cv2.GaussianBlur(frame, (21, 21), 0)
    #modified = cv2.Canny(modified, 30, 150)
    #(_, modified) = cv2.threshold(modified, mahotas.thresholding.rc(modified), 255, cv2.THRESH_BINARY_INV)
    #modified = cv2.cvtColor(modified, cv2.COLOR_BGR2HSV)
    #modified = cv2.cvtColor(modified, cv2.COLOR_BGR2GRAY)
    #modified = cv2.adaptiveThreshold(modified, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)

    modified = cv2.cvtColor(modified, cv2.COLOR_BGR2HSV)
    modified = imutils.resize(np.hstack(cv2.split(modified)), width=1000)

    cv2.imshow('frame', frame)
    cv2.imshow('modified', modified)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()