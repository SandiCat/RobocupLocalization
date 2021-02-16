import cv2
import imutils
import numpy as np
from matplotlib import pyplot
import localization


def nothing(x):
    pass

window_name = 'modified'
channel_names = ['H', 'S', 'V']
min_suffix = "min"
max_suffix = "max"

cv2.namedWindow(window_name)

circle_img = np.zeros([100, 100], dtype="uint8")
cv2.circle(circle_img, (50, 50), 40, 255, thickness=-1)
_, [circle_cnt], _ = cv2.findContours(circle_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


def get_mask(image):
    modified = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    minHSV= np.array([cv2.getTrackbarPos(channel + min_suffix, window_name) for channel in channel_names], dtype = "uint8")
    maxHSV = np.array([cv2.getTrackbarPos(channel + max_suffix, window_name) for channel in channel_names], dtype = "uint8")
    return cv2.inRange(modified, minHSV, maxHSV)


def transform_image(image):
    #modified = cv2.GaussianBlur(image, (21, 21), 0)
    #modified = cv2.bilateralFilter(image, 5, 500, 500)
    modified = cv2.medianBlur(image, 21)
    #modified = cv2.fastNlMeansDenoisingColored(image, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    return modified


def run_on_sample_images():
    import os
    dir_name = "sample_images"

    for channel in channel_names:
        cv2.createTrackbar(channel + min_suffix, window_name, 0, 255, nothing)
        cv2.createTrackbar(channel + max_suffix, window_name, 0, 255, nothing)

    while True:
        images = [cv2.imread(os.path.join(dir_name, filename)) for filename in os.listdir(dir_name)]
        transformed = imutils.resize(np.hstack([np.vstack((image, transform_image(image), cv2.cvtColor(get_mask(transform_image(image)), cv2.COLOR_GRAY2BGR))) for image in images]), width=1000)
        cv2.imshow(window_name, transformed)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cv2.destroyAllWindows()


def loop_step(frame, keycode):
    modified = transform_image(frame)
    cv2.imshow('after transform', modified)
    modified = cv2.cvtColor(modified, cv2.COLOR_BGR2HSV)
    minHSV = np.array([0, 0, 0], dtype="uint8")
    maxHSV = np.array([100, 255, 255], dtype="uint8")
    mask = cv2.bitwise_not(cv2.inRange(modified, minHSV, maxHSV))

    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    circles = []
    for contour in contours:
        if cv2.contourArea(contour) > np.pi * 20**2 and cv2.matchShapes(contour, circle_cnt, 1, 0) < 0.05:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            circles.append([x, y, radius])
    if circles:
        x, y, radius = sorted(circles, key=lambda c: c[2])[-1]
        coords = localization.camera_coords_to_world_coords(
            point=localization.localize_ball(x, y, radius, ball_radius=7.4, focal_length=3.5),
            cam_height=20,
            cam_angle= np.deg2rad(56)
        )
        cv2.putText(frame, str(list(coords)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow(window_name, frame)

    if imutils.is_keycode_char(keycode, 'h'):
        xrange = [0, 180]
        hist = cv2.calcHist([modified], [0], None, [256], xrange)
        pyplot.plot(hist)
        pyplot.xlim(xrange)
        pyplot.show()


#imutils.testing_loop(1, loop_step)
run_on_sample_images()