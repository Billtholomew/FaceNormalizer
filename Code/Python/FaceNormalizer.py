import sys, traceback, time
import cv2
import numpy as np

import image_processing as ip

face_cascade = cv2.CascadeClassifier('C:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/opencv/sources/data/haarcascades/haarcascade_eye.xml')


def detect_faces(im):
    face_boxes = face_cascade.detectMultiScale(im, 1.3, 5)
    face_boxes = filter(lambda (x, y, w, h): w * h > 128 * 128, face_boxes)
    return face_boxes


def face_ellipse_to_points(face_ellipse, vertices=10):
    (cx, cy), (w, h), rotation = face_ellipse
    rotation *= np.pi / 180
    face_points = []
    for theta in xrange(0, 360, 360 / vertices):
        theta *= np.pi / 180
        px = int(np.cos(theta - rotation) * w / 2 + cx)
        py = int(np.sin(theta - rotation) * h / 2 + cy)
        face_points.append([px, py])
    return face_points


def threshold_image(im, eyes, mu, sig):
    mu = mu.astype(int)
    sig = sig.astype(int)
    for (ex, ey, ew, eh) in eyes:
        im[ey:ey + eh, ex:ex + ew, 0] = mu[0]
        im[ey:ey + eh, ex:ex + ew, 1] = mu[1]
        im[ey:ey + eh, ex:ex + ew, 2] = mu[2]
    thresh = np.ones((im.shape[0], im.shape[1]), dtype=np.uint8) * 255
    for i in xrange(3):
        low = mu[i] - sig[i] * .75
        high = mu[i] + sig[i] * .75
        _, threshold_low = cv2.threshold(im[:, :, i], low, 255, cv2.THRESH_BINARY)
        _, threshold_high = cv2.threshold(im[:, :, i], high, 255, cv2.THRESH_BINARY_INV)
        t = cv2.bitwise_and(threshold_low, threshold_high)
        thresh = cv2.bitwise_and(thresh, t)
    return thresh


def get_face_points(face_boxes, im, im_grayscale):
    for face_box in face_boxes:
        x, y, w, h = face_box
        # adjust height and origin y
        y -= h * 0.25
        h *= 1.5
        im_grayscale_face_box = im_grayscale[y: y + h, x: x + w]
        im2 = im[y: y + h, x: x + w]

        eyes = eye_cascade.detectMultiScale(im_grayscale_face_box)
        # eyes should not be  below the "halfway" line of face
        eyes = filter(lambda (eye_x, eye_y, eye_w, eye_h): eye_y < im_grayscale_face_box.shape[0] / 2, eyes)
        if len(eyes) != 2:
            continue

        thresh = ip.im_mask(im_grayscale_face_box, sigma=0.5, image_is_grayscale=True)
        thresh = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # only keep contours that are big enough
        contours = filter(lambda contour: cv2.contourArea(contour) > 0.33 * im_grayscale_face_box.size, contours)
        contours = sorted(contours, key=lambda contour: cv2.contourArea(contour))
        contours = map(lambda contour: cv2.convexHull(contour), contours)
        # face contour is largest contour
        if len(contours) == 0:
            continue
        face_ellipse = cv2.fitEllipse(contours[0])
        face_points = face_ellipse_to_points(face_ellipse)
        # add eye points, this is a diamond around the eye, not a rectangle
        for (ex, ey, ew, eh) in eyes:
            cx, cy = int(ex), int(ey + 0.5 * eh)
            face_points.append([cx, cy])
            cx, cy = int(ex + ew), int(ey + 0.5 * eh)
            face_points.append([cx, cy])
            cx, cy = int(ex + 0.5 * ew), int(ey)
            face_points.append([cx, cy])
            cx, cy = int(ex + 0.5 * ew), int(ey + eh)
            face_points.append([cx, cy])
        # find  mouth point
        # not sure how to do this...

        # draw facial points onto image
        for pt in face_points:
            # (img, center, radius, color, thickness=1, lineType=8, shift=0
            cv2.circle(im2, (pt[0], pt[1]), 1, (0, 255, 255), 3)
    return im


def find_faces(fps, frame_count):
    while frame_count > 0:
        return_value, im = camera.read()
        image_with_face_points = np.copy(im)
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        face_boxes = detect_faces(imgray)
        image_with_face_points = get_face_points(face_boxes, image_with_face_points, imgray)
        frame_count -= 1
        cv2.imshow('', image_with_face_points)
        cv2.waitKey(1000/fps)
    return

try:
    camera = []
    camera_port = 0
    camera = cv2.VideoCapture(camera_port)
    find_faces(fps=45, frame_count=30)
except Exception, e:
    print e
    traceback.print_exc(file=sys.stdout)
finally:
    cv2.destroyAllWindows()
    del camera
