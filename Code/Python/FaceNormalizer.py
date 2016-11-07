import sys, traceback, time
import cv2
import numpy as np

camera_port = 0
# FPS to use when ramping the camera
fps = 30
# Number of frames to take during camera ramps
ramp_frames = 30

face_cascade = cv2.CascadeClassifier('C:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/opencv/sources/data/haarcascades/haarcascade_eye.xml')


def detect_faces(im):
    face_boxes = face_cascade.detectMultiScale(im, 1.3, 5)
    if len(face_boxes) == 0:
        return []
    face_boxes[:, 2:] += face_boxes[:, :2]
    face_boxes = filter(lambda (x1, y1, x2, y2): (x2 - x1) * (y2 - y1) > 128 * 128, face_boxes)
    return face_boxes


# interpolate new points in a convex polygon
def interpolate_convex_poly(contour, vertices=10):
    m = cv2.moments(contour)
    cy = int(m['m01'] / m['m00'])
    cx = int(m['m10'] / m['m00'])
    polar = []
    for i, pt in enumerate(contour):
        pt = pt[0]
        dy = pt[0] - cy
        dx = pt[1] - cx
        theta = np.arctan2(dy, dx)
        radius = np.sqrt(dx ** 2 + dy ** 2)
        polar.append((theta, radius))
    polar = sorted(polar, key=lambda (theta, radius): theta)
    cnt2 = []
    for interpolation_theta in xrange(-180, 180, 360 / vertices):
        interpolation_theta = interpolation_theta * np.pi / 180
        # interpolation_theta is theta to inperolate with
        interpolation_radius = 0  # interpolated radius
        p2 = zip([polar[-1]] + polar[:-1], polar)
        for pA, pB in p2:
            theta_a, radius_a = pA
            theta_b, radius_b = pB
            interpolation_radius = (radius_a + radius_b) / 2
            if theta_a <= interpolation_theta <= theta_b:
                interpolation_radius = (interpolation_theta - theta_a) / (theta_b - theta_a) * (
                radius_b - radius_a) + radius_a
                break
        x = int(interpolation_radius * np.cos(interpolation_theta) + cx)
        y = int(interpolation_radius * np.sin(interpolation_theta) + cy)
        cnt2.append([[y, x]])
    return cnt2


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
        x, y, x2, y2 = face_box
        w = x2 - x
        h = y2 - y
        x = int(x + 0.05 * w)
        w = int(0.90 * w)
        y = int(y - 0.10 * h)
        h = int(h * 1.3)
        x2 = x + w
        y2 = y + h
        im_grayscale_face_box = im_grayscale[y:y2, x:x2]
        im2 = im[y:y2, x:x2]
        im3 = im[y + 0.25 * h:y + 0.7 * h, x + 0.25 * w:x + 0.75 * w]
        mu, sig = cv2.meanStdDev(im3)
        eyes = eye_cascade.detectMultiScale(im_grayscale_face_box)
        eyes = [eye for eye in eyes if eye[1] < im_grayscale_face_box.shape[0] / 2]
        if len(eyes) != 2:
            continue
        thresh = threshold_image(im2.copy(), im_grayscale_face_box, eyes, mu, sig)
        if thresh is None:
            continue
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # only keep contours that are big enough
        contours = filter(lambda contour: cv2.contourArea(contour) > 0.33 * im_grayscale_face_box.size, contours)
        # contours = [cv2.approxPolyDP(cnt,0.025*cv2.arcLength(cnt,True),True) for cnt in contours]
        contours = map(lambda contour: cv2.convexHull(contour), contours)
        if len(contours) == 0:
            continue
        face_points = interpolate_convex_poly(contours[0])
        # add eye points
        for (ex, ey, ew, eh) in eyes:
            cx, cy = int(ex), int(ey + 0.5 * eh)
            face_points.append([[cx, cy]])
            cx, cy = int(ex + ew), int(ey + 0.5 * eh)
            face_points.append([[cx, cy]])
            cx, cy = int(ex + 0.5 * ew), int(ey)
            face_points.append([[cx, cy]])
            cx, cy = int(ex + 0.5 * ew), int(ey + eh)
            face_points.append([[cx, cy]])
        # find  mouth point
        # not sure how to do this...

        # draw facial points onto image
        for pt in face_points:
            pt = pt[0]
            # (img, center, radius, color, thickness=1, lineType=8, shift=0
            cv2.circle(im2, (pt[0], pt[1]), 1, (0, 255, 255), 3)

        cv2.imshow('FACE', im2)
        cv2.waitKey(30)
        pass
        # cv2.waitKey(1000/(fps))
        # cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
    return im


# Captures a single image from the camera and returns it in IplImage format
def get_image():
    # QueryFrame is the easiest way to get a full image out of a capture object
    return_value, im = camera.read()
    return im


def find_faces(fps, frame_count):
    while frame_count > 0:
        # Don't need to actually save these images
        im = get_image()
        image_with_face_points = np.copy(im)
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        face_boxes = detect_faces(imgray)
        image_with_face_points = get_face_points(face_boxes, image_with_face_points, imgray)
        frame_count -= 1
        # cv2.imshow('dst_rt', image_with_face_points)
        # cv2.waitKey(1000/(fps))
        # time.sleep(1/fps)
    return


# Now we can set up the camera with the CaptureFromCAM() function. All it needs is
# the index to a camera port. The 'camera' variable will be a cv2.capture object
try:
    camera = []
    camera = cv2.VideoCapture(camera_port)
    find_faces(60, 32)
except Exception, e:
    print e
    traceback.print_exc(file=sys.stdout)
finally:
    cv2.destroyAllWindows()
    del camera
