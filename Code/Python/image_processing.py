import numpy as np
import cv2


# auto canny function
# from http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
def auto_canny(im, sigma=0.33):
    v = np.median(im)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(im, lower, upper)


def bgr2gray(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def rgb2gray(im):
    return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)


def im_close(im, n=5):
    im = cv2.dilate(im, np.ones((n, n), np.uint8), iterations=1)
    im = cv2.erode(im, np.ones((n, n), np.uint8), iterations=1)
    return im


def im_open(im, n=5):
    im = cv2.erode(im, np.ones((n, n), np.uint8), iterations=1)
    im = cv2.dilate(im, np.ones((n, n), np.uint8), iterations=1)
    return im


def threshold_image(im, color_mu, color_std=0, sigma=1):
    nim = im.copy()
    _, ima = cv2.threshold(im, color_mu - (color_std * sigma), 255, cv2.THRESH_BINARY)
    _, imb = cv2.threshold(im, color_mu + (color_std * sigma), 255, cv2.THRESH_BINARY_INV)
    nim[np.equal(ima, imb)] = 255
    nim[np.not_equal(ima, imb)] = 0
    nim = im_open(nim)
    nim = im_open(nim)
    return nim


def im_mask(im, sigma=1, image_is_grayscale=False):
    if image_is_grayscale:
        im_gray = im.copy()
    else:
        im_gray = rgb2gray(im)
    card_color_mu, card_color_std = cv2.meanStdDev(im_gray)
    return threshold_image(im_gray, card_color_mu, card_color_std, sigma).astype(np.uint8)
