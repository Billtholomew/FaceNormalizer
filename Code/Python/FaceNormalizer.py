import sys
import cv2
import time
import numpy as np
import math

# so I don't have to keep saying np.pi
PI = np.pi

# Camera 0 is the integrated web cam on my netbook
camera_port = 0
# FPS to use when ramping the camera
fps = 30
# Number of frames to take during camera ramps
ramp_frames = 30

face_cascade = cv2.CascadeClassifier('C:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/opencv/sources/data/haarcascades/haarcascade_eye.xml')

def detect_faces(im):
    rects = face_cascade.detectMultiScale(im, 1.3, 5)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    rects = [[x1,y1,x2,y2] for [x1,y1,x2,y2] in rects if ((x2-x1)*(y2-y1)>128*128)]
    return rects

# interpolate new points in a convex polygon
def interpolate_cpoly(cnt,npts=10):
    m = cv2.moments(cnt)         
    cy = int(m['m01']/m['m00'])
    cx = int(m['m10']/m['m00'])
    polar = []
    for i,pt in enumerate(cnt):
        pt = pt[0]
        dy = pt[0]-cy
        dx = pt[1]-cx
        theta = np.arctan2(dy,dx)
        radius = np.sqrt(dx**2+dy**2)
        polar.append((theta,radius))
    polar = sorted(polar,key=lambda x:x[0])
    cnt2 = []
    for itheta in xrange(-180,180,360/npts):
        itheta = itheta*np.pi/180
        # itheta is theta to inperolate with
        irad = 0 # interpolated radius
        p2 = zip([polar[-1]]+polar[:-1],polar)
        for pA,pB in p2:
            thetaA,radiusA = pA
            thetaB,radiusB = pB
            irad = (radiusA+radiusB)/2
            if thetaA<=itheta and thetaB>=itheta:
                irad = (itheta-thetaA)/(thetaB-thetaA)*(radiusB-radiusA)+radiusA
                break
        x = int(irad*np.cos(itheta)+cx)
        y = int(irad*np.sin(itheta)+cy)
        cnt2.append([[y,x]])
    return cnt2

def threshold_image(im,imgray,eyes,mu,sig):
    mu = mu.astype(int)
    sig = sig.astype(int)
    for (ex,ey,ew,eh) in eyes:
        im[ey:ey+eh,ex:ex+ew,0] = mu[0]
        im[ey:ey+eh,ex:ex+ew,1] = mu[1]
        im[ey:ey+eh,ex:ex+ew,2] = mu[2]
    thresh = np.ones((im.shape[0],im.shape[1]),dtype='uint8')*255
    for i in xrange(3):
        low = mu[i]-sig[i]*.75
        high = mu[i]+sig[i]*.75
        _,threshLow = cv2.threshold(im[:,:,i],low,255,cv2.THRESH_BINARY)
        _,threshHigh = cv2.threshold(im[:,:,i],high,255,cv2.THRESH_BINARY_INV)
        T = cv2.bitwise_and(threshLow,threshHigh)
        thresh = cv2.bitwise_and(thresh,T)
    return thresh

def get_fpoints(rects, im, imgray):
    for rect in rects:
        x,y,x2,y2 = rect
        w = x2-x
        h = y2-y
        x = int(x+0.05*w)
        w = int(0.90*w)
        y = int(y-0.10*h)
        h = int(h*1.3)
        x2 = x+w
        y2 = y+h
        imgray2 = imgray[y:y2,x:x2]
        im2 = im[y:y2,x:x2]
        im3 = im[y+0.25*h:y+0.7*h,x+0.25*w:x+0.75*w]
        mu,sig = cv2.meanStdDev(im3)
        eyes = eye_cascade.detectMultiScale(imgray2)
        #print imgray.shape
        #print eyes
        eyes = [eye for eye in eyes if eye[1]<imgray2.shape[0]/2]
        if len(eyes)!=2:
            continue
        thresh = threshold_image(im2.copy(),imgray2,eyes,mu,sig)
        if thresh is None:
            continue
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        # only keep contours that are big enough
        contours = [cnt for cnt in contours if cv2.contourArea(cnt)>0.33*imgray2.size]
        #contours = [cv2.approxPolyDP(cnt,0.025*cv2.arcLength(cnt,True),True) for cnt in contours]
        contours = [cv2.convexHull(cnt) for cnt in contours]
        if len(contours)==0:
            continue
        facePoints = interpolate_cpoly(contours[0])
        # add eye points
        for (ex,ey,ew,eh) in eyes:
            cx,cy = int(ex),int(ey+0.5*eh)
            facePoints.append([[cx,cy]])
            cx,cy = int(ex+ew),int(ey+0.5*eh)
            facePoints.append([[cx,cy]])
            cx,cy = int(ex+0.5*ew),int(ey)
            facePoints.append([[cx,cy]])
            cx,cy = int(ex+0.5*ew),int(ey+eh)
            facePoints.append([[cx,cy]])
        # find  mouth point
        # not sure how to do this...

        # draw fiducial points onto image
        for pt in facePoints:
            pt = pt[0]
            # (img, center, radius, color, thickness=1, lineType=8, shift=0
            cv2.circle(im2, (pt[0],pt[1]), 1, (0,255,255), 3)
        
        cv2.imshow('FACE', im2)
        cv2.waitKey(0)
        #cv2.waitKey(1000/(fps))
        #cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
    return im

# Captures a single image from the camera and returns it in IplImage format
def get_image():
    # QueryFrame is the easiest way to get a full image out of a capture object
    retval,im = camera.read()
    return im

def find_faces(fps, nFrames):
    rects = []
    while nFrames > 0:
    # Don't need to actually save these images
        im = get_image()
        im2 = np.copy(im)
        imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        nRects = detect_faces(imgray)
        if nRects!=[]:
            rects = nRects
        im2 = get_fpoints(rects,im2,imgray)
        nFrames = nFrames - 1
        #cv2.imshow('dst_rt', im2)
        #cv2.waitKey(1000/(fps))
        #time.sleep(1/fps)
    return

import sys, traceback

# Now we can set up the camera with the CaptureFromCAM() function. All it needs is
# the index to a camera port. The 'camera' variable will be a cv2.capture object
try:
    camera = []
    camera = cv2.VideoCapture(camera_port)
    find_faces(60, 32)
except Exception,e:
    print e
    traceback.print_exc(file=sys.stdout)
finally:
    cv2.destroyAllWindows()
    del(camera)
