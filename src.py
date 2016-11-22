import numpy;
import cv2;
import os;

IMAGE_PATH_ROOT = "E:\YandexDisk\Learning\TechnoPark\CUDA\TP_CUDA\photos\\";
IMAGE_PATH_DETECTOR_IN = "detector\\";
IMAGE_PATH_DETECTOR_OUT = "detector\out\\";

def preprocess_image(image):
    smoother = numpy.ones((13, 13), numpy.float32) / (10*10);
    smoothed = cv2.filter2D(image, -1, smoother);
    blurred = cv2.GaussianBlur(smoothed, (15, 15), 0);
    #bilatered = cv2.bilateralFilter(image, 45, 160, 200);  # http://docs.opencv.org/3.0-beta/modules/imgproc/doc/filtering.html?highlight=bilateral#cv2.bilateralFilter
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 1.7);
    return thresholded;
"""
    cv2.namedWindow('original', cv2.WINDOW_NORMAL);
    cv2.namedWindow('smoothed', cv2.WINDOW_NORMAL);
    #cv2.namedWindow('bilatered', cv2.WINDOW_NORMAL);
    cv2.namedWindow('blurred', cv2.WINDOW_NORMAL);
    cv2.namedWindow('thresholded', cv2.WINDOW_NORMAL);
    cv2.imshow('original', image);
    cv2.imshow('smoothed', smoothed);
    #cv2.imshow('bilatered', bilatered);
    cv2.imshow('blurred', blurred);
    cv2.imshow('thresholded', thresholded);
    cv2.resizeWindow('original', 1024, 768);
    cv2.resizeWindow('smoothed', 1024, 768);
    #cv2.resizeWindow('bilatered', 1024, 768);
    cv2.resizeWindow('blurred', 1024, 768);
    cv2.resizeWindow('thresholded', 1024, 768);
"""

def get_circles(image):
    circles = cv2.HoughCircles(image, cv2.cv.CV_HOUGH_GRADIENT, 1, 90, param1=100, param2=30, minRadius=0, maxRadius=150);
    return numpy.uint16(numpy.around(circles));


path = IMAGE_PATH_ROOT+IMAGE_PATH_DETECTOR_IN;
filenames = next(os.walk(path))[2];
for image in filenames:
    img_gray = cv2.imread(path+image, 0);
    circles = get_circles(preprocess_image(img_gray));

    img = cv2.imread(path+image, -1);
    for i in circles[0,:]:
        cv2.circle(img,(i[0],i[1]),i[2],(40,150,40),5)
        cv2.circle(img,(i[0],i[1]),2,(10,140,170),10)

    cv2.namedWindow('circles', cv2.WINDOW_NORMAL);
    cv2.imshow('circles', img);
    cv2.imwrite(IMAGE_PATH_ROOT+IMAGE_PATH_DETECTOR_OUT+image, img)
    cv2.resizeWindow('circles', 1024, 768);

    cv2.waitKey(0);
    cv2.destroyAllWindows();
