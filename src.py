import numpy;
import cv2;
import os;

IMAGE_PATH_ROOT = "E:\YandexDisk\Learning\TechnoPark\CUDA\TP_CUDA\photos\\"
IMAGE_PATH_DETECTOR_IN = "detector_new\\"
IMAGE_PATH_DETECTOR_OUT = "detector_new\out\\"
YELLOW_COIN_COLORS_BGR = [
    # 10 kops
    ([45, 65, 85], [45, 65, 85]),       #55412d
    ([24, 32, 45], [24, 32, 45]),       #2d2018
    ([30, 45, 59], [30, 45, 59]),       #3b2d1e
    ([29, 65, 88], [29, 65, 88]),       #58411d
    ([24, 32, 44], [24, 32, 44]),       #2c2018
    ([29, 44, 72], [29, 44, 72]),       #482c1d
    ([35, 58, 88], [35, 58, 88]),       #583a23
    # 50 kops
    ([32, 54, 72], [32, 54, 72]),       #483620
    ([28, 56, 80], [28, 56, 80]),       #50381c
    ([33, 48, 60], [33, 48, 60]),       #3c3021
        # enlightened
    ([96, 167, 208], [96, 167, 208]),   #d0a760
    ([63, 121, 157], [63, 121, 157]),   #9d793f
    # 10 roubles
    ([28, 55, 74], [28, 55, 74]),       #4a371c
]
YELLOW_COIN_COLOR_BOUNDARY = ([-15, -15, -20], [30, 35, 20])
SILVER_COIN_COLORS_BGR = [
    # 5 roubles
    ([69, 77, 86], [69, 77, 86]),       #564d45
    ([68, 73, 76], [68, 73, 76]),       #4c4944
    ([59, 60, 66], [59, 60, 66]),       #423c3b
        # enlightened
    ([124, 173, 202], [124, 173, 202]), #caad7c
    ([181, 155, 134], [181, 155, 134]), #869bb5
    ([173, 150, 130], [173, 150, 130]), #8296ad
    ([141, 126, 112], [141, 126, 112]), #707e8d
    # 1 rouble
    ([55, 63, 74], [55, 63, 74]),       #4a3f37
    # 2 roubles
    ([59, 65, 72], [59, 65, 72]),       #48413b
        # enlightened
    ([136, 188, 210], [136, 188, 210]), #d2bc88
    # 10 roubles
    ([82, 85, 86], [82, 85, 86]),       #565552
]
SILVER_COIN_COLOR_BOUNDARY = ([-15, -15, -15], [30, 30, 30])

def print_image(image):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', image)
    cv2.resizeWindow('image', 1024, 768)
    cv2.waitKey(0)

def apply_masks(image):
    masked_image = numpy.zeros(image.shape, numpy.uint8)
    for (lower, upper) in YELLOW_COIN_COLORS_BGR:
        lower = list(map(sum, zip(lower, YELLOW_COIN_COLOR_BOUNDARY[0])))
        lower = numpy.array(lower, dtype="uint8")
        upper = list(map(sum, zip(upper, YELLOW_COIN_COLOR_BOUNDARY[1])))
        upper = numpy.array(upper, dtype="uint8")

        mask = cv2.inRange(image, lower, upper)
        current_masked_image = cv2.bitwise_and(image, image, mask=mask)
        masked_image = cv2.bitwise_or(masked_image, current_masked_image)

    for (lower, upper) in SILVER_COIN_COLORS_BGR:
        lower = list(map(sum, zip(lower, SILVER_COIN_COLOR_BOUNDARY[0])))
        lower = numpy.array(lower, dtype="uint8")
        upper = list(map(sum, zip(upper, SILVER_COIN_COLOR_BOUNDARY[1])))
        upper = numpy.array(upper, dtype="uint8")

        mask = cv2.inRange(image, lower, upper)
        current_masked_image = cv2.bitwise_and(image, image, mask=mask)
        masked_image = cv2.bitwise_or(masked_image, current_masked_image)

    return masked_image
"""
    cv2.namedWindow('images', cv2.WINDOW_NORMAL)
    cv2.imshow("images", numpy.hstack([image, masked_image]))
    cv2.resizeWindow('images', 2048, 768)
    cv2.waitKey(0)
"""

def preprocess_image(image):
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    masked = apply_masks(blurred)
    masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    #thresholded = cv2.adaptiveThreshold(masked, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 1.7)
    _,  thresholded = cv2.threshold(masked, 10, 255, cv2.THRESH_BINARY_INV)
    _,  thresholded2 = cv2.threshold(masked, 120, 255, cv2.THRESH_BINARY)
    thresholded3 = cv2.bitwise_xor(thresholded, thresholded2)
    blurred_again = cv2.GaussianBlur(thresholded3, (13, 13), 0)
    return blurred_again
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
    circles = cv2.HoughCircles(image, cv2.cv.CV_HOUGH_GRADIENT, 2, 120, param1=100, param2=75, minRadius=15, maxRadius=150);
    if circles != None:
        circles = numpy.uint16(numpy.around(circles))
    return circles


path = IMAGE_PATH_ROOT+IMAGE_PATH_DETECTOR_IN;
filenames = next(os.walk(path))[2];
for image in filenames:
    img_gray = cv2.imread(path+image, -1);
    filtered_image = preprocess_image(img_gray);

    circles = get_circles(filtered_image)

    img = cv2.imread(path+image, -1);
    if circles != None:
        for i in circles[0,:]:
            cv2.circle(img,(i[0],i[1]),i[2],(40,150,40),5)
            cv2.circle(img,(i[0],i[1]),2,(10,140,170),10)

    cv2.namedWindow(image, cv2.WINDOW_NORMAL);
    cv2.imshow(image, img);
    cv2.imwrite(IMAGE_PATH_ROOT+IMAGE_PATH_DETECTOR_OUT+image, img)
    cv2.resizeWindow(image, 1024, 768);

    #cv2.waitKey(0);
    cv2.destroyAllWindows();
