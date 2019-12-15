import cv2
import numpy as np
import imutils

img = cv2.imread("tests/test_sample1.jpg", 1)
# cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img", img)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# filter red colors only
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
cv2.imshow("mask blue", mask)


lower_red = np.array([160, 50, 50])
upper_red = np.array([180, 255, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)
cv2.imshow("mask red", mask)
ret, thresh = cv2.threshold(mask, 127, 255, 0)
cv2.imshow("thresh", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
# find contours in the filtered image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


def get_vertex_count(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    return len(approx)


contours_rect = [c for c in contours if get_vertex_count(c) == 4]
mask = np.zeros(img.shape, np.uint8)
cv2.drawContours(mask, contours, -1, (0, 255, 0), 1)
cv2.imshow("mask red", mask)
mask = np.zeros(img.shape, np.uint8)
cv2.drawContours(mask, contours_rect, -1, (0, 255, 0), 1)
cv2.imshow("mask red squares", mask)
largest_square_contour = sorted(contours_rect, key=cv2.contourArea, reverse=True)[0]
mask = np.zeros(img.shape, np.uint8)
cv2.drawContours(mask, [largest_square_contour], -1, (0, 0, 255), 1)
cv2.imshow("largest red square", mask)

_, (width, height), angle = cv2.minAreaRect(largest_square_contour)
x, y, w, h = cv2.boundingRect(contours[0])
# # make the angle in the [0, 180) range *-ve
# if width < height:
#     angle = angle - 90
# cv2.imshow("cnt", contours[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
