import cv2


def get_vertex_count(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.2 * peri, True)
    return len(approx)


import numpy as np

src = cv2.imread("tests/test_sample4.jpg", 1)
# src = cv.resize(src, (800, 1200))
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# contours_lines = [c for c in contours if get_vertex_count(c) == 2]
# gray = gray - contours_lines

# gray = cv2.medianBlur(gray, (9,9))
gray1 = cv2.medianBlur(gray, 5)
cv2.namedWindow("gray1", cv2.WINDOW_NORMAL)
cv2.imshow("gray1k", gray1)
cv2.imwrite("gray median.jpg", gray1)
# gray = cv2.GaussianBlur(gray, (9,9), 0)
ret, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
cv2.imshow("gray", thresh)
circle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
eroded_thresh = cv2.erode(thresh, circle_kernel)

height, width = thresh.shape
eroded_thresh = eroded_thresh[height // 4:height, width // 2:width]
img_display = src[height // 4:height, width // 2:width]
circles = cv2.HoughCircles(eroded_thresh, cv2.HOUGH_GRADIENT, 1, 10,
                           param1=10, param2=11,
                           minRadius=1, maxRadius=15)

thresh = thresh[height // 4:height, width // 2:width]
circles2 = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 10,
                            param1=10, param2=15,
                            minRadius=9, maxRadius=15)


def loop_circles(circles):
    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(type(circles))
        for x, y, radius in circles2[0, :]:
            list_s.append((x, y))
            print(f"x {x}, y {y}")
            center = (x, y)
            # circle center
            cv2.circle(img_display, center, 1, (0, 100, 100), 3)
            # circle outline
            cv2.circle(img_display, center, radius, (255, 0, 255), 3)


list_s = []
loop_circles(circles)
loop_circles(circles2)

cv2.namedWindow("detected circles", cv2.WINDOW_NORMAL)
cv2.imshow("detected circles", src)
cv2.imwrite("detected circles.jpg", img_display)
cv2.imwrite("thresh.jpg", thresh)
cv2.waitKey(0)
