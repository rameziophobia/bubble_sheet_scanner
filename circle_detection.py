import cv2
import numpy as np
import math

ROW_DIFF = np.uint16(15)
src = cv2.imread("tests/test_sample6.jpg", 1)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
ret, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
cv2.imshow("gray", thresh)
circle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
eroded_thresh = cv2.erode(thresh, circle_kernel)

height, width = thresh.shape
eroded_thresh = eroded_thresh[height // 4:height, width // 2:width]
img_display = src[height // 4:height, width // 2:width]

black_circles = cv2.HoughCircles(eroded_thresh, cv2.HOUGH_GRADIENT, 1, 10,
                                 param1=10, param2=11,
                                 minRadius=1, maxRadius=15)

thresh = thresh[height // 4:height, width // 2:width]
mixed_circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 10,
                                 param1=10, param2=15,
                                 minRadius=9, maxRadius=15)


# def get_circle_centers(circles):
#     if circles is not None:
#         return [(center[0], center[1]) for center in circles[0]]


def get_centers_x_sorted(circles):
    if circles is not None:
        return sorted([center[0] for center in circles])


def get_centers_y_sorted(circles):
    if circles is not None:
        return sorted([center[1] for center in circles])


all_circles = []
all_circles.extend(black_circles[0])
all_circles.extend(mixed_circles[0])

centers_sorted_by_y = get_centers_y_sorted(all_circles)
questions_y_list = [centers_sorted_by_y[0]]
last_y = centers_sorted_by_y[0]

for y in centers_sorted_by_y:
    y_diff = y - last_y
    if y_diff < ROW_DIFF:
        continue
    else:
        questions_y_list.append(y)
        last_y = y
# last y feha el y's bta3et kol row we sorted kman shoft ba2a el 7lawa

centers_sorted_by_x = get_centers_x_sorted(all_circles)
max_x = centers_sorted_by_x[-1]
min_x = centers_sorted_by_x[0]
span = (max_x - min_x) / 5

answers = []
black_circles = black_circles.tolist()[0]
black_circles.sort(key=lambda circle: circle[1])
c = 0
for y in questions_y_list:
    ans_num = (black_circles[c][0] - min_x) / span
    if abs(y - black_circles[c][1]) < 5:
        answers.append(int(1 + math.floor(ans_num)))
        c = c + 1
    else:
        answers.append("Unanswered")

print(answers)

# todo
# nafs elli ta7t ne3mlo fo2
