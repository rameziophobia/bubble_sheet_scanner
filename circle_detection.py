import cv2
import numpy as np
ROW_DIFF = np.uint16(15)
src = cv2.imread("tests/test_sample1.jpg", 1)
# src = cv.resize(src, (800, 1200))
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
        # print(type(circles))
        for x, y, radius in circles[0, :]:
            all_centers_XandY.append((x, y))
            # print(f"x {x}, y {y}")
            center = (x, y)
            # circle center
            cv2.circle(img_display, center, 1, (0, 100, 100), 3)
            # circle outline
            cv2.circle(img_display, center, radius, (255, 0, 255), 3)


all_centers_XandY = []

loop_circles(circles)
loop_circles(circles2)

cv2.namedWindow("detected circles", cv2.WINDOW_NORMAL)
cv2.imshow("detected circles", src)
cv2.imwrite("detected circles.jpg", img_display)
cv2.imwrite("thresh.jpg", thresh)
cv2.waitKey(0)

# todo
# 1. sort list of all detected circles based on x
# 2. get max_x, min_x
# 3. define span as (max_x - min_x) / 5
# loop over black_circles
#   ans_num = (circle.x - min.x) / span
centers_sorted_by_x = sorted(all_centers_XandY, key=lambda circle : circle[0])
max_x = centers_sorted_by_x[-1][0]
min_x = centers_sorted_by_x[0][0]
span = (max_x - min_x) / 5



# todo
# sort all circles 7asab el y

centers_sorted_by_y = sorted(all_centers_XandY, key=lambda circle: circle[1])
centers_sorted_by_y = [center[1] for center in centers_sorted_by_y]
y_list = [centers_sorted_by_y[0]]
last_y = centers_sorted_by_y[0]
print(type(last_y))
for y in centers_sorted_by_y:
    # print(type(y))
    y_diff = np.uint16(last_y - y)
    print(y_diff)
    if y_diff < ROW_DIFF:
        continue
    else:
        y_list.append(y)
        last_y = y
# last y feha el y's bta3et kol row we sorted kman shoft ba2a el 7lawa

print(y_list)
print(len(y_list))





# use black lines to mark circle_y
