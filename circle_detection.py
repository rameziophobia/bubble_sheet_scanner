import cv2
import numpy as np
import math
ROW_DIFF = np.uint16(15)
src = cv2.imread("tests/test_sample4.jpg", 1)
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
cv2.imshow("asd", eroded_thresh)
black_circles = cv2.HoughCircles(eroded_thresh, cv2.HOUGH_GRADIENT, 1, 10,
                           param1=10, param2=11,
                           minRadius=9, maxRadius=15)

thresh = thresh[height // 4:height, width // 2:width]
mixed_circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 10,
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

loop_circles(black_circles)
#loop_circles(mixed_circles)

cv2.namedWindow("detected circles", cv2.WINDOW_NORMAL)
cv2.imshow("detected circles", src)
cv2.imwrite("detected circles.jpg", img_display)
cv2.imwrite("thresh.jpg", thresh)
cv2.waitKey(0)

# todo
# sort all circles 7asab el y

centers_sorted_by_y = sorted(all_centers_XandY, key=lambda circle: circle[1])
centers_sorted_by_y = [(center[1]) for center in centers_sorted_by_y]
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


# todo
# 1. sort list of all detected circles based on x
# 2. get max_x, min_x
# 3. define span as (max_x - min_x) / 5

centers_sorted_by_x = sorted(all_centers_XandY, key=lambda circle : circle[0])
max_x = centers_sorted_by_x[-1][0]
min_x = centers_sorted_by_x[0][0]
span = (max_x - min_x) / 5
print("span =", span)
print("centers =", centers_sorted_by_x)
# loop over black_circles
#   ans_num = (circle.x - min.x) / span
#
answers = []
# print(type(black_circles))
black_circles = black_circles.tolist()[0]
# print(type(black_circles))

black_circles.sort(key=lambda circle: circle[1])
c = 0
# print(len(black_circles))
for y in questions_y_list:

    ans_num = (black_circles[c][0] - min_x)/ span
    print(ans_num,end =" ")
    # print(f"min: {min_x}, max: {max_x}, span: {span}, reg: {max_x - min_x}")
    # print(f"x{c}: {black_circles[c][0]}, y: {black_circles[c][1]}")

    if abs(y - black_circles[c][1]) < 5:
        answers.append(int(1 + math.floor(ans_num)))
        c = c + 1
    else:
        answers.append("Unanswered")

print(" ")
print(answers)

# todo
# nafs elli ta7t ne3mlo fo2
# crop el 7eta el fo2 fel sora

# img_display = src[0:height// 4, 0:width]# [y1:y2, x1:x2]

ret, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
eroded_thresh = cv2.erode(thresh, circle_kernel) # obtain sora ndefa

eroded_thresh = eroded_thresh[0:height//4]
img_display = src[0:height//4]



# a3mel black circle detection and all circles detection
cv2.imshow("tis",eroded_thresh)
black_circles = cv2.HoughCircles(eroded_thresh, cv2.HOUGH_GRADIENT, 1, 10,
                           param1=10, param2=11,
                           minRadius=9, maxRadius=15)
thresh = thresh[0:height//4-20]
cv2.imshow("this 20", thresh)

mixed_circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 10,
                            param1=10, param2=11,
                            minRadius=11, maxRadius=15)
all_centers_XandY = []

loop_circles(black_circles)
loop_circles(mixed_circles)


# bos ba2a hnsgel el mkan el feh 2 circles overlap, ba3d kda hnshof fe circles 3ala el shemal


centers_sorted_by_y = sorted(all_centers_XandY, key =lambda circle: circle[1])
# print(centers_sorted_by_y)
#sort based n circles y
last_x = centers_sorted_by_y[0][0]
c=0

y_centers = []
# print(centers_sorted_by_y)
x_centers =[]
x_centers = [(center[0]) for center in centers_sorted_by_y]
centers_sorted_by_y = [(center[1]) for center in centers_sorted_by_y]
# print(last_y)
last_y = 0

for y in centers_sorted_by_y:
    y_diff = y - last_y
    if y_diff < ROW_DIFF:
        continue
    else:
        y_centers.append(y)
        last_y = y

# print(y_centers)
# print(x_centers)
x = x_centers[0]
unique_y_list = [0]*4
# print(unique_y_list)

last_y = int(y_centers[0])
c = 0
# print("ere")
# print(centers_sorted_by_y)
intcenters = np.array(centers_sorted_by_y)
intcenters - intcenters.astype(int)

for y in intcenters:
    diff = y- last_y
    # print(diff)
    if(diff <5):
        unique_y_list[c] = unique_y_list[c] +1
    else:
        last_y = y
        c = c+1
        unique_y_list[c] = unique_y_list[c] + 1

# print(unique_y_list)
max = x_centers[0]
print(unique_y_list)
if(unique_y_list[0] == 3):
    gender_xVals = []
    gender_xVals = x_centers[0:3]
    gender_xVals.sort(key=lambda circle: circle)
    # print(gender_xVals)

    if(gender_xVals[1] == gender_xVals[2]): gender = "Female"
    else: gender = "Male"
else:
    gender = "no gender"

print("Gender:", gender)

def firstDuplicate(a):
    set_ = set()
    for item in a:
        if item in set_:
            return item
        set_.add(item)
    return None
if(unique_y_list[1] == 4):
    semster_xVals = []
    semster_xVals = x_centers[3:7]
    val1 = unique_y_list[0]
    val2 = unique_y_list[2]
    semster_xVals = x_centers[val1:val1+4]
    semster_xVals.sort(key=lambda circle: circle)
    duplicate = firstDuplicate(semster_xVals)
    semster_xVals.remove(duplicate)
    if(duplicate == semster_xVals[0]): semster = "Fall"
    if(duplicate == semster_xVals[1]): semster = "Spring"
    if(duplicate == semster_xVals[2]): semster = "Summer"

else:
    semster = "no ans"
print("Semster: ",semster)
#shoof fe kam circle fo2 and ta7t if 8 yeb2a egaba fo2

#if 5 yeb2a ta7t then a3mel el kona 3amleno fo2 ma3 ramez noshy
if(unique_y_list[2]+unique_y_list[3] == 12):
    # program_xVals = x_centers[6:]
    program_xVals = x_centers[unique_y_list[0]+unique_y_list[1] - 1: ]

    program_xVals.sort(key=lambda circle: circle)

    dup =  firstDuplicate(program_xVals)
    if(unique_y_list[2] == 8): # egaba in fo2

        span = (program_xVals[-1] - min(program_xVals) )/7
        ans_num = (dup - min(program_xVals) ) / span
       # answers.append(int(1 + math.floor(ans_num)))
       #  print(int(1 + math.floor(ans_num)))

    # if(unique_y_list[3] == 4) #egaba in ta7t
# else:
    # program = no ans
# hnege 3and ael one and nshof el mttkrara akbar or as8yar law as8yar yeb2a male else femal

cv2.imshow("ne",img_display)
cv2.imwrite("ne.jpeg", img_display)
cv2.waitKey(0)

# last todo
# 1. fix the qesution equation?
# 2. make the program
# 3. input img
# 4. output img
