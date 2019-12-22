import cv2
import numpy as np
import math
import imutils
import unittest

ROW_DIFF = np.uint16(15)
LOWER_RED = np.array([160, 50, 50])
UPPER_RED = np.array([180, 255, 255])


def get_vertex_count(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    return len(approx)


def main(img_num):
    src = cv2.imread(f"tests/test_sample{img_num}.jpg", 1)
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_RED, UPPER_RED)
    _, thresh_red = cv2.threshold(mask, 127, 255, 0)
    contours, _ = cv2.findContours(thresh_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_rect = [c for c in contours if get_vertex_count(c) == 4]
    largest_square_contour = sorted(contours_rect, key=cv2.contourArea, reverse=True)[0]
    _, (width, height), angle = cv2.minAreaRect(largest_square_contour)
    # x, y, w, h = cv2.boundingRect(contours[0])
    # make the angle in the [0, 180) range *-ve
    if width < height:
        angle = angle - 90

    if 2 < abs(angle) < 178:
        angle = angle + 180 if abs(angle) < 5 else angle
        src = imutils.rotate(src, angle + 180)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    # cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
    # cv2.imshow("gray", thresh)
    circle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    eroded_thresh = cv2.erode(thresh, circle_kernel)

    height, width = thresh.shape
    eroded_thresh_questions = eroded_thresh[height // 4:height, width // 2:width]
    img_display = src[height // 4:height, width // 2:width]

    black_circles_questions = cv2.HoughCircles(eroded_thresh_questions, cv2.HOUGH_GRADIENT, 1, 10,
                                               param1=10, param2=11,
                                               minRadius=1, maxRadius=15)

    thresh_questions = thresh[height // 4:height, width // 2:width]
    mixed_circles_questions = cv2.HoughCircles(thresh_questions, cv2.HOUGH_GRADIENT, 1, 10,
                                               param1=10, param2=15,
                                               minRadius=9, maxRadius=15)

    def get_circle_centers(circles):
        if circles is not None:
            return [(center[0], center[1]) for center in circles]

    def get_centers_x_sorted(circles):
        if circles is not None:
            return sorted([center[0] for center in circles])

    def get_centers_y_sorted(circles):
        if circles is not None:
            # for x, y, radius in circles:
            #     center = (x, y)
            #     # circle center
            #     cv2.circle(img_display, center, 1, (0, 100, 100), 3)
            #     # circle outline
            #     cv2.circle(img_display, center, radius, (255, 0, 255), 3)
            return sorted([center[1] for center in circles])

    all_circles_questions = []
    all_circles_questions.extend(black_circles_questions[0])
    all_circles_questions.extend(mixed_circles_questions[0])

    centers_sorted_by_y = get_centers_y_sorted(all_circles_questions)
    questions_y_list = [centers_sorted_by_y[0]]
    last_y = centers_sorted_by_y[0]

    for y in centers_sorted_by_y:
        y_diff = y - last_y
        if y_diff < ROW_DIFF:
            continue
        else:
            questions_y_list.append(y)
            last_y = y

    centers_sorted_by_x = get_centers_x_sorted(all_circles_questions)
    max_x = centers_sorted_by_x[-1]
    min_x = centers_sorted_by_x[0]
    span = (max_x - min_x) / 5

    answers = []
    answers2 = []
    black_circles = black_circles_questions.tolist()[0]
    black_circles.sort(key=lambda circle: circle[1])
    c = 0
    for y in questions_y_list:
        ans_num = (black_circles[c][0] - min_x) / span
        if abs(y - black_circles[c][1]) < 5:
            answers.append(int(1 + math.floor(ans_num)))
            answers2.append(ans_num)
            c = c + 1
        else:
            answers.append("Unanswered")

    eroded_thresh_upper = eroded_thresh[0:height // 4]
    thresh_upper = thresh[0:height // 4 - 20]
    img_display = src[0:height // 4]

    black_circles_upper = cv2.HoughCircles(eroded_thresh_upper, cv2.HOUGH_GRADIENT, 1, 10,
                                           param1=10, param2=11,
                                           minRadius=9, maxRadius=15)
    mixed_circles_upper = cv2.HoughCircles(thresh_upper, cv2.HOUGH_GRADIENT, 1, 10,
                                           param1=10, param2=11,
                                           minRadius=11, maxRadius=15)
    all_circles_upper = []
    all_circles_upper.extend(black_circles_upper[0])
    all_circles_upper.extend(mixed_circles_upper[0])

    centers_sorted_by_y = sorted(get_circle_centers(all_circles_upper), key=lambda center: center[1])

    y_centers = []
    x_centers = [(round(center[0])) for center in centers_sorted_by_y]
    centers_sorted_by_y = [(center[1]) for center in centers_sorted_by_y]

    last_y = 0
    for y in centers_sorted_by_y:
        y_diff = y - last_y
        if y_diff < ROW_DIFF:
            continue
        else:
            y_centers.append(y)
            last_y = y

    unique_y_list = [0] * 10  # when that num was 4 it caused a bug

    last_y = int(y_centers[0])
    c = 0

    intcenters = np.array(centers_sorted_by_y)
    intcenters = intcenters.astype(int)

    for y in intcenters:
        diff = y - last_y
        if diff < 5:
            unique_y_list[c] = unique_y_list[c] + 1
        else:
            last_y = y
            c = c + 1
            unique_y_list[c] = unique_y_list[c] + 1

    unique_y_list = [y for y in unique_y_list if y > 1]
    gender = "no gender"
    if unique_y_list[0] == 3:
        gender_xVals = x_centers[0:3]
        gender_xVals.sort(key=lambda circle: circle)
        if gender_xVals[1] == gender_xVals[2]:
            gender = "Female"
        else:
            gender = "Male"

    def firstDuplicate(a):
        differnce_between_elements = np.diff(a)
        for i, element_diff in enumerate(differnce_between_elements):
            if abs(element_diff) < 5:
                return i
        # set_ = set()
        # for item in a:
        #     if item in set_:
        #         return item
        #     set_.add(item)
        return None

    semester = "no semester"
    if unique_y_list[1] == 4:
        val1 = unique_y_list[0]
        semster_xVals = x_centers[val1:val1 + 4]
        semster_xVals.sort(key=lambda circle: circle)
        duplicate_index = firstDuplicate(semster_xVals)

        semster_xVals.remove(semster_xVals[duplicate_index])
        if duplicate_index == 0:
            semester = "Fall"
        if duplicate_index == 1:
            semester = "Spring"
        if duplicate_index == 2:
            semester = "Summer"

    if unique_y_list[2] + unique_y_list[3] == 12:
        program_xVals = x_centers[unique_y_list[0] + unique_y_list[1] - 1:]

        program_xVals.sort(key=lambda circle: circle)

        dup = firstDuplicate(program_xVals)
        if unique_y_list[2] == 8:
            span = (program_xVals[-1] - min(program_xVals)) / 7
            ans_num = (dup - min(program_xVals)) / span
        # answers.append(int(1 + math.floor(ans_num)))
        #  print(int(1 + math.floor(ans_num)))

        # if(unique_y_list[3] == 4) #egaba in ta7t
    # else:
    # program = no ans
    # hnege 3and ael one and nshof el mttkrara akbar or as8yar law as8yar yeb2a male else femal

    answers.extend([gender, semester])
    return answers


test_ans = [[4, 1, 4, 2, 1, 2, 5, 4, 2, 4, 2, 2, 1, 4, 3, 1, 3, 1, 3, "Female", "Fall"],
            [3, 1, 3, 1, 2, 4, 4, 4, 2, 2, 1, 3, 1, 2, 1, 4, 1, 3, 2, "Male", "Summer"],
            [2, 1, 3, 4, 2, 4, 4, 4, 2, 2, 1, 2, 1, 3, 1, 4, 3, 1, 2, "Male", "Summer"],
            [1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 4, 2, 2, "Male", "Fall"],
            [1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 2, 1, 2, 3, 4, 5, 4, 2, 1, "Male", "Fall"],
            [2, 3, 3, 2, 5, 5, 4, 1, 2, 1, 1, 1, 2, 4, 4, 1, 2, 2, 2, "Female", "Fall"],
            [1, 3, 4, 1, 3, 2, 4, 3, 2, 3, 4, 5, 1, 5, 3, 1, 4, 1, 3, "Female", "Fall"],
            [4, 3, 4, 2, 3, 1, 5, 4, 1, 4, 2, 2, 1, 3, 3, 2, 3, 1, 2, "Female", "Fall"],  # rotation
            [1, 1, 2, 1, 2, 4, 3, 4, 3, 2, 2, 3, 3, 3, 2, 3, 2, 3, 1, "Male", "Spring"],  # rotation
            [5, 1, 4, 2, 4, 2, 4, 1, 3, 2, 3, 3, 2, 1, 4, 3, 1, 4, 1, "Female", "Summer"],
            [4, 1, 4, 2, 1, 2, 5, 4, 'Unanswered', 4, 2, 2, 1, 4, 3, 1, 3, 1, 3, "Female", "Fall"]]


class TestAnswers(unittest.TestCase):
    def tests(self):
        for i in range(1, 12):
            with self.subTest(i=i):
                self.assertEqual(main(i), test_ans[i - 1])


if __name__ == '__main__':
    unittest.main()
