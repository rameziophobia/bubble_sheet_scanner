import cv2
import numpy as np
import math
import imutils
import unittest
import json

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



    eroded_thresh_upper = eroded_thresh[225:height // 4 -30] # crop more to avoid detection of example's circle
    thresh_upper = thresh[225:height // 4 - 30]
    img_display = src[225:height // 4 -30]

    black_circles_upper = cv2.HoughCircles(eroded_thresh_upper, cv2.HOUGH_GRADIENT, 1, 10,
                                           param1=10, param2=11,
                                           minRadius=9, maxRadius=15)
    mixed_circles_upper = cv2.HoughCircles(thresh_upper, cv2.HOUGH_GRADIENT, 1, 10,
                                           param1=10, param2=11,
                                           minRadius=11, maxRadius=15)
    all_circles_upper = []
    all_circles_upper.extend(black_circles_upper[0])
    all_circles_upper.extend(mixed_circles_upper[0])

    def loop_circles(circles):  # I added this method for debugging, please don't remove it ramez.
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # print(type(circles))
            for x, y, radius in circles[0, :]:
                #all_centers_XandY.append((x, y))
              #  print(f"x {x}, y {y}")
                center = (x, y)
                # circle center
                cv2.circle(img_display, center, 1, (0, 100, 100), 3)
                # circle outline
                cv2.circle(img_display, center, radius, (255, 0, 255), 3)
    # loop_circles(black_circles_upper)
    # loop_circles(mixed_circles_upper)


    # if(img_num ==8 ):
    #       cv2.imshow(f"test_sample{img_num}.jpg",img_display)
    #       cv2.waitKey(0)

    centers_sorted_by_y = sorted(get_circle_centers(all_circles_upper), key=lambda center: center[1])
    # print(centers_sorted_by_y) #contains coord of all circles sorted by y
    y_centers = []
    x_centers = [(round(center[0])) for center in centers_sorted_by_y]
    # print(x_centers) #contains the x coord of all circles sorted by y
    centers_sorted_by_y = [(center[1]) for center in centers_sorted_by_y]
    #print(centers_sorted_by_y) #now it contains the y-coord of all circles sorted

    last_y = 0

    #this loop extracts the y-vals of the different ans sets

    for y in centers_sorted_by_y:
        y_diff = y - last_y
        if y_diff < ROW_DIFF:
            continue
        else:
            y_centers.append(y)
            last_y = y
   # print(y_centers)
    unique_y_list = [0] * 10  # when that num was 4 it caused a bug     # ok Ramez, thanks for the comment :)

    last_y = int(y_centers[0])
    c = 0

    intcenters = np.array(centers_sorted_by_y)
    intcenters = intcenters.astype(int)

    #this loop extracts the number of circles in each set of ans
    for y in intcenters:
        diff = y - last_y
        if diff < 5:
            unique_y_list[c] = unique_y_list[c] + 1
        else:
            last_y = y
            c = c + 1
            unique_y_list[c] = unique_y_list[c] + 1
    # print(unique_y_list,"test: ", img_num)
    unique_y_list = [y for y in unique_y_list if y > 1]


    gender = "Unanswered"
    if unique_y_list[0] == 3: # if there is 3 circle in the first line then the q is ansered
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

    semester = "Unanswered"
    if unique_y_list[1] == 4:
        val1 = unique_y_list[0]
        semster_xVals = x_centers[val1:val1 + 4]
        semster_xVals.sort(key=lambda circle: circle)
        duplicate_index = firstDuplicate(semster_xVals)
       # print(duplicate_index)
        for i in range(len(semster_xVals)-1) :
           # print(abs(semster_xVals[i] - semster_xVals[i+1]) <5)
            if(abs(semster_xVals[i] - semster_xVals[i+1]) <5):
                semster_xVals[i] = semster_xVals[i+1]
      #  print(semster_xVals)
        semster_xVals.remove(semster_xVals[duplicate_index])
       # print(semster_xVals)
        if duplicate_index == 0:
            semester = "Fall"
        if duplicate_index == 1:
            semester = "Spring"
        if duplicate_index == 2:
            semester = "Summer"

    program = "Unanswered"
    if unique_y_list[2] + unique_y_list[3] == 12:
        program_xVals = x_centers[unique_y_list[0] + unique_y_list[1]  :] #-1 hena removed(solved bug in program x vals)

        xtest= []
        xtest = x_centers
        xtest.sort(key=lambda circle:circle)
        #print(xtest)
        program_xVals.sort(key=lambda circle: circle)
        # print(program_xVals)

        if(len(program_xVals) >12): # akbar men 12 and not betenn 456 and 1256???
            program = "Unanswered"
        else:
            right_ans_program_xVals = program_xVals[8:]
            dup_index = firstDuplicate(right_ans_program_xVals)
            # print(dup_index)
            if(dup_index ==0):
                program = "ERGY"
            elif(dup_index ==1):
                program = "COMM"
            elif(dup_index ==2):
                program = "MANF"
            else: # *insert crying cat pic here*
                input = program_xVals[0:9]
                n = 3
                prev = -1
                count = 0
                flag = 0
                index =0

                # Iterating
                for idx,item in enumerate(input):
                    if abs(item - prev) < 10:
                        count = count + 1
                        index = idx
                    else:
                        count = 1

                    prev = item

                    if count == n:
                        flag = 1
                        index = index - 2
                       ##un comment this print("There are {} occurrences of {} in index {} in {} in sample {} ".format(n, item, index, input,img_num))
                        if(unique_y_list[3] ==5 and unique_y_list[2] == 7):
                            if(index ==0):
                                program = "LAAR"
                            elif(index ==2):
                                program = "MATL"
                            elif (index == 4):
                                program = "CISE"
                            elif (index == 6):
                                program = "HAUD"
                        elif (unique_y_list[3] == 4 and unique_y_list[2] == 8):
                            if (index == 0):
                                program = "MCTA"
                            elif (index == 2):
                                program = "ENVER"
                            elif (index == 4):
                                program = "BLDG"
                            elif (index == 6):
                                program = "CESS"
                        else:
                            program = "Unanswered"

                        break


    answers.extend([gender, semester, program])
    questions_Sets = [5, 6, 3, 3, 2]
    with open('Output.txt', 'a') as outfile:
        json.dump(f"test_sample{img_num}.jpg", outfile, indent=4)
        c=0
        for i in range(5):
            for j in range(questions_Sets[i]):
                #print(c)
                json.dump({f"Q{i+1}.{j+1}": answers[c]}, outfile, indent=0)
                c=c+1
        json.dump({f"Gender: ": answers[19]}, outfile, indent=0)
        json.dump({f"Semester: ": answers[20]}, outfile, indent=0)
        json.dump({f"Program: ": answers[21]}, outfile, indent=2)


    return answers


test_ans = [[4, 1, 4, 2, 1, 2, 5, 4, 2, 4, 2, 2, 1, 4, 3, 1, 3, 1, 3, "Female", "Fall", "ERGY"],#1
            [3, 1, 3, 1, 2, 4, 4, 4, 2, 2, 1, 3, 1, 2, 1, 4, 1, 3, 2, "Male", "Summer", "MANF"], #2
            [2, 1, 3, 4, 2, 4, 4, 4, 2, 2, 1, 2, 1, 3, 1, 4, 3, 1, 2, "Male", "Summer", "HAUD"],#3
            [1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 4, 2, 2, "Male", "Fall", "MATL"],#4
            [1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 2, 1, 2, 3, 4, 5, 4, 2, 1, "Male", "Fall", "ENVER"],#5
            [2, 3, 3, 2, 5, 5, 4, 1, 2, 1, 1, 1, 2, 4, 4, 1, 2, 2, 2, "Female", "Fall", "BLDG"],#6
            [1, 3, 4, 1, 3, 2, 4, 3, 2, 3, 4, 5, 1, 5, 3, 1, 4, 1, 3, "Female", "Fall", "BLDG"],#7
            [4, 3, 4, 2, 3, 1, 5, 4, 1, 4, 2, 2, 1, 3, 3, 2, 3, 1, 2, "Female", "Fall", "BLDG"], # 8 # rotation
            [1, 1, 2, 1, 2, 4, 3, 4, 3, 2, 2, 3, 3, 3, 2, 3, 2, 3, 1, "Male", "Spring", "COMM"],  #9 # rotation
            [5, 1, 4, 2, 4, 2, 4, 1, 3, 2, 3, 3, 2, 1, 4, 3, 1, 4, 1, "Female", "Summer", "ERGY"],# 10
            [4, 1, 4, 2, 1, 2, 5, 4, 'Unanswered', 4, 2, 2, 1, 4, 3, 1, 3, 1, 3, "Female", "Fall", "ERGY"],# 11
            [4, 1, 4, 2, 1, 2, 5, 4, 2, 4, 2, 2, 1, 4, 3, 1, 3, 1, 3, "Unanswered", "Fall", "ERGY"], # added test 12, same as test 1, but removed gender lol, next transgender test
            [3, 1, 3, 1, 2, 4, 4, 4, 2, 2, 1, 3, 1, 2, 1, 4, 1, 3, 2, "Male", "Unanswered", "MANF"], #test 13, same as test 2, but removed sem
            [1, 1, 2, 1, 2, 4, 3, 4, 3, 2, 2, 3, 3, 3, 2, 3, 2, 3, 1, "Unanswered", "Unanswered", "COMM"], # test 14, same as test 9 but removed gender and sem
            [4, 1, 4, 2, 1, 2, 5, 4, 2, 4, 2, 2, 1, 4, 3, 1, 3, 1, 3, "Female", "Unanswered", "ERGY"], # multiple sem
            [4, 1, 4, 2, 1, 2, 5, 4, 2, 4, 2, 2, 1, 4, 3, 1, 3, 1, 3, "Female", "Unanswered", "ERGY"], #multiple sem
            [4, 1, 4, 2, 1, 2, 5, 4, 2, 4, 2, 2, 1, 4, 3, 1, 3, 1, 3, "Unanswered", "Unanswered", "Unanswered"], # multiple gen and sem
            [4, 3, 4, 2, 3, 1, 5, 4, 1, 4, 2, 2, 1, 3, 3, 2, 3, 1, 2, "Female", "Unanswered", "Unanswered"]] # multiple programs

class TestAnswers(unittest.TestCase):
    def tests(self):
        for i in range(1, 12+3+3+1): # added 3+3+1 more tests, and guess whatttttt!!!. they all worked fine :).
            with self.subTest(i=i):
                self.assertEqual(main(i), test_ans[i - 1])


if __name__ == '__main__':
    unittest.main()
