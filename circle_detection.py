import cv2
import numpy as np
import math
import imutils
import json
import argparse
import os

ROW_DIFF = np.uint16(15)
CIRCLE_DIFF = 5

LOWER_RED = np.array([160, 50, 50])
UPPER_RED = np.array([180, 255, 255])
img_num = 0


def main(img_path):
    global img_num
    img_num += 1
    src = cv2.imread(f"{img_path}", 1)
    src = rotate_img(src)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) # convert the image to gray image
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV) # reverse the intenisty of the pixels
    circle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    eroded_thresh = cv2.erode(thresh, circle_kernel) # erode image to remove lines and other noise pixels

    answers = extract_question_answers(eroded_thresh, thresh)

    gender, program, semester = extract_upper_answers(eroded_thresh, img_num, src, thresh)

    answers.extend([gender, semester, program])
    answers_dict = get_ans_dict_from_list(answers, gender, img_path, program, semester)
    return answers, answers_dict


def write_answers_to_json(answers_dict):
    with open('Output.json', 'w') as outfile:
        json.dump(answers_dict, outfile, indent=4)


def get_ans_dict_from_list(answers, gender, img_path, program, semester):
    questions_num_per_set = [5, 6, 3, 3, 2]
    answers_dict = {"img_name": os.path.split(img_path)[1],
                    "gender": gender,
                    "semester": semester,
                    "program": program,
                    f"Questions Set0": answers[0: questions_num_per_set[0]]}
    for i in range(len(questions_num_per_set)):
        if i == 0:
            continue
        answers_dict[f"Questions Set{i}"] = \
            answers[questions_num_per_set[i - 1]: questions_num_per_set[i - 1] + questions_num_per_set[i]]
    return answers_dict


def get_vertex_count(cnt): # return the number of vertex of a countur
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    return len(approx)


def get_circle_centers(circles):
    if circles is not None:
        return [(center[0], center[1]) for center in circles]


def get_centers_x_sorted(circles): # return a sorted list of circle center coordiantes sorted by x-value
    if circles is not None:
        return sorted([center[0] for center in circles])


def get_centers_y_sorted(circles):  # return a sorted list of circle center coordiantes sorted by y-value
    if circles is not None:
        return sorted([center[1] for center in circles])


def first_duplicate_index(a): # return index of the first duplicate elemnt in a list
    difference_between_elements = np.diff(a)
    for i, element_diff in enumerate(difference_between_elements):
        if abs(element_diff) < 5:
            return i
    return None


def extract_upper_answers(eroded_thresh, img_num, src, thresh): #
    all_circles_upper = get_upper_circles(eroded_thresh, src, thresh, img_num) # get the upper cirles in a list
    x_centers_sorted_by_y, y_centers_sorted_by_y = zip(
        *sorted(get_circle_centers(all_circles_upper), key=lambda center: center[1]))
    x_centers_sorted_by_y = list(x_centers_sorted_by_y) # all circles x centers sorted by asc from up to bot
    y_centers_sorted_by_y = list(y_centers_sorted_by_y) # all circles y centers sorted by asc from up to bot

    y_centers = []
    last_y = 0

    # this loop extracts the y-value of each row in the upper part in y_centers
    for y in y_centers_sorted_by_y:
        y_diff = y - last_y
        if y_diff < ROW_DIFF:
            continue
        else:
            y_centers.append(y)
            last_y = y



    unique_y_list = [0] * 10  # when that num was 4 it caused a bug     # ok Ramez, thanks for the comment :)
    last_y = int(y_centers[0])
    c = 0
    intcenters = np.array(y_centers_sorted_by_y)
    intcenters = intcenters.astype(int)
    # this loop extracts the number of circles in each question row in unique_y_list
    for y in intcenters:
        diff = y - last_y
        if diff < 5:
            unique_y_list[c] = unique_y_list[c] + 1
        else:
            last_y = y
            c = c + 1
            unique_y_list[c] = unique_y_list[c] + 1


    unique_y_list = [y for y in unique_y_list if y > 1] # remove values less than 1 from the list

    gender = extract_gender(unique_y_list[0], x_centers_sorted_by_y)
    semester = extract_semester(unique_y_list, x_centers_sorted_by_y)
    program = extract_program(unique_y_list, x_centers_sorted_by_y)
    return gender, program, semester





def get_upper_circles(eroded_thresh, src, thresh, img_num): ## return black and mixed circles coorinates in a list # done
    height, width = thresh.shape
    eroded_thresh_upper = eroded_thresh[0:height // 4 - 30] # crop the image to remove the upper and lowwer noise
    thresh_upper = thresh[0:height // 4 - 30]
    #img_display = src[0:height // 4 - 30]
    black_circles_upper = cv2.HoughCircles(eroded_thresh_upper, cv2.HOUGH_GRADIENT, 1, 10, ## detect black circles in the upper region
                                           param1=10, param2=11,
                                           minRadius=9, maxRadius=15)
    mixed_circles_upper = cv2.HoughCircles(thresh_upper, cv2.HOUGH_GRADIENT, 1, 10, ## detect mixed circles in the upper region
                                           param1=10, param2=12,
                                           minRadius=11, maxRadius=15)
    all_circles_upper = []
    # add the circles to the list
    all_circles_upper.extend(black_circles_upper[0])
    all_circles_upper.extend(mixed_circles_upper[0])
    # debug_imshow(black_circles_upper, img_display, img_num, mixed_circles_upper)
    return all_circles_upper # return black and mixed circles coorinates in a list


def extract_question_answers(eroded_thresh, thresh): #pass the proccesed image before erison and after
    height, width = thresh.shape # returns a tuple of the number of rows and coloumns
    eroded_thresh_questions = eroded_thresh[height // 4:height, width // 2:width]
    # img_display = src[height // 4:height, width // 2:width]
    black_circles_questions = cv2.HoughCircles(eroded_thresh_questions, cv2.HOUGH_GRADIENT, 1, 10, #detect black circels only
                                               param1=10, param2=11,
                                               minRadius=1, maxRadius=15)
    thresh_questions = thresh[height // 4:height, width // 2:width]
    mixed_circles_questions = cv2.HoughCircles(thresh_questions, cv2.HOUGH_GRADIENT, 1, 10, #detect black and white circels
                                               param1=10, param2=15,
                                               minRadius=9, maxRadius=15)
    all_circles_questions = []
    all_circles_questions.extend(black_circles_questions[0])
    all_circles_questions.extend(mixed_circles_questions[0])

    centers_sorted_by_y = get_centers_y_sorted(all_circles_questions)  # list of y values of all circles sorted from 1.1 to 5.3
    questions_y_list = [centers_sorted_by_y[0]] # same list as above (to not change the values after the loop)
    last_y = centers_sorted_by_y[0]

    # iterate over the sorted y-vals to produce a list with uniquie y vals of each row(each question) and store the ans in qestuon_y_list
    for y in centers_sorted_by_y:
        y_diff = y - last_y
        if y_diff < ROW_DIFF:
            continue
        else:
            questions_y_list.append(y)
            last_y = y

    centers_sorted_by_x = get_centers_x_sorted(all_circles_questions) # x-vals stored in asc order
    max_x = centers_sorted_by_x[-1] # largest x- value
    min_x = centers_sorted_by_x[0] # smallest x-value
    span = (max_x - min_x) / 5 # the distance between answer bubbles in te sheet
    answers = []
    black_circles = black_circles_questions.tolist()[0]
    black_circles.sort(key=lambda circle: circle[1])   # sorted black circles list by y value (from first q to last)
    c = 0
    # iterate over each row, calc the distance beteen te black and the first buuble in the row, check if tthe black circle is on the same row as the q ,if yes then record the answer
    for y in questions_y_list:
        ans_num = (black_circles[c][0] - min_x) / span
        if abs(y - black_circles[c][1]) < 5: # if the answer is on te same line as the question
            answers.append(int(1 + math.floor(ans_num)))
            c = c + 1
        else:#  else the q is unaswwered
            answers.append("Unanswered")
    return answers


SEMESTERS = ["Fall", "Spring", "Summer"]
def extract_semester(unique_y_list, x_centers_sorted_by_y): #done
    semester = "Unanswered" # default answer
    if unique_y_list[1] == 4: # if there are 4 circles in the semester fields then there is an ans( 3 white + 1 black)
        val1 = unique_y_list[0] # number of circles in the gender section
        semster_xVals = x_centers_sorted_by_y[val1:val1 + 4] # to get the of circles in the semster section
        semster_xVals.sort(key=lambda circle: circle) # sort the circles in the semster section by the c

        duplicate_index = first_duplicate_index(semster_xVals) # index of first duplicate element in the list(duplicates values = black + white circle overlap)

        for i in range(len(semster_xVals) - 1):# useless?
            if abs(semster_xVals[i] - semster_xVals[i + 1]) < 5: # if 2 circles are very close, then equate their x vals
                semster_xVals[i] = semster_xVals[i + 1]

        semster_xVals.remove(semster_xVals[duplicate_index]) # useless?

        if duplicate_index is not None and duplicate_index < 3:
            semester = SEMESTERS[duplicate_index]

    return semester


def extract_gender(unique_y_list_0, x_centers_sorted_by_y): #done
    gender = "Unanswered" #default answer

    if unique_y_list_0 == 3:  # if there are 3 circles in the first line then the q is answered(1 black and 2 white)
        gender_xVals = x_centers_sorted_by_y[0:3] #takes the first 3 circles from the top(the gender circles)
        gender_xVals.sort()
        if abs(gender_xVals[1] - gender_xVals[2]) < 5: # if there are two circles on the right (1 black and 1 white)
            gender = "Female"
        else: # if there are two circles on the left
            gender = "Male"
    return gender


def extract_program(unique_y_list, x_centers_sorted_by_y):
    program = "Unanswered" #default answer

    if unique_y_list[2] + unique_y_list[3] == 12: # if there are exactly 12 circles in the program section(11 white + 1 black) else there are no or multiiple ans
        program_xVals = x_centers_sorted_by_y[unique_y_list[0] + unique_y_list[1]:]   # the circles in the program section


        program_xVals.sort(key=lambda circle: circle)


        if len(program_xVals) > 12:  # akbar men 12
            program = "Unanswered"
        else:
            right_ans_program_xVals = program_xVals[8:] # the 3 ans of the right side
            dup_index = first_duplicate_index(right_ans_program_xVals) # return the index of first duplicate elemnt in the list
            # print(dup_index)
            right_side_programs = ["ERGY", "COMM", "MANF"]
            if dup_index is not None and dup_index < 3:
                program = right_side_programs[dup_index]
            else: # continue if the ans is not in the right side
                # the following segment checks the position of the 3 circles sharing the same x value(2 white +1 black)
                input = program_xVals[0:9]
                n = 3
                prev = -1
                count = 0
                index = 0

                for idx, item in enumerate(input):
                    if abs(item - prev) < 10:
                        count = count + 1
                        index = idx
                    else:
                        count = 1

                    prev = item

                    left_side_upper_programs = ["LAAR", "MATL", "CISE", "HAUD"]
                    left_side_lower_programs = ["MCTA", "ENVER", "BLDG", "CESS"]
                    if count == n:
                        index = index - 2
                        # print("There are {} occurrences of {} in index {} in {} in sample {} ".format(n, item, index, input,img_num))
                        if unique_y_list[3] == 5 and unique_y_list[2] == 7:
                            program = left_side_upper_programs[index // 2]
                        elif unique_y_list[3] == 4 and unique_y_list[2] == 8:
                            program = left_side_lower_programs[index // 2]
                        else:
                            program = "Unanswered"
                        break
    return program


def rotate_img(src): # rotaes the image by 1- filtering the image by leaving only the red colored pixels in a spcefic range
    # 2- finding contrours with 4 vertex( rectange) in the filtered image
    # 3- sorting the contures based on area to obatin the  smallest rect
    # 4-obtain te idth height and angle of the rect
    # 5- rotate the image based on the detected angle
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_RED, UPPER_RED)
    _, thresh_red = cv2.threshold(mask, 127, 255, 0)
    contours, _ = cv2.findContours(thresh_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_rect = [c for c in contours if get_vertex_count(c) == 4]
    largest_square_contour = sorted(contours_rect, key=cv2.contourArea, reverse=True)[0]
    _, (width, height), angle = cv2.minAreaRect(largest_square_contour)
    # make the angle in the [0, 180) range *-ve
    if width < height:
        angle = angle - 90
    if 2 < abs(angle) < 178:
        angle = angle + 180 if abs(angle) < 5 else angle
        src = imutils.rotate(src, angle + 180)
    return src


def draw_circles_on_img(circles, img_display):  # for debugging
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, radius in circles[0, :]:
            #  print(f"x {x}, y {y}")
            center = (x, y)
            # circle center
            cv2.circle(img_display, center, 1, (0, 100, 100), 3)
            # circle outline
            cv2.circle(img_display, center, radius, (255, 0, 255), 3)


def debug_imshow(black_circles_upper, img_display, img_num, mixed_circles_upper):
    draw_circles_on_img(black_circles_upper, img_display)
    draw_circles_on_img(mixed_circles_upper, img_display)
    if img_num == 9:
        cv2.imshow(f"test_sample{img_num}.jpg", img_display)
        cv2.waitKey(0)


def parse_arguments():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--img_folder", default="",
                       help="usage: '-f path' please specify a folder path containing the images to be analyzed",
                       type=str)
    group.add_argument("-i", "--img_path", default="",
                       help="usage: '-i path' please specify a full image path to analyze it",
                       type=str)
    return ap


if __name__ == '__main__':
    arg_parser = parse_arguments()
    args = vars(arg_parser.parse_args())
    folder_path = args["img_folder"]
    img_path = args["img_path"]
    if img_path != "":
        write_answers_to_json(main(img_path)[1])
    else:
        answers_dicts = []
        for filename in os.listdir(folder_path):
            answers_dicts.append(main(f"{folder_path}/{filename}")[1])
        write_answers_to_json(answers_dicts)
