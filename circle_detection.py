import cv2
import numpy as np
import math
import imutils
import json
import argparse
import os

ROW_DIFF = np.uint16(15)
LOWER_RED = np.array([160, 50, 50])
UPPER_RED = np.array([180, 255, 255])
img_num = 0


def main(img_path):
    global img_num
    img_num += 0
    src = cv2.imread(f"{img_path}", 1)
    src = rotate_img(src)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    circle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    eroded_thresh = cv2.erode(thresh, circle_kernel)

    answers = extract_question_answers(eroded_thresh, thresh)
    gender, program, semester = extract_upper_answers(eroded_thresh, img_num, src, thresh)
    answers.extend([gender, semester, program])
    answers_dict = get_ans_dict_from_list(answers, gender, img_num, program, semester)
    return answers, answers_dict


def write_answers_to_json(answers_dict):
    with open('Output.json', 'w') as outfile:
        json.dump(answers_dict, outfile, indent=4)


def get_ans_dict_from_list(answers, gender, img_num, program, semester):
    questions_num_per_set = [5, 6, 3, 3, 2]
    answers_dict = {"test_sample": img_num,
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


def get_vertex_count(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    return len(approx)


def get_circle_centers(circles):
    if circles is not None:
        return [(center[0], center[1]) for center in circles]


def get_centers_x_sorted(circles):
    if circles is not None:
        return sorted([center[0] for center in circles])


def get_centers_y_sorted(circles):
    if circles is not None:
        return sorted([center[1] for center in circles])


def first_duplicate_index(a):
    difference_between_elements = np.diff(a)
    for i, element_diff in enumerate(difference_between_elements):
        if abs(element_diff) < 5:
            return i
    return None


def extract_upper_answers(eroded_thresh, img_num, src, thresh):
    all_circles_upper = get_upper_circles(eroded_thresh, src, thresh, img_num)
    x_centers_sorted_by_y, y_centers_sorted_by_y = zip(
        *sorted(get_circle_centers(all_circles_upper), key=lambda center: center[1]))
    x_centers_sorted_by_y, y_centers_sorted_by_y = list(x_centers_sorted_by_y), list(y_centers_sorted_by_y)
    y_centers = []
    last_y = 0
    # this loop extracts the y-vals of the different ans sets
    for y in y_centers_sorted_by_y:
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
    intcenters = np.array(y_centers_sorted_by_y)
    intcenters = intcenters.astype(int)
    # this loop extracts the number of circles in each set of ans
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
    gender = extract_gender(unique_y_list[0], x_centers_sorted_by_y)
    semester = extract_semester(unique_y_list, x_centers_sorted_by_y)
    program = extract_program(unique_y_list, x_centers_sorted_by_y)
    return gender, program, semester


def get_upper_circles(eroded_thresh, src, thresh, img_num):
    height, width = thresh.shape
    eroded_thresh_upper = eroded_thresh[0:height // 4 - 30]
    thresh_upper = thresh[0:height // 4 - 30]
    img_display = src[0:height // 4 - 30]
    black_circles_upper = cv2.HoughCircles(eroded_thresh_upper, cv2.HOUGH_GRADIENT, 1, 10,
                                           param1=10, param2=11,
                                           minRadius=9, maxRadius=15)
    mixed_circles_upper = cv2.HoughCircles(thresh_upper, cv2.HOUGH_GRADIENT, 1, 10,
                                           param1=10, param2=12,
                                           minRadius=11, maxRadius=15)
    all_circles_upper = []
    all_circles_upper.extend(black_circles_upper[0])
    all_circles_upper.extend(mixed_circles_upper[0])
    # debug_imshow(black_circles_upper, img_display, img_num, mixed_circles_upper)
    return all_circles_upper


def extract_question_answers(eroded_thresh, thresh):
    height, width = thresh.shape
    eroded_thresh_questions = eroded_thresh[height // 4:height, width // 2:width]
    # img_display = src[height // 4:height, width // 2:width]
    black_circles_questions = cv2.HoughCircles(eroded_thresh_questions, cv2.HOUGH_GRADIENT, 1, 10,
                                               param1=10, param2=11,
                                               minRadius=1, maxRadius=15)
    thresh_questions = thresh[height // 4:height, width // 2:width]
    mixed_circles_questions = cv2.HoughCircles(thresh_questions, cv2.HOUGH_GRADIENT, 1, 10,
                                               param1=10, param2=15,
                                               minRadius=9, maxRadius=15)
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
    black_circles = black_circles_questions.tolist()[0]
    black_circles.sort(key=lambda circle: circle[1])
    c = 0
    for y in questions_y_list:
        ans_num = (black_circles[c][0] - min_x) / span
        if abs(y - black_circles[c][1]) < 5:
            answers.append(int(1 + math.floor(ans_num)))
            c = c + 1
        else:
            answers.append("Unanswered")
    return answers


SEMESTERS = ["Fall", "Spring", "Summer"]


def extract_semester(unique_y_list, x_centers):
    semester = "Unanswered"
    if unique_y_list[1] == 4:
        val1 = unique_y_list[0]
        semster_xVals = x_centers[val1:val1 + 4]
        semster_xVals.sort(key=lambda circle: circle)
        duplicate_index = first_duplicate_index(semster_xVals)
        # print(duplicate_index)
        for i in range(len(semster_xVals) - 1):
            # print(abs(semster_xVals[i] - semster_xVals[i+1]) <5)
            if abs(semster_xVals[i] - semster_xVals[i + 1]) < 5:
                semster_xVals[i] = semster_xVals[i + 1]
        #  print(semster_xVals)
        semster_xVals.remove(semster_xVals[duplicate_index])
        # print(semster_xVals)
        if duplicate_index is not None and duplicate_index < 3:
            semester = SEMESTERS[duplicate_index]
    return semester


def extract_gender(unique_y_list_0, x_centers):
    gender = "Unanswered"
    if unique_y_list_0 == 3:  # if there are 3 circle in the first line then the q is answered
        gender_xVals = x_centers[0:3]
        gender_xVals.sort()
        if abs(gender_xVals[1] - gender_xVals[2]) < 5:
            gender = "Female"
        else:
            gender = "Male"
    return gender


def extract_program(unique_y_list, x_centers):
    program = "Unanswered"
    if unique_y_list[2] + unique_y_list[3] == 12:
        program_xVals = x_centers[unique_y_list[0] + unique_y_list[1]:]  # -1 hena removed(solved bug in program x vals)

        xtest = x_centers
        xtest.sort(key=lambda circle: circle)
        # print(xtest)
        program_xVals.sort(key=lambda circle: circle)
        # print(program_xVals)

        if len(program_xVals) > 12:  # akbar men 12 and not between 456 and 1256???
            program = "Unanswered"
        else:
            right_ans_program_xVals = program_xVals[8:]
            dup_index = first_duplicate_index(right_ans_program_xVals)
            # print(dup_index)
            right_side_programs = ["ERGY", "COMM", "MANF"]
            if dup_index is not None and dup_index < 3:
                program = right_side_programs[dup_index]
            else:
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


def rotate_img(src):
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
    group = ap.add_mutually_exclusive_group(required=False)
    group.add_argument("-f", "--img_folder", default="tests",
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
