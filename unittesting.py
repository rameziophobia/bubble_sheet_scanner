import unittest
from circle_detection import main

test_ans = [[4, 1, 4, 2, 1, 2, 5, 4, 2, 4, 2, 2, 1, 4, 3, 1, 3, 1, 3, "Female", "Fall", "ERGY"],  # 1
            [3, 1, 3, 1, 2, 4, 4, 4, 2, 2, 1, 3, 1, 2, 1, 4, 1, 3, 2, "Male", "Summer", "MANF"],  # 2
            [2, 1, 3, 4, 2, 4, 4, 4, 2, 2, 1, 2, 1, 3, 1, 4, 3, 1, 2, "Male", "Summer", "HAUD"],  # 3
            [1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 2, 2, 2, 4, 4, 4, 2, 2, "Male", "Fall", "MATL"],  # 4
            [1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 2, 1, 2, 3, 4, 5, 4, 2, 1, "Male", "Fall", "ENVER"],  # 5
            [2, 3, 3, 2, 5, 5, 4, 1, 2, 1, 1, 1, 2, 4, 4, 1, 2, 2, 2, "Female", "Fall", "BLDG"],  # 6
            [1, 3, 4, 1, 3, 2, 4, 3, 2, 3, 4, 5, 1, 5, 3, 1, 4, 1, 3, "Female", "Fall", "BLDG"],  # 7
            [4, 3, 4, 2, 3, 1, 5, 4, 1, 4, 2, 2, 1, 3, 3, 2, 3, 1, 2, "Female", "Fall", "BLDG"],  # 8 # rotation
            [1, 1, 2, 1, 2, 4, 3, 4, 3, 2, 2, 3, 3, 3, 2, 3, 2, 3, 1, "Male", "Spring", "COMM"],  # 9 # rotation
            [5, 1, 4, 2, 4, 2, 4, 1, 3, 2, 3, 3, 2, 1, 4, 3, 1, 4, 1, "Female", "Summer", "ERGY"],  # 10
            [4, 1, 4, 2, 1, 2, 5, 4, 'Unanswered', 4, 2, 2, 1, 4, 3, 1, 3, 1, 3, "Female", "Fall", "ERGY"],  # 11
            # added test 12, same as test 1, but removed gender
            [4, 1, 4, 2, 1, 2, 5, 4, 2, 4, 2, 2, 1, 4, 3, 1, 3, 1, 3, "Unanswered", "Fall", "ERGY"],
            # test 13, same as test 2, but removed semester
            [3, 1, 3, 1, 2, 4, 4, 4, 2, 2, 1, 3, 1, 2, 1, 4, 1, 3, 2, "Male", "Unanswered", "MANF"],
            # test 14, same as test 9 but removed gender and semester
            [1, 1, 2, 1, 2, 4, 3, 4, 3, 2, 2, 3, 3, 3, 2, 3, 2, 3, 1, "Unanswered", "Unanswered", "COMM"],
            [4, 1, 4, 2, 1, 2, 5, 4, 2, 4, 2, 2, 1, 4, 3, 1, 3, 1, 3, "Female", "Unanswered", "ERGY"],  # multiple sem
            [4, 1, 4, 2, 1, 2, 5, 4, 2, 4, 2, 2, 1, 4, 3, 1, 3, 1, 3, "Female", "Unanswered", "ERGY"],  # multiple sem
            [4, 1, 4, 2, 1, 2, 5, 4, 2, 4, 2, 2, 1, 4, 3, 1, 3, 1, 3, "Unanswered", "Unanswered", "Unanswered"],
            # multiple gender and semesters
            [4, 3, 4, 2, 3, 1, 5, 4, 1, 4, 2, 2, 1, 3, 3, 2, 3, 1, 2, "Female", "Unanswered",
             "Unanswered"]]  # multiple programs


class TestAnswers(unittest.TestCase):
    def tests(self):
        for i in range(1, 12 + 7):  # added 3+3+1 more tests, and guess whatttttt!!!. they all worked fine :).
            with self.subTest(i=i):
                self.assertEqual(main(f"tests/test_sample{i}.jpg")[0], test_ans[i - 1])
