import math
import time

import cv2
import numpy as np
from mss import mss
import pyautogui

stop = False
previous_jump = "medium"
font = cv2.FONT_HERSHEY_SIMPLEX
thresh_1 = 100
thresh_2 = 200
x1_offset = 28
x2_offset = 61

y1_offset = 34
y2_offset = 62

point_x = 174
point_y = 363

last_dist_values = []
jump_coefficient = 140


def coefficient_callback(value):
    global jump_coefficient
    jump_coefficient = value


def point_x_callback(value):
    global point_x
    point_x = value


def point_y_callback(value):
    global point_y
    point_y = value


def thresh_1_callback(value):
    global thresh_1
    thresh_1 = value


def thresh_2_callback(value):
    global thresh_2
    thresh_2 = value


def callback_x1_offset(val):
    global x1_offset
    x1_offset = val


def callback_x2_offset(val):
    global x2_offset
    x2_offset = val


def callback_y1_offset(val):
    global y1_offset
    y1_offset = val


def callback_y2_offset(val):
    global y2_offset
    y2_offset = val


def process_frame(img, debug_img=False):
    x = img.shape[0]
    y = img.shape[1]
    # cutting unnecessary parts of the frame
    x_1 = int(x * (x1_offset / 100))
    x_2 = int(x * (x2_offset / 100))
    y_1 = int(y * (y1_offset / 100))
    y_2 = int(y * (y2_offset / 100))
    img = img[x_1:x_2, y_1:y_2]
    width = img.shape[1]
    half_width = int(width / 2)

    if len(img) == 0:
        return img, img
    # passing frame through filters
    img = cv2.medianBlur(img, 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(img, thresh_1, thresh_2)

    # finding both points
    upper_point = find_upper_point(canny)
    lower_point = find_lower_point(upper_point, width)

    # drawing
    draw_point(lower_point, gray)
    draw_point(upper_point, gray)
    cv2.line(gray, (half_width, 0), (half_width, img.shape[0]), (0, 255, 0), 3)

    # calculating distance and checking if a jump could be performed
    global last_dist_values
    dist = math.hypot(upper_point[0] - lower_point[0], upper_point[1] - upper_point[1])
    last_dist_values.append(dist)
    # jump only if last 15 measures are the same, means all the points are stable
    if len(last_dist_values) > 15 and all_equal(last_dist_values[-15:]):
        last_dist_values = []  # cleaning distance cache to prevent jumping on the next frame
        jump(dist)

    if debug_img:
        cv2.imwrite('output/output_gray.jpg', gray)
        cv2.imwrite('output/output_canny.jpg', canny)

    return gray, canny


def jump(distance):
    # calculates delay of the button release
    # depends on previous jump

    global previous_jump
    if 380 > distance > 310:
        print("medium jump, distance= " + str(distance))
        if previous_jump == "medium":
            jump_correction = -5
        else:
            jump_correction = 5
        delay = (distance / 1000) * ((jump_coefficient + jump_correction) / 100)
        previous_jump = "medium"
    elif distance >= 380:
        print("large jump, distance= " + str(distance))
        delay = (distance / 1000) * ((jump_coefficient - 10) / 100)
        previous_jump = "large"
    elif 270 < distance <= 310:
        print("small jump, distance= " + str(distance))
        if previous_jump == "small":
            jump_correction = -5
        else:
            jump_correction = -20

        delay = (distance / 1000) * ((jump_coefficient + jump_correction) / 100)
        previous_jump = "small"
    else:
        print("extra-small jump, distance= " + str(distance))
        jump_correction = -15

        delay = (distance / 1000) * ((jump_coefficient + jump_correction) / 100)
        previous_jump = "extra-small"
    pyautogui.mouseDown(x=900, y=100)
    start = time.time()
    while True:
        # time.sleep() is not accurate when it comes to milliseconds,
        # so that's the best way
        now = time.time()
        if now - start > delay:
            break
    pyautogui.mouseUp(x=900, y=100)


def draw_point(point, img):
    cv2.circle(img, point, radius=5, color=(0, 0, 255), thickness=-1)
    cv2.putText(img, '  ' + 'w: ' + str(point[0]) + ' ' + 'h: ' + str(point[1]), (point[0] - 150, point[1] + 50), font,
                1, (255, 255, 255), 2,
                cv2.LINE_AA)


def find_lower_point(upper_point, width):
    # this code finds the position of the lower point
    # basically just calculates the position of upper one
    # and set the position of lower one on the other side
    # some additional checks are performed to make sure
    # it's not a fluke

    half_width = int(width / 2)
    right_lower_point = width - point_x, point_y
    left_lower_point = point_x, point_y
    right_shift_check = upper_point[0] < half_width + 30
    left_shift_check = upper_point[0] < half_width - 30
    center_check = upper_point[0] < half_width

    if center_check:
        if right_shift_check:
            return right_lower_point
        else:
            return left_lower_point
    else:
        if left_shift_check:
            return right_lower_point
        else:
            return left_lower_point


def find_upper_point(canny):
    # this is a frame:
    #
    #    1 2 3 4 5 6 7
    #  ---------------
    # 1| 0 0 0 0 0 0 0
    # 2| 0 0 0 0 0 0 0
    # 3| 0 0 1 1 1 1 1
    # 4| 0 0 1 1 1 1 1
    # 5| 0 0 1 1 1 1 1
    # 6| 0 0 1 1 1 1 1
    #
    # code below finds element number [3][3]
    i = 0
    for row in canny:
        j = np.nonzero(row)[0]
        if len(j) != 0:
            return j[0], i
        else:
            i = i + 1


def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)


def process_window(debug=False):
    sct = mss()
    bounding_box = sct.monitors[1]
    source_window = 'Controls'
    cv2.namedWindow(source_window, cv2.WINDOW_NORMAL)

    # canny filter params
    cv2.createTrackbar('Canny Thresh Lower:', source_window, thresh_1, 225, thresh_1_callback)
    cv2.createTrackbar('Canny Thresh Upper:', source_window, thresh_2, 450, thresh_2_callback)

    # window capture sizes
    # x1/y1 must be always less than x2/y2 otherwise crash will happen
    cv2.createTrackbar('Window X1 Offset:', source_window, x1_offset, 100, callback_x1_offset)
    cv2.createTrackbar('Window X2 Offset:', source_window, x2_offset, 100, callback_x2_offset)
    cv2.createTrackbar('Window Y1 Offset:', source_window, y1_offset, 100, callback_y1_offset)
    cv2.createTrackbar('Window Y2 Offset:', source_window, y2_offset, 100, callback_y2_offset)

    # default place of the lower point
    cv2.createTrackbar('Lower point X:', source_window, point_x, 1000, point_x_callback)
    cv2.createTrackbar('Lower point Y:', source_window, point_y, 1000, point_y_callback)

    # coefficient of jump power
    cv2.createTrackbar('jump_coefficient:', source_window, jump_coefficient, 200, coefficient_callback)
    while True:
        img = sct.grab(bounding_box)  # grabbing image from monitor 1
        img = np.array(img)

        frames = process_frame(img)
        if debug:
            stack = np.vstack(frames)  # combine 2 frames vertical for debug reasons
            cv2.imshow("Result", stack)

        if cv2.waitKey(1) & 0xFF == ord('q') or stop:
            print("stopped")
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # debug=False disables video output
    process_window(debug=True)
