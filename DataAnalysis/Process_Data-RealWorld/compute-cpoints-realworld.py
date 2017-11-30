import csv
import numpy as np
import string
import sys
from numpy import genfromtxt
# Set the global path for loading the input
global_path = r"C:\Users\SSRL\Desktop\VRDepthTest\Testbed\Testbed-RealWorld"

import os
right_eye = {}
left_eye = {}
alternate = False

try:
    print("1:Testing Data Validity ....\n")
    path_to_gaze_file = os.path.join(global_path + r"\exports", os.listdir(global_path + r"\exports").pop()) + "\gaze_positions.csv"
    gaze_data = np.genfromtxt(path_to_gaze_file, delimiter=',',
                              skip_header=1,
                              names=['timestamp', 'index', 'confidence', 'norm_pos_x', 'norm_pos_y', 'base_data'],
                              dtype=(float, int, float, float, float, "|S50"))
    # Iterate over the data to filter based on confidence value
    for p in gaze_data:
        if (p['confidence'] >= 0.70):
            alternate = (not alternate)
            # Split 'base_data' string using ' ' character to get the eye_index
            eye_data = p['base_data'].decode().split(" ")
            if (len(eye_data) == 2):
                eye_data_l = eye_data[0]
                eye_data_r = eye_data[1]
                if (alternate == True):
                    eye_index = int(eye_data_l.split("-")[1])
                else:
                    eye_index = int(eye_data_r.split("-")[1])
                # print((p['timestamp']*1000).astype(int))
                if (eye_index == 0):
                    left_eye[(p['timestamp'] * 1000).astype(int)] = p
                if (eye_index == 1):
                    right_eye[(p['timestamp'] * 1000).astype(int)] = p
    print("Total recorded data:", len(gaze_data))
    print("Total valid data(confidence >0.70):", len(right_eye) + len(left_eye))
    print("Valid data-percentage:", (len(right_eye) + len(left_eye)) / len(gaze_data) * 100)
    print("Valid data - Right eye", len(right_eye))
    print("Valid data - Left eye", len(left_eye))
    print()
except FileNotFoundError:
    print("Error:")
    print("Oops! Cound not test data validity...")
    print("No 'exports' folder found. Did you process the recording with 'Pupil-Player'?")
    exit()

import collections
diff_list = {}
distance_ball = {}
time_in_seconds = {}

print("2:Computing convergence points ....\n")
try:
    start_time = gaze_data['timestamp'][0]

    def match_record(key):
        if(key in left_eye):
            return key
        elif(key+1 in left_eye):
            return key+1
        elif(key-1 in left_eye):
            return key-1
        elif(key+2 in left_eye):
            return key+2
        elif(key-2 in left_eye):
            return key-2
        elif(key+3 in left_eye):
            return key+3
        elif(key-3 in left_eye):
            return key-3
        elif(key+4 in left_eye):
            return key+4
        elif(key-4 in left_eye):
            return key-4
        elif(key+5 in left_eye):
            return key+5
        elif(key-5 in left_eye):
            return key-5
        else:
            return 0

    for key,value in right_eye.items():
        result = match_record(key)
        if(result != 0):
            xpos_leye=left_eye[result]['norm_pos_x']
            xpos_reye=right_eye[key]['norm_pos_x']
            diff_xpos = xpos_leye - xpos_reye
            time_in_seconds[key] = (key / 1000 )- start_time

            # Convergence point calculation (E * D) / (E + M)
            conv_point = 65 * 500/(65 + (600 *diff_xpos))
            diff_list[key] = conv_point
    od = collections.OrderedDict(sorted(diff_list.items()))

    path_to_convpt_file = os.path.join(global_path + r"\exports", os.listdir(global_path + r"\exports").pop()) + "\convergence_points.csv"
    with open(path_to_convpt_file, 'w',newline="\n", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in od.items():
           writer.writerow([time_in_seconds[key], value])
    print("Successfully computed Convergence points, 'convergence_points.csv' file written to " + path_to_convpt_file)
except PermissionError:
    print("Error:")
    print("Oops! Cound not write to convergence_points.csv...")
    print("Is the file open in the system? Please close the file and try again")
    exit()


