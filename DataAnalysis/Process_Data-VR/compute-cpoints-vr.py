import csv
import numpy as np
import string
import sys
from numpy import genfromtxt
# Set the global path for loading the input
global_path = r"C:\Users\SSRL\Desktop\VRDepthTest\Testbed\Testbed-VR"
try:
    print("1:Converting timestamps format ....\n")
    input_data = genfromtxt(global_path + '/world_timestamps.npy', delimiter=',')
    # Convert to numpy array after deleting the first value(Which is a test string)
    output_data = np.delete(input_data, [0])
    # Save output on to the same folder(or replace)
    np.save(global_path + '/world_timestamps', output_data)
    print("Successfully converted the timestamps. Ready for processing by 'pupil-player' \n")
except ValueError:
    # handle ValueError exception
    print("TimeStamps already converted. Skipping to next step...\n")
except OSError:
    print("Error:")
    print("No Input file found for processing. Is the folder empty?\n")
    exit()


import os
right_eye = []
left_eye = []

try:
    print("2:Testing Data Validity ....\n")
    path_to_gaze_file = os.path.join(global_path + r"\exports", os.listdir(global_path + r"\exports").pop()) + "\gaze_positions.csv"
    gaze_data = np.genfromtxt(path_to_gaze_file, delimiter=',',
                              skip_header=1,
                              names=['timestamp', 'index', 'confidence', 'norm_pos_x', 'norm_pos_y', 'base_data'],
                              dtype=(float, int, float, float, float, "|S50"))
    # Iterate over the data to filter based on confidence value
    for p in gaze_data:
        if (p['confidence'] >= 0.70):
            # Split 'base_data' string using '-' character to get the eye_index
            eye_index = int(p['base_data'].decode().split("-")[1])
            if (eye_index == 0):
                left_eye.append(p)
            if (eye_index == 1):
                right_eye.append(p)
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
right_eye_convpt = {}
left_eye_convpt = {}
diff_list = {}
distance_ball = {}
time_in_seconds = {}

print("3:Computing convergence points ....\n")
try:
    distance_data = np.genfromtxt(global_path  + "\distance.npy", delimiter=',',
                          skip_header=1,names = ['Distance'], dtype=(float))
    start_time = gaze_data['timestamp'][0]
    for p in gaze_data:
        if(p['confidence'] >= 0.70):
            #Split 'base_data' string using '-' character to get the eye_index
            eye_index = int(p['base_data'].decode().split("-")[1])
            #print((p['timestamp']*1000).astype(int))
            if(eye_index == 0):
                left_eye_convpt[(p['timestamp']*1000).astype(int)] = p
            if (eye_index == 1):
                right_eye_convpt[(p['timestamp']*1000).astype(int)] = p

    def match_record(key):
        if(key in left_eye_convpt):
            return key
        elif(key+1 in left_eye_convpt):
            return key+1
        elif(key-1 in left_eye_convpt):
            return key-1
        elif(key+2 in left_eye_convpt):
            return key+2
        elif(key-2 in left_eye_convpt):
            return key-2
        elif(key+3 in left_eye_convpt):
            return key+3
        elif(key-3 in left_eye_convpt):
            return key-3
        elif(key+4 in left_eye_convpt):
            return key+4
        elif(key-4 in left_eye_convpt):
            return key-4
        elif(key+5 in left_eye_convpt):
            return key+5
        elif(key-5 in left_eye_convpt):
            return key-5
        else:
            return 0

    for key,value in right_eye_convpt.items():
        result = match_record(key)
        if(result != 0):
            xpos_leye=left_eye_convpt[result]['norm_pos_x']
            xpos_reye=right_eye_convpt[key]['norm_pos_x']
            diff_xpos = xpos_leye - xpos_reye
            time_in_seconds[key] = (key / 1000 )- start_time

            # Convergence point calculation (E * D) / (E + M)
            conv_point = 65 * 18/(65 + (3.15 *diff_xpos))

            distance_ball[key] = distance_data['Distance'][right_eye_convpt[key]['index']]
            diff_list[key] = conv_point
    od = collections.OrderedDict(sorted(diff_list.items()))

    path_to_convpt_file = os.path.join(global_path + r"\exports", os.listdir(global_path + r"\exports").pop()) + "\convergence_points.csv"
    with open(path_to_convpt_file, 'w',newline="\n", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in od.items():
           writer.writerow([time_in_seconds[key], value, distance_ball[key]])
    print("Successfully computed Convergence points, 'convergence_points.csv' file written to " + path_to_convpt_file)
except PermissionError:
    print("Error:")
    print("Oops! Cound not write to convergence_points.csv...")
    print("Is the file open in the system? Please close the file and try again")
    exit()


