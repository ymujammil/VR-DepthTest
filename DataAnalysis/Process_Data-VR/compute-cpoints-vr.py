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
    subject_id = input("Set the subject ID: ")
    exp_type = input("Set the experiment type: ")
    int_pup_distance = 0
    while True:
        try:
            int_pup_distance= int(input("Set the measured Interpupillary distance (mm): "))
        except ValueError:
            print("\t\t\tNot a valid input. Please enter an integer value")
            continue
        else:
            break
    for p in gaze_data:
        if(p['confidence'] >= 0.70):
            #Split 'base_data' string using '-' character to get the eye_index
            eye_index = int(p['base_data'].decode().split("-")[1])
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
            conv_point = int_pup_distance * 18/(int_pup_distance + (99.5 *diff_xpos))

            distance_ball[key] = distance_data['Distance'][right_eye_convpt[key]['index']]
            diff_list[key] = conv_point
    od = collections.OrderedDict(sorted(diff_list.items()))

    path_to_convpt_file = os.path.join(global_path + r"\exports", os.listdir(global_path + r"\exports").pop()) + r"\output" + "\conv_points_" + exp_type + "_" +subject_id + "_VR.csv"
    path_to_plot_file = os.path.join(global_path + r"\exports", os.listdir(global_path + r"\exports").pop()) + r"\output" + "\plot_" + exp_type + "_" +subject_id + "_VR.png"
    if not os.path.exists(os.path.dirname(path_to_convpt_file)):
        try:
            os.makedirs(os.path.dirname(path_to_convpt_file))
        except OSError as exc:  # Guard against race condition
                raise
    with open(path_to_convpt_file, 'w',newline="\n", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["time","convergence_point","ball_distance"])
        for key, value in od.items():
           writer.writerow([time_in_seconds[key], value, distance_ball[key]])
    print("\t\t\tConvergence points saved to " + path_to_convpt_file)
except PermissionError:
    print("Error:")
    print("Oops! Cound not write output to file...")
    print("Is the file open in the system? Please close the file and try again")
    exit()

# Create a plot with the output of "convergence_points.csv"
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(path_to_convpt_file)

# Create a figure
plt.figure(figsize=(10,8))

if(exp_type == "static"):
    # Create a scatterplot of,
                # time with ball distance 0.5m as the x axis
    plt.scatter(df['time'][df['ball_distance'] == 0.5],
                # convergence_point with ball distance 0.5m as the y axis
                df['convergence_point'][df['ball_distance'] == 0.5],
                # the marker as
                marker='x',
                # the color
                color='b',
                # the alpha
                alpha=0.7,
                # with size
                s = 124,
                # labelled this
                label='Distance 0.5m')
    # Repeat For Distance 1m
    plt.scatter(df['time'][df['ball_distance'] == 1],
                df['convergence_point'][df['ball_distance'] == 1],
                marker='o',
                color='r',
                alpha=0.7,
                s = 124,
                label='Distance 1')
    # Repeat For Distance 1.5m
    plt.scatter(df['time'][df['ball_distance'] == 1.5],
                df['convergence_point'][df['ball_distance'] == 1.5],
                marker='+',
                color='y',
                alpha=0.7,
                s = 124,
                label='Distance 1.5')
    # Repeat For Distance 2m
    plt.scatter(df['time'][df['ball_distance'] == 2],
                df['convergence_point'][df['ball_distance'] == 2],
                marker='*',
                color='c',
                alpha=0.7,
                s = 124,
                label='Distance 2')
    # Repeat For Distance 3m
    plt.scatter(df['time'][df['ball_distance'] == 3],
                df['convergence_point'][df['ball_distance'] == 3],
                marker='s',
                color='m',
                alpha=0.7,
                s = 124,
                label='Distance 3')
    # Repeat For Distance 4m
    plt.scatter(df['time'][df['ball_distance'] == 4],
                df['convergence_point'][df['ball_distance'] == 4],
                marker='v',
                color='k',
                alpha=0.7,
                s = 124,
                label='Distance 4')
    # Repeat For Distance 6m
    plt.scatter(df['time'][df['ball_distance'] == 6],
                df['convergence_point'][df['ball_distance'] == 6],
                marker='>',
                color='r',
                alpha=0.7,
                s = 124,
                label='Distance 6')
    # Repeat For Distance 8m
    plt.scatter(df['time'][df['ball_distance'] == 8],
                df['convergence_point'][df['ball_distance'] == 8],
                marker='^',
                color='g',
                alpha=0.7,
                s = 124,
                label='Distance 8')
    # Chart title
    plt.title('Convergence Point Vs Distance Vs Time (Static-VR)')
    # x label
    plt.xlabel('Time(sec)')
    # set the figure boundaries x
    plt.xlim([min(df['time']), max(df['time'])])
    # set the figure boundaries y
    plt.ylim([min(df['convergence_point']), max(df['convergence_point'])])
else:
    # Create a scatterplot of,
                # ball_distance as the x axis
    plt.scatter(df['ball_distance'],
                # convergence_point as the y axis
                df['convergence_point'],
                # the marker as
                marker='o',
                # the color
                color='b',
                # the alpha
                alpha=0.7,
                # with size
                s = 124)
    # Chart title
    plt.title('Convergence Point Vs Distance (Dynamic-VR)')
    # x label
    plt.xlabel('Distance(m)')
    # set the figure boundaries x
    plt.xlim([min(df['ball_distance']), max(df['ball_distance'])])
    # set the figure boundaries y
    plt.ylim([15,21])



# y label
plt.ylabel('Convergence Point(mm)')

# and a legend
plt.legend(loc='upper right')

# Save the output plot
plt.savefig(path_to_plot_file)
print("\t\t\tScatter plot saved to " + path_to_plot_file)
print("Successfully processed!\n")

