import numpy as np
import os
import csv
import collections
from numpy import array
#Set the date of the recording
date = r"\2017_11_25"
#Set the recording no
recording_no = r"\005"
path = r"C:\Users\SSRL\Documents\VRDepthTest\2017_11_24\010\exports"
#distance_path = r"C:\Users\SSRL\Documents\VRDepthTest" + date + recording_no
new_path = os.path.join(path, os.listdir(path).pop())
data = np.genfromtxt(new_path  + "\gaze_positions.csv", delimiter=',',
                      skip_header=1,names = ['timestamp','index','confidence','norm_pos_x','norm_pos_y','base_data'], dtype=(float, int, float,float,float,"|S50"))
#distance = np.genfromtxt(distance_path  + "\distance.npy", delimiter=',',
#                      skip_header=1,names = ['Distance'], dtype=(float))
right_eye = {}
left_eye = {}
diff_list = {}
distance_ball = {}
alternate = False
#Iterate over the data to filter based on confidence value
print(max(data['timestamp']))
for p in data:
    if(p['confidence'] >= 0.70):
        alternate = (not alternate)
        #Split 'base_data' string using '-' character to get the eye_index
        eye_data = p['base_data'].decode().split(" ")
        if(len(eye_data) == 2):
            eye_data_l = eye_data[0]
            eye_data_r = eye_data[1]
            if(alternate == True):
                eye_index = int(eye_data_l.split("-")[1])
            else:
                eye_index = int(eye_data_r.split("-")[1])
            #print((p['timestamp']*1000).astype(int))
            if(eye_index == 0):
                left_eye[(p['timestamp']*1000).astype(int)] = p
            if (eye_index == 1):
                right_eye[(p['timestamp']*1000).astype(int)] = p
#print("value",left_eye[14252579])
# print("Total recorded data:",len(data))
# print("Total valid data(confidence >0.70):", len(right_eye)+len(left_eye))
# print("Valid data - Right eye",len(right_eye))
# print("Valid data - Left eye",len(left_eye))

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
        conv_point = 65 * 500/(65 + (1920 *diff_xpos))

        #distance_ball[key] = distance['Distance'][right_eye[key]['index']]
        diff_list[key] = conv_point
#foundkey =matchkey(14252625)
od = collections.OrderedDict(sorted(diff_list.items()))

with open(new_path  + "\convergence_points.csv", 'w',newline="\n", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    for key, value in od.items():
       writer.writerow([key, value])
print("Computed convergence points written to csv file")