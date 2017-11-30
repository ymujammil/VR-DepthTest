import numpy as np
import os
from numpy import array
right_eye = []
left_eye = []
alternate = False
#Set the date of the recording
date = r"\2017_11_24"
#Set the recording no
recording_no = r"\000"
path = r"C:\Users\SSRL\Documents\VRDepthTest\2017_11_24\009\exports"
new_path = os.path.join(path, os.listdir(path).pop()) + "\gaze_postions.csv"
data = np.genfromtxt(new_path, delimiter=',',
                      skip_header=1,names = ['timestamp','index','confidence','norm_pos_x','norm_pos_y','base_data'], dtype=(float, int, float,float,float,"|S50"))

for p in data:
    if(p['confidence'] >= 0.70):
        alternate = (not alternate)
        #Split 'base_data' string using '-' character to get the eye_index
        eye_data = p['base_data'].decode().split(" ")
        eye_data_l = eye_data[0]
        eye_data_r = eye_data[1]
        if(alternate == True):
            eye_index = int(eye_data_l.split("-")[1])
        else:
            eye_index = int(eye_data_r.split("-")[1])

        if(eye_index == 0):
            left_eye.append(p)
        if (eye_index == 1):
            right_eye.append(p)

print("Total recorded data:",len(data))
print("Total valid data(confidence >0.70):", len(right_eye)+len(left_eye))
print("Valid data-percentage:", (len(right_eye)+len(left_eye))/len(data) * 100)
print("Valid data - Right eye",len(right_eye))
print("Valid data - Left eye",len(left_eye))