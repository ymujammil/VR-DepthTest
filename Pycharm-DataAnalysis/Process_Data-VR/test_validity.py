import numpy as np
import os
from numpy import array
right_eye = []
left_eye = []
#Set the date of the recording
date = r"\2017_11_28"
#Set the recording no
recording_no = r"\003"
path = r"C:\Users\SSRL\Documents\VRDepthTest" + date + recording_no + r"\exports"
new_path = os.path.join(path, os.listdir(path).pop()) + "\gaze_positions.csv"
data = np.genfromtxt(new_path, delimiter=',',
                      skip_header=1,names = ['timestamp','index','confidence','norm_pos_x','norm_pos_y','base_data'], dtype=(float, int, float,float,float,"|S50"))
#Iterate over the data to filter based on confidence value

for p in data:
    if(p['confidence'] >= 0.70):
        #Split 'base_data' string using '-' character to get the eye_index
        eye_index = int(p['base_data'].decode().split("-")[1])
        if(eye_index == 0):
            left_eye.append(p)
        if (eye_index == 1):
            right_eye.append(p)

print("Total recorded data:",len(data))
print("Total valid data(confidence >0.70):", len(right_eye)+len(left_eye))
print("Valid data-percentage:", (len(right_eye)+len(left_eye))/len(data) * 100)
print("Valid data - Right eye",len(right_eye))
print("Valid data - Left eye",len(left_eye))