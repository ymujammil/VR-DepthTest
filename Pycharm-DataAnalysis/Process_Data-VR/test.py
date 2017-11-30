import os
import numpy as np
date = r"\2017_11_22"
recording_no = r"\000"
path = r"C:\Users\SSRL\Documents\VRDepthTest" + date + recording_no
new_path = os.path.join(path, os.listdir(path).pop())
data = np.genfromtxt(path  + "\distance.npy", delimiter=',',
                      skip_header=1,names = ['Distance'], dtype=(float))

print(data['Distance'][1445])