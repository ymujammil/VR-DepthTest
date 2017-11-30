import csv
import numpy as np
import string
from numpy import genfromtxt
path = r"C:\Users\SSRL\Documents\VRDepthTest\2017_11_28\003"
#path.replace("\\", "/")
print(path)
my_data = genfromtxt(path + '/world_timestamps.npy', delimiter=',')
new_data = np.delete(my_data, [0])
#print(wines)
np.save(path + '/world_timestamps', new_data)
#wines = np.array(wines[1:], dtype=np.float)
#print(wines)
data = np.load(path + '/world_timestamps.npy')
print(data)
print(new_data)