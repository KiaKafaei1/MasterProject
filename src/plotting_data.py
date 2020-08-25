import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from matplotlib.pyplot import plot, axis, show

#Importing values and changing from df to numpy array
df = pd.read_csv('room_coordinates.csv',sep=',', header=None)
array = df.to_numpy()
array = np.delete(array,0)#deleting the header

# The coordinates are in string form and will have to be changed to float

# Splitting the strings in the array with spaces
array_new = []
for ele in array:
    array_new.append(ele.split(' '))


#Splitting the strings with comma
array_new2 = []
ele_temp = []
for ele in array_new:
    ele.pop()
    for pair in ele:
        ele_temp.append(pair.split(','))
    array_new2.append(ele_temp)
    ele_temp = []


#From string to float
array_new3 = []
for ele in array_new2:
    for pair in ele:
        pair_temp=list(map(float, pair))
        ele_temp.append(pair_temp)
    array_new3.append(ele_temp)
    ele_temp = []



#Putting 3 following coordinates together to make up triangles
array_new4= []
for ele in array_new3:
    for i in range(0,len(ele)-1,3):
         ele_temp.append([ele[i],ele[i+1],ele[i+2]])
    array_new4.append(ele_temp)
    ele_temp = []


#plotting triangles
tempx=[]
tempy=[]
temp = array_new4[3]
for ele in array_new4:
    for tri in ele:
        for pair in tri:
            print(pair)
            for i in range(0,len(pair)):
                print(i)
                print(len(pair))
                if i ==0:
                    tempx.append(pair[i])
                else:
                    tempy.append(pair[i])
        tempx.append(tempx[0])
        tempy.append(tempy[0])
        plt.plot(tempx,tempy)
        tempx=[]
        tempy=[]
    
plt.show()






























