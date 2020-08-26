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

# The array after processing the data
array = array_new3

#Putting 3 following coordinates together to make up triangles
array_tri= []
for ele in array:
    for i in range(0,len(ele)-1,3):
         ele_temp.append([ele[i],ele[i+1],ele[i+2]])
        # hash(frozenset(ele[i+2]))# identifying the hypotenuse coordinate
         
    array_tri.append(ele_temp)
    ele_temp = []

#
##plotting triangles
tempx=[]
tempy=[]
for ele in array_tri:
    for tri in ele:
        for pair in tri:
            for i in range(0,len(pair)):
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


# From list to tuple since we have to use tuple for hashing
array_tup =[]
tup = ()
for ele in array:
    tup = tuple(tuple(x) for x in ele)
    array_tup.append(tup)
    tup = ()
print(array_tup[3])



#Making Frozenset
fro1=None 
fro2 =None 
temp1 = []
temp2 = []
for ele in array_new3:
    for i in range(0,len(ele)):
        if i%2==0:
            temp1 = ele[i]
        else:
            temp2 = ele[i]
        if temp1==None or temp2==None:
            continue;
       # print(hash(frozenset([temp1,temp2])))
