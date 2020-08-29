import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from matplotlib.pyplot import plot, axis, show
import collections

#Importing values and changing from df to numpy array
df = pd.read_csv('room_coordinates.csv',sep=',', header=None)
array = df.to_numpy()
# Deleting the header
array = np.delete(array,0)


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
    array_tri.append(ele_temp)
    ele_temp = []


#plotting triangles
tempx=[]
tempy=[]
Dic =collections.defaultdict(list) 

for room in array_tri:
    for tri in room:
        for pair in tri:
            tempx.append(pair[0])
            tempy.append(pair[1])

        #hashing each line in triangle
        hash1= hash(((tempx[0],tempy[0]),(tempx[1],tempy[1])))
        hash12= hash(((tempx[1],tempy[1]),(tempx[0],tempy[0])))
        hash2= hash(((tempx[0],tempy[0]),(tempx[2],tempy[2])))
        hash22= hash(((tempx[2],tempy[2]),(tempx[0],tempy[0])))
        hash3= hash(((tempx[1],tempy[1]),(tempx[2],tempy[2])))
        hash32= hash(((tempx[2],tempy[2]),(tempx[1],tempy[1])))
        
        #Adding the hashes with their corresponding line to a Dict
        Dic[hash1].append(((tempx[0],tempy[0]),(tempx[1],tempy[1])))
        Dic[hash12].append(((tempx[1],tempy[1]),(tempx[0],tempy[0])))
        Dic[hash2].append(((tempx[0],tempy[0]),(tempx[2],tempy[2])))
        Dic[hash22].append(((tempx[2],tempy[2]),(tempx[0],tempy[0])))
        Dic[hash3].append(((tempx[1],tempy[1]),(tempx[2],tempy[2])))
        Dic[hash32].append(((tempx[2],tempy[2]),(tempx[1],tempy[1])))    
        
        tempx = []
        tempy = []

# Removing lines that are reoccuring from the Dictionary
[Dic.pop(x) for x in list(Dic) if len(Dic[x])>1]

# Plotting the lines
for line in Dic.values():
    plt.plot([line[0][0][0], line[0][1][0]],[line[0][0][1], line[0][1][1]],'b')    
plt.show()


