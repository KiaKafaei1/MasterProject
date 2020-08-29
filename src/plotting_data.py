import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from matplotlib.pyplot import plot, axis, show
import collections
import coordinate_processing as cop

#Importing values and changing from df to numpy array
df = pd.read_csv('room_coordinates.csv',sep=',', header=None)
array = df.to_numpy()
# Deleting the header
array = np.delete(array,0)

array_tri = cop.cor_processing(array)

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


