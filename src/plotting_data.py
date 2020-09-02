import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from matplotlib.pyplot import plot, axis, show
import collections
# This is my own library
import coordinate_processing as cop


def hashing_values(tempx,tempy):
    hash_values=[]
    hash_values.append(hash(((tempx[0],tempy[0]),(tempx[1],tempy[1]))))
    hash_values.append(hash(((tempx[1],tempy[1]),(tempx[0],tempy[0]))))
    hash_values.append(hash(((tempx[0],tempy[0]),(tempx[2],tempy[2]))))
    hash_values.append(hash(((tempx[2],tempy[2]),(tempx[0],tempy[0]))))
    hash_values.append(hash(((tempx[1],tempy[1]),(tempx[2],tempy[2]))))
    hash_values.append(hash(((tempx[2],tempy[2]),(tempx[1],tempy[1]))))
    
    return hash_values


def dict_hashing(Dic,hash_values):
    Dic[hash_values[0]].append(((tempx[0],tempy[0]),(tempx[1],tempy[1])))
    Dic[hash_values[1]].append(((tempx[1],tempy[1]),(tempx[0],tempy[0])))
    Dic[hash_values[2]].append(((tempx[0],tempy[0]),(tempx[2],tempy[2])))
    Dic[hash_values[3]].append(((tempx[2],tempy[2]),(tempx[0],tempy[0])))
    Dic[hash_values[4]].append(((tempx[1],tempy[1]),(tempx[2],tempy[2])))
    Dic[hash_values[5]].append(((tempx[2],tempy[2]),(tempx[1],tempy[1]))) 
    return Dic


#### Plotting Rooms ####

#Importing values and changing from df to numpy array
df = pd.read_csv('room_coordinates.csv',sep=',', header=None)
array = df.to_numpy()
# Deleting the header
array = np.delete(array,0)
# Preprocessing the csv file such that it is in the right format for plotting
array = cop.cor_processing(array)
array_tri = cop.tri_processing(array)

#plotting triangles
tempx=[]
tempy=[]
Dic =collections.defaultdict(list) 

i=0
for room in array_tri:
    
    #Not including every room to simplify the drawing
    if(i>4):
        continue
    for tri in room:
        for cor in tri:
            tempx.append(cor[0])
            tempy.append(cor[1])
        #hashing each line in triangle
        hash_values = hashing_values(tempx,tempy)
        #Adding the hashes with their corresponding line to a Dict
        Dic = dict_hashing(Dic,hash_values)
        
        tempx = []
        tempy = []
    i+=1

# Removing lines that are reoccuring from the Dictionary
[Dic.pop(x) for x in list(Dic) if len(Dic[x])>1]

# Plotting the lines
for line in Dic.values():
    plt.plot([line[0][0][0], line[0][1][0]],[line[0][0][1], line[0][1][1]],'b')    
#plt.show()




#### Plotting Doors ####
df = pd.read_csv('door_coordinates.csv',sep=',', header=None)
#print(df)
array = df.to_numpy()
array = np.delete(array,0)
array = cop.cor_processing(array)

i=0
for cor in array:
    i+=1
#    print(cor[0][0])
    #plt.plot(-1*(cor[0][0]/1000),cor[1][0]/1000,marker='o')
    plt.plot((cor[0][0]+75000)+117,cor[1][0]-153000-30,marker='o')
    if i>10:
        break
plt.show()

