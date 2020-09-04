import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import pandas as pd
from matplotlib.pyplot import plot, axis, show
import collections
# This is my own library
import coordinate_processing as cop
from matplotlib.animation import FuncAnimation

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

# Doing the animation, moving from one point to another

fig, ax = plt.subplots()
ax.set(xlim=(-0.1,2*np.pi+0-1),ylim = (-1.1,1.1))
line, = ax.plot( [],[])









#### Plotting Doors
#### Plotting using CSV####
#df = pd.read_csv('door_coordinates.csv',sep=',', header=None)
##print(df)
#array = df.to_numpy()
#array = np.delete(array,0)
#array = cop.cor_processing(array)
#
#i=0
#for cor in array:
#    i+=1
#    #plt.plot(-1*(cor[0][0]/1000),cor[1][0]/1000,marker='o')
#    plt.plot((cor[0][0]+75000)+117,cor[1][0]-153000-30,marker='o')
#    if i>10:
#        break
#plt.show()

## Plotting manually ##

#plt.plot(75,6,marker='o')
#plt.plot(60,10.2,marker='o')
#plt.plot(46,14,marker='o')
#plt.plot(61,1.6,marker='o')
#plt.show()

#### Plotting points in each room ###
plt.plot(48,21,marker='o')
plt.plot(60,17,marker='o')
plt.plot(76,13,marker='o')
plt.plot(72,2,marker='o')
plt.plot(46,10,marker='o')
plt.plot(40,22,marker='o')
plt.show()













## DOING THE MINIMUMSPANNING TREE ##
#A = [48,21]
#B = [60,17]
#C = [76,13]
#D = [72,2]
#E = [46,10]
#F = [40,22]
#
### Calculating distances between each point assuming straight lines in a distance table##
#
#points = [A,B,C,D,E,F]
#dist_table={}
#for p1 in points:
#    for p2 in points:
#        i1 = points.index(p1)
#        i2 = points.index(p2)
#        if i1==i2 or i1>i2:
#            continue
#        dist_table[str(i1)+str(i2)]=math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
#        
##        dist_table[str(i0
#dist_table = sorted(dist_table.items(),key=lambda x: x[1])
#print(dist_table)
#
#### Making Minimum spanning tree ###


















