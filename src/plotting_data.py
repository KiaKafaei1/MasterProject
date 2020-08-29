import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from matplotlib.pyplot import plot, axis, show
import collections

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

## From list to tuple since we have to use tuple for hashing
#array_tup =[]
#tup = ()
#for ele in array:
#    tup = tuple(tuple(x) for x in ele)
#    array_tup.append(tup)
#    tup = ()
#array = array_tup


#Putting 3 following coordinates together to make up triangles
array_tri= []
for ele in array:
    for i in range(0,len(ele)-1,3):
         ele_temp.append([ele[i],ele[i+1],ele[i+2]])
    array_tri.append(ele_temp)
    ele_temp = []


#plotting triangles
hash_values = []
tempx=[]
tempy=[]
i=0
fig, ax = plt.subplots()
Dic =collections.defaultdict(list) 
for room in array_tri:
    for tri in room:
        for pair in tri:
            tempx.append(pair[0])
            tempy.append(pair[1])
        #tempx.append(tempx[0])
        #tempy.append(tempy[0])
        #print(pair) 
        #hashing each line in triangle
        hash1= hash(((tempx[0],tempy[0]),(tempx[1],tempy[1])))
        hash12= hash(((tempx[1],tempy[1]),(tempx[0],tempy[0])))
        hash2= hash(((tempx[0],tempy[0]),(tempx[2],tempy[2])))
        hash22= hash(((tempx[2],tempy[2]),(tempx[0],tempy[0])))
        hash3= hash(((tempx[1],tempy[1]),(tempx[2],tempy[2])))
        hash32= hash(((tempx[2],tempy[2]),(tempx[1],tempy[1])))
       #line1, = ax.plot([tempx[0],tempx[1]],[tempy[0],tempy[1]])
        
        
        
        if (hash1 in hash_values or hash12 in hash_values):
            i+=1
            #print(i)
            #del ax.lines[0]
            #ax.lines.remove(line1)
            #l1 = line1.pop(0)
            #l1.remove()
            #del l1
        if (hash2 in hash_values or hash22 in hash_values):
            k=3
            #ax.lines.remove(line2)
            #l2 = line2.pop(0)
            #l2.remove()              
            #del l2                               
        if (hash3 in hash_values or hash32 in hash_values):
            k=3
            #ax.lines.remove(line3)
            #l3 = line3.pop(0)
            #l3.remove()
            #del l3
        Dic[hash1].append(((tempx[0],tempy[0]),(tempx[1],tempy[1])))
        Dic[hash12].append(((tempx[1],tempy[1]),(tempx[0],tempy[0])))
        Dic[hash2].append(((tempx[0],tempy[0]),(tempx[2],tempy[2])))
        Dic[hash22].append(((tempx[2],tempy[2]),(tempx[0],tempy[0])))
        Dic[hash3].append(((tempx[1],tempy[1]),(tempx[2],tempy[2])))
        Dic[hash32].append(((tempx[2],tempy[2]),(tempx[1],tempy[1])))    
        
        
        
        #hash_values.append(hash1) 
        #hash_values.append(hash12)
        #hash_values.append(hash2) 
        #hash_values.append(hash22) 
        #hash_values.append(hash3) 
        #hash_values.append(hash32) 
        tempx = []
        tempy = []

#print(len(Dic.keys()))
[Dic.pop(x) for x in list(Dic) if len(Dic[x])>1]
#print(len(Dic.keys()))


for line in Dic.values():
    #print(line[0][0][0])
    plt.plot([line[0][0][0], line[0][1][0]],[line[0][0][1], line[0][1][1]])    








plt.show()
#print(Dic)


test_dic = {1: [((1,2),(2,1))]}
#print("This is the length of the test array",len(test_dic[1]))
#print(len(Dic[3516082391940370633]))
#print(len(Dic[-1449807290174480895])) 

seen = []
for number in hash_values:
    if number in seen:
        k=3
        #print( "Number repeated!")
    else:
        seen.append(number)



#print(hash(((1,2),(3,1))))
#print(hash(((3,1),(1,2))))
l = [1,2,3,4,4,3,1,2,4,4]
l2 =[]
#seen = set()
#for i in range(0,len(l)-1):
#    x=l[i]
#    z=l[i+1]
#    
#    if(l
##for x,z in zip(l[0::2],l[1::2]):
#    if x in l2 and z in l2:
#      print(x)
#      print(z)
#      continue
# l2.append(x)
# l2.append(z)
#  #seen.add(x)
#  #seen.add(z)

#print(hash(((3,3),(4,3))))
#print(hash(((3,3),(4,3))))



       # hash_values.append(hash1) 
       # hash_values.append(hash2) 
       # hash_values.append(hash3) 

        #plt.plot(tempx,tempy)
       # line1= plt.plot([tempx[0],tempx[1]],[tempy[0],tempy[1]])
       # line2= plt.plot([tempx[0],tempx[2]],[tempy[0],tempy[2]])
       # line3= plt.plot([tempx[1],tempx[2]],[tempy[1],tempy[2]])



