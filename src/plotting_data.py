import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import csv
import pandas as pd
from matplotlib.pyplot import plot, axis, show
import collections
import coordinate_processing as cop # This is my own library
from matplotlib.animation import FuncAnimation
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
import dwave_networkx as dnx
from scipy.spatial import distance
import random
from time import process_time
from rtree import index
from rtree.index import Rtree
import copy
################# Defining functions and classes ###################

#1
def dict_hashing(Dic,tempx,tempy):
    '''We hash 6 lines because the lines may be in reverse coordinate order for different triangles, meaning  that a line from x1,y1 to x2,y2 in one triangle is the line x2,y2 to x1,y1 in another triangle.'''

    h1=hash(((tempx[0],tempy[0]),(tempx[1],tempy[1])))
    h2=hash(((tempx[1],tempy[1]),(tempx[0],tempy[0])))
    h3=hash(((tempx[0],tempy[0]),(tempx[2],tempy[2])))
    h4=hash(((tempx[2],tempy[2]),(tempx[0],tempy[0])))
    h5=hash(((tempx[1],tempy[1]),(tempx[2],tempy[2])))
    h6=hash(((tempx[2],tempy[2]),(tempx[1],tempy[1])))

    Dic[h1].append(((tempx[0],tempy[0]),(tempx[1],tempy[1])))
    Dic[h2].append(((tempx[1],tempy[1]),(tempx[0],tempy[0])))
    Dic[h3].append(((tempx[0],tempy[0]),(tempx[2],tempy[2])))
    Dic[h4].append(((tempx[2],tempy[2]),(tempx[0],tempy[0])))
    Dic[h5].append(((tempx[1],tempy[1]),(tempx[2],tempy[2])))
    Dic[h6].append(((tempx[2],tempy[2]),(tempx[1],tempy[1]))) 
    return Dic


#2
class Point:
    ''' A class used to save node/point objects'''
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __repr__(self):
        return "Point({},{})".format(self.x,self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def round_to_half(self):
        self.x = round(self.x*2)/2
        self.y = round(self.y*2)/2

    def __add__(self,other):
        self.x = self.x + other.x
        self.y = self.y + other.y
        return self



#3
def ccw(A,B,C):
    ''' 
Checking if 3 points are in counterclockwise order
Dertimining line intersection 
Code taken from https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
'''
    return (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)

def intersect(A,B,C,D):
    '''
Finding if 4 points are intersecting. Se above link for explanation.
'''
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

#4
def plot_point(point,door=True,starting_node=False):
    ''' 
Function for plotting points
'''
    if door:
        plt.plot(point.x,point.y,marker='o',color='black',markerfacecolor='black')
    elif starting_node:
        plt.plot(point.x,point.y,marker='o',color='green', markerfacecolor='green')
    else:
        plt.plot(point.x,point.y,marker='o',color='red', markerfacecolor='red')


#5
def is_traversable(p1,q1,Dic_lines):
    '''
Function for plotting path.
The function returns true if there is a traversable connection between 2 points else returns False.
'''
    epsilon = 2
    # If this is true it means that it is 2 doors right next to eachother and the path should therefore be traversable
    # To make sure that the doors are the correctly aligned such that it doesn't walk diagonally to another door we make sure that the x or y values are the same.
    if (math.sqrt((p1.x-q1.x)**2+(p1.y-q1.y)**2)<epsilon) and (-0.01<p1.x-q1.x<0.01 or -0.01<p1.y-q1.y<0.01):
        return True
    for line in Dic_lines.values():
        p2 = Point(line[0][0][0],line[0][0][1])
        q2 = Point(line[0][1][0],line[0][1][1])
        val= intersect(p1,q1,p2,q2)
        if val:
            #They do intersect
            return False
        if not val:
            continue
    return True


# Code taken from https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
def point_line_dist(x1,y1,x2,y2,x3,y3): #x3,y3 is the point
    px = x2-x1
    py = y2-y1
    norm = px*px+py*py
    u = ((x3-x1)*px+(y3-y1)*py)/float(norm)
    if u>1:
        u=1
    elif u<0:
        u=0
    x = x1+u*px
    y = y1+u*py
    dx = x-x3
    dy = y-y3
    # Note: If the actual distance does not matter,
    # if you only want to compare what this function
    # returns to other results of this function, you
    # can just return the squared distance instead
    # (i.e. remove the sqrt) to gain a little performance
    dist = (dx*dx+dy*dy)**0.5
    return dist

# Rounds the limits to nearest 10, to leave a little space in the plot on the sides.
def roundUp10(x):
    return int(math.ceil(x/10.0))*10
def roundDown10(x):
    return int(math.floor(x/10.0))*10

# Rounds the limits to nearest 5, to leave a little space in the plot on the sides.
def roundUp5(x):
    return int(math.ceil(x/5.0))*5
def roundDown5(x):
    return int(math.floor(x/5.0))*5


def plot_grid(ax,x_min,x_max,y_min,y_max):
    #major_ticks = np.arange(x_min, y_max, 20)
    #minor_ticks = np.arange(, y, 5)

    #ax.set_xticks(major_ticks)
    #ax.set_xticks(minor_ticks, minor=True)
    #ax.set_yticks(major_ticks)
    #ax.set_yticks(minor_ticks, minor=True)


    #ax.set_xlim(roundDown5(x_min),roundUp5(x_max))
    #ax.set_ylim(roundDown5(y_min),roundUp5(y_max))
    ax.set_xlim((x_min),(x_max))
    ax.set_ylim((y_min),(y_max))
    ## Change major ticks 
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ## Minor locator at every 0.5 (20/40)
    ax.xaxis.set_minor_locator(AutoMinorLocator(40))
    ax.yaxis.set_minor_locator(AutoMinorLocator(40))
    ## Minor locator at every 0.2 (20/100)
    #ax.xaxis.set_minor_locator(AutoMinorLocator(100))
    #ax.yaxis.set_minor_locator(AutoMinorLocator(100))


    ax.grid(which = 'minor')
    plt.grid()

def rec_dd():
    return collections.defaultdict(rec_dd)



## Testing area



### Implementing spatial datastrukture for rooms ###
p = index.Property()
idx_rooms = index.Index(interleaved=False)
# Making a dictionary for the rectangles, such that they can later be retrieved when wanting to branch out
rectangles_rooms = collections.defaultdict(list)
Dic_rooms = rec_dd()
Dic_rooms_edges = collections.defaultdict(list)
##################### Plotting Rooms ###########################
#6
#Importing values and changing from df to numpy array
#t0 = time.time()
df = pd.read_csv('room_coordinates.csv',sep=',', header=None)
room_127 = False
array = df.to_numpy()
# Deleting the header
array = np.delete(array,0)
# Preprocessing the csv file such that it is in the right format for plotting
array = cop.cor_processing(array)
array_tri = cop.tri_processing(array)

# Getting the room min elevation values
df_elevation = pd.read_csv('room_elevation.csv',sep=',', header=None)
array_elevation = df_elevation.to_numpy()
#array_elevation = np.delete(array_elevation,0)

# Getting the room numbers
df_number = pd.read_csv('room_number.csv',sep=',', header=None)
array_number = df_number.to_numpy()



#print(array_elevation)
#7
#plotting triangles
tempx=[]
tempy=[]
Dic =collections.defaultdict(list) 
#fig, ax = plt.subplots()
Dic_all = collections.defaultdict(list)


#Finding all possible elevation combos
elevation_combos = []
#floor_number = building_floors[1]

for i,ele in enumerate(array_elevation):
    min_ele = ele[0]
    max_ele = ele[1]

    if [min_ele,max_ele] not in elevation_combos:
        elevation_combos.append([min_ele,max_ele])

# Choosing random clipping height, by taking avg of a floor min and max height
random_floor = random.randint(0, len(elevation_combos)-1)
#clippingHeight = (elevation_combos[random_floor][0]+elevation_combos[random_floor][1])/2 
#clippingHeight = 0.965
#print(clippingHeight)


#print(elevation_combos)
# # Choosing random floor
# random_floor = random.randint(0, len(elevation_combos)-1)
min_elevation = elevation_combos[random_floor][0]
max_elevation = elevation_combos[random_floor][1]
min_elevation = -8.311
print(min_elevation)
print(elevation_combos)

building_min_height = min([x[0] for x in elevation_combos])
building_max_height = max([x[1] for x in elevation_combos])



for i,room in enumerate(array_tri):
    if array_elevation[i][0]!=min_elevation:
       continue
    #if array_elevation[i][0]!=min_elevation or array_elevation[i][1]!=max_elevation:
    #    continue
    #if (clippingHeight <= array_elevation[i][0] or clippingHeight >= array_elevation[i][1]): 
    #    continue
    #print(i)
    for tri in room:
        for cor in tri:
            tempx.append(cor[0])
            tempy.append(cor[1])
        # Hashing each line in both directions and adding the hashes with their corresponding line to a Dict
        Dic = dict_hashing(Dic,tempx,tempy)
        tempx = []
        tempy = []
    # Removing lines that are reoccuring from the Dictionary
    for x in list(Dic):
        if len(Dic[x])>1:
            Dic.pop(x)            
#9
    # All lines are plotted twice due to the way we are hashing, it is more appropriate to remove replicate values that are in more than 1 hash.
    # Remove replicate lines  
    temp = Dic.copy()
    visited_lines = []
    for key,line in temp.items():
        for key2,line2 in temp.items():
            if line[0][0] == line2[0][1] and line[0][1] == line2[0][0] and line not in visited_lines and line2 not in visited_lines:
                Dic.pop(key2)
                visited_lines.append(line)
                break
    Dic_all.update(Dic)
    #if array_number[i][0] == 102 or array_number[i][0] == 107:
    # fig, ax = plt.subplots()
    # #plot_grid(ax,x_min,x_max,y_min,y_max)

    # for line in Dic_all.values():
    #     ax.plot([line[0][0][0], line[0][1][0]],[line[0][0][1], line[0][1][1]],'b')  
    #     ax.set_title(array_number[i][0])  
    # plt.show()
    #### Making a dictionary containing all the rooms and their lines.
    p_x_max = 0
    p_y_max = 0
    p_x_min = 1000
    p_y_min = 1000
    for line in Dic.values():
        p1 = line[0][0]
        p2 = line[0][1]
        for p in (p1,p2):
            if p[0]>p_x_max:
                p_x_max = p[0] #p1
            if p[0]<p_x_min:
                p_x_min = p[0] #p1
            if p[1]>p_y_max:
                p_y_max = p[1]
            if p[1]<p_y_min:
                p_y_min = p[1]
    p_y_min = p_y_min - 1
    p_x_min = p_x_min - 1
    p_y_max = p_y_max + 1
    p_x_max = p_x_max + 1
    bottom = p_y_min
    left = p_x_min
    top = p_y_max
    right = p_x_max
    idx_rooms.insert(i,(left,right,bottom,top))
    rectangles_rooms[i] = [left,right,bottom,top]
    # I save the maximum points in another dictionary
    Dic_rooms_edges[i] = [p_x_min, p_y_min,p_x_max,p_y_max]
    Dic_rooms[i] = Dic
    Dic = collections.defaultdict(list)



print("PART 1")
###### MAKING THE HIGH RESOLUTION GRID #######
# Finding the keys for the largest and smallest x and y values from each point in each line
x_max_idx1 = max(Dic_all, key=lambda key: Dic_all[key][0][1][0])
x_max_idx2 = max(Dic_all, key=lambda key: Dic_all[key][0][0][0])
x_min_idx1 = min(Dic_all, key=lambda key: Dic_all[key][0][1][0])
x_min_idx2 = min(Dic_all, key=lambda key: Dic_all[key][0][0][0])
# y_values
y_max_idx1 = max(Dic_all, key=lambda key: Dic_all[key][0][0][1])
y_max_idx2 = max(Dic_all, key=lambda key: Dic_all[key][0][1][1])
y_min_idx1 = min(Dic_all, key=lambda key: Dic_all[key][0][0][1])
y_min_idx2 = min(Dic_all, key=lambda key: Dic_all[key][0][1][1])
# Getting the corresponding max and min values
x_max = math.ceil(max([Dic_all[x_max_idx1][0][1][0],Dic_all[x_max_idx2][0][0][0]]))
x_min = math.floor(min([Dic_all[x_min_idx1][0][1][0],Dic_all[x_min_idx2][0][0][0]]))
y_max = math.ceil(max([Dic_all[y_max_idx1][0][0][1],Dic_all[y_max_idx2][0][1][1]]))
y_min = math.floor(min([Dic_all[y_min_idx1][0][0][1],Dic_all[y_min_idx2][0][1][1]]))


#10
### Extracting the door information
df = pd.read_csv('door_coordinates.csv',sep=',', header=None)
array = df.to_numpy()
array = [item for sublist in array for item in sublist]
array = cop.cor_processing(array,room=0)

#Checking if there are 2 door coordinates or 3 door coordinates
#This is the translation needed to make doors align with floorplan
translation =  [array[-1][i] for i in (0,-1)]
translation_height = array[-1][1][0]
# Remove the translation from the array of door coordinates
array = array[:-1]
# Split the doors in two lists: the points of the doors and the facing of the doors.
list_points_doors = [ele for i,ele in enumerate(array) if i%2==0]
list_facing_doors = [ele for i,ele in enumerate(array) if i%2==1]
# Door elevation
df2 = pd.read_csv('doors_elevation.csv', sep=',', header=None)
array2 = df2.to_numpy()
#Finding all possible elevation combos
elevation_combos = []
for i,ele in enumerate(array2):
    min_ele = ele[0]
    max_ele = ele[1]
    if [min_ele,max_ele] not in elevation_combos:
        elevation_combos.append([min_ele,max_ele])
points_doors = []
facing_doors = []

# Here what you can do instead is to take the median value and see if it is within the boundaries
translation_usage = False
num_door_coord = 3
#print("[x_min,x_max]", [x_min,x_max])
#print("[y_min,y_max]", [y_min,y_max])
if list_points_doors[0][0][0]>x_max or list_points_doors[0][0][0]<x_min or list_points_doors[0][1][0]>y_max or list_points_doors[0][1][0]<y_min:
    print("The x and y of the doors are translated")
    translation_usage_xy = True

# Checking if the height needs to be translated
if array2[0][0]>building_max_height or array2[0][0]<building_min_height:
    print("The height is translated")
    min_elevation = min_elevation+translation_height#+0.0001


clippingHeight = min_elevation
print(min_elevation)
translation_usage_xy = False
#print(translation)
#print(translation[0][0])
#print(translation[1][0])
if translation_usage_xy:
    for i,elev in enumerate(array2):
        if (clippingHeight-0.01<=elev[0]<=clippingHeight+0.01):
            #print(elev)
            # This is the standard
            points_doors.append(Point(list_points_doors[i][0][0]-translation[0][0],list_points_doors[i][1][0]+translation[1][0]))
            #points_doors.append(Point(list_points_doors[i][0][0]-translation[0][0],list_points_doors[i][1][0]+translation[1][0]))
            facing_doors.append([list_facing_doors[i][0][0],list_facing_doors[i][1][0]])
else:
    for i,elev in enumerate(array2):
        if (clippingHeight-0.01<=elev[0]<=clippingHeight+0.01):
            points_doors.append(Point(list_points_doors[i][0][0],list_points_doors[i][1][0]))
            facing_doors.append([list_facing_doors[i][0][0],list_facing_doors[i][1][0]])

# The new way to append opposite doors
temp_points = []#points_doors.copy()
points_doors_opposite = []
for i,door in enumerate(points_doors):
    # print(i)
    # Dividing by 2 such that it doesn't go a unit vector away from the original point but only a euclidean distance of 0.5
    temp_points.append(Point(door.x+facing_doors[i][0]/2,door.y+facing_doors[i][1]/2))
    points_doors_opposite.append(Point(door.x-facing_doors[i][0]/2,door.y-facing_doors[i][1]/2))
points_doors = temp_points.copy()
#print("length points doors",len(points_doors))

print("PART 2")
# Grid height is used when making the grid nodes
grid_height = len(np.linspace(y_min,y_max,(y_max-y_min)*2+1))
# Keeping the original door coordinates
points_doors_discrete = copy.deepcopy(points_doors)
points_doors_opposite_discrete = copy.deepcopy(points_doors_opposite)
#print(points_doors_discrete)
# Rounding the discrete doors to nearest 0.5 since that is the resolution of the grid.
[p.round_to_half() for p in points_doors_discrete]
[p.round_to_half() for p in points_doors_opposite_discrete]
# Doing the same for the room nodes
#[p.round_to_half() for p in points_rooms]

# Adding the door nodes
G_grid = nx.Graph()
idx_doors = index.Index(interleaved=False)
for idx_d in range(1,len(points_doors)):
    # The distance will be a dummy number, since it is only needed as a place holder
    dist = 2
    p_float= points_doors[idx_d]
    G_grid.add_node(idx_d,att =("door",p_float,dist,idx_d))
    idx_doors.insert(idx_d,(p_float.x,p_float.x,p_float.y,p_float.y)) 
number_of_nodes = len(G_grid.nodes)

for idx_d in range(1,len(points_doors)):
    dist = 2
    p_float = points_doors_opposite[idx_d]
    G_grid.add_node(number_of_nodes+idx_d,att =("door",p_float,dist,idx_d))
    idx_doors.insert(number_of_nodes+idx_d,(p_float.x,p_float.x,p_float.y,p_float.y))


#temp_dic = Dic_all.deepcopy()
# Removin all one line walls, since these do not represent real walls and therefore obstruct the path.
temp_dic = copy.deepcopy(Dic_all)
for key1,line1 in temp_dic.items():
    for key2,line2 in temp_dic.items():
        if key1==key2:
            #print("same key")
            continue
        #print(key1)
        #print(line1)
        p1 = Point(line1[0][0][0],line1[0][0][1])
        p2 = Point(line1[0][1][0],line1[0][1][1])
        p3 = Point(line2[0][0][0],line2[0][0][1])
        p4 = Point(line2[0][1][0],line2[0][1][1])
        # If the two points that make up the line is in the same order or opposite order
        #if ((p1 == p3 and p2==p4) or (p1==p4 and p2==p3)):
        #if (line1[0][0] == line2[0][0] and line1[0][1]== line2[0][1]) or (line1[0][0] == line2[0][1] and line1[0][1]== line2[0][0]):
            #print("testen virker")
            ##print("key1", key1)
            ##print(Dic_all[key1])
        #    del Dic_all[key1]
        #    break
        # Checking if the two lines (or the four points they are made of) 
        # are colinear, by checking the slope of three or more points are the same.
        slope_lines = []
        if (p2.x-p1.x) == 0:
            slope_lines.append(1000)
        else:
            slope_lines.append(abs((p2.y-p1.y))/(abs((p2.x-p1.x))))
        if (p4.x-p3.x) == 0:
            slope_lines.append(1000)
        else:
            slope_lines.append(abs((p4.y-p3.y))/(abs((p4.x-p3.x))))
        if (p4.x-p1.x) == 0:
            slope_lines.append(1000)
        else:
            slope_lines.append(abs((p4.y-p1.y))/(abs((p4.x-p1.x))))
        # They are coolinear now we check if they overlap by looking at their projections on the x axis
        intersecting = False
        if all(x==slope_lines[0] for x in slope_lines):
            # We have complete vertical and colinear which are a special exception
            # We now check if they intersect by looking at their y values. 
            
            if slope_lines[0]==1000:
                # If the lines are inbetween eachother
                if (p3.y < p1.y < p4.y or p4.y < p1.y < p3.y) or \
                (p3.y < p2.y < p4.y or p4.y < p2.y < p3.y) or \
                (p1.y < p3.y < p2.y or p2.y < p3.y < p1.y) or \
                (p1.y < p4.y < p2.y or p2.y < p4.y < p1.y):
                    intersecting = True
                # If the lines are exactly the same
                elif ((p1 == p3 and p2==p4) or (p1==p4 and p2==p3)):
                    intersecting = True

            #    del Dic_all[key1]
            #    break  
            #for p in [p1,p2,p3,p4]:
            #    if 123.9< p.x <124 and 60.9 <p.y < 61.2:
            #        print([p1,p2,p3,p4])
            #print("VI ER HER")

            # If not we check by projection on the x axis
            elif(p3.x < p1.x < p4.x or p4.x < p1.x < p3.x) or \
            (p3.x < p2.x < p4.x or p4.x < p2.x < p3.x) or \
            (p1.x < p3.x < p2.x or p2.x < p3.x < p1.x) or \
            (p1.x < p4.x < p2.x or p2.x < p4.x < p1.x):
                intersecting = True

                # We check if there are doors near the wall, if that is the case we don't remove the walls.
                # We have to index the points properly
                # Checking the order of the points in the lines
                # if p1.x<p2.x:
                #     if p1.y<

            # This is the special scenario where the lines are exactly the same
            elif ((p1 == p3 and p2==p4) or (p1==p4 and p2==p3)):
                intersecting = True

            if intersecting:
                if p1.x < p2.x:
                    if p1.y<=p2.y:
                        k = list(idx_doors.nearest((p1.x,p2.x, p1.y, p2.y),1))
                    else:
                        k = list(idx_doors.nearest((p1.x,p2.x, p2.y, p1.y),1))
                else:
                    if p1.y<=p2.y:
                        k = list(idx_doors.nearest((p2.x,p1.x, p1.y, p2.y),1))
                    else:
                        k = list(idx_doors.nearest((p2.x,p1.x, p2.y, p1.y),1))

                dist_list = []
                for idx_ in k:
                    door_p = G_grid.nodes[idx_]['att'][1]
                    dist = point_line_dist(p1.x,p1.y,p2.x,p2.y,door_p.x,door_p.y)
                    dist_list.append(dist)

                dist = min(dist_list)
                #print(dist_list)
                if dist<=0.5:
                #    print("dør for tæt på")
                    continue
                #print("vi er her")                
                del Dic_all[key1]
                break

            # If the two points that make up the line is in the same order or opposite order
            # This is the special scenario where all points are exactly the same
            # elif ((p1 == p3 and p2==p4) or (p1==p4 and p2==p3)):
            # #if (line1[0][0] == line2[0][0] and line1[0][1]== line2[0][1]) or (line1[0][0] == line2[0][1] and line1[0][1]== line2[0][0]):
            #     #print("testen virker")
            #     ##print("key1", key1)
            #     ##print(Dic_all[key1])
            #     del Dic_all[key1]
            #     break  

# Making datastructure for the walls
p = index.Property()
idx = index.Index(interleaved=False)
# Making a dictionary for the rectangles, such that they can later be retrieved when wanting to branch out
rectangles = collections.defaultdict(list)
Dic_all_unhashed = collections.defaultdict(list)
for i,line in enumerate(Dic_all.values()):
    p1 = line[0][0]
    p2 = line[0][1]
    bottom = min(p1[1],p2[1])
    left = min(p1[0],p2[0])
    top = max(p1[1],p2[1])
    right = max(p1[0],p2[0])
    rectangles[i] = [left,right,bottom,top]
    Dic_all_unhashed[i] = line
    idx.insert(i,(left,right,bottom,top),obj = line)
## Splitting the leaf nodes (indexes) up in branches
left_branch = index.Index(interleaved = False)
right_branch = index.Index(interleaved= False)
for id in idx.intersection((x_min,(x_max+x_min)/2,y_min,(y_max+y_min)/2)):
    [left,right,bottom,top] = rectangles[id]
    left_branch.insert(id, (left,right,bottom,top))
for id in idx.intersection(((x_max+x_min)/2,x_max,(y_max+y_min)/2,y_max)):
    [left,right,bottom,top] = rectangles[id]
    right_branch.insert(id,(left,right,bottom,top))



grid_height = len(np.linspace(y_min,y_max,(y_max-y_min)*2+1))
G_grid = nx.Graph()
i = 0
counter_room = 0 
print("PART 3")
#Making the grid nodes
#number_of_nodes = len(G_grid.nodes)
#i = number_of_nodes 
idx_nodes = index.Index(interleaved=False)
for x in np.linspace(x_min,x_max,(x_max-x_min)*2+1):
    for y in np.linspace(y_min,y_max,(y_max-y_min)*2+1):
        i = i+1
        p = Point(x,y)
        k = list(idx.nearest((p.x,p.x, p.y, p.y),1))
        # if p.x == -1.5 and p.y==-0.5:
        #     print("k",k)
        #     print(Dic_all_unhashed.get(k[0]))
        #     print(Dic_all_unhashed.get(k[1]))
        #     print(Dic_all_unhashed.get(k[2]))
            #print("line",line)
        idx_nodes.insert(i,(p.x,p.x,p.y,p.y)) # Making spatial datastructure for the nodes
        dist_list = []
        # Finding the wall with the shortest distance
        for idx_ in k:
            line = Dic_all_unhashed.get(idx_)
            p1 = line[0][0]
            p2 = line[0][1]
            dist = point_line_dist(p1[0],p1[1],p2[0],p2[1],p.x,p.y)
            dist_list.append(dist)
        dist = min(dist_list)

        #k = max(k) #If multiple walls are close we just choose a random wall (the wall with the highest index)
        #line = Dic_all_unhashed.get(k)
        #p1 = line[0][0]
        #p2 = line[0][1]
        #dist = point_line_dist(p1[0],p1[1],p2[0],p2[1],p.x,p.y)
        
        #Testing bug
        # if p.x == -1.5 and p.y==-0.5:
        #     print("k",k)
        #     print("line",line)
        
        G_grid.add_node(i,att =("grid",p,dist))       
        # Not adding edges to and from nodes that are too close to the wall
        # This is not possible because then the other nodes will have weird edges to and from eachother
        # THe only reliable way is to add the edges and then remove them again.

        # Vertical and horizontal neighbours
        if x>x_min:
            G_grid.add_edge(i,i-grid_height,weight=10)
        if y>y_min:
            G_grid.add_edge(i,i-1,weight=10)
        # Diagonal neighbours
        # Diagonal down left which is the same as up right
        if x>x_min and y>y_min:
            G_grid.add_edge(i,i-grid_height-1,weight=14)
        # Diagonal up left which is the same as right down
        if x>x_min and y<y_max: 
           G_grid.add_edge(i,i-grid_height+1,weight=14)


# Adding the door nodes again, the first time they were used to check for walls that werent supposed to be there.
# now they are made again to fit into the grid.
#G_grid = nx.Graph()
number_of_nodes = len(G_grid.nodes)
idx_doors = index.Index(interleaved=False)
for idx_d in range(1,len(points_doors)):
    # The distance will be a dummy number, since it is only needed as a place holder
    dist = 2
    p_float= points_doors[idx_d]
    G_grid.add_node(number_of_nodes+idx_d,att =("door",p_float,dist,idx_d))
    idx_doors.insert(number_of_nodes+idx_d,(p_float.x,p_float.x,p_float.y,p_float.y)) 
number_of_nodes = len(G_grid.nodes)

for idx_d in range(1,len(points_doors)):
    dist = 2
    p_float = points_doors_opposite[idx_d]
    G_grid.add_node(number_of_nodes+idx_d,att =("door",p_float,dist,idx_d))
    idx_doors.insert(number_of_nodes+idx_d,(p_float.x,p_float.x,p_float.y,p_float.y))


# # Debugging
# for node,at in sorted(G_grid.nodes(data=True)):
#     p = at['att'][1] 
#     if 65.72<p.x<65.74:
#         idx_d = at['att'][3]
#         print(idx_d)
#     if 66.69<p.x<66.7:
#         idx_d = at['att'][3]
#         print(idx_d)


# Changing all door nodes that are floating to regular grid nodes.
# This might not be important after all since the program doesn't care if a point in the middle of the room is a door node or a grid node
# This is only visible when plotting.
# G_grid_temp = G_grid.copy()
# for node,at in sorted(G_grid_temp.nodes(data=True)):
#     node_type = at['att'][0]
#     if node_type != 'door':
#         continue
#     dist = at['att'][2]
#     if dist >2:
#         p = at['att'][1]
#         idx_d = at['att'][3]
#         G_grid.add_node(node,att=("grid",p,dist))


    

# Remove non traversable nodes
removable_edge_list = []
G_grid_cpy = G_grid.copy()
inside_building = 0
for node,at in sorted(G_grid.nodes(data=True)):
   dist = at['att'][2]
   point = at['att'][1]
   node_type = at['att'][0]
   # We remove all the edges connected to and from the nodes that are too close to a wall
   if dist<0.5:
    if node_type == "door":
        continue
    G_grid_cpy.remove_node(node)
 
G_grid = G_grid_cpy.copy()
t1_start = process_time()
G_grid_cpy = G_grid.copy()



print("PART 4")
# Placing a room label on all nodes in the grid, such that we know which rooms they belong to
inside_building=0
intersect_flag = False
new_room_flag = False
intersected_line = 0
vote = []
room_indexes = []
for node,at in sorted(G_grid_cpy.nodes(data=True)):
    p = at['att'][1]
    k = list(idx_rooms.nearest((p.x,p.x, p.y, p.y), 1)) #len(Dic_rooms)))
    # This is for the case that the closest room to a gridpoint is not the room that the point is within
    room_index_sq_feet = collections.defaultdict(list)
    list_rooms = []
    # Finding the area of the rooms such that we can begin assigning to the smallest rooms first
    for test,room_index in enumerate(k):
        [room_x_min, room_y_min,room_x_max,room_y_max] = Dic_rooms_edges[room_index]
        sq_feet = (room_y_max - room_y_min)*(room_x_max-room_x_min)
        room_index_sq_feet[room_index] = sq_feet
    # Dictionary of rooms 
    temp_dict = {k: v for k, v in sorted(room_index_sq_feet.items(), key=lambda item: item[1])}
    # Converting back to list
    k_new = list(temp_dict.keys())
    for j,room_index in enumerate(k_new):
    #Doing line intersection with the gridpoint and all walls in the room in 4 directions
        #[room_x_min,room_y_min,room_x_max,room_y_max] = Dic_rooms_edges[room_index] # remember that the edges are +- a small sigma
        #four_points = [Point(p.x,room_y_max),Point(p.x,room_y_min),Point(room_x_max,p.y),Point(room_x_min,p.y)]
        four_points = [Point(p.x,y_max),Point(p.x,y_min),Point(x_max,p.y),Point(x_min,p.y)] # using global boundaries
        #four_points = [Point(p.x,y_max),Point(p.x-10,y_max),Point(p.x+10,y_max),Point(p.x-5,y_max)] # Shooting different angles in the same direction
        # We go through each of the 4 edge points and check for line intersection with all the walls in the room
        for i,p1 in enumerate(four_points):
            for line in Dic_rooms[room_index].values():
                p2 = Point(line[0][0][0],line[0][0][1])
                p3 = Point(line[0][1][0],line[0][1][1])
                if intersect(p,p1,p2,p3)==True:
                    # The point intersecteded with a line in the given direction
                    intersected_line +=1

            # Plotting the cases with more than 2 intersections
            #if intersected_line >2:
            # if p.x == 133 and p.y == 89:
            #     print("K_new", len(k_new))
            #     print("j",j)
            #     print("Dic_rooms_lines",Dic_rooms[room_index].values())
            #     print("p1",p1)
            #     print("p",p)
            #     print("intersected_line",intersected_line)
            #     fig, ax = plt.subplots()
            #     plot_grid(ax,x_min,x_max,y_min,y_max)
            #     for line in Dic_rooms[room_index].values():
            #         ax.plot([line[0][0][0], line[0][1][0]],[line[0][0][1], line[0][1][1]],'b')    
            #     plot_point(p)
            #     plot_point(p1)
            #     plt.show()

            # The node is outside the building
            if intersected_line %2 ==0:
                vote.append(0)
            else: #inside the building
                vote.append(1)
            intersected_line = 0
        # After checking all directions we use majority vote to figure out if inside or outside room
        # If not inside room we check the other rooms
        if sum(vote)>=3:
            vote = []
            # We are inside room
            node_type = at['att'][0]
            dist = at['att'][2]
            room_label = room_index
            # This is used when doing k means. It is usefull to have a list of all room indexes
            if room_index not in room_indexes:
                room_indexes.append(room_index)

            # We add a room label to the node
            if node_type == 'door':
                idx_d = at['att'][3]
                G_grid.add_node(node, att=(node_type,p,dist,room_label,idx_d))
            else:
                G_grid.add_node(node, att=(node_type,p,dist,room_label))
            break
        # If none of the rooms belong to the node, we remove the node since it means it is outside the building
        if j==len(k_new)-1:
            G_grid.remove_node(node)

#print(room_indexes)
#test = [at['att'] for node,at in G_grid.nodes(data=True)]# if at['att'][3]==1]

#print("test", test)
#print("len dic rooms", len(Dic_rooms))
#print("len dic rooms edges",len(Dic_rooms_edges))
#print("len room indexes", len(room_indexes))
print("PART 5")

## Doing K means clustering
room_nodes = []
cent_ratio = 50 # For every 100 nodes we have 1 room node
points_rooms = []
points_rooms_dic = collections.defaultdict(list)
#print("Dic_rooms",len(Dic_rooms))
# Iterating over all labeled rooms. We are not iterating over all rooms since some
# of the rooms might not be labeled. 
#print(len(Dic_rooms))
#print(len(room_indexes))
#print(range(len(room_indexes)-1))
#for i in range(13):
    #print(i)

#print(len(room_indexes))
for i in range(len(room_indexes)):
    # We find all the points in the room that are not doors
    points_temp = [at['att'][1] for node,at in G_grid.nodes(data=True) if (at['att'][3]==room_indexes[i] and at['att'][0]!='door')]
    num_of_nodes = len(points_temp)
    # If there are no nodes in the room
    #print("num of nodes",num_of_nodes)
    if room_indexes[i]==8:
        print("points_temp",points_temp)
    if num_of_nodes == 0:
        continue
    # Number of centroids, 1 in every cent_ratio
    num_cent = math.ceil(len(points_temp)/cent_ratio)
    centroids = []
    centroid_dict = collections.defaultdict(list)
    # Generating random integer to indicate which node should be centroids
    for j in range(num_cent):
        cent_idx = random.randint(0,num_of_nodes-1)
        centroids.append(points_temp[cent_idx])
    delta = 0.6 # This indicates when we stop the K means algorithm
    counter = 0

    while counter < 5:#delta>0.5:
        #Associating each node with the nearest centroid
        for p in points_temp:
            # Checking the distance to each centroid
            dist = []
            for centroid in centroids:
                p1 = centroid
                dist.append(round(distance.euclidean([p.x,p.y],[p1.x,p1.y]),2))
            # Finding the index of the closest centroid
            centr_idx = dist.index(min(dist))
            # Adding the point to the closest centroid
            centroid_dict[centr_idx].append(p)
        # Calculating new centroid
        centroids_new = []
        for centroid_idx in range(len(centroids)):
            points_centroid = centroid_dict[centroid_idx]
            # If there are no points associated with the centroid we skip it
            if len(points_centroid)==0:
                continue
            # Finding the avg of the points which will be the new centroid
            avg_x = round(np.mean([p.x for p in points_centroid]),2)
            avg_y = round(np.mean([p.y for p in points_centroid]),2)
            avg_point = Point(avg_x,avg_y)
            avg_point.round_to_half()


            if avg_point not in points_temp:
                # If there are no centroids we choose a random node in the room as centroid
                #if len(centroids_new)==0:
                #print("centroids_new",centroids_new)
                points_temp_idx = random.randint(0,len(points_temp)-1)
                #print("points_temp_idx",points_temp_idx)
                #print("len points temp",len(points_temp))
                avg_point = points_temp[points_temp_idx]
            centroids_new.append(avg_point)
        centroids = copy.deepcopy(centroids_new)
        #if room_indexes[i] == 85:
            #print("centroids for room 85", centroids)
        # Resetting which nodes are associated to which centroid
        centroid_dict = collections.defaultdict(list)
        counter +=1
        # If we only have 1 centroid it will converge after one iteration
        if len(centroids)==1:
            counter = 6
    # Adding the centroids to the correct room
    #points_rooms.extend(centroids)
    points_rooms_dic[i] = centroids
    if room_indexes[i]==8:
        print("centroid room 8", points_rooms_dic[i])

#print("points_rooms_dic",points_rooms_dic)
print("PART 6")
# Connecting all the room nodes to the overall graph
grid_len = len(G_grid)
j = 0
for room_label,room_points in points_rooms_dic.items():
    for p in room_points:
        j = j+1
        node_and_at = [[node,at] for node,at in G_grid.nodes(data=True) if at['att'][1]==p]
        # This is the case if the point is not in the grid. This is probably because room point is outside the building
        if not node_and_at:
            continue
        node = node_and_at[0][0]
        at = node_and_at[0][1]
        dist = at['att'][2]
        G_grid.add_node(node,att = ("room",p,dist,room_label))
t1_stop = process_time()
#print(t1_stop - t1_start)



t1_start = process_time()
# Connecting all doors that are opposite from eachother. This is because we want connection between the doors outisde and inside a room.
for node,at in sorted(G_grid.nodes(data=True)):
    # First door
    node_type = at['att'][0]
    if node_type == 'door':
        #p = at['att'][1]
        idx = at['att'][4]
        # Finding the corresponding door
        for node1,at1 in sorted(G_grid.nodes(data=True)):
            node_type1 = at1['att'][0]
            if node_type1 == 'door':
                idx1 = at1['att'][4]
                if node == node1 or idx1 != idx:
                    continue
                #if idx == 57:
                #    print ("hej")
                G_grid.add_edge(node,node1, weight = 10)
                break
t1_stop = process_time() 

#print("len grid", len(G_grid.nodes))


# For all doors connect them to a node in their respective room.
for node,at in sorted(G_grid.nodes(data=True)):
    node_type = at['att'][0]
    if node_type != "door":
        continue
    num_edges = G_grid.edges(node)
    #print("num edges", len(num_edges))
    # This is the case when the door is only connected to its opposite door, and to no other nodes in the room
    # In this case we find the room it belongs to and connect it to one of the room nodes of the room.
    #if len(num_edges) <=1:
    #print("hej")
    #print("len idx nodes", len(list(idx_nodes)))
    room_num_door = at['att'][3]
    door_idx = at['att'][4]
    p = at['att'][1]
    # Find the nearest nodes to the door
    nearest_nodes= list(idx_nodes.nearest((p.x,p.x, p.y, p.y), 1000000))#en(G_grid.nodes)))
    if node == 2071:
        print("nearest nodes",len(nearest_nodes))
    # For all the nearest node the first one that is in the same room as the door will be connected to the door
    for node1 in nearest_nodes:
        #if node == 5471:
            
            #print(node1)
        # The nearest node will be itself, and therefore we skip this node
        if node == node1:
            continue
        # Check if the node exists in the grid graph, this is because it might be a node that has been
        # removed from the grid graph
        if not G_grid.has_node(node1):
            # if node == 5471:
            #     print(node1)
            continue
        room_num_node = G_grid.nodes[node1]['att'][3]
        # If the grid node is not in the same room as the door node find another node
        if room_num_door != room_num_node:
            continue
        # Doing a line intersection check here, 
        # such that if it is a node in the same room but the node is obstructed by a wall
        # we find another node
        #if node == 5471:
        #    print(node1)
        p1 = G_grid.nodes[node1]['att'][1]
        if not is_traversable(p,p1,Dic_rooms[room_num_door]):
            #print("hej")
            continue
        # If there is a wall find a new node.
        #if not connection_to_room:
        #    continue
        G_grid.add_edge(node,node1,weight=14)
        point_node = G_grid.nodes[node1]['att'][1]
        break

# Debugging
for node, at in G_grid.nodes(data=True):
    node_type = at['att'][0]
    if node == 3740:
        print("point 3740", at['att'][1])
    if node_type == 'door':
        p_door = at['att'][1]
        idx_d = at['att'][4]
        #if 1.9 < p_door.x <2:# and 16<p_door.y<17:  
        #if node == 5485:
        if node == 2071:
        #if idx_d == 57:
            node_edges = G_grid.edges(node)
            print("node_edges",node_edges)
            print("node", node)
            print("room_label", at['att'][3])
            #node2 = 4511
            #point2 = G_grid.nodes[node2]['att'][1]
            #room_label = G_grid.nodes[node2]['att'][3]
            #print("room_label",room_label)
            #print("point", point2)


#node_edges = G_grid.edges(4511)
#print(node_edges)



print("PART 7")
####### Approximate solution to the TSP problem #######
G = G_grid.copy()
test = [at['att'] for node,at in G.nodes(data=True) if(at['att'][1].x ==4.5 and at['att'][1].y ==2)]# if at['att'][3]==1]
#print(G.edges[4])
#test = [at['att'] for node,at in G.edges(data=True if(at['att'][1].x ==5.0 and at['att'][1].y ==2.5)]
#print(test)
#Finding shortest path between all room nodes using the astar algorithm
# Get all the traversable nodes in the points_all_incl_trav
# Then calculate the entire walkable path for all the connected nodes while you also check for traversability.
# This is not the fastest method since it needs to calculate the distance between all nodes. 


# Find the shortest path between all room nodes using the a star algoritm
astar_path_dic = rec_dd()
t1_start = process_time()
G_rooms = nx.Graph()
for node,at in sorted(G.nodes(data=True)):
    if at['att'][0]!= "room":
        continue
    #print("ny G_rooms node")
    #Make complete subgraph of all room nodes
    #Make a new graph only including the room nodes called G_rooms
    G_rooms.add_node(node,att = at['att'])  
   # identify the essential nodes in the graph
   # Checking nodes are traversable to node p

 
Dic_connectivity = collections.defaultdict()
# First check for connectivity between all room nodes
from networkx.algorithms import approximation as approx
for node,at in sorted(G_rooms.nodes(data=True)):
    Dic_connectivity[node] = 0
    for node1,at1 in sorted(G_rooms.nodes(data=True)):
        if node==node1:
            continue
        # This functions checks how many nodes need to be removed to disconnect 2 nodes
        connectivity = approx.local_node_connectivity(G, node, node1)
        #if connectivity == 0 and at['att'][3]==at1['att'][]

        if connectivity > 0:
            connectivity = 1
        Dic_connectivity[node] += connectivity
# Removing room nodes that are not connected to other room nodes, also change its description from roon node to grid node in G


#print(G.nodes)
#print(Dic_connectivity)
max_connection = max(Dic_connectivity.values())
#print(len(Dic_connectivity))




#### TESTING
# fig, ax = plt.subplots()
# plot_grid(ax,x_min,x_max,y_min,y_max)

# for line in Dic_all.values():
#    ax.plot([line[0][0][0], line[0][1][0]],[line[0][0][1], line[0][1][1]],'b')    


# for node,at in sorted(G_rooms.nodes(data=True)):
#     p = at['att'][1]
#     node_type = at['att'][0]
#     print(node,p)
#     if node_type == "room":
#         plt.plot(p.x,p.y,marker='o',color='red',markerfacecolor='red')

# plt.show()

    #elif node_type == 'door':
    #    plt.plot(p.x,p.y,marker='o',color='black',markerfacecolor='black')




# This section is for the rare case that there are 2 identical max connections, for example two isolated rooms of the same size.
number_of_nodes = 0
list_of_keys = []
for key,value in Dic_connectivity.items():
    if value == max_connection:
        number_of_nodes += 1
        list_of_keys.append(key)
# if the number of nodes are bigger than the number of max connections we remove the n biggest nodes with n being the max connection.
# Since we have clusters of bigger nodes.
if number_of_nodes > max_connection+1:
    #print("hej")
    list_of_keys = sorted(list_of_keys)[:len(list_of_keys)-(max_connection+1)]
    for key in list_of_keys:
        G_rooms.remove_node(key)
        p = G.nodes[key]['att'][1]
        dist = G.nodes[key]['att'][2]
        room_label = G.nodes[key]['att'][3]
        G.remove_node(key)
        G.add_node(key,att=("grid",p,dist,room_label))

# This is the normal case where we dont have to large clusters of equal size nodes that are unconnected.
# This is performed no matter the case.
for key,value in Dic_connectivity.items():
    if value == max_connection:
        continue
    G_rooms.remove_node(key)
    p = G.nodes[key]['att'][1]
    dist = G.nodes[key]['att'][2]
    room_label = G.nodes[key]['att'][3]
    G.remove_node(key)
    G.add_node(key,att=("grid",p,dist,room_label))


for node,at in sorted(G_rooms.nodes(data=True)):
    for node1,at1 in sorted(G_rooms.nodes(data=True)):
        if node==node1:
            continue
        p = at['att'][1]
        q = at1['att'][1]
        #dist = round(distance.euclidean([p.x,p.y],[q.x,q.y]),2)
        temp = nx.astar_path(G, node, node1, weight = 'weight')# heuristic=distance.euclidean, weight='weight')
        astar_path_dic[node][node1] = temp



# Getting the edge data between all connected nodes in the a star path.
# Getting the distance of the shortest path between each room node
astar_path_length_dic = rec_dd()
for room_node1 in astar_path_dic:
    for room_node2 in astar_path_dic[room_node1]:
        dist = 0
        for i,node_path in enumerate(astar_path_dic[room_node1][room_node2]):
            if i==0:
                node_start = node_path
                continue
            node_dest = node_path
            dist = dist + G.get_edge_data(node_start,node_dest)['weight']
            node_start = node_path
        G_rooms.add_edge(room_node1,room_node2,weight=dist)


#Checking if the graph is indeed complete
# print("Is G graph connected? Returns 1 if graph is complete", nx.density(G))
# print("Is G_rooms graph connected? Returns 1 if graph is complete", nx.density(G_rooms))


print("PART 8")
# #Make a minimum spanning tree
mst_G_rooms=nx.minimum_spanning_tree(G_rooms)

source_node = random.choice(list(G_rooms.nodes))

# Solve the TSP problem for subgraph using DFS traversal
dfs_edges_list = list(nx.dfs_edges(mst_G_rooms,source=source_node))


# Remove double vertices
tsp_edges = []
for i in range(len(dfs_edges_list)):
   if i== 0:
       # This is the top node of the dfs traversal
       node_pair = dfs_edges_list[i]
   elif dfs_edges_list[i-1][1]!=dfs_edges_list[i][0]:
       # If the 'to node' in the previous node pair (from node, to node)
       # is not the same as the from node in the next node pair then the new 
       # 'from node' should be the previous 'to node' and the new to node 
       # should be the current 'to node'.
       # For example if we have (1,2), (1,3) then because we go back to 1
       # we change it to (1,2), (2,3).
       node_pair = tuple([dfs_edges_list[i-1][1],dfs_edges_list[i][1]])
   else:
      node_pair = dfs_edges_list[i] 
   tsp_edges.append(node_pair)
#Adding the last edge from end to start node in tsp edges       
tsp_edges.append(tuple([tsp_edges[-1][1],tsp_edges[0][0]]))

print("PART 9")
# ###### PLOTTING #######
# ####Plotting the TSP solution for room and other nodes
# #Make new figure with the floorplan
fig, ax = plt.subplots()
plot_grid(ax,x_min,x_max,y_min,y_max)

for line in Dic_all.values():
   ax.plot([line[0][0][0], line[0][1][0]],[line[0][0][1], line[0][1][1]],'b')
   #ax.arrow(line[0][0][0], line[0][1][0],line[0][0][1]-line[0][0][0], line[0][1][1]-line[0][1][0])    

#Plotting all the nodes of the graph on the map
for node,at in sorted(G.nodes(data=True)):
    p = at['att'][1]
    node_type = at['att'][0]
    if node_type == "room":
        plt.plot(p.x,p.y,marker='o',color='red',markerfacecolor='red')
    elif node_type == 'door':
        plt.plot(p.x,p.y,marker='o',color='black',markerfacecolor='black')
    #else:
    #    plt.plot(p.x,p.y,marker='o',color='black',markerfacecolor='black')


# #Plotting all the nodes with their room labels of the graph on the map
# color_list = ['k','b','y','g','r']
# for node,at in sorted(G_grid.nodes(data=True)):
#     p = at['att'][1]
#     node_type = at['att'][0]
#     room = at['att'][3]
#     color = color_list[room%5]
#     plt.plot(p.x,p.y,marker='o',color=color,markerfacecolor=color)

print("starting node", source_node)
#Plotting starting node
p = G_rooms.nodes(data=True)[source_node]['att'][1]
plot_point(p,door=False,starting_node = True)
#Making a mapping from node values to index values
from scipy.interpolate import interp1d
min_node = min(list(G_rooms.nodes))
max_node = max(list(G_rooms.nodes))
range_val = max_node-min_node
m = interp1d([min_node,max_node],[0,range_val])

# Then we plot the edges of the tsp approximate solution
for edge in tsp_edges:
    start_node = edge[0]
    end_node = edge[1]
    pred = astar_path_dic[start_node][end_node]
    node_prev = end_node
    #loop that plots line from end node to all its predecessors until it reaches start node
    count=len(pred)
    while node_prev != edge[0]:
        node_new = node_prev
        #node_prev = pred[node_prev][0]
        node_prev = pred[count-1]
        p = G.nodes(data=True)[node_new]['att'][1]
        q = G.nodes(data=True)[node_prev]['att'][1]
        #plt.plot([p.x,q.x],[p.y,q.y],'g')
        ax.arrow(p.x, p.y,q.x-p.x, q.y-p.y,shape='full', length_includes_head=True, head_width=.1, color='g')
        count = count-1
plt.show()
