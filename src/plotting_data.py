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
    #print(Dic_lines.values())
    #print("size of dict",len(Dic_lines))
    for line in Dic_lines.values():
        #print(line)
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


def plot_grid(ax,x_min,x_max,y_min,y_max):
    ax.set_xlim(roundDown10(x_min),roundUp10(x_max))
    ax.set_ylim(roundDown10(y_min),roundUp10(y_max))
    #ax.set_xlim((x_min),(x_max))
    #ax.set_ylim((y_min),(y_max))
    # Change major ticks 
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(20))


    # Change minor ticks to show every 5 (20/4 = 5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(40))
    ax.yaxis.set_minor_locator(AutoMinorLocator(40))
    ax.grid(which = 'minor')
    plt.grid()

def rec_dd():
    return collections.defaultdict(rec_dd)

### Implementing spatial datastrukture for rooms ###
p = index.Property()
idx_rooms = index.Index(interleaved=False)
# Making a dictionary for the rectangles, such that they can later be retrieved when wanting to branch out
rectangles_rooms = collections.defaultdict(list)
Dic_rooms = rec_dd()

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

#7
#plotting triangles
tempx=[]
tempy=[]
Dic =collections.defaultdict(list) 
#fig, ax = plt.subplots()
Dic_all = collections.defaultdict(list)
i=0
for room in array_tri:
    #Not including every room to simplify the drawing
    if room_127:
        if(i>4): break 
    else:
        #ojoj=1
        if(i>35): break 
    for tri in room:
        for cor in tri:
            tempx.append(cor[0])
            tempy.append(cor[1])
        # Hashcing each line in both directions and adding the hashes with their corresponding line to a Dict
        Dic = dict_hashing(Dic,tempx,tempy)
        tempx = []
        tempy = []

    # Removing lines that are reoccuring from the Dictionary
    # Removing small circles and small unusefull lines
    for x in list(Dic):
        if len(Dic[x])>1 or distance.euclidean(Dic[x][0][0],Dic[x][0][1])<0.1:
            Dic.pop(x)            
#8        
    # Manually removing lines    
    for x in list(Dic):
        coord1 = Dic[x][0][0]
        coord2 = Dic[x][0][1]
            #Dic.pop(x)
        if (123.82<coord1[0]<123.86 and 95.6<coord1[1]<95.75) or (123.82<coord2[0]<123.86 and 95.6<coord2[1]<95.75): 
            Dic.pop(x) 
        if (117<coord1[0]<118 and 95<coord1[1]<95.5) or (117<coord2[0]<118 and 95<coord2[1]<95.5): 
            Dic.pop(x)
        if (114<coord1[0]<115 and 98.5<coord1[1]<99) or (114<coord2[0]<115 and 98.5<coord2[1]<99):
            Dic.pop(x)
        if (123.75<coord1[0]<123.95 and 63.6<coord1[1]<63.67) or (123.75<coord2[0]<123.95 and 63.6<coord2[1]<63.67): 
            Dic.pop(x)     
#9
    # All lines are plotted twice due to the way we are hashing, it is more appropriate to remove replicate values that are in more than 1 hash.
    # Remove replicate lines  
    #print("length of dic before",len(Dic.keys()))
    temp = Dic.copy()
    visited_lines = []
    for key,line in temp.items():
        for key2,line2 in temp.items():
            #print(line)
            #print(line[0][0])
            #print("udenfor")
            if line[0][0] == line2[0][1] and line[0][1] == line2[0][0] and line not in visited_lines and line2 not in visited_lines:
                #print("indenfor")
                #print(line[0])
                #print(line2[0])
                Dic.pop(key2)
                visited_lines.append(line)
                break
    #print("length of dic after", len(Dic.keys()))


    Dic_all.update(Dic)
    # Plotting the lines
    #for line in Dic.values():
    #    ax.plot([line[0][0][0], line[0][1][0]],[line[0][0][1], line[0][1][1]],'b')
    #Dic_all_unhashed = collections.defaultdict(list)
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
                p_x_max = p1[0]
            if p[0]<p_x_min:
                p_x_min = p1[0]
            if p[1]>p_y_max:
                p_y_max=p[1]
            if p[1]<p_y_min:
                p_y_min = p[1]

    bottom = p_y_min
    left = p_x_min
    top = p_y_max
    right = p_x_max
    idx_rooms.insert(i,(left,right,bottom,top),obj = room)
    rectangles_rooms[i] = [left,right,bottom,top]

    Dic_rooms[i] = Dic
    Dic = collections.defaultdict(list)
    i+=1

### Plotting the rooms from the R tree datastructure
# fig, ax = plt.subplots()
# for room in rectangles_rooms.values():
#     plt.plot([room[0],room[0],room[1],room[1],room[0]],[room[2],room[3],room[3],room[2],room[2]])

#plt.show()   



#10
### Plotting Doors
### Plotting using CSV####
df = pd.read_csv('door_coordinates.csv',sep=',', header=None)
#print(df)
array = df.to_numpy()
array = np.delete(array,0)
array = cop.cor_processing(array,room=0)
#print(array)

#This is the translation needed to make doors align with floorplan
translation =  [array[-1][i] for i in (0,2)]

# Remove the translation from the array of door coordinates
array = array[:-1]
i=0
points_doors = []
facing_doors = []


for i,cor in enumerate(array):
    if i%2 == 0:
        points_doors.append(Point(cor[0][0]-translation[0][0],cor[1][0]+translation[1][0]))
    if i%2 == 1:
        facing_doors.append([cor[0],cor[1]])




# Checking if the point is within range of wall, if not the door is not usefull and should therefore be removed
#for i in range(len(points_doors)):
#    p1 = points_doors[i]

# placing a door on the other side of the wall
temp_points = []#points_doors.copy()
points_doors_opposite = []

for i,door in enumerate(points_doors):
    #print(facing_doors[i])
    if facing_doors[i][0][0] == -1:
        temp_points.append(Point(door.x-0.5,door.y))
        points_doors_opposite.append(Point(door.x+0.5,door.y))
    if facing_doors[i][0][0] == 1:
        temp_points.append(Point(door.x+0.5,door.y))
        points_doors_opposite.append(Point(door.x-0.5,door.y)) 
    if facing_doors[i][1][0] == -1:
        temp_points.append(Point(door.x,door.y-0.5))
        points_doors_opposite.append(Point(door.x,door.y+0.5))
    if facing_doors[i][1][0] == 1:
        temp_points.append(Point(door.x,door.y+0.5))
        points_doors_opposite.append(Point(door.x,door.y-0.5))

    
points_doors = temp_points.copy()





# Making the room points
# First I do it manually
points_rooms = [Point(165,56),Point(154,63),Point(168,54),Point(110,56),Point(120,72),Point(110,85),Point(110,104),Point(120,93),Point(110,84),Point(135,103),Point(126,60),Point(118,92),Point(141,57)]


# Now I try to autogenerate them
points_rooms = []
for room in Dic_rooms.values():
    p_x_max = 0
    p_y_max = 0
    p_x_min = 1000
    p_y_min = 1000
    for line in room.values():
        p1 = line[0][0]
        p2 = line[0][1]
        for p in (p1,p2):
            if p[0]>p_x_max:
                p_x_max = p1[0]
            if p[0]<p_x_min:
                p_x_min = p1[0]
            if p[1]>p_y_max:
                p_y_max=p[1]
            if p[1]<p_y_min:
                p_y_min = p[1]

    room_node = Point((p_x_max-p_x_min)/2+p_x_min,(p_y_max-p_y_min)/2+p_y_min)
    room_node.round_to_half()
    points_rooms.append(room_node)

#Some of the room nodes are identical since two overlapping rooms have identical centers
#print(len(points_rooms))




    # else:
    #    plot_point(p)

#points_corners = [Point(170.3,61),Point(170.3,60.14),Point(173.15,61.1),Point(129.3,61),Point(124,61),Point(117.9,61.1),Point(123.9,63.7),Point(123.2,69.2),Point(123.1,75.2),Point(123.1,85.8),Point(123.1,93.6),Point(123.9,95.7),Point(141.3,98.2),Point(137.8,100.8),Point(115.1,98.4),Point(120.5,58.7)]#,Point(124,61),Point(124,61),Point(124,61),Point(124,61),Point(124,61),Point(124,61),Point(124,61),Point(124,61),Point(124,61)]

points_all = points_doors+points_rooms #+points_corners



###### MAKING THE HIGH RESOLUTION GRID #######
# Finding the keys for the largest and smallest x and y values from each point in each line
x_max_idx1 = max(Dic_all, key=lambda key: Dic_all[key][0][1][0])
x_max_idx2 = max(Dic_all, key=lambda key: Dic_all[key][0][0][0])
x_min_idx1 = min(Dic_all, key=lambda key: Dic_all[key][0][1][0])
x_min_idx2 = min(Dic_all, key=lambda key: Dic_all[key][0][0][0])

y_max_idx1 = max(Dic_all, key=lambda key: Dic_all[key][0][0][1])
y_max_idx2 = max(Dic_all, key=lambda key: Dic_all[key][0][1][1])
y_min_idx1 = min(Dic_all, key=lambda key: Dic_all[key][0][0][1])
y_min_idx2 = min(Dic_all, key=lambda key: Dic_all[key][0][1][1])


# Getting the corresponding max and min values
x_max = math.ceil(max([Dic_all[x_max_idx1][0][0][0],Dic_all[x_max_idx2][0][1][0]]))
x_min = math.floor(min([Dic_all[x_min_idx1][0][0][0],Dic_all[x_min_idx2][0][1][0]]))
y_max = math.ceil(max([Dic_all[y_max_idx1][0][0][1],Dic_all[y_max_idx2][0][1][1]]))
y_min = math.floor(min([Dic_all[y_min_idx1][0][0][1],Dic_all[y_min_idx2][0][1][1]]))

# fig, ax = plt.subplots()
# plot_grid(ax,x_min,x_max,y_min,y_max)

# for line in Dic_all.values():
#    ax.plot([line[0][0][0], line[0][1][0]],[line[0][0][1], line[0][1][1]],'b')    

# #Plotting all the nodes of the graph on the map
# for point in points_rooms:
#     plot_point(point)
# for point in points_doors:
#     plot_point(point,False)
# for point in points_doors_opposite:
#     plot_point(point,False)

# plt.show()


######## IMPLEMENTING THE DATASTRUCTURE FOR THE WALLS ########
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
for id in idx.intersection((106,145,52,75)):
    [left,right,bottom,top] = rectangles[id]
    left_branch.insert(id, (left,right,bottom,top))

for id in idx.intersection((145,174,75,107)):
    [left,right,bottom,top] = rectangles[id]
    right_branch.insert(id,(left,right,bottom,top))


# Grid height is used when making the grid nodes
grid_height = len(np.linspace(y_min,y_max,(y_max-y_min)*2+1))


#points_rooms = [Point(165,56),Point(154,63),Point(168,54),Point(110,56),Point(120,72),Point(110,85),Point(110,104),Point(120,93),Point(110,84),Point(135,103),Point(126,60),Point(118,92),Point(141,57)]




# Keeping the original door coordinates
points_doors_discrete = copy.deepcopy(points_doors)
points_doors_opposite_discrete = copy.deepcopy(points_doors_opposite)


# Rounding the discrete doors to nearest 0.5 since that is the resolution of the grid.
[p.round_to_half() for p in points_doors_discrete]
[p.round_to_half() for p in points_doors_opposite_discrete]

# Doing the same for the room nodes
[p.round_to_half() for p in points_rooms]

points_all = points_doors+points_rooms+ points_doors_opposite #+points_corners
#Plotting the points


# #### Removing doors that lead to the outside
# # I try to do this manually first
# temp = points_doors.copy()
# for i,p in enumerate(temp):
#     if p.x ==144.5 and p.y==100.5:
#         points_doors.remove(p)
#     if p.x == 173.5 and p.y ==60:
#         points_doors.remove(p)

#     # if p.x >144 and p.x <145 and p.y>100 and p.y<101:
#     #     points_doors.pop(i)
#     # if p.x >173 and p.x <174 and p.y>59 and p.y<61:
#     #     points_doors.pop(i)

    
# temp = points_doors_opposite.copy()
# for i,p in enumerate(temp):
#     if p.x ==144.5 and p.y==100.5:
#         points_doors_opposite.remove(p)
#     if p.x == 173.5 and p.y ==60:
#         points_doors_opposite.remove(p)
    #if p.x >144 and p.x <145 and p.y>100 and p.y<101:
     #   points_doors_opposite.pop(i)
    #if p.x >173 and p.x <174 and p.y>59 and p.y<61:
      #  points_doors_opposite.pop(i)








G_grid = nx.Graph()
i = 0
counter_room = 0 
for x in np.linspace(x_min,x_max,(x_max-x_min)*2+1):
    for y in np.linspace(y_min,y_max,(y_max-y_min)*2+1):
        i = i+1
        p = Point(x,y)
        k = list(idx.nearest((p.x,p.x, p.y, p.y), 1))
        k = max(k) #If multiple walls are close we just choose a random wall (the wall with the highest index)
        line = Dic_all_unhashed.get(k)
        p1 = line[0][0]
        p2 = line[0][1]
        dist = point_line_dist(p1[0],p1[1],p2[0],p2[1],p.x,p.y)
        

        #Removing floating doors by removing all doors that are far away from the nearest wall
        # We use the information that length of the lists are the same and each index corresponds to opposite points.
        if p in points_doors_discrete:# and dist<1.1: 
            idx_d = points_doors_discrete.index(p)
            p_float = points_doors[idx_d]
            dist = point_line_dist(p1[0],p1[1],p2[0],p2[1],p_float.x,p_float.y)
            G_grid.add_node(i,att =("door",p_float,dist,idx_d))
        elif p in points_doors_opposite_discrete: # and dist<1.1:
            idx_d = points_doors_opposite_discrete.index(p)
            p_float = points_doors_opposite[idx_d]
            dist = point_line_dist(p1[0],p1[1],p2[0],p2[1],p_float.x,p_float.y)
            G_grid.add_node(i,att = ("door",p_float,dist,idx_d))


        elif p in points_rooms:
            G_grid.add_node(i,att=("room",p,dist))
        else:
            G_grid.add_node(i,att =("grid",p,dist))       

       

        # Not adding edges to and from nodes that are too close to the wall
        # This is not possible because then the other nodes will have weird edges to and from eachother
        # THe only reliable way is to add the edges and then remove them again.

        # Vertical and horizontal neighbours
        if x>x_min:
            #Making sure that the nodes are the length we think.
            grid_height = len(np.linspace(y_min,y_max,(y_max-y_min)*2+1))
            #if len(G_grid.nodes)> grid_height:
            G_grid.add_edge(i,i-grid_height,weight=10)
        if y>y_min:
            G_grid.add_edge(i,i-1,weight=10)

        # Diagonal neighbours
        # Diagonal down left which is the same as up right
        if x>x_min and y>y_min:
            G_grid.add_edge(i,i-grid_height-1,weight=14)

        # Diagonal up left which is the same as right down
        if x<x_max and y>y_min:
            G_grid.add_edge(i,i-grid_height+1,weight=14)




# Adressing a very weird bug where the graph would add these random nodes.
# print("number of room nodes detected", counter_room)
# print("number of room nodes that actually exist", len(points_rooms))
bug_nodes = list(range(0,-110,-1))
G_grid.remove_nodes_from(bug_nodes)



# Changing all door nodes that are floating to regular grid nodes.
# This might not be important after all since the program doesn't care if a point in the middle of the room is a door node or a grid node
# This is only visible when plotting.
G_grid_temp = G_grid.copy()
for node,at in sorted(G_grid_temp.nodes(data=True)):
    node_type = at['att'][0]
    if node_type != 'door':
        continue
    dist = at['att'][2]
    if dist >2:
        p = at['att'][1]
        idx_d = at['att'][3]
        G_grid.add_node(node,att=("grid",p,dist))



# # Removing doors connected to the outside, this is done using if statements.
# G_grid_cpy = G_grid.copy()
# for node,at in sorted(G_grid.nodes(data=True)):
#     node_type = at['att'][0]
#     if node_type == "door":
#         p = at['att'][1]
#         if p.x == 144.5 and p.y == 100.5:
#             dist = at['att'][2]
#             #G_grid.remove_node(node)
#             G_grid.add_node(node,att=("grid",p,dist))
#         elif p.x == 173.5 and p.y == 60:
#             dist = at['att'][2]
#             #G_grid.remove_node(node)
#             G_grid.add_node(node,att=("grid",p,dist))


## Removing doors connected to the outside, using the line intersection algorithm
# If the door is inside the building it will interesect with wall lines in all four directions. If it does not intersect with walls 
# in a given direction it means that the door is outside the building. And it should therefore be removed.

inside_building = 0
#G_grid_cpy = G_grid.copy()
for node,at in sorted(G_grid.nodes(data=True)):
    node_type = at['att'][0]
    if node_type == "door":
        p = at['att'][1]
        four_points = [Point(p.x,y_max),Point(p.x,y_min),Point(x_max,p.y),Point(x_min,p.y)]
        for p1 in four_points:
            for line in Dic_all.values():
                p2 = Point(line[0][0][0],line[0][0][1])
                p3 = Point(line[0][1][0],line[0][1][1])
                if intersect(p,p1,p2,p3)==True:
                    #print("lol")
                    # The point intersecteded with a line in the given direction, we therefore move on to the next direction
                    inside_building = 1
                    break
            if inside_building==1:
                inside_building=0
                continue
            # The door node didn't intersect with any lines in the given direction meaning it is outside the building.
            dist = at['att'][2]
            G_grid.add_node(node,att=("grid",p,dist))
            #inside_building = 1
            break
        


                    







# Remove non traversable nodes
removable_edge_list = []
G_grid_cpy = G_grid.copy()
for node,at in sorted(G_grid.nodes(data=True)):
   dist = at['att'][2]
   point = at['att'][1]
   node_type = at['att'][0]

   # We remove all the edges connected to and from the nodes that are too close to a wall
   if dist<0.5:
    if node_type == "door":
        continue

    #removable_edge_list.append(G_grid.edges(node))
    
    #removable_edge_list = G_grid.edges(node)
    #G_grid_cpy.remove_edges_from(removable_edge_list)
    G_grid_cpy.remove_node(node)
    #print(removable_edge_list)
 








G_grid = G_grid_cpy.copy()
t1_start = process_time()
# Connecting all doors that are opposite from eachother. This is because we want connection between the doors outisde and inside a room.
for node,at in sorted(G_grid.nodes(data=True)):
    node_type = at['att'][0]
    if node_type == 'door':

        #p = at['att'][1]
        idx = at['att'][3]
        #print(idx)
        for node1,at1 in sorted(G_grid.nodes(data=True)):
            node_type1 = at1['att'][0]
            

            if node_type1 == 'door':
                idx1 = at1['att'][3]
                #print(idx1)
                if node == node1 or idx1 != idx:
                    continue

                #print("First door",at)
                #print("Second door",at1)
                G_grid.add_edge(node,node1, weight = 10)
                break


t1_stop = process_time() 







####### Approximate solution to the TSP problem #######

G = G_grid.copy()


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
    #Make complete subgraph of all room nodes
    #Make a new graph only including the room nodes called G_rooms
    G_rooms.add_node(node,att = at['att'])  

   # identify the essential nodes in the graph
   # Checking nodes are traversable to node p

# #Checking for descendants for each room node   
# for node,at in sorted(G_rooms.nodes(data=True)):
#     for node1,at1 in sorted(G_rooms.nodes(data=True)):
#         pass
#         #nx.algorithms.descendants(, target_nodes)

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
        if connectivity > 0:
            connectivity = 1

        Dic_connectivity[node] += connectivity

#print(G.nodes[501]['att'][0])

# Removing room nodes that are not connected to other room nodes, also change its description from roon node to grid node in G
#print(Dic_connectivity)
max_connection = max(Dic_connectivity.values())
#print(Dic_connectivity)
print(len(Dic_connectivity))
for key,value in Dic_connectivity.items():
    if value == max_connection:
        continue
    G_rooms.remove_node(key)
    p = G.nodes[key]['att'][1]
    dist = G.nodes[key]['att'][2]
    G.remove_node(key)
    G.add_node(key,att=("grid",p,dist))


 


for node,at in sorted(G_rooms.nodes(data=True)):
    for node1,at1 in sorted(G_rooms.nodes(data=True)):
        # if at1['att'][0]!= "room":
        #     continue
        if node==node1:
            continue

        p = at['att'][1]
        q = at1['att'][1]


        #dist = round(distance.euclidean([p.x,p.y],[q.x,q.y]),2)

        temp = nx.astar_path(G, node, node1, weight = 'weight')# heuristic=distance.euclidean, weight='weight')

        astar_path_dic[node][node1] = temp



# t1_stop = process_time() 

# #print(astar_path_dic)
      
# print("Elapsed time for checking traversability:", t1_stop-t1_start)




# Getting the edge data between all connected nodes in the a star path.
# Getting the distance of the shortest path between each room node
astar_path_length_dic = rec_dd()
for room_node1 in astar_path_dic:
    #print("The key is: ", key)
    for room_node2 in astar_path_dic[room_node1]:
        # print("This is the destination node ", room_node2)
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

#Make a minimum spanning tree
mst_G_rooms=nx.minimum_spanning_tree(G_rooms)

source_node = random.choice(list(G_rooms.nodes))


# #Make a minimum spanning tree
mst_G_rooms=nx.minimum_spanning_tree(G_rooms)

source_node = random.choice(list(G_rooms.nodes))
source_node = 1141
# #source_node = 189

# Solve the TSP problem for subgraph using DFS traversal
dfs_edges_list = list(nx.dfs_edges(mst_G_rooms,source=source_node))


#print("The DFS traversal before removing double vertices:")
#print(dfs_edges_list)

# Remove double vertices
tsp_edges = []
for i in range(len(dfs_edges_list)):
   #print(dfs_edges_list[i])
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
   #print(node_pair)
   tsp_edges.append(node_pair)
#Adding the last edge from end to start node in tsp edges       
tsp_edges.append(tuple([tsp_edges[-1][1],tsp_edges[0][0]]))




###### PLOTTING #######
####Plotting the TSP solution for room and other nodes
#Make new figure with the floorplan
fig, ax = plt.subplots()
plot_grid(ax,x_min,x_max,y_min,y_max)

for line in Dic_all.values():
   ax.plot([line[0][0][0], line[0][1][0]],[line[0][0][1], line[0][1][1]],'b')    

#Plotting all the nodes of the graph on the map
for node,at in sorted(G.nodes(data=True)):
    p = at['att'][1]
    node_type = at['att'][0]
    if node_type == "room":
        plot_point(p,False)
    elif node_type == 'door':
        plot_point(p)
    # else:
    #     plot_point(p)

#print(points_rooms)
for p in points_rooms:
    plot_point(p,False)

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

    #print("start node: ", start_node)
    #print("end node: ", end_node)
    while node_prev != edge[0]:
        node_new = node_prev
        #node_prev = pred[node_prev][0]
        node_prev = pred[count-1]
        p = G.nodes(data=True)[node_new]['att'][1]
        q = G.nodes(data=True)[node_prev]['att'][1]
        plt.plot([p.x,q.x],[p.y,q.y],'g')

        count = count-1

plt.show()
