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
    #print(Dic)
    # Plotting the lines
    #for line in Dic.values():
    #    ax.plot([line[0][0][0], line[0][1][0]],[line[0][0][1], line[0][1][1]],'b')    
    Dic = collections.defaultdict(list)
    i+=1

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
for i,door in enumerate(points_doors):
    #print(facing_doors[i])
    if facing_doors[i][0][0] == -1:
        temp_points.append(Point(door.x-0.5,door.y))
        temp_points.append(Point(door.x+0.5,door.y))
    if facing_doors[i][0][0] == 1:
        temp_points.append(Point(door.x+0.5,door.y))
        temp_points.append(Point(door.x-0.5,door.y)) 
    if facing_doors[i][1][0] == -1:
        temp_points.append(Point(door.x,door.y-0.5))
        temp_points.append(Point(door.x,door.y+0.5))
    if facing_doors[i][1][0] == 1:
        temp_points.append(Point(door.x,door.y+0.5))
        temp_points.append(Point(door.x,door.y-0.5))
    #print(facing_doors[i])
    
points_doors = temp_points.copy()


# Doing some work on the doors
# Remove one of the doors that are within certain radius of another door
# This is to cut down on the number of doors
#temp_points =  points_doors.copy()
#print("Before: ",len(temp_points))
#for i in range(len(points_doors)):
#    p1 = points_doors[i]
#    if p1 not in temp_points:
#        continue
#    for j in range(len(points_doors)):
#            if i ==j:
#                continue
#            p2 = points_doors[j]
#            if math.sqrt((p2.x-p1.x)**2+(p2.y-p1.y)**2)<4:
#            #if -0.001<(p1.x-p2.x)<0.001 and -1<(p1.y-p2.y)<1: 
#                if p2 in temp_points:
#                    temp_points.remove(p2)
#print("After: ",len(temp_points))
#points_doors = temp_points.copy()            





#remove 2 outside doors
points_doors = [ p for p in points_doors if not (123<p.x<128 and 74<p.y<77)]

# remove all doors that are floating
temp_doors = points_doors.copy()
for p in temp_doors:
    if 112<p.x<116 and 79<p.y<81:
        points_doors.remove(p)
    if 106<p.x<108 and 81<p.y<83:
        points_doors.remove(p)
    if 112<p.x<114 and 91<p.y<94:
        points_doors.remove(p)
    if 114<p.x<117 and 93<p.y<94:
        points_doors.remove(p)
    if 114<p.x<117 and 94<p.y<95:
        points_doors.remove(p)
    if 120<p.x<140 and 99.5<p.y<106:
        points_doors.remove(p)


points_rooms = [Point(165,56),Point(154,63),Point(168,54),Point(110,56),Point(120,72),Point(110,85),Point(110,104),Point(120,93),Point(110,84),Point(135,103),Point(126,60),Point(118,92),Point(141,57)]

#points_corners = [Point(170.3,61),Point(170.3,60.14),Point(173.15,61.1),Point(129.3,61),Point(124,61),Point(117.9,61.1),Point(123.9,63.7),Point(123.2,69.2),Point(123.1,75.2),Point(123.1,85.8),Point(123.1,93.6),Point(123.9,95.7),Point(141.3,98.2),Point(137.8,100.8),Point(115.1,98.4),Point(120.5,58.7)]#,Point(124,61),Point(124,61),Point(124,61),Point(124,61),Point(124,61),Point(124,61),Point(124,61),Point(124,61),Point(124,61)]

points_all = points_doors+points_rooms #+points_corners

#Plotting the points
#for p in points_doors:
#   plot_point(p)
#for p in points_rooms:
#   plot_point(p,False)
# for p in points_corners:
#    plot_point(p)
#plt.show()

#for p in points_doors:
#    plot_point(p,door=True)
#plt.show()


#t1 = time.time()
#print("Time for section 1:", t1-t0)
#t0 = time.time()

## Creating the point lists for the door, room and corner points.
#if room_127:
#    points_doors = [Point(80.5,4.3), Point(60.8,9.9),Point(50.6,12.3),Point(69.1,19.7),Point(56.3,22.9),Point(42.6,25.8),Point(80.5,5.1),Point(60.8,10.7),Point(50.6,13.4),Point(68.0,19.7),Point(54.9,22.9),Point(43.4,25.8)]
#    points_rooms= [Point(48,21),Point(60,17),Point(76,13),Point(72,2),Point(46,10),Point(40,22),Point(36.2,12.5),Point(67.8,10.4),Point(84.9,16.1),Point(84.3,1.9),Point(64.1,12.6)]
#    points_corners = [Point(40,15.8),Point(60,2.5),Point(81,-3.2),Point(36.2,17.9)]
#    points_all = points_doors+points_rooms+points_corners
#
#else: 


# Finding the largest and smallest points in the dictionary which will be used as 
# limits for the x axis and y axis



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


#print(Dic_all[938979898160025640])
#print(Dic_all[938979898160025640][0][0][:][1])
#print(Dic_all[938979898160025640][0][0][0])

# Getting the corresponding max and min values
x_max = math.ceil(max([Dic_all[x_max_idx1][0][0][0],Dic_all[x_max_idx2][0][1][0]]))
x_min = math.floor(min([Dic_all[x_min_idx1][0][0][0],Dic_all[x_min_idx2][0][1][0]]))
y_max = math.ceil(max([Dic_all[y_max_idx1][0][0][1],Dic_all[y_max_idx2][0][1][1]]))
y_min = math.floor(min([Dic_all[y_min_idx1][0][0][1],Dic_all[y_min_idx2][0][1][1]]))




#plot_grid(ax,x_min,x_max,y_min,y_max)

# ax.set_xlim(roundDown10(x_min),roundUp10(x_max))
# ax.set_ylim(roundDown10(y_min),roundUp10(y_max))

# # Change major ticks 
# ax.xaxis.set_major_locator(MultipleLocator(20))
# ax.yaxis.set_major_locator(MultipleLocator(20))


# # Change minor ticks to show every 5 (20/4 = 5)
# ax.xaxis.set_minor_locator(AutoMinorLocator(40))
# ax.yaxis.set_minor_locator(AutoMinorLocator(40))
# ax.grid(which = 'minor')
# plt.grid()
#plt.show()






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

#print(len(list(idx.intersection((110,150,55,100)))))
#print(idx)

## Splitting the leaf nodes (indexes) up in branches
left_branch = index.Index(interleaved = False)
right_branch = index.Index(interleaved= False)
for id in idx.intersection((106,145,52,75)):
    [left,right,bottom,top] = rectangles[id]
    left_branch.insert(id, (left,right,bottom,top))

for id in idx.intersection((145,174,75,107)):
    [left,right,bottom,top] = rectangles[id]
    right_branch.insert(id,(left,right,bottom,top))


# Making the grid nodes.
# Making the new grid nodes
#grid_length = len(np.linspace(x_min,x_max,(x_max-x_min)*2+1))
grid_height = len(np.linspace(y_min,y_max,(y_max-y_min)*2+1))

#G_grid1 = nx.grid_2d_graph(grid_length,grid_height)


# G = G_grid.copy()
# points_all = points_rooms+points_doors
# i = len(G)-1
# for p in points_all:
#     i = i+1
#     # identify the essential nodes in the graph
#     if p in points_rooms:
#         G.add_node(i,att=("room", p))
#     else:
#         G.add_node(i,att=("door",p))

# Making the map as a grid 
# This is done by placing points at every cell center
#points_grid = []

points_rooms = [Point(165,56),Point(154,63),Point(168,54),Point(110,56),Point(120,72),Point(110,85),Point(110,104),Point(120,93),Point(110,84),Point(135,103),Point(126,60),Point(118,92),Point(141,57)]

#points_corners = [Point(170.3,61),Point(170.3,60.14),Point(173.15,61.1),Point(129.3,61),Point(124,61),Point(117.9,61.1),Point(123.9,63.7),Point(123.2,69.2),Point(123.1,75.2),Point(123.1,85.8),Point(123.1,93.6),Point(123.9,95.7),Point(141.3,98.2),Point(137.8,100.8),Point(115.1,98.4),Point(120.5,58.7)]#,Point(124,61),Point(124,61),Point(124,61),Point(124,61),Point(124,61),Point(124,61),Point(124,61),Point(124,61),Point(124,61)]

#test = Point(165,56)

#print(test in points_rooms)

#Plotting the points
#for p in points_doors:
#   plot_point(p)
#for p in points_rooms:
#   plot_point(p,False)


# Rounding the doors to nearest 0.5 since that is the resolution of the grid.
[p.round_to_half() for p in points_doors]
# Doing the same for the room nodes
[p.round_to_half() for p in points_rooms]

points_all = points_doors+points_rooms #+points_corners
#Plotting the points



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
        #print('Line',line)
        p1 = line[0][0]
        p2 = line[0][1]
        dist = point_line_dist(p1[0],p1[1],p2[0],p2[1],p.x,p.y)
        
        #if i>20:
        #   break
        #points_grid.append(Point(x,y))
        #print(i)


        if p in points_doors:
            G_grid.add_node(i,att =("door",p,dist))
        elif p in points_rooms:
            G_grid.add_node(i,att=("room",p,dist))
        else:
            G_grid.add_node(i,att =("grid",p,dist))
        #print(G_grid.number_of_nodes())
        #print(G_grid.nodes(data=True))       

        #print(G_grid.nodes)
        # Not adding edges to and from nodes that are too close to the wall
        # This is not possible because then the other nodes will have weird edges to and from eachother
        # THe only reliable way is to add the edges and then remove them again.

        # if dist<0.5:
        #     continue

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

# adressing a very weird bug where the graph would add these random nodes.

# print("number of room nodes detected", counter_room)
# print("number of room nodes that actually exist", len(points_rooms))
bug_nodes = list(range(0,-110,-1))
G_grid.remove_nodes_from(bug_nodes)

#temp = [node for node,at in sorted(G_grid.nodes(data=True))]
#print(temp)
#print(x_min)
#print(y_min)

# Remove non traversable nodes
removable_edge_list = []
G_grid_cpy = G_grid.copy()
for node,at in sorted(G_grid.nodes(data=True)):
   dist = at['att'][2]
   point = at['att'][1]
   node_type = at['att'][0]

   # We remove all the edges connected to and from the nodes that are too close to a wall
   if dist<0.5:
    if node_type == "door" or node_type == "room":
        continue
    #if node_type == "door"

    #removable_edge_list.append(G_grid.edges(node))
    removable_edge_list = G_grid.edges(node)
    G_grid_cpy.remove_edges_from(removable_edge_list)
    G_grid_cpy.remove_node(node)
    #print(removable_edge_list)
 

G_grid = G_grid_cpy.copy()
t1_start = process_time()
# Connecting all doors that are 1 m from eachother. This is because we want connection between the doors outisde and inside a room.
for node,at in sorted(G_grid.nodes(data=True)):
    node_type = at['att'][0]
    if node_type == 'door':
        p = at['att'][1]
        for node1,at1 in sorted(G_grid.nodes(data=True)):
            node_type1 = at['att'][0]

            if node_type1 == 'door':
                if node == node1:
                    continue
                q = at['att'][1]
                if distance.euclidean([p.x,p.y],[q.x,q.y])<1.1:
                    print(p)
                    print(q)
                    print(distance.euclidean([p.x,p.y],[q.x,q.y]))
                    G_grid.add_edge(node,node1, weight = 10)
                    break

t1_stop = process_time() 

#print(astar_path_dic)
      
print("Elapsed time for connecting doors:", t1_stop-t1_start)

#print(removable_edge_list[:200])

#G_grid_cpy = G_grid.copy()


# Plotting to see if all nodes are indeed reachable

# for p in points_doors:
#   plot_point(p)
# for p in points_rooms:
#   plot_point(p,False)

# Plotting all the nodes of the graph on the map
# for node,at in sorted(G_grid.nodes(data=True)):
#     p = at['att'][1]
#     node_type = at['att'][0]
#     if node_type == "room":
#         plot_point(p,False)
#     elif node_type == 'door':
#         plot_point(p,False)
    #else:
    #    plot_point(p)

#plt.show()
# # Debugging: Finding out where the points that are not discretize are from
# points_temp = []
# for node,at in sorted(G_grid.nodes(data=True)):
#     p = at['att'][1]
#     points_temp.append(p)
# print(points_temp)
#G_grid.remove_edges_from(removable_edge_list)


#G_grid.remove_edges_from([(x, y) for x, y, t in G_grid.edges.data('type') if t == 1])

#print(G_grid.edges.data())





        #nx.set_node_attributes(G_grid1,{}{})



# print(np.linspace(x_min,x_max,(x_max-x_min)*2+1))
# print(x_min)


# G_grid = nx.Graph()
# for i,p in enumerate(points_grid):
#     k = list(idx.nearest((p.x,p.x, p.y, p.y), 1))
#     #print(k)
#     k = max(k) #If multiple walls are close we just choose a random wall (the wall with the highest index)
#     #print(rectangles[k])
#     #print(p)
#     line = Dic_all_unhashed.get(k)
#     #print('Line',line)
#     p1 = line[0][0]
#     p2 = line[0][1]
#     dist = point_line_dist(p1[0],p1[1],p2[0],p2[1],p.x,p.y)
#     G_grid.add_node(i,att =("grid",p,dist))


# This is to visualize the border to the walls
# for node,at in sorted(G_grid.nodes(data=True)):
#     dist = at['att'][2]
#     point = at['att'][1]
    # if dist<0.5:
    #     plot_point(point)
#plt.show()


#print(Dic_all_unhashed)

# Plotting the points
#for p in points_grid:
#   plot_point(p,door=True,starting_node=False)




#### Compute distance to nearest walls. This is the brute force way of comparing every point to every wall to find nearest wall.
# Make a graph object with all the grid points as nodes
# t0 = time.time()
# #Compute distance to nearest walls. This is the brute force way of comparing every point to every wall to find nearest wall.
# all_point_wall_dist = []
# for i,p in enumerate(points_grid):
#     if i == 200:
#         break;
#     for line in Dic_all.values():
#         p1 = line[0][0]
#         p2 = line[0][1]
#         dist = point_line_dist(p1[0],p1[1],p2[0],p2[1],p.x,p.y)
#         all_point_wall_dist.append(dist)  
    
#     min_dist = min(all_point_wall_dist)
#     G_grid.add_node(i,att =("grid",p,min_dist))   

# t1 = time.time()
# #print("Time for section 4:", t1-t0)

# #Plot all the points that are close to the wall for visual purposes
# for node,at in sorted(G_grid.nodes(data=True)):
#     dist = at['att'][2]
#     point = at['att'][1]
#     if dist<0.5:
#         plot_point(point)
#plt.show()
#print("# lines: ",len(Dic_all))
#print("# gridpoints: ",len(points_grid))


## Having a list of traversable nodes and list of non traversable nodes.
#G_traversable = nx.Graph()
#G = nx.Graph()
#points_trav = []

#for node,at in sorted(G_grid.nodes(data=True)):
    #dist = at['att'][2]
    #point = at['att'][1]
    
#     if dist<0.5:
#     	continue
#     	#plot_point(point)
#     points_trav.append(point)


# for i,p in enumerate(points_trav):
#     G.add_node(i,att=("traversable",p))


# Distance function used for A star




####### Approximate solution to the TSP problem #######
#node_rooms = [node for node,at in sorted(G.nodes(data=True)) if at['att'][0]=="room"]
#points_trav = [at for node,at in sorted(G.nodes(data=True))]
#print(points_trav)
G = G_grid.copy()
# points_all = points_rooms+points_doors
# i = len(G)-1
# for p in points_all:
#     i = i+1
#     # identify the essential nodes in the graph
#     if p in points_rooms:
#         G.add_node(i,att=("room", p))
#     else:
#         G.add_node(i,att=("door",p))



#points_all = points_rooms+points_doors+points_trav


   # Checking nodes are traversable to node p

#Finding shortest path between all room nodes using the astar algorithm
# Get all the traversable nodes in the points_all_incl_trav
# Then calculate the entire walkable path for all the connected nodes while you also check for traversability.
# This is not the fastest method since it needs to calculate the distance between all nodes. 

def rec_dd():
    return collections.defaultdict(rec_dd)



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
    for node1,at1 in sorted(G.nodes(data=True)):
        if at1['att'][0]!= "room":
            continue
        if node==node1:
            continue

        p = at['att'][1]
        q = at1['att'][1]

        # if node == 897:
        #     if node1 == 953:
        #         print(p)
        #         print(q)

        dist = round(distance.euclidean([p.x,p.y],[q.x,q.y]),2)

        temp = nx.astar_path(G, node, node1, weight = 'weight')# heuristic=distance.euclidean, weight='weight')

        astar_path_dic[node][node1] = temp

        #G_rooms.add_edge(node,node1,weight = temp)


t1_stop = process_time() 

#print(astar_path_dic)
      
print("Elapsed time for checking traversability:", t1_stop-t1_start)


#print(astar_path_dic[897])
# print([at for node,at in G_rooms.nodes(data=True)])

#Make complete subgraph of all room nodes
#Make a new graph only including the room nodes called G_rooms
# G_rooms = nx.Graph()
# for node,at in G.nodes(data=True):
#     #print(at['att'][0])
#     if at['att'][0]=="room":
#         G_rooms.add_node(node,att = at['att'])  

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
            # print(temp.get("weight"))
            #print(temp)
            #temp2 = temp['weight']
            #print(temp)
            #print(G.get_edge_data(node_start,node_dest)['weight'])

            node_start = node_path

        #astar_path_length_dic[room_node1][room_node2] = dist
        G_rooms.add_edge(room_node1,room_node2,weight=dist)


#Checking if the graph is indeed complete
#print("Is G graph connected? Returns 1 if graph is complete", nx.density(G))
#print("Is G_rooms graph connected? Returns 1 if graph is complete", nx.density(G_rooms))

#Make a minimum spanning tree
mst_G_rooms=nx.minimum_spanning_tree(G_rooms)

source_node = random.choice(list(G_rooms.nodes))

 

            
            #print('with the nodes:',path)


# Compute the distance between each node in G_rooms using the dijk_distance
# for i,node in enumerate(sorted(G_rooms)):
#    for node2 in sorted(G_rooms):
#        if node==node2:
#            continue
#        #print(i)
#        #print(dijk_dist[i][node2])
#        distance= dijk_dist[i][node2]
#        G_rooms.add_edge(node,node2,weight = distance)


####### Approximate solution to the TSP problem #######

# Plotting the entire walkable path and making graph of connected nodes 
# with weighted edges (euclidean distance)
# G = nx.Graph()
# s = 0
# a = 0
# #print(Dic_all)
# for i,p in enumerate(points_all):
#    # This is used only for debugging, such that we dont have to run through all doors 
#    #if i>2:
#    #    break
#    # identify the essential nodes in the graph
#    if p in points_rooms:
#        G.add_node(i,att=("room", p)) #pos=(p.x,p.y))
#    else:
#        G.add_node(i,att=("other",p))
#    # Checking nodes are traversable to node p
#    for j,q in enumerate(points_all):
#        if p==q:
#            continue
#        #print(p)
#        #print(q)
#        #print("-------------------------")
#        #print(Dic_all)
#        is_walkable = is_traversable(p,q,Dic_all)

#        bool_dist_wall = True
#        if is_walkable:
#            s +=1
#            eucl_dist = round(distance.euclidean([p.x,p.y],[q.x,q.y]),2)
#            G.add_edge(i,j,weight=eucl_dist)
#            plt.plot([p.x,q.x],[p.y,q.y],'b')
#        else:
#            a+=1
# nodes_ordered = sorted(G.nodes())
# print("# walkable paths", s)
# print("# non walkable paths",a)


# Find shortest path between all "room" nodes using dijkstras algorithm
#node_rooms = [at for node,at in sorted(G.nodes(data=True))]
#print(node_rooms)
#print('att' in node_rooms[1])
# node_rooms = [node for node,at in sorted(G.nodes(data=True)) if at['att'][0]=="room"]
#dijk_dist = []
#dijk_pred = []
#node_rooms.sort()
# for node in node_rooms:
#    pred,distance = nx.dijkstra_predecessor_and_distance(G,node,weight='weight')
#    dijk_dist.append(distance)
#    dijk_pred.append(pred)

# #Make complete subgraph of all room nodes
# #Make a new graph only including the room nodes called G_rooms
# G_rooms = nx.Graph()
# for node,at in G.nodes(data=True):
#    if at['att'][0]=="room":
#        G_rooms.add_node(node,att = at['att']) 

# #print(dijk_dist)
# #print("pred: ")
# #print(pred)
# # Compute the distance between each node in G_rooms using the dijk_distance
# for i,node in enumerate(sorted(G_rooms)):
#    for node2 in sorted(G_rooms):
#        if node==node2:
#            continue
#        #print(i)
#        #print(dijk_dist[i][node2])
#        distance= dijk_dist[i][node2]
#        G_rooms.add_edge(node,node2,weight = distance)


# # Checking if the graph is indeed complete
# #print("Is G graph connected? Returns 1 if graph is complete", nx.density(G))
# #print("Is G_rooms graph connected? Returns 1 if graph is complete", nx.density(G_rooms))


# #Make a minimum spanning tree
mst_G_rooms=nx.minimum_spanning_tree(G_rooms)

source_node = random.choice(list(G_rooms.nodes))
# #source_node = 189

# Solve the TSP problem for subgraph using DFS traversal
dfs_edges_list = list(nx.dfs_edges(mst_G_rooms,source=source_node))



#print(list(tsp_tree.nodes()))
#print(list(nx.dfs_postorder_nodes(G_rooms)))

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

#print("The DFS traversal after removing double vertices:")
#print(tsp_edges)

#print('source node:', source_node)


###### PLOTTING #######

### Plotting only the "room" points and their TSP solution
#fig, ax = plt.subplots()
# for line in Dic.values():
#    ax.plot([line[0][0][0], line[0][1][0]],[line[0][0][1], line[0][1][1]],'b')


# # First we plot just the relevant points
# for p in points_rooms:
#    plot_point(p,False)
# # Plotting starting node
# p = G_rooms.nodes(data=True)[source_node]['att'][1]
# plot_point(p,door=False,starting_node = True)

# # Then we plot the edges
# for edge in tsp_edges:
#    #print(edge)
#    p = G_rooms.nodes(data=True)[edge[0]]['att'][1]
#    q = G_rooms.nodes(data=True)[edge[1]]['att'][1]
#    plt.plot([p.x,q.x],[p.y,q.y],'b')


####Plotting the TSP solution for room and other nodes
#Make new figure with the floorplan
fig, ax = plt.subplots()
plot_grid(ax,x_min,x_max,y_min,y_max)

for line in Dic_all.values():
   ax.plot([line[0][0][0], line[0][1][0]],[line[0][0][1], line[0][1][1]],'b')    

#Plotting all the nodes of the graph on the map
for node,at in sorted(G_grid.nodes(data=True)):
    p = at['att'][1]
    node_type = at['att'][0]
    if node_type == "room":
        plot_point(p,False)
    elif node_type == 'door':
        plot_point(p,False)
    else:
       plot_point(p)



# #First we plot all points
# for p in points_doors:
#    plot_point(p)
# for p in points_rooms:
#    plot_point(p,False)
# # #for p in points_corners:
# #  #  plot_point(p)

#Plotting starting node
p = G_rooms.nodes(data=True)[source_node]['att'][1]
plot_point(p,door=False,starting_node = True)


#Making a mapping from node values to index values
from scipy.interpolate import interp1d
min_node = min(list(G_rooms.nodes))
max_node = max(list(G_rooms.nodes))
range_val = max_node-min_node


m = interp1d([min_node,max_node],[0,range_val])
#fig, ax = plt.subplots()
#plot_grid(ax,x_min,x_max,y_min,y_max)

# Then we plot the edges of the tsp approximate solution
for edge in tsp_edges:
    #print(edge[0])
    #index = int(m(edge[0]))
    #pred = dijk_pred[index] # These are the predecessor nodes

    start_node = edge[0]
    end_node = edge[1]
    pred = astar_path_dic[start_node][end_node]
    #print(edge)
    #print(index)
    # pred = 
    node_prev = end_node
    #loop that plots line from end node to all its predecessors until it reaches start node
    count=0
    while node_prev != edge[0]:
         node_new = node_prev
         #node_prev = pred[node_prev][0]
         node_prev = pred[count]
         p = G.nodes(data=True)[node_new]['att'][1]
         q = G.nodes(data=True)[node_prev]['att'][1]
         plt.plot([p.x,q.x],[p.y,q.y],'b')

         count = count+1

plt.show()































#If we want to include the return to starting node
#p = q
#q = G_rooms.nodes(data=True)[source_node]['att'][1]
#plt.plot([p.x,q.x],[p.y,q.y],'b')



    #p = G_rooms.nodes(data=True)[edge[0]]['att'][1]
    #q = G_rooms.nodes(data=True)[edge[1]]['att'][1]
    #plt.plot([p.x,q.x],[p.y,q.y],'b')
#plt.show()

#for node,at in G_rooms.nodes(data=True):
#   for node2,at in G_rooms.nodes(data=True):
#       i=1
        #if
        #plt.plot([p.x,q.x],[p.y,q.y],'b') 
        #print(node,at)
        #print(node2,at)
        #if G_rooms.edges()


#print(G_rooms.edges())








# Solve the TSP problem using networkx's heuristic
#from dwave.system import DWaveSampler
#import dimod
#sampler = DWaveSampler()
#print(dnx.traveling_salesperson(G_rooms,dimod.ExactSolver(),start=12))



#T = nx.minimum_spanning_tree(G)
#print("minimum spanning tree", T)
#print(sorted(T.edges(data=True)))
#dfs = nx.dfs_edges(T)
##print("DFS", list(dfs))
#print(sorted(T.edges(data=True)))




######## SOLVING THE TSP PROBLEM ##############
#G = nx.Graph()
#for i,p in enumerate(points_all):
#    G.add_node(i,pos=(p.x,p.y))
#    G.add_edge(1,2,weight = 4.7)
#print(G.adj)
######## MINIMUM SPANNING TREEE ###############


##### Plotting Doors ####
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
#    plt.plot((cor[0][0]+75000)+117,cor[1][0]-153000-30,color='black',marker='o',markerfacecolor='black')
#    #plt.plot(cor[0][0]+75000,cor[1][0]-153000,color='black',marker='o',markerfacecolor='black')
#    if i>10:
#        break
#plt.show()



########### DOING THE MINIMUMSPANNING TREE ################
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






##################### SIMULATION ########################################


#class Ani():
#    def __init__(self,x_dest,y_dest, nsteps, line):
#        self.nsteps = nsteps
#        self.x_dest =x_dest
#        self.y_dest = y_dest
#        self.line=line
#        self.step = 0
#        self.i = 0
#    
#    def getdata(self,j):
#        for i in range(len(self.x_dest)):
#            x_goal = self.x_dest[i]
#            y_goal = self.y_dest[i]
#            if x_data[-1]>x_goal:
#                x_data.append(x_data[-1]-i)
#            elif x_data[-1]<x_goal:
#                x_data.append(x_data[-1]+i)
#            else:# x_data[-1]==x_goal:
#                x_data.append(x_data[-1])
#            
#            
#            if y_data[-1]>y_goal:
#                y_data.append(y_data[-1]-0.25*i)
#            elif y_data[-1]<y_goal:
#                y_data.append(y_data[-1]+0.25*i) 
#            else:# y_data[-1]==y_goal:
#                y_data.append(y_data[-1])
#
#
#        #t = np.arange(0,j)/float(self.nsteps)*2*np.pi
#        #x = np.sin(self.omegas[self.step]*t)
#        return t,x
#        
#    def gen(self):
#        for i in range(len(self.omegas)):
#            tit = u"animated sin(${:g}\cdot t$)".format(self.omegas[self.step])
#            self.line.axes.set_title(tit)
#            for j in range(self.nsteps):
#                yield j
#            self.step += 1
#            
#    def animate(self,j):
#        x,y = self.getdata(j)
#        self.line.set_data(x,y)
#
#fig, ax = plt.subplots()
#ax.axis([0,2*np.pi,-1,1])
#title = ax.set_title(u"")
#line, = ax.plot([],[], lw=3)
#
#
#
##omegas= [1,2,4,5]
#x_dest = [60,76]
#y_dest = [17,13]
#
#a = Ani(omegas,50,line)
#ani = FuncAnimation(fig,a.animate, a.gen, repeat=False, interval=60)
#plt.show()
#



# Doing the animation, moving from one point to another
##ax.set(xlim=(-0.1,2*np.pi+0-1),ylim = (-1.1,1.1))
#x_data = [48]
#y_data = [21]
#
#
#x =48
#y=21
#line, = ax.plot( 0,0)
#
#def animate(i):
#   # if y_data[-1]==y_goal and x_data[-1]==x_goal:
#   #     break
#    print("X_data: ", x_data[-1])
#    print("Y_data: ", y_data[-1],"\n")
#    if x_data[-1]>x_goal:
#        x_data.append(x_data[-1]-i)
#    elif x_data[-1]<x_goal:
#        x_data.append(x_data[-1]+i)
#    else:# x_data[-1]==x_goal:
#        x_data.append(x_data[-1])
#    
#    
#    if y_data[-1]>y_goal:
#        y_data.append(y_data[-1]-0.25*i)
#    elif y_data[-1]<y_goal:
#        y_data.append(y_data[-1]+0.25*i) 
#    else:# y_data[-1]==y_goal:
#        y_data.append(y_data[-1])
#    
#    line.set_xdata(x_data)
#    line.set_ydata(y_data)
#    return line,
#    
#
#
#
#
#
#
#x_points =[60, 76]
#y_points =[17, 13]
#for j in range(len(x_points)):
#    x_goal = x_points[j]
#    y_goal = y_points[j]
#    print("X_goal: ", x_goal)
#    print("Y_goal: ", y_goal)
#
#
#    animation = FuncAnimation(fig,func=animate, frames = np.ones(40), interval =30,repeat=False)
#    print("hej")
#    plt.show()
#
#





#### Plotting Doors
#### Plotting using CSV####
#df = pd.read_csv('door_coordinates.csv',sep=',', header=None)
#print(df)
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


#### Removing all unconnected lines
#    remove_indices = []
#    for x in list(Dic):
#        counter = 0
#        coord1 = Dic[x][0][0]
#        coord2 = Dic[x][0][1]
#        if 117<coord1[0]<117.5 or 117<coord2[0]<117.5:
#            k=1
#            print("x",x)
#            print("coord1: ",coord1)
#            print("coord2: ",coord2)
#        for y in list(Dic):
#            #print("counter", counter)
#            
#            if x==y:
#                continue
#            # The first coordinate of one line should be the last coordinate of another line
#            # We thereby remove all lines that are not connected to another line
#            if coord1 in Dic[y][0] or coord2 in Dic[y][0]:
#                counter+=1
#            if 117<coord1[0]<117.5 or 117<coord2[0]<117.5:
#                k=1
#                print("Dic[y][0] ",Dic[y][0])
#                print("counter: ",counter)
#                print("y:",y)
#            #print(coord1)
#            #print(Dic[y][0])
#            #print(coord1 in Dic[y][0])
#        if counter<3:
#            remove_indices.append(x)
#            if 117<coord1[0]<117.5 or 117<coord2[0]<117.5:
#                #print(remove_indices[-1])
#                k=1
#
#    #print("removed indices", remove_indices)
#    for x in remove_indices:
#        Dic.pop(x)
