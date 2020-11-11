import numpy as np
import math
import matplotlib.pyplot as plt
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

################# Defining functions and classes ###################

# We hash 6 lines because the lines may be in reverse coordinate order for different triangles, meaning that a line from x1,y1 to x2,y2 in one triangle is the line x2,y2 to x1,y1 in another triangle.
def dict_hashing(Dic,tempx,tempy):
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

#Dertimining line intersection 
# Code taken from https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
class Point:
	def __init__(self,x,y):
		self.x = x
		self.y = y
# Checking if 3 points are in counterclockwise order
def ccw(A,B,C):
	return (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)
# Finding if 4 points are intersecting. Se above link for explanation.
def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

# Function for plotting points
def plot_point(point,door=True,starting_node=False):
    if door:
        plt.plot(point.x,point.y,marker='o',color='black',markerfacecolor='black')
    elif starting_node:
        plt.plot(point.x,point.y,marker='o',color='green', markerfacecolor='green')
    else:
        plt.plot(point.x,point.y,marker='o',color='red', markerfacecolor='red')


# Function for plotting path 
def plot_path(p1,q1,Dic_lines):
    epsilon = 2
    # Function that returns true if there is a traversable connection between 2 points
    # else returns False

    # If this is true it means that it is 2 doors right next to eachother and the path should therefore be traversable
    if (math.sqrt((p1.x-q1.x)**2+(p1.y-q1.y)**2)<epsilon):
        return True

    for line in Dic.values():
        p2 = Point(line[0][0][0],line[0][0][1])
        q2 = Point(line[0][1][0],line[0][1][1])
        
        val= intersect(p1,q1,p2,q2)
        if val:
            #They do intersect
            break
    if not val:
        return True
    return False


##################### Plotting Rooms ###########################
#Importing values and changing from df to numpy array
df = pd.read_csv('room_coordinates.csv',sep=',', header=None)
room_127 = False
#room_big = False 
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
fig, ax = plt.subplots()

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
        
    # Manually removing lines    
    for x in list(Dic):
        coord1 = Dic[x][0][0]
        coord2 = Dic[x][0][1]
        if (123.8<coord1[0]<123.85 and 63.6<coord1[1]<63.65) or (123.8<coord2[0]<123.85 and 63.6<coord2[1]<63.65):
            Dic.pop(x)
        if (123.82<coord1[0]<123.86 and 95.6<coord1[1]<95.75) or (123.82<coord2[0]<123.86 and 95.6<coord2[1]<95.75): 
            Dic.pop(x)
        
        if (117<coord1[0]<118 and 95<coord1[1]<95.5) or (117<coord2[0]<118 and 95<coord2[1]<95.5): 
            Dic.pop(x)
        if (114<coord1[0]<115 and 98.5<coord1[1]<99) or (114<coord2[0]<115 and 98.5<coord2[1]<99):
            Dic.pop(x)
    

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



    

    # Plotting the lines
    for line in Dic.values():
        ax.plot([line[0][0][0], line[0][1][0]],[line[0][0][1], line[0][1][1]],'b')    
    Dic = collections.defaultdict(list)
    i+=1


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
    print(facing_doors[i])
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



# Place 2 doors on each side of the wall

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







for p in points_doors:
    plot_point(p,door=True)
plt.show()




#Removing unconnected lines



## Creating the point lists for the door, room and corner points.
#if room_127:
#    points_doors = [Point(80.5,4.3), Point(60.8,9.9),Point(50.6,12.3),Point(69.1,19.7),Point(56.3,22.9),Point(42.6,25.8),Point(80.5,5.1),Point(60.8,10.7),Point(50.6,13.4),Point(68.0,19.7),Point(54.9,22.9),Point(43.4,25.8)]
#    points_rooms= [Point(48,21),Point(60,17),Point(76,13),Point(72,2),Point(46,10),Point(40,22),Point(36.2,12.5),Point(67.8,10.4),Point(84.9,16.1),Point(84.3,1.9),Point(64.1,12.6)]
#    points_corners = [Point(40,15.8),Point(60,2.5),Point(81,-3.2),Point(36.2,17.9)]
#    points_all = points_doors+points_rooms+points_corners
#
#else: 
#points_rooms = [Point(165,56),Point(154,63),Point(168,54),Point(110,56),Point(120,72),Point(110,85),Point(110,104),Point(120,93),Point(110,84),Point(135,103),Point(126,60),Point(118,92),Point(141,57)]











##plt.show()
#    
##Plotting the points
#for p in points_doors:
#    plot_point(p)
#for p in points_rooms:
#    plot_point(p,False)
#for p in points_corners:
#    plot_point(p)
#
#
######## Approximate solution to the TSP problem #######
#
## Plotting the entire walkable path and making graph of connected nodes 
## with weighted edges (euclidean distance)
#G = nx.Graph()
#for i,p in enumerate(points_all):
#    # identify the essential nodes in the graph
#    if p in points_rooms:
#        G.add_node(i,att=("room", p)) #pos=(p.x,p.y))
#    else:
#        G.add_node(i,att=("other",p))
#    # Checking nodes are traversable to node p
#    for j,q in enumerate(points_all):
#        if p==q:
#            continue
#        is_walkable = plot_path(p,q,Dic)
#        if is_walkable:
#            eucl_dist = round(distance.euclidean([p.x,p.y],[q.x,q.y]),2)
#            G.add_edge(i,j,weight=eucl_dist)
#            plt.plot([p.x,q.x],[p.y,q.y],'b')
#nodes_ordered = sorted(G.nodes())
#
## Find shortest path between all "room" nodes using dijkstras algorithm
#node_rooms = [node for node,at in sorted(G.nodes(data=True)) if at['att'][0]=="room"]
#dijk_dist = []
#dijk_pred = []
##node_rooms.sort()
#for node in node_rooms:
#    pred,distance = nx.dijkstra_predecessor_and_distance(G,node,weight='weight')
#    dijk_dist.append(distance)
#    dijk_pred.append(pred)
#
#
##Make complete subgraph of all room nodes
##Make a new graph only including the room nodes called G_rooms
#G_rooms = nx.Graph()
#for node,at in G.nodes(data=True):
#    if at['att'][0]=="room":
#        G_rooms.add_node(node,att = at['att']) 
#
#
#
## Compute the distance between each node in G_rooms using the dijk_distance
#for i,node in enumerate(sorted(G_rooms)):
#    for node2 in sorted(G_rooms):
#        if node==node2:
#            continue
#        distance= dijk_dist[i][node2]
#        G_rooms.add_edge(node,node2,weight = distance)
#
#
## Checking if the graph is indeed complete
##print("Is G graph connected? Returns 1 if graph is complete", nx.density(G))
##print("Is G_rooms graph connected? Returns 1 if graph is complete", nx.density(G_rooms))
#
#
##Make a minimum spanning tree
#mst_G_rooms=nx.minimum_spanning_tree(G_rooms)
#source_node = 18
#
## Solve the TSP problem for subgraph using DFS traversal
#dfs_edges_list = list(nx.dfs_edges(mst_G_rooms,source=source_node))
#
#
#
##print(list(tsp_tree.nodes()))
##print(list(nx.dfs_postorder_nodes(G_rooms)))
#
##print("The DFS traversal before removing double vertices:")
##print(dfs_edges_list)
#
## Remove double vertices
#tsp_edges = []
#for i in range(len(dfs_edges_list)):
#    if i== 0:
#        node_pair = dfs_edges_list[i]
#    elif dfs_edges_list[i-1][1]!=dfs_edges_list[i][0]:
#        node_pair = tuple([dfs_edges_list[i-1][1],dfs_edges_list[i][1]])
#    else:
#       node_pair = dfs_edges_list[i] 
#    #print(node_pair)
#    tsp_edges.append(node_pair)
##Adding the last edge from end to start node in tsp edges       
#tsp_edges.append(tuple([tsp_edges[-1][1],tsp_edges[0][0]]))
#
##print("The DFS traversal after removing double vertices:")
##print(tsp_edges)
#
#
##### Plotting only the "room" points and their TSP solution
## Make new figure with the floorplan
#fig, ax = plt.subplots()
#for line in Dic.values():
#    ax.plot([line[0][0][0], line[0][1][0]],[line[0][0][1], line[0][1][1]],'b')    
#
## First we plot just the relevant points
#for p in points_rooms:
#    plot_point(p,False)
## Plotting starting node
#p = G_rooms.nodes(data=True)[source_node]['att'][1]
#plot_point(p,door=False,starting_node = True)
#
## Then we plot the edges
#for edge in tsp_edges:
#    p = G_rooms.nodes(data=True)[edge[0]]['att'][1]
#    q = G_rooms.nodes(data=True)[edge[1]]['att'][1]
#    plt.plot([p.x,q.x],[p.y,q.y],'b')
#
#
#####Plotting the TSP solution for room and other nodes
##Make new figure with the floorplan
#fig, ax = plt.subplots()
#for line in Dic.values():
#    ax.plot([line[0][0][0], line[0][1][0]],[line[0][0][1], line[0][1][1]],'b')    
#
##First we plot all points
#for p in points_doors:
#    plot_point(p)
#for p in points_rooms:
#    plot_point(p,False)
#for p in points_corners:
#    plot_point(p)
#
##Plotting starting node
#p = G_rooms.nodes(data=True)[source_node]['att'][1]
#plot_point(p,door=False,starting_node = True)
#
#
##Making a mapping from node values to index values
#from scipy.interpolate import interp1d
#m = interp1d([12,22],[0,10])
#
#
## Then we plot the edges of the tsp approximate solution
#for edge in tsp_edges:
#    index = int(m(edge[0]))
#    pred = dijk_pred[index] # These are the predecessor nodes
#    node_prev = edge[1]
#    #loop that plots line from end node to all its predecessors until it reaches start node
#    while node_prev != edge[0]:
#        node_new = node_prev
#        node_prev = pred[node_prev][0]
#        p = G.nodes(data=True)[node_new]['att'][1]
#        q = G.nodes(data=True)[node_prev]['att'][1]
#        plt.plot([p.x,q.x],[p.y,q.y],'b')
#plt.show()
#





























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
