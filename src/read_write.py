
#Using the ElemenTree library to handle XML data
import xml.etree.ElementTree as ET
import csv


#GETTING COORDINATES FOR ROOMS
#path ='WayFinding/127/641300/'
path ='WayFinding/127/641299/rooms.xml'
tree = ET.parse(path)
rooms_coord = []

elements = tree.findall('.Room/Triangles')
for i in elements:
    rooms_coord.append({'Coords': i.attrib['Coords']})
    #print("Room: ", type(i.attrib['Coords']))
#    print(i.attrib['Coords'])

#print( len(rooms_coord))
#Getting the coordinates by looping through each room
#for child in root:
#    for grandchild in child:
#        if(grandchild.tag=='Triangles'): 
#            rooms_cord.append(grandchild.attrib)



#print(rooms_cord)
#Writing to CSV file. Used this for help: https://www.geeksforgeeks.org/working-csv-files-python/
fields = ['Coords']
filename = "room_coordinates.csv"

with open(filename,'w') as csvfile:
    writer = csv.DictWriter(csvfile,fieldnames = fields)
    writer.writeheader()
    writer.writerows(rooms_coord)
#with open(filename,'w') as csvfile:
    #w = csv.writer(csvfile)
    #w.writerows(rooms_coord)
    #for line in rooms_coord:
    #    csvfile.write(line)
    #writer = csv.writer(csvfile,delimiter=',')
#    writer.writeheader()
    #writer.writerows(rooms_coord)



## GETTING COORDINATES FOR DOORS
path = 'WayFinding/127/641299/floorplaninfo.xml'
tree = ET.parse(path)
doors_coord = []
elements = tree.findall('.elements/door/definition')
for i in elements:
   doors_coord.append({'Coords': i.attrib['origin']})
   #print("Door: ", type(i.attrib['origin']))
#   print(i.attrib['origin'])
#print(door_cord)






#for child in root:
#    for grandchild in child:
#        for great_grandchild in grandchild:
#            if(great_grandchild
#            door_cord.append(great_grandchild.attrib['origin'])
#
#print(door_cord)
fields = ['Coords']
filename = "door_coordinates.csv"

with open(filename,'w') as csvfile:
    writer = csv.DictWriter(csvfile,fieldnames = fields)
    writer.writeheader()
    writer.writerows(doors_coord)






#with open(filename,'w') as csvfile:
#    writer = csv.writer(csvfile)
#    writer.writeheader()
#    writer.writerows(doors_coord)



