
#Using the ElemenTree library to handle XML data
import xml.etree.ElementTree as ET
import csv


def get_root(path):
    tree = ET.parse(path)
    root = tree.getroot()
    return root


#GETTING COORDINATES FOR ROOMS
#path ='WayFinding/127/641300/'
path ='WayFinding/127/641299/rooms.xml'
root = get_root(path)
rooms_cord = []

#Getting the coordinates by looping through each room
for child in root:
    for grandchild in child:
        if(grandchild.tag=='Triangles'): 
            rooms_cord.append(grandchild.attrib)
#print(rooms_cord)
#Writing to CSV file. Used this for help: https://www.geeksforgeeks.org/working-csv-files-python/
fields = ['Coords']
filename = "room_coordinates.csv"

with open(filename,'w') as csvfile:
    writer = csv.DictWriter(csvfile,fieldnames = fields)
    writer.writeheader()
    writer.writerows(rooms_cord)







## GETTING COORDINATES FOR DOORS
path = 'WayFinding/127/641299/floorplaninfo.xml'
root = get_root(path)
door_cord = []

tree = ET.parse(path)
test = tree.findall('.elements/door/definition')
for i in test:
   door_cord.append(i.attrib['origin'])

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
    writer = csv.writer(csvfile)
#    writer.writeheader()
    writer.writerows(door_cord)


