#Using the ElemenTree library to handle XML data
import xml.etree.ElementTree as ET
import csv


#GETTING COORDINATES FOR ROOMS
path ='WayFinding/127/641299/rooms.xml'
tree = ET.parse(path)
rooms_coord = []

elements = tree.findall('.Room/Triangles')
for i in elements:
    rooms_coord.append({'Coords': i.attrib['Coords']})



# Writing to csv file
fields = ['Coords']
filename = "room_coordinates.csv"

with open(filename,'w') as csvfile:
    writer = csv.DictWriter(csvfile,fieldnames = fields)
    writer.writeheader()
    writer.writerows(rooms_coord)


## GETTING COORDINATES FOR DOORS
path = 'WayFinding/127/641299/floorplaninfo.xml'
tree = ET.parse(path)
doors_coord = []
elements = tree.findall('.elements/door/definition')
for i in elements:
   doors_coord.append({'Coords': i.attrib['origin']})




fields = ['Coords']
filename = "door_coordinates.csv"

with open(filename,'w') as csvfile:
    writer = csv.DictWriter(csvfile,fieldnames = fields)
    writer.writeheader()
    writer.writerows(doors_coord)

