#Using the ElemenTree library to handle XML data
import xml.etree.ElementTree as ET
import csv


#GETTING COORDINATES FOR ROOMS
bd127 = '127/641299'
bdBig = 'rac' 
path ='WayFinding/'+bdBig
room_path = path+'/rooms.xml'
tree = ET.parse(room_path)
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
door_path = path+'/floorplaninfo.xml'
tree = ET.parse(door_path)
doors_coord = []
elements = tree.findall('.elements/door/definition')
for i in elements:
   doors_coord.append({'Coords': i.attrib['origin']})
#print(doors_coord)



## GETTING THE TRANSLATION OF DOOR COORDINATES
output_path = path+'/output.txt'
with open(output_path,'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for line in csv_reader:
        line = ''.join(line)
        text = line.split(' ',1)[0]
        if text == 'Translation:':
            line = line.split(' ',1)[1]
            doors_coord.append({'Coords': line})




fields = ['Coords']
filename = "door_coordinates.csv"

with open(filename,'w') as csvfile:
    writer = csv.DictWriter(csvfile,fieldnames = fields)
    writer.writeheader()
    writer.writerows(doors_coord)

