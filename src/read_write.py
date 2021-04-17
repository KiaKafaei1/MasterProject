#Using the ElemenTree library to handle XML data
import xml.etree.ElementTree as ET
import csv


#GETTING COORDINATES FOR ROOMS
bd127 = '127/641299'
bdBig = 'rac' 
bdHospital = 'NNH/Hospital'
bdNy = 'Ny_byg'
bdNy2 = 'Ny_byg2'
path ='WayFinding/'+bdNy2
#path = bdHospital
room_path = path+'/rooms.xml'
tree = ET.parse(room_path)
rooms_coord = []

elements = tree.findall('.Room/Triangles')
for i in elements:
    rooms_coord.append({'Coords': i.attrib['Coords']})

#print(rooms_coord)

# Writing to csv file
fields = ['Coords']
filename = "room_coordinates.csv"

with open(filename,'w') as csvfile:
    writer = csv.DictWriter(csvfile,fieldnames = fields)
    writer.writeheader()
    writer.writerows(rooms_coord)

#-------------

# Getting the min and max elevations
elements = tree.findall('.Room')
room_elevation =[]
for i in elements:
    #room_elevation.append({'MinElevation': i.attrib['MinElevation']})
    #room_elevation.append({'MaxElevation': i.attrib['MaxElevation']})
    room_elevation.append([i.attrib['MinElevation'],i.attrib['MaxElevation']])
    #room_elevation.append(i.attrib['MaxElevation'])


#fields = ['MinElevation','MaxElevation']
filename = "room_elevation.csv"
with open(filename,'w') as csvfile:
    #writer = csv.DictWriter(csvfile)
    writer = csv.writer(csvfile)
    #csvfile.writeheader()
    writer.writerows(room_elevation)
    #writer.writerows(max_elevation)

#------------
room_number = []
for i in elements:
    room_number.append([i.attrib["Number"]])


filename = "room_number.csv"
with open(filename,'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(room_number)








## GETTING COORDINATES FOR DOORS
door_path = path+'/floorplaninfo.xml'
tree = ET.parse(door_path)
doors_coord = []
elements = tree.findall('.elements/door/definition')
for i in elements:
    doors_coord.append([i.attrib['origin']])#,i.attrib['facing']])
    doors_coord.append([i.attrib['facing']])
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
            doors_coord.append([line])




#fields = ['Coords']
filename = "door_coordinates.csv"

with open(filename,'w') as csvfile:
    writer = csv.writer(csvfile)#,fieldnames = fields)
    #writer.writeheader()
    writer.writerows(doors_coord)

#-----
# Getting the min and max elevation of the doors.
elements = tree.findall('.elements/door')
doors_elevation = []
for i in elements:
   doors_elevation.append([i.attrib['minheight'], i.attrib['maxheight']])
   #doors_elevation.append(i.attrib['maxheight'])

filename = "doors_elevation.csv"
with open(filename,'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(doors_elevation)
