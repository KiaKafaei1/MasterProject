#Using the ElemenTree library to handle XML data
import xml.etree.ElementTree as ET
import csv

path ='WayFinding/127/641299/'
tree = ET.parse(path+'rooms.xml')
root = tree.getroot()
rooms_cord = []
#Getting the coordinates by looping through each room
for child in root:
    for grandchild in child:
        if(grandchild.tag=='Triangles'): 
            rooms_cord.append(grandchild.attrib)




#Writing to CSV file. Used this for help: https://www.geeksforgeeks.org/working-csv-files-python/
fields = ['Coords']
filename = "room_coordinates.csv"

with open(filename,'w') as csvfile:
    writer = csv.DictWriter(csvfile,fieldnames = fields)
    writer.writeheader()
    writer.writerows(rooms_cord)








#coord =[]
#for dic in rooms_cord:
#    coord.append(dic['Coords'])
#print(coord)
#







#with open('data.b00','rb') as reader:j
    #Further file processing goes here
 #   print(reader.readline(5))


