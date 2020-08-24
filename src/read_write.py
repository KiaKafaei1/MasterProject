import xml.etree.ElementTree as ET


path ='WayFinding/127/641299/'
tree = ET.parse(path+'rooms.xml')
root = tree.getroot()
print(root.tag)
rooms_cord = []
for child in root:
    for grandchild in child:
        if(grandchild.tag=='Triangles'): 
            #print(grandchild.attrib)
            rooms_cord.append(grandchild.attrib)
print(rooms_cord)
coord =[]
for dic in rooms_cord:
    coord.append(dic['Coords'])
print(coord)








#with open('data.b00','rb') as reader:j
    #Further file processing goes here
 #   print(reader.readline(5))


