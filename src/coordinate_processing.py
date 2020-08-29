
# The coordinates are in string form and will have to be changed to float

# Splitting the strings in the array with spaces
def cor_processing(array):
    array_new = []
    for ele in array:
        array_new.append(ele.split(' '))
    
    
    #Splitting the strings with comma
    array_new2 = []
    ele_temp = []
    for ele in array_new:
        ele.pop()
        for pair in ele:
            ele_temp.append(pair.split(','))
        array_new2.append(ele_temp)
        ele_temp = []
    
    
    #From string to float
    array_new3 = []
    for ele in array_new2:
        for pair in ele:
            pair_temp=list(map(float, pair))
            ele_temp.append(pair_temp)
        array_new3.append(ele_temp)
        ele_temp = []
    
    # The array after processing the data
    array = array_new3
    
    
    #Putting 3 following coordinates together to make up triangles
    array_tri= []
    for ele in array:
        for i in range(0,len(ele)-1,3):
             ele_temp.append([ele[i],ele[i+1],ele[i+2]])
        array_tri.append(ele_temp)
        ele_temp = []
   
    return array_tri
