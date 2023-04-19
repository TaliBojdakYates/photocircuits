import cv2
import numpy as np
from ultralytics import YOLO
import time
import cv2
import random
from numberDetection import detect_numbers
import math
from boxOverlap import remove_boxes_inside
# Load a model
model = YOLO()
model = YOLO("runs/detect/train2/weights/best.pt")

# model = YOLO("runs\detect/train7/weights/best.pt")
# 3not all nodes,6 too many noise,31 still the no good area problem
# problem solved but check   10 same node seperated,14 nodes, 16 nodes and noise, 21 too many nodes for one, 25 nodes think shape no closed,  28 nodes same as 25,
# skinny lines 11,18, 25
image = "circuitTest2.png"
# image = "draw2.jpg"
# skinny lines 11,18, 37,41,46,48
# noisy 6, 31, 34
# image = "data/imgs/circuit6.png"

results = model.predict(source=image)  # can also put ,save=True
results = results[0].boxes
boxesOriginal = results.xyxy.tolist() # this holds the bounding box coordinates
classes = results.cls.tolist() # this holds the classes for the boxes

boxes, removed_indices = remove_boxes_inside(boxesOriginal, 0.85)
removed_indices.sort(reverse=True)

# remove elements from the list based on indices
for i in removed_indices:
    del classes[i]
    

def assignUnit(componentClass, number):
    
    unit = ''
    
    if (componentClass == 0):
        # This is a voltage source
        if not number[-1].isalpha():
            unit = 'V'
        else:
            prefix = number[-2] if len(number) >= 2 and number[-2].isalpha() else number[-1]
            prefix = prefix.lower()
            if prefix == 'u':
                unit = 'u_V'
            elif prefix == 'm':
                unit = 'm_V'
            elif prefix == 'v':
                unit = 'V'
            elif prefix == 'k':
                unit = 'k_V'
            elif prefix == 'm':
                unit = 'M_V'
                
    elif(componentClass == 1):
        if not number[-1].isalpha():
            unit = 'ohm'
        else:
            prefix = number[-2] if len(number) >= 2 and number[-2].isalpha() else number[-1]
            prefix = prefix.lower()
            if prefix == 'u':
                unit = 'uohm'
            elif prefix == 'm':
                unit = 'mohm'
            elif prefix == 'k':
                unit = 'kohm'
            elif prefix == 'm':
                unit = 'Mohm'
    elif(componentClass == 2):
        #current source
        pass

    if(unit == ''):
        prefix = number[-2:].lower() if len(number) >= 2 and number[-2:].isalpha() else number[-1].lower()
        prefix = prefix.lower()
        if prefix == 'uv':
                unit = 'u_V'
        elif prefix == 'mv':
            unit = 'm_V'
        elif prefix == 'v':
            unit = 'V'
        elif prefix == 'kv':
            unit = 'k_V'
        elif prefix == 'mv':
            unit = 'M_V'
        
   
    return unit
    


centers = []
componentID = 0
x=0
for box in boxes:
    x1, y1, x2, y2 = box
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    centers.append([componentID,(x_center, y_center), classes[x]])
    componentID+=1
    x +=1
numbers = values = detect_numbers(image)

components = []  # list to hold component information



# iterate through numbers of components
print(numbers)
for number in numbers:
    closest_center = None
    closest_distance = math.inf
    
    # iterate through centers to find closest one
    tempID = None
    tempClass = None
    for i in range(len(centers)):
        unitValue = assignUnit(classes[i],number[0])
        distance = math.sqrt((centers[i][1][0] - number[1][0]) ** 2 + (centers[i][1][1] - number[1][1]) ** 2)
        if distance < closest_distance:
            closest_distance = distance
            closest_center = centers[i][1]
            tempID = centers[i][0]
            tempClass = centers[i][2]
            
        
    
    components.append([tempID, number[0],unitValue,tempClass])

print(components)




img = cv2.imread(image)
imgOrg = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (7,7), 1)
# cv2.imshow("cor",img)
(thresh, blackAndWhiteImage) = cv2.threshold(img, 225, 255, cv2.THRESH_BINARY)# was lower 245
# cv2.imshow("blk",blackAndWhiteImage)
img = cv2.cvtColor(blackAndWhiteImage, cv2.COLOR_GRAY2RGB)

for idx in range(len(boxes)):
    box = boxes[idx]
    print(classes[idx])
    if classes[idx] == 0 or classes[idx] == 3:
        topCorner = [int(box[0])-2,int(box[1])-2]
        bottomCorner = [int(box[2]+2),int(box[3])+2]

        boxes[idx][0] = int(box[0]) - 2.0
        boxes[idx][1] = int(box[1]) - 2.0
        boxes[idx][2] = int(box[2]) + 2.0
        boxes[idx][3] = int(box[3]) + 2.0
    else:
        topCorner = [int(box[0]), int(box[1])]
        bottomCorner = [int(box[2]), int(box[3])]

    cv2.rectangle(img, topCorner, bottomCorner, color=(255, 255, 255), thickness=-1)
    cv2.rectangle(imgOrg, topCorner, bottomCorner, color=(0, 0, 255), thickness=1)

# cv2.imshow("yolo",imgOrg)
# cv2.waitKey(0)

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# cv2.imshow("hsv",img)
# Threshold of blk in HSV space
lower_blk = np.array([0,0,0])
upper_blk = np.array([360,255,80])
# upper_blk = np.array([170,260,260])

# mask, blur, canny
img = cv2.inRange(img, lower_blk, upper_blk)
# cv2.imshow("mask", img)
img = cv2.GaussianBlur(img, (7,7), 1)
# cv2.imshow("blur", img)
# img = cv2.Canny(img, 200, 200)
img = cv2.Canny(img, 20, 250)
cnts, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
counter = 0


countours = []
for cnt in cnts:
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    print(counter, area, int(peri))
    if area > 50 or peri > 500:
        color = (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))
        cv2.putText(imgOrg, f'n:{counter}', (int(cnt[0][0][0]), int(cnt[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, .7, color)
        # print(counter, colors[counter])
        cv2.fillPoly(imgOrg, pts=[cnt], color=color)
        # peri = cv2.arcLength(cnt,True)
        # approx = cv2.approxPolyDP(cnt, .02*peri, True)
        # objCor = len(approx)
        # x , y , w, h = cv2.boundingRect(approx)
        counter += 1
        countours.append(cnt.tolist())
    # else:
    #     cv2.fillPoly(imgOrg, pts=[cnt], color=(128, 0, 128))


    # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    # cv2.putText(imgOrg, f'n:{counter}', (int(cnt[0][0][0]), int(cnt[0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, .7,
    #             color)
    # cv2.fillPoly(imgOrg, pts=[cnt], color=color)
    # counter += 1
    # countours.append(cnt.tolist())

# cv2.imshow("canny", img)
# cv2.imshow("out1", imgOrg)
# cv2.waitKey(0)

conectionDict = {}
biggestNode = 0


# loop through the features
for featureInd in range(len(classes)):
    print(classes)
    coor = boxes[featureInd]
    
    if classes[featureInd] == 0:
        print("voltage", featureInd)
        cv2.putText(imgOrg, f"f{featureInd}", (int(coor[0]), int(coor[3])), cv2.FONT_HERSHEY_SIMPLEX, .7, 255)
    elif classes[featureInd] == 1:
        print("res", featureInd)
        cv2.putText(imgOrg, f"f{featureInd}", (int(coor[0]), int(coor[3])), cv2.FONT_HERSHEY_SIMPLEX, .7, 255)
    elif classes[featureInd] == 3:
        print("curr", featureInd)
        cv2.putText(imgOrg, f"f{featureInd}", (int(coor[0]), int(coor[3])), cv2.FONT_HERSHEY_SIMPLEX, .7, 255)

    # then loop through the nodes
    for node in range(len(countours)):
        cnt = countours[node]
        # print(cnt)
        # point in this format [[x y]]
        for point in cnt:
            # if a point in the node is within the x and within the y
            if int(coor[0])-3 < point[0][0] and point[0][0] < int(coor[2])+3 and int(coor[1])-3 < point[0][1] and point[0][1] < int(coor[3])+3:
               
                # print(point[0][0] - (int(coor[0])-5), point[0][1] - (int(coor[1])-5), (int(coor[2])+5) -point[0][0], (int(coor[3])+5)-point[0][1])
                
                # adding to dictionary
                if featureInd in conectionDict.keys() and node not in conectionDict[featureInd]:
                    conectionDict[featureInd].append(node)
                    
                elif featureInd not in conectionDict.keys():
                    conectionDict[featureInd] = [node,]
                    
                
                # keeping track of lowest node used
                if node > biggestNode:
                    biggestNode = node
        
print(conectionDict)
print(biggestNode)

cv2.imshow("out1", imgOrg)
cv2.waitKey(0)


from pyspice import solve
solve(conectionDict,components)



# todo need a way to determine the closest line to a feature
# todo gets confused when more symbols on the circuit, when too much noise

# todo for user
# make sure some white space between circuit and edge
# best if done on plain light background
# remove extra symbols, just values of the components