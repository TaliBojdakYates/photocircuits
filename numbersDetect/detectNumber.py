from ultralytics import YOLO

model = YOLO()
model = YOLO("train5/weights/best.pt")

# model = YOLO("runs\detect/train7/weights/best.pt")
# 3not all nodes,6 too many noise,31 still the no good area problem
# problem solved but check   10 same node seperated,14 nodes, 16 nodes and noise, 21 too many nodes for one, 25 nodes think shape no closed,  28 nodes same as 25,
# skinny lines 11,18, 25
image = "34.png"
# image = "draw2.jpg"
# skinny lines 11,18, 37,41,46,48
# noisy 6, 31, 34
# image = "data/imgs/circuit6.png"

results = model.predict(source=image, save=True)  # can also put ,save=True

results = results[0].boxes
boxesOriginal = results.xyxy.tolist() # this holds the bounding box coordinates
classes = results.cls.tolist() # this holds the classes for the boxes

centers = []
i = 0
print(classes)
for box in boxesOriginal:
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    centers.append([center_x,center_y, classes[i]])
    i+=1
   
values = ['V','2', '1', '0', '3', 'I', 'A', '4', '6', '8', '7', '5', '9', 'k', 'M', '.', 'x', 'u', 'm','n']
print(centers)
# create a new array to store the mapped values



    
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def group_boxes(boxes):
    X = np.array(boxes)
    scores = []
    for n_clusters in range(2, len(boxes)):
        kmeans = KMeans(n_clusters=n_clusters, max_iter=1000, tol=1e-6, n_init=20)
        y_kmeans = kmeans.fit_predict(X)
        score = silhouette_score(X, y_kmeans)
        scores.append(score)
    best_n_clusters = np.argmax(scores) + 2
    print(f"Best number of clusters: {best_n_clusters}")
    kmeans = KMeans(n_clusters=best_n_clusters, max_iter=1000, tol=1e-6, n_init=20)
    y_kmeans = kmeans.fit_predict(X)
    plt.scatter(X[:,0], X[:,1], c=y_kmeans)
    plt.show()
    groups = [[] for i in range(best_n_clusters)]
    
    for i in range(len(y_kmeans)):
        label = y_kmeans[i]
        groups[label].append([boxes[i],boxes[i][2]])

    # sort boxes within each group based on x-coordinate
    for i in range(best_n_clusters):
        groups[i] = sorted(groups[i], key=lambda box: box[0])
    return groups

# Example usage

groups = group_boxes(centers)
print(groups)

groupMap = []
for group in groups:
    mapped_arr = []
    for x in group:
        
        i = x[1]
        if i == 0:
            mapped_arr.append(values[0])
        elif i == 1:
            mapped_arr.append(values[1])
        elif i == 2:
            mapped_arr.append(values[2])
        elif i == 3:
            mapped_arr.append(values[3])
        elif i == 4:
            mapped_arr.append(values[4])
        elif i == 5:
            mapped_arr.append(values[5])
        elif i == 6:
            mapped_arr.append(values[6])
        elif i == 7:
            mapped_arr.append(values[7])
        elif i == 8:
            mapped_arr.append(values[8])
        elif i == 9:
            mapped_arr.append(values[9])
        elif i == 10:
            mapped_arr.append(values[10])
        elif i == 11:
            mapped_arr.append(values[11])
        elif i == 12:
            mapped_arr.append(values[12])
        elif i == 13:
            mapped_arr.append(values[13])
        elif i == 14:
            mapped_arr.append(values[14])
        elif i == 15:
            mapped_arr.append(values[15])
        elif i == 16:
            mapped_arr.append(values[16])
        elif i == 17:
            mapped_arr.append(values[17])
        elif i == 18:
            mapped_arr.append(values[18])
        elif i == 19:
            mapped_arr.append(values[19])
    groupMap.append(mapped_arr)
   
    
print(groupMap)
