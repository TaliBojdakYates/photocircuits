def calculate_box_area(box):
    # calculates the area of a box given its four (x, y) corner coordinates
    
    x1, y1, x2, y2 = box
    
    return abs(x2 - x1) * abs(y2 - y1)

def calculate_intersection_area(box_a, box_b):
    # calculates the area of the intersection between two boxes
    
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b
    
    dx = min(x2_a, x2_b) - max(x1_a, x1_b)
    dy = min(y2_a, y2_b) - max(y1_a, y1_b)
    
    if dx <= 0 or dy <= 0:
        return 0
    
    return dx * dy

def check_box_overlap(box_a, box_b, threshold):
    # checks if box_a and box_b overlap by at least threshold percent
    
    intersection_area = calculate_intersection_area(box_a, box_b)
    a_area = calculate_box_area(box_a)
    b_area = calculate_box_area(box_b)
    
    if intersection_area / a_area >= threshold or intersection_area / b_area >= threshold:
        return True
    
    return False

def check_box_inside(box_a, box_b, threshold):
    # checks if box_a is mostly inside box_b (i.e., at least threshold percent of box_a is inside box_b)
    
    intersection_area = calculate_intersection_area(box_a, box_b)
    a_area = calculate_box_area(box_a)
    
    if intersection_area / a_area >= threshold:
        return True
    
    return False

def remove_boxes_inside(boxes, threshold):
    # removes boxes that overlap past the threshold or are completely inside another box
    
    num_boxes = len(boxes)
    inside_boxes = []
    remove_boxes = []
    
    for i in range(num_boxes):
        for j in range(num_boxes):
            if i == j:
                continue
                
            if check_box_inside(boxes[i], boxes[j], threshold):
                inside_boxes.append(i)
                break
            elif check_box_overlap(boxes[i], boxes[j], threshold):
                if calculate_box_area(boxes[i]) <= calculate_box_area(boxes[j]):
                    remove_boxes.append(i)
                else:
                    remove_boxes.append(j)
                
    for box_index in sorted(list(set(inside_boxes + remove_boxes)), reverse=True):
        if box_index in remove_boxes:
            del boxes[box_index]
                
    return boxes, remove_boxes



