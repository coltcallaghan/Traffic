import cv2
import datetime
import sys
import numpy as np

# Get list of object labels 
def get_labels(list_of_labels):
    with open(list_of_labels) as fp:
        labels = fp.read().split("\n")
        return labels

# Checks if detection is truck
def check_truck(confidence, detections, frame_width, frame_height, i, confidence_threshold, labels, classId):
    if confidence > confidence_threshold and "{}".format(labels[classId]) == "truck":
        x_left_top = int(detections[0, 0, i, 3] * frame_width)     
        y_left_top = int(detections[0, 0, i, 4] * frame_height) 
        x_right_bottom = int(detections[0, 0, i, 5] * frame_width) 
        y_right_bottom = int(detections[0, 0, i, 6] * frame_height)
        y_mid_point = int(y_right_bottom - y_left_top)
        return True, (y_mid_point, x_left_top, y_left_top, x_right_bottom, y_right_bottom)
    else:
        return False, ()

# Draws around cars so we know they are different from trucks
def check_car(confidence, detections, frame_width, frame_height, i, labels, classId, frame):
    if confidence > .039 and "{}".format(labels[classId]) == "car":
        x_left_top = int(detections[0, 0, i, 3] * frame_width)     
        y_left_top = int(detections[0, 0, i, 4] * frame_height) 
        x_right_bottom = int(detections[0, 0, i, 5] * frame_width) 
        y_right_bottom = int(detections[0, 0, i, 6] * frame_height)
        cv2.rectangle(frame, (x_left_top, y_left_top), (x_right_bottom, y_right_bottom), (0, 0, 255),2)
        label = "{}".format(labels[classId])
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)    
        cv2.rectangle(frame, (x_left_top, y_left_top - label_size[1]), (x_left_top + label_size[0], y_left_top + base_line), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, label, (x_left_top, y_left_top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    

# Counts trucks and saves image of each that passes defined space
def count_trucks(y_mid_point, trucks, frame):
    if y_mid_point in range(145, 155):
            trucks += 1
            file_name = f"trucks/Truck{trucks}.jpg"
            cv2.imwrite(file_name, frame)
            return trucks
    else:
        return trucks

# Gets current time (As if this was a live feed)
def get_time():
    today = datetime.datetime.now()
    date_time = today.strftime("%d/%m/%Y, %H:%M:%S")
    return date_time

# Shows live tracking of truck count, time and direction of travel (Time & Direction fictional in this instance)
def show_data(frame, date_time, trucks, frame_width):
    cv2.rectangle(frame, (frame_width-600,0), (frame_width, 180), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, f"{date_time}", (frame_width-590,150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255),3)
    cv2.putText(frame, f"Number of trucks: {trucks}", (frame_width-590,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255),3)
    cv2.putText(frame, "Southbound A40", (frame_width-590,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255),3)

# Run the Network and make all detections
def run_nn(frame, net, in_width, in_height, mean):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB = True, crop = False)
    net.setInput(blob)
    detections = net.forward()
    return detections

# Finds confidence and ID of detections
def check_detecions(detections):
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        classId = int(detections[0, 0, i, 1])
    return confidence, classId, i

# Frame and label detections from ID
def label(frame, x_left_top, y_left_top, x_right_bottom, y_right_bottom, labels, classId):
    cv2.rectangle(frame, (x_left_top, y_left_top), (x_right_bottom, y_right_bottom), (0, 255, 0),2)
    label = "{}".format(labels[classId])
    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)    
    cv2.rectangle(frame, (x_left_top, y_left_top - label_size[1]), (x_left_top + label_size[0], y_left_top + base_line), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, label, (x_left_top, y_left_top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

def main():
    # Variables - model, config, list_of_labels can be changed for different datasets (of same structure)
    model = "models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"
    config = "models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
    list_of_labels = "coco_class_labels.txt"
    net = cv2.dnn.readNetFromTensorflow(model, config)
    s = "traffic_cropped.mp4"
    win_name = 'Traffic camera'
    in_width = 300
    in_height = 300
    mean = [100,120,120]
    confidence_threshold = 0.03
    trucks = 0
    
    source = cv2.VideoCapture(s)

    while cv2.waitKey(1) != 27:
        has_frame, frame = source.read()
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        grey = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        labels = get_labels(list_of_labels)
        date_time = get_time()
        show_data(frame, date_time, trucks,frame_width)
        detections = run_nn(grey, net, in_width, in_height, mean) 
        confidence, classId, i = check_detecions(detections)
        success, vars = check_truck(confidence, detections, frame_width, frame_height, i, confidence_threshold, labels, classId)
        if success:
            y_mid_point, x_left_top, y_left_top, x_right_bottom, y_right_bottom = vars
            label(frame, x_left_top, y_left_top, x_right_bottom, y_right_bottom, labels, classId)
            trucks = count_trucks(y_mid_point, trucks, frame)
        check_car(confidence, detections, frame_width, frame_height, i, labels, classId, frame)
        cv2.imshow(win_name, frame)
    source.release()
    cv2.destroyWindow(win_name)
    if not has_frame:
        sys.exit()
    
    
main()    
