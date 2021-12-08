import cv2
import datetime

# Files
model = "models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"
config = "models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
labels = "coco_class_labels.txt"
s = "traffic_cropped.mp4"
# Model Params
in_width = 300
in_height = 300
mean = [104, 117, 123]
confidence_threshold = 0.25
trucks = 0
source = cv2.VideoCapture(s)
win_name = 'Traffic camera'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Get list of object labels 
with open(labels) as fp:
    labels = fp.read().split("\n")

# Read the Tensorflow network
net = cv2.dnn.readNetFromTensorflow(model, config)

while cv2.waitKey(1) != 27:
    has_frame, frame = source.read()
    today = datetime.datetime.now()
    date_time = today.strftime("%d/%m/%Y, %H:%M:%S")
    cv2.rectangle(frame, (0,0), (600, 180), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, f"{date_time}", (25,150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255),3)
    cv2.putText(frame, f"Number of trucks: {trucks}", (25,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255),3)
    cv2.putText(frame, "Southbound A40", (25,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255),3)

    if not has_frame:
        break

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
        
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB = False, crop = False)

    # Run a model
    net.setInput(blob)
    detections = net.forward()
    
    # For every Detected Object
    for i in range(detections.shape[2]):
    
        # Find the class and confidence 
        confidence = detections[0, 0, i, 2]
        classId = int(detections[0, 0, i, 1])

        # Check if the detection is of good quality and is truck
        if "{}".format(labels[classId]) == "truck" and confidence > confidence_threshold:

                x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
                y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
                x_right_top = int(detections[0, 0, i, 5] * frame_width)
                y_right_top = int(detections[0, 0, i, 6] * frame_height)
                x_mid_point = int( x_right_top - x_left_bottom)
                y_mid_point = int( y_right_top - y_left_bottom)
                
                # Counting trucks
                if y_mid_point in range(290, 295):
                    trucks += 1
                    name = f"trucks/Truck{trucks}.jpg"
                    cv2.imwrite(name, frame)
                    
                # Frame around the detected object
                cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0),2)
                label = "{}".format(labels[classId])
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Add label to object
                cv2.rectangle(frame, (x_left_bottom, y_left_bottom - label_size[1]), (x_left_bottom + label_size[0], y_left_bottom + base_line), (0, 0, 0), cv2.FILLED)
                cv2.putText(frame, label, (x_left_bottom, y_left_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        
        cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)