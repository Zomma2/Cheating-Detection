import numpy as np
import argparse
import time


import cv2
import os

def detect(imagePath, yoloPath, confidenceNeeded=0.5, threshold=0.3):

    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([yoloPath, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([yoloPath, "yolov3.weights"])
    configPath = os.path.sep.join([yoloPath, "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # load our input image and grab its spatial dimensions
    image = cv2.imread(imagePath)
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confidenceNeeded:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidenceNeeded,
        threshold)

    # ensure at least one detection exists
    result=[]
    if len(idxs) > 0:
        for i in idxs.flatten():
            result.append([LABELS[classIDs[i]], confidences[i]])

            

    # return the output
    return result



def get_palgarism_objects_txt(all_objects_that_can_be_detected , mode ):
    f = open(all_objects_that_can_be_detected, "r")
    list_of_palgarism_objects = [] 
    objects = f.read().split("\n") 
    for obj in objects : 
        if "*" in obj :
            list_of_palgarism_objects.append((''.join([i for i in obj if not i.isdigit()])).replace('*','').replace('-',''))
    if mode == 1 : 
        f = open("palgarism_objects_to_detect.txt", "w")
        for obj in list_of_palgarism_objects : 
            f.write(obj + "\n")
        f.close()
    if mode == 2 : 
        return list_of_palgarism_objects
    else : 
        return -1 
    
    
    
def filter_output_yolo(yolo_output , palgarism_objects) : 
    yolo_filtered_output = [] 
    for obj , conf in yolo_output : 
        if obj in palgarism_objects : 
            yolo_filtered_output.append([obj ,conf ])
    return yolo_filtered_output 
    
def run_yolo(image_path) : 
    output=detect(image_path,'/mnt/acess/')
    yolo_filtered = filter_output_yolo(output ,
                        get_palgarism_objects_txt("/mnt/acess/palgarism_objects.txt",2))
    return yolo_filtered

def is_cheating_yolo(image_path) :
    cheating_objs = []
    yolo_out = run_yolo(image_path)
    if len(yolo_out) > 0 :
        for item in yolo_out : 
            cheating_objs.append(item[0])  
        return cheating_objs , True 
    else:
        return 'None' , False 
        