import numpy as np
import cv2
import time
import os
from pprint import pprint

yolo_dir = '../yolo-coco/'
set_confidence = 0.5
set_threshold = 0.3
# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([yolo_dir, "yolov3.weights"])
configPath = os.path.sep.join([yolo_dir, "yolov3.cfg"])

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([yolo_dir, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

print(LABELS)

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

def object_detect(imagepath):
    # load our input image and grab its spatial dimensions
    image = cv2.imread(imagepath)
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding centers and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding centers, confidences, and
    # class IDs, respectively
    centers = []
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
            if confidence > set_confidence:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the centers' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                # x = int(centerX - (width / 2))
                # y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                centers.append([centerX, centerY])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # centers
    # idxs = cv2.dnn.NMSBoxes(centers, confidences, set_confidence,
    #                         set_threshold)

    # pprint(centers)
    # print('******************************************')
    # pprint(classIDs)
    # print('******************************************')
    # pprint(confidences)
    # print('******************************************')

    objects = []
    for i in range(len(classIDs)):
        x, y = centers[i]
        id = LABELS[classIDs[i]]
        objects.append([x, y, id])
    flattened_objs = [str(x) for sublist in objects for x in sublist]
    flattened_objs = "-".join(flattened_objs)
    return flattened_objs


images = os.listdir('../images/')
images = [x for x in images if x[-4:]=='.jpg']
print(images)
sorted_imgs = {}
for img in images:
    vid = img[:2]
    if vid not in sorted_imgs.keys():
        sorted_imgs[vid] = []
    sorted_imgs[vid].append(int(img[:-4].split('_')[-1]))
for k in sorted_imgs.keys():
    sorted_imgs[k].sort()
# pprint(sorted_imgs)

fout = open('objects', 'w')

for vid in sorted_imgs.keys():
    for f in sorted_imgs[vid]:
        img = '../images/'+vid+'_frame_'+str(f)+'.jpg'
        print(img)
        detections = object_detect(img)
        fout.write(detections+'\n')
fout.close()
