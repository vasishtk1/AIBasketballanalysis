# importing the necessary libraries
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pickle
# from google.colab.patches import cv2_imshow as cv2
def tensorflow_init():
    MODEL_NAME = 'inference_graph'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    return detection_graph, image_tensor, boxes, scores, classes, num_detections

# Creating a VideoCapture object to read the video
video_name = 'IMG_0369.mp4'
cap = cv2.VideoCapture(video_name)
 
count = 0
ball_position_coordinates = []
def detect_image(img):
    # obtains height and width of the image
    height, width = img.shape[:2]
    # initializes tensorflow 
    detection_graph, image_tensor, boxes, scores, classes, num_detections = tensorflow_init()
    # initiates tensorflow session
    with tf.Session(graph=detection_graph) as sess:
        # gets the expanded array
        img_expanded = np.expand_dims(img, axis=0)
        # creates a default graph 
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: img_expanded})
        valid_detections = 0

        xCoor = -1
        yCoor = -1
        for i, box in enumerate(boxes[0]):
            # print("detect")
            if (scores[0][i] > 0.5):
                valid_detections += 1
                ymin = int((box[0] * height))
                xmin = int((box[1] * width))
                ymax = int((box[2] * height))
                xmax = int((box[3] * width))
                xCoor = int(np.mean([xmin, xmax]))
                yCoor = int(np.mean([ymin, ymax]))
                # if it finds the basketball, it draws a cirlce around it
                if(classes[0][i] == 1):  # basketball
                    cv2.circle(img=img, center=(xCoor, yCoor), radius=25,
                               color=(255, 0, 0), thickness=-1)
                    cv2.putText(img, "BALL", (xCoor - 50, yCoor - 50),
                                cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 8)
                    print("add basketball")
            
    return xCoor, yCoor
# Loop until the end of the video
while (cap.isOpened()):
 
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.resize(frame, (540, 380), fx = 0, fy = 0,
                         # interpolation = cv2.INTER_CUBIC)
 
    # Display the resulting frame
    # cv2.imshow('Frame', frame)
    
    ret,frame = cap.read()
    xCoor, yCoor = detect_image(frame) 
    ball_position_coordinates += [xCoor, yCoor]
    cv2.imshow('window-name', frame)
    cv2.imwrite("frame%d.jpg" % count, frame)
    count = count + 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

print(ball_position_coordinates)
with open(video_name + '.pkl', 'wb') as f:
    pickle.dump(ball_position_coordinates, f)
# release the video capture object
cap.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()

