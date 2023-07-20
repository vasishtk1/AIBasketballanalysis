import time
from absl import app, logging
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os
import sys
from sys import platform
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statistics import mean
tf.disable_v2_behavior()

shooting_result = {
    'attempts': 0,
    'made': 0,
    'miss': 0,
    'avg_elbow_angle': 0,
    'avg_knee_angle': 0,
    'avg_release_angle': 0,
    'avg_ballInHand_time': 0
}

def detect_shot(frame, trace, width, height, sess, image_tensor, boxes, scores, classes, num_detections, previous, during_shooting, shot_result, fig, datum, opWrapper, shooting_pose):
    global shooting_result

    if(shot_result['displayFrames'] > 0):
        shot_result['displayFrames'] -= 1
    if(shot_result['release_displayFrames'] > 0):
        shot_result['release_displayFrames'] -= 1
    if(shooting_pose['ball_in_hand']):
        shooting_pose['ballInHand_frames'] += 1
        # print("ball in hand")

    # getting openpose keypoints
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])
    try:
        headX, headY, headConf = datum.poseKeypoints[0][0]
        handX, handY, handConf = datum.poseKeypoints[0][4]
        elbowAngle, kneeAngle, elbowCoord, kneeCoord = getAngleFromDatum(datum)
    except:
        print("Something went wrong with OpenPose")
        headX = 0
        headY = 0
        handX = 0
        handY = 0
        elbowAngle = 0
        kneeAngle = 0
        elbowCoord = np.array([0, 0])
        kneeCoord = np.array([0, 0])

    frame_expanded = np.expand_dims(frame, axis=0)
    # main tensorflow detection
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # displaying openpose, joint angle and release angle
    frame = datum.cvOutputData
    cv2.putText(frame, 'Elbow: ' + str(elbowAngle) + ' deg',
                (elbowCoord[0] + 65, elbowCoord[1]), cv2.FONT_HERSHEY_COMPLEX, 1.3, (102, 255, 0), 3)
    cv2.putText(frame, 'Knee: ' + str(kneeAngle) + ' deg',
                (kneeCoord[0] + 65, kneeCoord[1]), cv2.FONT_HERSHEY_COMPLEX, 1.3, (102, 255, 0), 3)
    if(shot_result['release_displayFrames']):
        cv2.putText(frame, 'Release: ' + str(during_shooting['release_angle_list'][-1]) + ' deg',
                    (during_shooting['release_point'][0] - 80, during_shooting['release_point'][1] + 80), cv2.FONT_HERSHEY_COMPLEX, 1.3, (102, 255, 255), 3)

    for i, box in enumerate(boxes[0]):
        if (scores[0][i] > 0.5):
            ymin = int((box[0] * height))
            xmin = int((box[1] * width))
            ymax = int((box[2] * height))
            xmax = int((box[3] * width))
            xCoor = int(np.mean([xmin, xmax]))
            yCoor = int(np.mean([ymin, ymax]))
            # Basketball (not head)
            if(classes[0][i] == 1 and (distance([headX, headY], [xCoor, yCoor]) > 30)):

                # recording shooting pose
                if(distance([xCoor, yCoor], [handX, handY]) < 120):
                    shooting_pose['ball_in_hand'] = True
                    shooting_pose['knee_angle'] = min(
                        shooting_pose['knee_angle'], kneeAngle)
                    shooting_pose['elbow_angle'] = min(
                        shooting_pose['elbow_angle'], elbowAngle)
                else:
                    shooting_pose['ball_in_hand'] = False

                # During Shooting
                if(ymin < (previous['hoop_height'])):
                    if(not during_shooting['isShooting']):
                        during_shooting['isShooting'] = True

                    during_shooting['balls_during_shooting'].append(
                        [xCoor, yCoor])

                    #calculating release angle
                    if(len(during_shooting['balls_during_shooting']) == 2):
                        first_shooting_point = during_shooting['balls_during_shooting'][0]
                        release_angle = calculateAngle(np.array(during_shooting['balls_during_shooting'][1]), np.array(
                            first_shooting_point), np.array([first_shooting_point[0] + 1, first_shooting_point[1]]))
                        if(release_angle > 90):
                            release_angle = 180 - release_angle
                        during_shooting['release_angle_list'].append(
                            release_angle)
                        during_shooting['release_point'] = first_shooting_point
                        shot_result['release_displayFrames'] = 30
                        print("release angle:", release_angle)

                    #draw purple circle
                    cv2.circle(img=frame, center=(xCoor, yCoor), radius=7,
                               color=(235, 103, 193), thickness=3)
                    cv2.circle(img=trace, center=(xCoor, yCoor), radius=7,
                               color=(235, 103, 193), thickness=3)

                # Not shooting
                elif(ymin >= (previous['hoop_height'] - 30) and (distance([xCoor, yCoor], previous['ball']) < 100)):
                    # the moment when ball go below basket
                    if(during_shooting['isShooting']):
                        if(xCoor >= previous['hoop'][0] and xCoor <= previous['hoop'][2]):  # shot
                            shooting_result['attempts'] += 1
                            shooting_result['made'] += 1
                            shot_result['displayFrames'] = 10
                            shot_result['judgement'] = "SCORE"
                            print("SCORE")
                            # draw green trace when miss
                            points = np.asarray(
                                during_shooting['balls_during_shooting'], dtype=np.int32)
                            cv2.polylines(trace, [points], False, color=(
                                82, 168, 50), thickness=2, lineType=cv2.LINE_AA)
                            for ballCoor in during_shooting['balls_during_shooting']:
                                cv2.circle(img=trace, center=(ballCoor[0], ballCoor[1]), radius=10,
                                           color=(82, 168, 50), thickness=-1)
                        else:  # miss
                            shooting_result['attempts'] += 1
                            shooting_result['miss'] += 1
                            shot_result['displayFrames'] = 10
                            shot_result['judgement'] = "MISS"
                            print("miss")
                            # draw red trace when miss
                            points = np.asarray(
                                during_shooting['balls_during_shooting'], dtype=np.int32)
                            cv2.polylines(trace, [points], color=(
                                0, 0, 255), isClosed=False, thickness=2, lineType=cv2.LINE_AA)
                            for ballCoor in during_shooting['balls_during_shooting']:
                                cv2.circle(img=trace, center=(ballCoor[0], ballCoor[1]), radius=10,
                                           color=(0, 0, 255), thickness=-1)

                        # reset all variables
                        trajectory_fit(
                            during_shooting['balls_during_shooting'], height, width, shot_result['judgement'], fig)
                        during_shooting['balls_during_shooting'].clear()
                        during_shooting['isShooting'] = False
                        shooting_pose['ballInHand_frames_list'].append(
                            shooting_pose['ballInHand_frames'])
                        print("ball in hand frames: ",
                              shooting_pose['ballInHand_frames'])
                        shooting_pose['ballInHand_frames'] = 0

                        print("elbow angle: ", shooting_pose['elbow_angle'])
                        print("knee angle: ", shooting_pose['knee_angle'])
                        shooting_pose['elbow_angle_list'].append(
                            shooting_pose['elbow_angle'])
                        shooting_pose['knee_angle_list'].append(
                            shooting_pose['knee_angle'])
                        shooting_pose['elbow_angle'] = 370
                        shooting_pose['knee_angle'] = 370

                    #draw blue circle
                    cv2.circle(img=frame, center=(xCoor, yCoor), radius=10,
                               color=(255, 0, 0), thickness=-1)
                    cv2.circle(img=trace, center=(xCoor, yCoor), radius=10,
                               color=(255, 0, 0), thickness=-1)

                previous['ball'][0] = xCoor
                previous['ball'][1] = yCoor

            if(classes[0][i] == 2):  # Rim
                # cover previous hoop with white rectangle
                cv2.rectangle(
                    trace, (previous['hoop'][0], previous['hoop'][1]), (previous['hoop'][2], previous['hoop'][3]), (255, 255, 255), 5)
                cv2.rectangle(frame, (xmin, ymax),
                              (xmax, ymin), (48, 124, 255), 5)
                cv2.rectangle(trace, (xmin, ymax),
                              (xmax, ymin), (48, 124, 255), 5)

                #display judgement after shot
                if(shot_result['displayFrames']):
                    if(shot_result['judgement'] == "MISS"):
                        cv2.putText(frame, shot_result['judgement'], (xCoor - 65, yCoor - 65),
                                    cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 8)
                    else:
                        cv2.putText(frame, shot_result['judgement'], (xCoor - 65, yCoor - 65),
                                    cv2.FONT_HERSHEY_COMPLEX, 3, (82, 168, 50), 8)

                previous['hoop'][0] = xmin
                previous['hoop'][1] = ymax
                previous['hoop'][2] = xmax
                previous['hoop'][3] = ymin
                previous['hoop_height'] = max(ymin, previous['hoop_height'])

    combined = np.concatenate((frame, trace), axis=1)
    return combined, trace

def detect_image(img, response):
    height, width = img.shape[:2]
    detection_graph, image_tensor, boxes, scores, classes, num_detections = tensorflow_init()

    with tf.Session(graph=detection_graph) as sess:
        img_expanded = np.expand_dims(img, axis=0)
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: img_expanded})
        valid_detections = 0

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
                if(classes[0][i] == 1):  # basketball
                    cv2.circle(img=img, center=(xCoor, yCoor), radius=25,
                               color=(255, 0, 0), thickness=-1)
                    cv2.putText(img, "BALL", (xCoor - 50, yCoor - 50),
                                cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 8)
                    print("add basketball")
                    response.append({
                        'class': 'Basketball',
                        'detection_detail': {
                            'confidence': float("{:.5f}".format(scores[0][i])),
                            'center_coordinate': {'x': xCoor, 'y': yCoor},
                            'box_boundary': {'x_min': xmin, 'x_max': xmax, 'y_min': ymin, 'y_max': ymax}
                        }
                    })
                if(classes[0][i] == 2):  # Rim
                    cv2.rectangle(img, (xmin, ymax),
                                  (xmax, ymin), (48, 124, 255), 10)
                    cv2.putText(img, "HOOP", (xCoor - 65, yCoor - 65),
                                cv2.FONT_HERSHEY_COMPLEX, 3, (48, 124, 255), 8)
                    print("add hoop")
                    response.append({
                        'class': 'Hoop',
                        'detection_detail': {
                            'confidence': float("{:.5f}".format(scores[0][i])),
                            'center_coordinate': {'x': xCoor, 'y': yCoor},
                            'box_boundary': {'x_min': xmin, 'x_max': xmax, 'y_min': ymin, 'y_max': ymax}
                        }
                    })
        
        if(valid_detections < 2):
            for i in range(2):
                response.append({
                    'class': 'Not Found',
                    'detection_detail': {
                        'confidence': 0.0,
                        'center_coordinate': {'x': 0, 'y': 0},
                        'box_boundary': {'x_min': 0, 'x_max': 0, 'y_min': 0, 'y_max': 0}
                    }
                })
            
    return img

def detect_API(response, img):
    height, width = img.shape[:2]
    detection_graph, image_tensor, boxes, scores, classes, num_detections = tensorflow_init()

    with tf.Session(graph=detection_graph) as sess:
        img_expanded = np.expand_dims(img, axis=0)
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: img_expanded})

        for i, box in enumerate(boxes[0]):
            if (scores[0][i] > 0.5):
                ymin = int((box[0] * height))
                xmin = int((box[1] * width))
                ymax = int((box[2] * height))
                xmax = int((box[3] * width))
                xCoor = int(np.mean([xmin, xmax]))
                yCoor = int(np.mean([ymin, ymax]))
                if(classes[0][i] == 1):  # basketball
                    response.append({
                        'class': 'Basketball',
                        'detection_detail': {
                            'confidence': float(scores[0][i]),
                            'center_coordinate': {'x': xCoor, 'y': yCoor},
                            'box_boundary': {'x_min': xmin, 'x_max': xmax, 'y_min': ymin, 'y_max': ymax}
                        }
                    })
                if(classes[0][i] == 2):  # Rim
                    response.append({
                        'class': 'Hoop',
                        'detection_detail': {
                            'confidence': float(scores[0][i]),
                            'center_coordinate': {'x': xCoor, 'y': yCoor},
                            'box_boundary': {'x_min': xmin, 'x_max': xmax, 'y_min': ymin, 'y_max': ymax}
                        }
                    })

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

def openpose_init():
    try:
        if platform == "win32":
            sys.path.append(os.path.dirname(os.getcwd()))
            import OpenPose.Release.pyopenpose as op
        else:
            path = os.path.join(os.getcwd(), 'OpenPose/openpose')
            print(path)
            sys.path.append(path)
            import pyopenpose as op
    except ImportError as e:
        print("Something went wrong when importing OpenPose")
        raise e

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "./OpenPose/models"

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    return datum, opWrapper

def getVideoStream(video_path):
    print("Starting the function")
    datum, opWrapper = openpose_init()
    detection_graph, image_tensor, boxes, scores, classes, num_detections = tensorflow_init()

    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    trace = np.full((int(height), int(width), 3), 255, np.uint8)

    fig = plt.figure()
    #objects to store detection status
    previous = {
    'ball': np.array([0, 0]),  # x, y
    'hoop': np.array([0, 0, 0, 0]),  # xmin, ymax, xmax, ymin
        'hoop_height': 0
    }
    during_shooting = {
        'isShooting': False,
        'balls_during_shooting': [],
        'release_angle_list': [],
        'release_point': []
    }
    shooting_pose = {
        'ball_in_hand': False,
        'elbow_angle': 370,
        'knee_angle': 370,
        'ballInHand_frames': 0,
        'elbow_angle_list': [],
        'knee_angle_list': [],
        'ballInHand_frames_list': []
    }
    shot_result = {
        'displayFrames': 0,
        'release_displayFrames': 0,
        'judgement': ""
    }

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.36

    skip_count = 0
    with tf.Session(graph=detection_graph, config=config) as sess:
        while True:
            ret, img = cap.read()
            print(ret)
            if ret == False:
                break
            skip_count += 1
            if(skip_count < 4):
                continue
            skip_count = 0
            detection, trace = detect_shot(img, trace, width, height, sess, image_tensor, boxes, scores, classes,
                                        num_detections, previous, during_shooting, shot_result, fig, datum, opWrapper, shooting_pose)

            detection = cv2.resize(detection, (0, 0), fx=0.83, fy=0.83)
            frame = cv2.imencode('.jpg', detection)[1].tobytes()
            result = (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            yield result


    # getting average shooting angle
    shooting_result['avg_elbow_angle'] = round(mean(shooting_pose['elbow_angle_list']), 2)
    shooting_result['avg_knee_angle'] = round(mean(shooting_pose['knee_angle_list']), 2)
    shooting_result['avg_release_angle'] = round(mean(during_shooting['release_angle_list']), 2)
    shooting_result['avg_ballInHand_time'] = round(mean(shooting_pose['ballInHand_frames_list']) * (4 / fps), 2)

    print("avg", shooting_result['avg_elbow_angle'])
    print("avg", shooting_result['avg_knee_angle'])
    print("avg", shooting_result['avg_release_angle'])
    print("avg", shooting_result['avg_ballInHand_time'])

    plt.title("Trajectory Fitting", figure=fig)
    plt.ylim(bottom=0, top=height)
    trajectory_path = os.path.join(
        os.getcwd(), "static/detections/trajectory_fitting.jpg")
    fig.savefig(trajectory_path)
    fig.clear()
    trace_path = os.path.join(os.getcwd(), "static/detections/basketball_trace.jpg")
    cv2.imwrite(trace_path, trace)

def get_image(image_path, img_name, response):
    output_path = './static/detections/'
    # reading the images & apply detection 
    image = cv2.imread(image_path)
    filename = img_name
    detection = detect_image(image, response)

    cv2.imwrite(output_path + '{}' .format(filename), detection)
    print('output saved to: {}'.format(output_path + '{}'.format(filename)))

def detectionAPI(response, image_path):
    image = cv2.imread(image_path)
    detect_API(response, image)

getVideoStream('IMG_0369.mp4')
print("Script done.")