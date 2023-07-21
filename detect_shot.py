import time
from absl import app, logging
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os
from .config import shooting_result
import sys
from sys import platform
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
tf.disable_v2_behavior()

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