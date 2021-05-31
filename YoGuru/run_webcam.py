import argparse
import logging
import time

import cv2
import numpy as np
import math

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
    
#Added code
def find_point(pose, p):
    for point in pose:
        try:
            body_part = point.body_parts[p]
            return (int(body_part.x * width + 0.5), int(body_part.y * height + 0.5))
        except:
            return (0,0)
    return (0,0)


def angle_calc(p0, p1, p2 ):
    '''
        p1 is center point from where we measured angle between p0 and
    '''
    try:
        a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
        b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
        c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
        angle = math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180/math.pi
    except:
        return 0
    return int(angle)

def eval_warrior_ii(comment = ''):
    right_hand = angle_calc(find_point(pose, 2), find_point(pose, 3), find_point(pose, 4))
    left_hand = angle_calc(find_point(pose, 5), find_point(pose, 6), find_point(pose, 7))
    right_leg = angle_calc(find_point(pose, 8), find_point(pose, 9), find_point(pose, 10))
    left_leg = angle_calc(find_point(pose, 11), find_point(pose, 12), find_point(pose, 13))
    if right_hand in range(170,180) and left_hand in range(170,180) and right_leg in range(80,90) and left_leg in range(170,180):
        comment = "Perfect"
    if right_hand < 170:
        comment = "You should straighten up your right forearm"
    elif left_hand < 170:
        comment = "You should straighten up your left forearm"
    elif right_leg > 90:
        comment = "You should get lower on your right leg"
    elif left_leg < 170:
        comment = "You should straighten up your left leg"
    return(comment)

def eval_rithwik(comment = ''):
    right_hand = angle_calc(find_point(pose, 2), find_point(pose, 3), find_point(pose, 4))
    left_hand = angle_calc(find_point(pose, 5), find_point(pose, 6), find_point(pose, 7))
    right_leg_hand = angle_calc(find_point(pose, 1), find_point(pose, 8), find_point(pose, 9))
    left_leg = angle_calc(find_point(pose, 11), find_point(pose, 12), find_point(pose, 13))
    right_leg = angle_calc(find_point(pose, 8), find_point(pose, 9), find_point(pose, 10))
    if right_hand in range(170,180) and left_hand in range(170,180) and right_leg in range(80,90) and left_leg in range(170,180) and right_leg_hand in range(60,70):
        comment = "Perfect"
    if right_hand < 170:
        comment = "You should straighten up your right forearm"
    elif left_hand < 170:
        comment = "You should straighten up your left forearm"
    elif right_leg < 170:
        comment = "You should straighten up your right leg"
    elif left_leg < 170:
        comment = "You should straighten up your left leg"
    elif right_leg_hand > 70:
        comment = "You should raise your right leg more till you touch your feet"
    return(comment)
    
#End of added code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    i = 0
    frm = 0
    y1 = [0,0]
    global height,width
    orange_color = (0,140,255)
    mode = int(input('Pose: 1-Warrior II, 2-Rithwik: '))

    while True:
        ret_val, image = cam.read()

        #logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        pose = humans #Added code
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        height,width = image.shape[0],image.shape[1] #Added code
        #logger.debug('postprocess+')
        
        #Added code
        if mode == 1:
            comment = eval_warrior_ii()
        elif mode == 2:
            comment = eval_rithwik()

        cv2.putText(image,
                    "Comment: %s" %(comment),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        #End of added code

        cv2.imshow('tf-pose-estimation result', image)
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')

    cv2.destroyAllWindows()
