import cv2
import mediapipe as mp
import numpy as np
import os
import re
import json
import argparse
from PIL import Image
from tqdm import tqdm
from glob import glob

def keypoint_idx(mp_pose):
    keypoint = {
                'NOSE': mp_pose.PoseLandmark.NOSE,
                'RIGHT_SHOULDER': mp_pose.PoseLandmark.RIGHT_SHOULDER,
                'RIGHT_ELBOW': mp_pose.PoseLandmark.RIGHT_ELBOW,
                'RIGHT_WRIST': mp_pose.PoseLandmark.RIGHT_WRIST,
                'LEFT_SHOULDER': mp_pose.PoseLandmark.LEFT_SHOULDER,
                'LEFT_ELBOW': mp_pose.PoseLandmark.LEFT_ELBOW,
                'LEFT_WRIST': mp_pose.PoseLandmark.LEFT_WRIST,
                'RIGHT_HIP': mp_pose.PoseLandmark.RIGHT_HIP,
                'RIGHT_KNEE': mp_pose.PoseLandmark.RIGHT_KNEE,
                'RIGHT_ANKLE': mp_pose.PoseLandmark.RIGHT_ANKLE,
                'LEFT_HIP': mp_pose.PoseLandmark.LEFT_HIP,
                'LEFT_KNEE': mp_pose.PoseLandmark.LEFT_KNEE,
                'LEFT_ANKLE': mp_pose.PoseLandmark.LEFT_ANKLE,
                'RIGHT_EYE': mp_pose.PoseLandmark.RIGHT_EYE,
                'LEFT_EYE': mp_pose.PoseLandmark.LEFT_EYE,
                'RIGHT_EAR': mp_pose.PoseLandmark.RIGHT_EAR,
                'LEFT_EAR': mp_pose.PoseLandmark.LEFT_EAR
                }
    return keypoint

def extract_keypoints(results, keypoint, key, img_dim):
    H, W = img_dim
    return [results[keypoint[key]].x * W, results[keypoint[key]].y * H, results[keypoint[key]].visibility]

def make_json(mp_pose, image_list, save_dir=None, confidence=0.6, save_json=True):
    # mp_pose = mp.solutions.pose
    keypoint = keypoint_idx(mp_pose)

    with mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=confidence, model_complexity=2) as pose:

        for image in tqdm(image_list):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (192, 256), interpolation = cv2.INTER_AREA)
            H, W = img.shape[:2]
            results = pose.process(img)
            landmarks = results.pose_landmarks.landmark
            blazepose_keypoints = [
                                    extract_keypoints(landmarks, keypoint, key='NOSE', img_dim=(H, W)),
                                    [(landmarks[keypoint['LEFT_SHOULDER']].x*W + landmarks[keypoint['RIGHT_SHOULDER']].x*W) * 0.5,
                                     (landmarks[keypoint['LEFT_SHOULDER']].y*H + landmarks[keypoint['RIGHT_SHOULDER']].y*H) * 0.5,
                                     landmarks[keypoint['RIGHT_SHOULDER']].visibility],
                                    extract_keypoints(landmarks, keypoint, key='RIGHT_SHOULDER', img_dim=(H, W)),
                                    extract_keypoints(landmarks, keypoint, key='RIGHT_ELBOW', img_dim=(H, W)),
                                    extract_keypoints(landmarks, keypoint, key='RIGHT_WRIST', img_dim=(H, W)),
                                    extract_keypoints(landmarks, keypoint, key='LEFT_SHOULDER', img_dim=(H, W)),
                                    extract_keypoints(landmarks, keypoint, key='LEFT_ELBOW', img_dim=(H, W)),
                                    extract_keypoints(landmarks, keypoint, key='LEFT_WRIST', img_dim=(H, W)),
                                    extract_keypoints(landmarks, keypoint, key='RIGHT_HIP', img_dim=(H, W)),
                                    extract_keypoints(landmarks, keypoint, key='RIGHT_KNEE', img_dim=(H, W)),
                                    extract_keypoints(landmarks, keypoint, key='RIGHT_ANKLE', img_dim=(H, W)),
                                    extract_keypoints(landmarks, keypoint, key='LEFT_HIP', img_dim=(H, W)),
                                    extract_keypoints(landmarks, keypoint, key='LEFT_KNEE', img_dim=(H, W)),
                                    extract_keypoints(landmarks, keypoint, key='LEFT_ANKLE', img_dim=(H, W)),
                                    extract_keypoints(landmarks, keypoint, key='RIGHT_EYE', img_dim=(H, W)),
                                    extract_keypoints(landmarks, keypoint, key='LEFT_EYE', img_dim=(H, W)),
                                    extract_keypoints(landmarks, keypoint, key='RIGHT_EAR', img_dim=(H, W)),
                                    extract_keypoints(landmarks, keypoint, key='LEFT_EAR', img_dim=(H, W))
                                  ]
            json_content = {
               "version": mp.__version__,
               "people": [
               {
               "person_id":[-1],
               "pose_keypoints_2d": blazepose_keypoints
               }
               ]
            }

            if save_json:
                img_name = image.split('/')[-1].split('.')[0]
                with open(os.path.join(save_dir, img_name + "_keypoints.json"), 'w') as f:
                   json.dump(json_content, f)
            else:
                return json_content['people'][0]["pose_keypoints_2d"]
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mediapipe Keypoint Generation")
    parser.add_argument('--image_dir', type=str, default="./dataset/image", help="path to the image directory")
    parser.add_argument('--save_dir', type=str, default="./out/pose", help="path to dump keypoint json")
    parser.add_argument('--conf', type=int, default=0.6, help="minimum detection confidence")
    args = parser.parse_args()

    extensions = r'\.jpg|\.png|\.jpeg'
    images = glob(os.path.join(args.image_dir, '*'))
    images = list(filter(lambda img_name: re.search(extensions, img_name, re.IGNORECASE), images))
    print(f"Total images identified (extensions: jpg, png, jpeg): {len(images)}")

    os.makedirs(args.save_dir, exist_ok=True)
    confidence = args.conf
    mp_pose = mp.solutions.pose
    make_json(mp_pose, images, args.save_dir, confidence)
    print("JSON files can be found at {}".format(args.save_dir))
