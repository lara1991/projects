import cv2
import numpy as np
import os

def get_image_names(img_dir):
    img_paths_list = []
    image_names_list = os.listdir(img_dir)
    
    for image_name in image_names_list:
        img_path = img_dir + "/" + image_name
        img_paths_list.append(img_path)
    
    return img_paths_list

def load_images_and_get_features(img_path_list):
    images = []
    keypoints = []
    descriptors = []

    for img_path in img_path_list:
        img = cv2.imread(img_path)
        images.append(img)
        kp,des = cv2.SIFT_create().detectAndCompute(img,None)
        keypoints.append(kp)
        descriptors.append(des)
    
    return images,keypoints,descriptors

def main():
    img_dir = "image_frames"

    img_path_list = get_image_names(img_dir)
    images,keypoints,descriptors = load_images_and_get_features(img_path_list)
    matcher = cv2.BFMatcher()

    matches = []
    for i in range(len(images)-1):
        matches.append(matcher.knnMatch(descriptors[i],descriptors[i+1],k=2))

    # for match in matches:
    #     print(f"match: {match}")

    # exit(0)
    
    good_matches = []
    pts_left = []
    pts_right = []

    print(len(keypoints),len(matches))
    # print()
    for idx,match in enumerate(matches):
        for m,n in match:
            if m.distance < 0.75*n.distance:
                good_matches.append(m)

                print(keypoints[idx][n.queryIdx].pt)
                print(len(keypoints[idx+1]),n.queryIdx)
                print()
                # # print(m)

                x1,y1 = keypoints[idx][m.queryIdx].pt
                x2,y2 = keypoints[idx+1][n.queryIdx].pt
                pts_left.append(keypoints[idx][m.queryIdx].pt)
                pts_right.append(keypoints[idx+1][n.queryIdx].pt)
        
    # # find the fundamental matrix
    # pts1 = np.float32([keypoints[i][m.queryIdx].pt for m in good_matches for i in range(len(images))])
    # pts2 = np.float32([keypoints[i][m.queryIdx].pt for m in good_matches for i in range(1,len(images))])

    F,mask = cv2.findFundamentalMat(pts_left,pts_right,cv2.FM_RANSAC)

    print(F)




if __name__=="__main__":
    main()

    