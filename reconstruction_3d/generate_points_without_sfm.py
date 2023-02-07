import cv2
import numpy as np
import os
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')

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

## plot 3d point cloud
def plot_3d(points_3d):
    
    x = points_3d[:,0]
    y = points_3d[:,1]
    z = points_3d[:,2]
    
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    ax.grid()
    
    ax.scatter(x, y, z, c = 'g', s = 50)
    ax.set_title('3D Scatter Plot')

    # Set axes label
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)
    
    low = -3000
    high = 3000
    ax.set_xlim(low,high)
    ax.set_ylim(low,high)
    ax.set_zlim(low,high)

    plt.show()
        

def main():
    
    img_dir = "image_frames"

    img_path_list = get_image_names(img_dir)
    images,keypoints,descriptors = load_images_and_get_features(img_path_list)
    
    # initialize intrinsic matrix, extrinsic matrices and 3D points
    fx,fy,cx,cy = 1.,1,0,0
    K = np.array(
        [
            [fx,0,cx],
            [0,fy,cy],
            [0,0,1]
        ]
    )
    Rs = [np.eye(3) for i in range(len(images))]
    ts = [np.zeros((3,1)) for i in range(len(images))]
    points3D = []
    
    ## match the points for each pair of images
    matcher = cv2.BFMatcher()

    matches = []
    # num_images = len(images)-1
    num_images = len(images) - 1
    for i in range(num_images):
        matched_features = matcher.knnMatch(descriptors[i],descriptors[i+1],k=2)
        
        good_matches = []
        for m,n in matched_features:
            if m.distance < 0.75*n.distance:
                good_matches.append(m)
        
        good_matches = sorted(good_matches,key=lambda x: x.distance)
        
        ## compute the Essential matrices
        E,mask = cv2.findEssentialMat(
            np.array([keypoints[i][m.queryIdx].pt for m in good_matches]),
            np.array([keypoints[i+1][m.trainIdx].pt for m in good_matches]),
            K,
            cv2.RANSAC
        )
        _,R,t,mask = cv2.recoverPose(
            E,
            np.array([keypoints[i][m.queryIdx].pt for m in good_matches]),
            np.array([keypoints[i+1][m.trainIdx].pt for m in good_matches]),
            K
        )
        
        #update the intrinsic and extrinsic matrices
        Rs[i+1] = R.dot(Rs[i])
        ts[i+1] = R.dot(ts[i]) + t
        
        # triangulate the points to get the 3D points cloud
        
        print(Rs[:2],ts[:2])
        p1 = np.hstack((Rs[i],ts[i]))
        
        print(p1)
        p2 = np.hstack((Rs[i+1],ts[i+1]))
        points4D = cv2.triangulatePoints(
            p1,p2,
            np.array([keypoints[i][m.queryIdx].pt for m in good_matches]).T,
            np.array([keypoints[i+1][m.trainIdx].pt for m in good_matches]).T
        )
        
        #normalize the points
        points4D /= points4D[-1]
        points3D.append(points4D[:3].T)
        
    # stack the 3D points to a single array
    points3D = np.vstack(points3D)
    
    print("points 3D")
    print(points3D)
   

if __name__=="__main__":
    main()

    