import cv2
import os

def load_images(image_dir):
    images = []
    image_names = os.listdir(image_dir)
    for image_name in image_names:
        image_path = image_dir + "/" + image_name
        img = cv2.imread(image_path)
        images.append(img)
        return images

def generate_points(image_list):
    sfm = cv2.sfm.create()
    
    for idx,image in enumerate(image_list):
        print(f"adding to sfm: {idx}")
        sfm.addImage(image)
    
    success,poses,points = sfm.compute()
    print(success)
    

def main():
    image_dir = "image_frames"
    image_set = load_images(image_dir=image_dir)
    generate_points(image_set)
    
if __name__ == '__main__':
    main()
    