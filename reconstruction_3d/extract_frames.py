import cv2

def extract_frames_from_video(video_path,frame_save_dir=None):
    cap = cv2.VideoCapture(video_path)
    
    ret = True
    frame_number = 0
    while ret:
        ret,frame = cap.read()
        
        if ret:
            frame = cv2.resize(frame,(600,600))
            cv2.imshow("Frame",frame)
            if frame_save_dir and (frame_number % 1 == 0):
                frame_name = f"frame_{frame_number}.jpg"
                frame_save_path = frame_save_dir + "/" + frame_name
                cv2.imwrite(frame_save_path,frame)
                
            frame_number += 1   
            cv2.waitKey(20)
            if frame_number == 120:
                break
            
    cap.release()
    cv2.destroyAllWindows()
    

def main():
    file_path = "video_directory/test1.mp4"
    save_dir = "image_frames"
    extract_frames_from_video(file_path,frame_save_dir=save_dir)
    
if __name__=="__main__":
    main()