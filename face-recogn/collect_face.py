import os
import cv2
import argparse
import numpy as np
from libs import FaceDetection, preprocessing
from torchvision import transforms, datasets
from PIL import Image
# from timeit import default_timer as timer
from utils import Draw_Bboxes

frame_count = 0
total_save = 0
def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default = 'image', nargs='?', choices= ['webcam', 'video', 'image'], help='need to set one of webcam, video, image mode ' )
    parser.add_argument('--total-images', default= 50, type=int, help= 'total required images per class for training')
    parser.add_argument('--video-path', type= str, help = 'video folder path if "mode" is set to "video", *no need to give video file name')
    parser.add_argument('--image-path', default= 'datasets/test_images',  type=str, help = 'input image path')
    parser.add_argument('--interval', default= 20, type = int , help= 'extract face every interval from video')
    parser.add_argument('--save-path', default= 'datasets/users', type= str, help='store location of extracted faces') 
    
    return parser.parse_args()
    
def extract_face_from_img(detector, img, args, user_name):
    trf = transforms.Compose([
                            preprocessing.ExifOrientationNormalize(),
                            transforms.Resize(960)])
    
    assert isinstance(img, np.ndarray), 'image should be numpy array'
    img = trf(Image.fromarray(img[:,:,::-1]))
    boxes, probs, points = detector.extract_faces(img, lm = True)
    if boxes is None:
        print('[Warning] No Faces Found...!')
        return img
    if boxes.shape[0] >1 :
        print('[Warning] Skipping this image as many faces are detected. User need to be verify only one face...')
        img = Draw_Bboxes(img, boxes, probs)
        return img
    
    if args.mode == 'webcam' or args.mode == 'video':  
        global frame_count, total_save
        
        if frame_count % args.interval == 0 and total_save != args.total_images:
            total_save +=1
            file_name = os.path.join(args.save_path, user_name, user_name + f'_{frame_count}.jpg')
            detector.face_cut(img, boxes[0], save_path= file_name)
            print(f'[INFO] Total collected face : {total_save}')
        
        elif total_save >= args.total_images:
            print(f'[Warning] Collected enough face')   
    
    elif args.mode == 'image':
        total_save += 1
        file_name = os.path.join(args.save_path, user_name, user_name + f'_{total_save}.jpg')
        detector.face_cut(img, boxes[0], save_path = file_name)
    
    img = Draw_Bboxes(img, boxes, probs)         
    return img
    
def show_img(*imgs, name = 'test'):
    imgs = [cv2.resize(img, (640,480)) for img in imgs]
    imgs = np.hstack(imgs)
    cv2.imshow( name, imgs); cv2.waitKey(500); cv2.destroyAllWindows()    
    
def main():

    args = parse_args()
    global frame_count, total_save
    if args.mode == 'webcam':
        user = str(input('Enter User Name : '))
        detector = FaceDetection(margin = 0, thresh=[0.9,0.9,0.9], img_size = 160, min_face =20, select_largest = False)
        print(f'[INFO] Preparing to collect {args.total_images} images for {user}..!')

        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            frame = frame[:,::-1,:]
            frame_count += 1
            if not ret:
                break
            result = extract_face_from_img( detector, frame, args, user_name = user)
            # img = img.resize((frame.shape[1], frame.shape[0]))
            cv2.imshow('webcam', np.array(result)[:,:,::-1])
            
            if cv2.waitKey(1)== 27 or cv2.waitKey(1) == ord('q') :
                break
           
        cv2.destroyAllWindows()
        cap.release()

    elif args.mode == 'video':
        detector = FaceDetection(margin = 0, thresh=[0.9,0.9,0.9], img_size = 160, min_face =100, select_largest = True)
        video_files = [os.path.join(args.video_path, file) 
                       for file in sorted(os.listdir(args.video_path))
                       if file.endswith(('.mp4','.mkv','.avi'))]
        
        for idx, video_file in enumerate(video_files):
            user = os.path.splitext(os.path.basename(video_file))[0]
            cap = cv2.VideoCapture(video_file)
            assert cap.isOpened(), "Video can't open!"
            total_save = 0
            while True:
                ret, frame = cap.read()
                frame_count += 1

                if not ret:
                    break
                # start = timer()
                result = extract_face_from_img(detector, frame, args, user_name = user)
                result = result.resize((640, 480))
                # print('Detection speed is {} seconds'.format(timer()- start))
                cv2.imshow(user, np.array(result)[:,:,::-1])
                
                if cv2.waitKey(1) ==27 or cv2.waitKey(1) == ord('q'):
                    break
                
            cv2.destroyAllWindows()
            cap.release()
            print(f'{idx} --> [INFO] Finished Video {os.path.basename(video_file)}')   
    
    elif args.mode == 'image':
        visualize = True
        detector = FaceDetection(margin = 0, thresh=[0.9,0.9,0.9], img_size = 160, min_face =20, select_largest = False)
        dataset = datasets.ImageFolder(args.image_path)
        print('[INFO] Found {} images to train for {} users'.format(dataset.__len__(), dataset.classes.__len__()))
        for img_path, label in dataset.samples:    
            img = cv2.imread(img_path)
            result = extract_face_from_img(detector, img, args, user_name = dataset.classes[label])
            if visualize:
                show_img(np.array(result)[:,:,::-1], name = dataset.classes[label])

if __name__== '__main__':
    main()
    
    
    
    
    
    
    
    
    