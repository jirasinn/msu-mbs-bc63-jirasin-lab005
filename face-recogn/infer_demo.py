import cv2
from libs import preprocessing
import joblib
import numpy as np
from PIL import Image
import argparse
from utils import Draw_face_names
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default = 'webcam', nargs='?', choices= ['webcam', 'video', 'image'], help= '3 mode to test(image, video, webcam)')
    parser.add_argument('--confidence', default = 0.8, type= float, help = 'set confidence threshold for recognizer')
    parser.add_argument('--video-path', type= str, help = 'video folder path')
    parser.add_argument('--image-path', type=str, help = 'input image file')
    parser.add_argument('--model', default= 'models/recogniser_v1.pkl', type=str, help= 'load model path')
    return parser.parse_args()
    
def main():

    args = parse_args()
    global frame_count, total_save
    face_recogniser = joblib.load(args.model)
    preprocess = preprocessing.ExifOrientationNormalize()

    if args.mode == 'webcam':
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            frame = frame[:,::-1,:]
            if not ret:
                break
            img = Image.fromarray(frame[..., ::-1])
            faces = face_recogniser(preprocess(img))
            if faces is not None:
                Draw_face_names(faces, img, confidence= args.confidence)

            # img = img.resize((frame.shape[1], frame.shape[0]))
            cv2.imshow('webcam', np.array(img)[:,:,::-1])
            if cv2.waitKey(1)== 27 or cv2.waitKey(1) == ord('q') :
                break
        cv2.destroyAllWindows()
        cap.release()

    elif args.mode == 'video':
        video_files = [os.path.join(args.video_path, file) 
                       for file in sorted(os.listdir(args.video_path))
                       if file.endswith(('.mp4','.mkv','.avi'))]
        
        for idx, video_file in enumerate(video_files):
            cap = cv2.VideoCapture(video_file)
            assert cap.isOpened(), "video file can't open!"
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # start = timer()
                img = Image.fromarray(frame[..., ::-1])
                faces = face_recogniser(preprocess(img))
                if faces is not None:
                    Draw_face_names(faces, img, confidence= args.confidence)
                # img = img.resize((320, 480))
                cv2.imshow(os.path.basename(video_file), np.array(img)[:,:,::-1])
                if cv2.waitKey(1)== 27 or cv2.waitKey(1) == ord('q') :
                    break
                
            cv2.destroyAllWindows()
            cap.release()
            print(f'{idx} --> [INFO] Finished Video {os.path.basename(video_file)}')   
    
    elif args.mode == 'image':
        frame = cv2.imread(args.img_path)
        faces = face_recogniser(preprocess(img))
        if faces is not None:
            Draw_face_names(faces, img, confidence= args.confidence)
        cv2.imshow(os.path.basename(args.img_path), np.array(img)[...,::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':

    main()


