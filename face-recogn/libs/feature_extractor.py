import torch 
from facenet_pytorch import MTCNN,InceptionResnetV1
from .preprocessing import Whitening
from facenet_pytorch.models.utils.detect_face import extract_face
from torchvision import transforms
import os
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import functional as F

class FaceDetection:
    def __init__(self,  margin = 0, thresh =[0.6,0.7, 0.8],  img_size = 160, min_face= 20, select_largest = False, keep_all = True):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'[INFO] Detector is running on device --> {self.device}')
        self.img_size = img_size
        print('[INFO] Loading Detector Model ...', end = '\t')
        self.mtcnn = MTCNN(self.img_size, keep_all = keep_all , thresholds = thresh, device = self.device, min_face_size = min_face, select_largest = select_largest).eval()
        print('Done!')
    def extract_faces(self, img, lm= True):
        if lm:
            bbs, probs, landmarks = self.mtcnn.detect(img, landmarks= lm)
            if bbs is None:
                return None, None, None
            return bbs, probs, landmarks
        else:
            bbs, _ = self.mtcnn.detect(img)
            if bbs is None:
                return None
            return bbs
   
    def face_cut(self, img, box, save_path=None, margin=0,):
        margin = [
            margin * (box[2] - box[0]) / (self.img_size - margin),
            margin * (box[3] - box[1]) / (self.img_size - margin),
        ]
        raw_image_size = self._get_size(img)
        box = [
            int(max(box[0] - margin[0] / 2, 0)),
            int(max(box[1] - margin[1] / 2, 0)),
            int(min(box[2] + margin[0] / 2, raw_image_size[0])),
            int(min(box[3] + margin[1] / 2, raw_image_size[1])),
        ]
        face = self._crop_resize(img, box, self.img_size)
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) + "/", exist_ok=True)
            self._save_img(face, save_path)
        face = F.to_tensor(np.float32(face)) 
        return face
   
    def _crop_resize(self,img, box, image_size):
        if isinstance(img, np.ndarray):
            out = cv2.resize(
                img[box[1]:box[3], box[0]:box[2]],
                (image_size, image_size),
                interpolation=cv2.INTER_AREA
            ).copy()
        else:
            out = img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)
        return out
    
    def _save_img(self, img, path):
        if isinstance(img, np.ndarray):
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            img.save(path)
    
    def _get_size(self,img):
        if isinstance(img, np.ndarray):
            return img.shape[1::-1]
        else:
            return img.size
    
class FaceRecognition:
    
    def __init__(self, thresh=[0.6, 0.7, 0.9], img_size= 160):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
        print(f'[INFO] Recognizer is running on device --> {self.device}')
        self.img_size = img_size
        self.facenet_preprocess = transforms.Compose([Whitening()])
        self.facenet = InceptionResnetV1(pretrained= 'vggface2', device = self.device).eval()
        print('[INFO] Loading Recognizer Model ...', end = '\t')
        self.detector = FaceDetection(thresh= thresh, img_size = self.img_size)
        print('Done!')

    def detect_and_extract_feature(self, img):
        bbs = self.detector.extract_faces(img, lm= False)
        if bbs is None:
            return None, None
        faces = torch.stack([extract_face(img,box, self.img_size) for box in bbs])
        embeddings = self.facenet(self.facenet_preprocess(faces).to(self.device)).detach().to('cpu').numpy() 
        return bbs, embeddings
    
    #extract face features from cropped face for training model
    def extract_features(self,face):
        embedding = self.facenet(self.facenet_preprocess(face.to(self.device))).detach().to('cpu').numpy()
        return embedding

if __name__ == '__main__':    
    recognizer = FaceRecognition()
       
    
        