import argparse
import joblib
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from libs import FaceRecognition, Recognition_Pipeline
from timeit import default_timer as timer

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default = 'datasets/users', help= 'Training Dataset Path')
    parser.add_argument('--save_model', default = 'models/recogniser_v1.pkl', type = str , help= 'save model file name with (.pkl) extension')
    parser.add_argument('--grid_search', action = 'store_true', help= 'trained with grid search to estimate "C" parameter of Logistic Regression Classifier.')
    
    return parser.parse_args()

def dataset_to_embeddings(dataset, face_recognizer):
    
    trf = transforms.Compose([transforms.RandomHorizontalFlip(),
                              transforms.ToTensor()])
    embeddings = []
    labels = []
    for img_path, label in dataset.samples:
        pil_img = Image.open(img_path).convert('RGB')
        tensor_img = trf(pil_img).unsqueeze(0)
        embedding = face_recognizer.extract_features(tensor_img) 
        embeddings.append(embedding.flatten())
        labels.append(label)
    
    return np.stack(embeddings), labels

def load_data(args, face_recognizer):
    
    face_dataset = datasets.ImageFolder(args.dataset)
    print('[INFO] Found {} users with {} images to train the model'.format(len(face_dataset.classes), len(face_dataset.samples)))
    embeddings, labels = dataset_to_embeddings(face_dataset, face_recognizer)
    
    return embeddings, labels, face_dataset.class_to_idx

def train(args, embeddings, labels):
    
    softmax = LogisticRegression(solver='lbfgs', multi_class ='multinomial', C=10, max_iter = 10000)
    
    if args.grid_search:
        print('\n[INFO] Using Grid_Search Training ...!')
        clf = GridSearchCV(
            estimator = softmax,
            param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
            cv = 3
            )
    else:
        clf = softmax
    
    clf.fit(embeddings, labels)
    return clf.best_estimator_ if args.grid_search else clf

def main():
    
    args = parse_args()
    assert args.save_model.endswith('.pkl'), 'model file extension need to be end with ".pkl"'
    
    start = timer()
    face_recognizer = FaceRecognition(thresh=[0.9, 0.9, 0.9], img_size= 160)    
    embeddings, labels, class_to_idx = load_data(args, face_recognizer)
    clf = train(args, embeddings, labels)
    idx_to_class = {v: k for k,v in class_to_idx.items()}
    
    target_names = map(lambda i: i[1], sorted(idx_to_class.items(), 
                       key = lambda i: i[0]))
    print(metrics.classification_report(labels, clf.predict(embeddings), target_names= list(target_names)))
    
    joblib.dump(Recognition_Pipeline(face_recognizer, clf, idx_to_class), args.save_model)
    
    print('\n[INFO] Model Training Time : %.4f seconds'%( timer()- start))
    
if __name__ == '__main__':
    main()
    
  
    