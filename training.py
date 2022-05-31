#%% Import lib
import io

import cv2
import dlib
import joblib
import matplotlib.pyplot as plt
import numpy as np
import PIL
from keras.models import load_model
from numpy import expand_dims
from PIL import Image
from sklearn.datasets import load_files
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import LinearSVC

from util_funcs import get_embedding, load_datasets_facenet

#%% Load dataset and generate train, test set
NEW_RUN = False   

trainX, y, true_name, id = load_datasets_facenet()
X_train, X_test, y_train, y_test = train_test_split(
    trainX, y, test_size=0.25, random_state=42
)

# X_train, y_train = data_aug(X_train, y_train)
# X_test, y_test = data_aug(X_test, y_test)

#%% Load model and get embedded vector from faces
#load model
model = load_model('./model/facenet_keras.h5')
print('Loaded model')

def get_emb_from_dataset(dataset):
    newDataset = []
    for face_pixcels in dataset:
        embedding = get_embedding(model, face_pixcels)
        newDataset.append(embedding)
        
    return np.asarray(newDataset)
        
newTrainX = get_emb_from_dataset(X_train)
print(newTrainX.shape)

newTestX = get_emb_from_dataset(X_test)
print(newTestX.shape)

# %% Normalize input vector and train classifier (Linear SVM)
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(newTrainX)
testX = in_encoder.transform(newTestX)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(y_train)
trainy = out_encoder.transform(y_train)
testy = out_encoder.transform(y_test)

SVM_clf = LinearSVC(C=np.inf, loss='hinge', max_iter=100000, random_state=300)

if NEW_RUN:
    SVM_clf.fit(trainX, trainy)
    joblib.dump(SVM_clf, './model/svm_face_recognitor')
else:
    SVM_clf = joblib.load('./model/svm_face_recognitor')

# predict
yhat_train = SVM_clf.predict(trainX)
yhat_test = SVM_clf.predict(testX)
# score
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)
# summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

#%% Predict on an image
from util_funcs import convert_and_trim_bb

def predict_image(image, boundingColor = (0, 0, 255), color = (0, 255, 0), font = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, lineStroke = 2):
    hog = dlib.get_frontal_face_detector()
    rects = hog(image, 1)
    detections = [convert_and_trim_bb(image, r) for r in rects]
    
    for (x, y, w, h) in  detections:
        face = image[y:y + h, x:x + w]
        face = cv2.resize(face, (160, 160))
        face_embed = get_embedding(model, face)
        face_encode = in_encoder.transform(np.expand_dims(face_embed, axis=0))
        pred_name = out_encoder.inverse_transform(SVM_clf.predict(face_encode))
        # yhat_prob = SVM_clf._predict_proba_lr(face_encode)[0][0]
        cv2.putText(image, f'{true_name[pred_name][0]}', (x + 5, y - 5), font, fontScale, color, lineStroke)
        cv2.rectangle(image, (x, y), (x + w, y + h), boundingColor, lineStroke)
        
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

#%% Prediction
for i in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
    image = cv2.imread(f'./test_img/test{i}.jpg')
    plt.imsave(f'./test{i}.jpg', predict_image(image))

# %%
