import io
import PIL
from PIL import Image
import cv2
import joblib
import dlib
import numpy as np
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import LinearSVC
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

FACENET_PATH = './model/facenet_keras.h5'
SVM_PATH = './model/svm_face_recognitor'

def convert_and_trim_bb(image, rect):
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	# compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY
	# return our bounding box coordinates
	return (startX, startY, w, h) 

def predict_image(image, boundingColor = (0, 0, 255), color = (0, 255, 0), font = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, lineStroke = 2):
    hog = dlib.get_frontal_face_detector()
    rects = hog(image, 1)
    detections = [convert_and_trim_bb(image, r) for r in rects]
    true_name = load_true_name()
    model = load_model('./model/facenet_keras.h5')
    SVM_clf = joblib.load('./model/svm_face_recognitor')
    in_encoder = joblib.load('./model/in_encoder')
    out_encoder = joblib.load('./model/out_encoder')
    
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

def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = np.expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

def load_svm_model():
    return joblib.load('./model/svm_face_recognitor')

def load_true_name():
    data = load_files('./datasets')
    return np.array(data.target_names)

def blob_to_image(blob):
    image = Image.open(io.BytesIO(blob))
    return np.asarray(image)

def load_datasets_facenet(kernel = np.ones((5,5),np.float32)/25):
    train = load_files('./datasets')
    trainX = []

    for i in range(len(train.data)):
        try:
            image = blob_to_image(train.data[i])
            image = cv2.filter2D(image, -1, kernel)
            image = cv2.resize(image, (160, 160))
            trainX.append(image)
        except PIL.UnidentifiedImageError:
            print(f"error at {i}")

    return np.array(trainX), np.array(train.target), np.array(train.target_names), np.array(list(set(train.target)))

def data_aug(X, y):
    datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
    
    datagen.fit(X)
    X_au = []
    y_au = []
    for i in np.arange(len(X)):
        no_img = 0
        for x in datagen.flow(np.expand_dims(X[i], axis = 0), batch_size = 1):
            X_au.append(x[0])
            y_au.append(y[i])
            no_img += 1
            if no_img == 5:
                break

    return np.asarray(X_au), np.array(y_au)

def get_emb_from_dataset(dataset, model):
    newDataset = []
    for face_pixcels in dataset:
        embedding = get_embedding(model, face_pixcels)
        newDataset.append(embedding)
        
    return np.asarray(newDataset)

def train_model():
	try:
		trainX, y, _, _1 = load_datasets_facenet()
		X_train, _2, y_train, _3 = train_test_split(
			trainX, y, test_size=0.25, random_state=42
		)
		
		model = load_model(FACENET_PATH)
		newTrainX = get_emb_from_dataset(X_train, model)
		in_encoder = Normalizer()
		trainX = in_encoder.transform(newTrainX)
		
		out_encoder = LabelEncoder()
		out_encoder.fit(y_train)
		trainy = out_encoder.transform(y_train)
		SVM_clf = LinearSVC(C=np.inf, loss='hinge', max_iter=100000, random_state=300)
		SVM_clf.fit(trainX, trainy)
		joblib.dump(SVM_clf, './model/svm_face_recognitor')

		return True
	except:
		return False

def get_face_image(username, path = './model/haarcascade.xml'):
    faceClassifier = cv2.CascadeClassifier(path)
    cap = cv2.VideoCapture(f'./videos/{username}.mp4')
    cap.set(3,640) # set Width
    cap.set(4,480) # set Height

    count = 0
    while(True):
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceClassifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20,20)
        )

        for (x,y,w,h) in faces:
            count += 1
            cv2.imwrite(f'./datasets/{username}/{count}.jpg', frame[y:y+h, x:x+w])
        
        cv2.imshow('frame', gray)
        k = cv2.waitKey(30) & 0xff
        if count >= 30: # press 'ESC' to quit
            break

    cap.release()
    cv2.destroyAllWindows()