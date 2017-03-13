#!/usr/bin/python
import cv2, os
import numpy as np
from PIL import Image
import csv
import pandas as pd
import pdb as pdb


# Different recognizers
recognizer_LBPH = cv2.face.createLBPHFaceRecognizer()
recognizer_Fisher = cv2.face.createFisherFaceRecognizer()
recognizer_Eigen = cv2.face.createEigenFaceRecognizer()

# haar cascades for recognizing different angles
cascade_paths = ['haarcascade_frontalface_alt.xml', 'haarcascade_frontalface_default.xml', 'haarcascade_profileface.xml']

face_alt = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
face_default = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_profile = cv2.CascadeClassifier("haarcascade_profileface.xml")

#Detect face using 4 different classifiers


settings = {
    'minNeighbors': 4, 
    'minSize': (40,40)
}

# Import both the training and test datasets
train = pd.read_csv('train.csv', header=None, delimiter=",")
train.columns=['emotion','pixels','data_type']
test = pd.read_csv('test_public.csv', header=None, delimiter=",")
test.columns=['emotion','pixels','data_type']

# Extract relevant data
training_data = train['pixels']
training_labels = train['emotion']
prediction_data = test['pixels']
prediction_labels = test['emotion']

train_X=[]
train_Y=[]
test_X=[]
test_Y=[]

num_detect = 0
num_not_detect = 0

print 'detecting faces from training data'
for j, pixels in enumerate(training_data):
    # greyscale, resize, convert image to numpy array
    image = map(int, pixels.split(' '))
    image = np.array(image).astype('uint8')
    image = np.reshape(image, (48, 48))
    pil_image = Image.fromarray(image, 'L')
    image = np.array(pil_image, 'uint8')

    #detect using different classifiers
    face = face_alt.detectMultiScale(image, **settings)
    face2 = face_default.detectMultiScale(image, **settings)
    face3 = face_profile.detectMultiScale(image, **settings)

    #Go over detected faces, stop at first detected face, return empty if no face.
    if len(face) == 1:
        facefeatures = face
    elif len(face2) == 1:
        facefeatures = face2
    elif len(face3) == 1:
        facefeatures = face3
    else:
        facefeatures = ""
        num_not_detect = num_not_detect + 1


    for (x, y, w, h) in facefeatures:
        num_detect = num_detect + 1
        train_X.append(image)#[y: y + h, x: x + w])
        train_Y.append(train['emotion'][j])
        cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
        #cv2.waitKey(100)

percentage = num_detect/float((num_detect + num_not_detect))
print 'detected {} faces, out of {} ({:0.2f})'.format(num_detect, num_detect + num_not_detect, percentage)


# print 'predicting using LBPH'
# correct = 0
# incorrect = 0 
# recognizer_LBPH.train(train_X,np.array(train_Y))

# for i, image in enumerate(prediction_data):

#     image = map(int, image.split(' '))
#     image = np.array(image).astype('uint8')
#     image = np.reshape(image, (48, 48))
#     pil_image = Image.fromarray(image, 'L')
#     image = np.array(pil_image, 'uint8')

#     pred, conf = recognizer_LBPH.predict(image)

#     if pred == prediction_labels[i]:
#         correct += 1
#     else:
#         incorrect += 1
# print 'accuracy using LBPH: {}%'.format((100*correct)/(correct + incorrect))

print 'predicting using Fisher'
correct = 0
incorrect = 0 
recognizer_Fisher.train(train_X,np.array(train_Y))

for i, image in enumerate(prediction_data):

    image = map(int, image.split(' '))
    image = np.array(image).astype('uint8')
    image = np.reshape(image, (48, 48))
    pil_image = Image.fromarray(image, 'L')
    image = np.array(pil_image, 'uint8')

    pred, conf = recognizer_Fisher.predict(image)

    if pred == prediction_labels[i]:
        correct += 1
    else:
        incorrect += 1
print 'accuracy using Fisher: {}%'.format((100*correct)/(correct + incorrect))


print 'predicting using Eigen'
correct = 0
incorrect = 0 
recognizer_Eigen.train(train_X,np.array(train_Y))
for i, image in enumerate(prediction_data):

    image = map(int, image.split(' '))
    image = np.array(image).astype('uint8')
    image = np.reshape(image, (48, 48))
    pil_image = Image.fromarray(image, 'L')
    image = np.array(pil_image, 'uint8')

    pred, conf = recognizer_Eigen.predict(image)

    if pred == prediction_labels[i]:
        correct += 1
    else:
        incorrect += 1
print 'accuracy using Eigen: {}%'.format((100*correct)/(correct + incorrect))






