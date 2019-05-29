
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout

# Initialising the CNN
model = Sequential()
model.add(Conv2D(64, kernel_size=4, strides=1, activation='relu', input_shape=(64,64,3)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, kernel_size=4, strides=1, activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dense(27, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('asl_alphabet_train/asl_alphabet_train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('asl_alphabet_test/asl_alphabet_test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

model.fit_generator(training_set,
                         steps_per_epoch = 32571,
                         epochs = 3,
                         validation_data = test_set,
                         validation_steps = 27)

from keras.models import load_model

model.save('my_model.h5')
my_model = load_model('my_model.h5')


    
from glob import glob
import numpy as np
from keras.preprocessing import image

import time
import cv2

def video_generation():
    
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('Output.avi',fourcc,20.0,(640,480))
    
    while(True):
        ret, frame = cap.read()
        #path = 'predict/image'+str(i)+'.png'
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def image_generation():
    cap = cv2.VideoCapture('Output.avi')
    i=1
    while(True):
        ret, frame = cap.read()
        path = 'predict/image'+str(i)+'.png'
        i+=1
        cv2.imwrite(path,frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(1)
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

video_generation()   
image_generation()    

test = image.load_img('predict/image12.png',target_size=(64,64))
test = image.img_to_array(test)
test = np.expand_dims(test,axis=0)
res = my_model.predict(test)
y_pred = np.argmax(res,axis=1)

'''    
data = []
files = glob('predict/*.png')
for file in files:
    test = image.load_img(file,target_size=(64,64))
    test = image.img_to_array(test)
    test = np.expand_dims(test,axis=0)
    res = my_model.predict(test)
    y_pred = np.argmax(res,axis=1)
    data.append(y_pred)
    
'''  

data = np.array(data)
    #data.append(test)
       


y_hist = training_set.class_indices
result = []
for i in y_pred:
    if i==0:
        result.append('A')
    elif i==1:
        result.append('B')
    elif i==2:
        result.append('C')
    elif i==3:
        result.append('D')
    elif i==4:
        result.append('E')
    elif i==5:
        result.append('F')
    elif i==6:
        result.append('G')
    elif i==7:
        result.append('H')
    elif i==8:
        result.append('I')
    elif i==9:
        result.append('J')
    elif i==10:
        result.append('K')
    elif i==11:
        result.append('L')
    elif i==12:
        result.append('M')
    elif i==13:
        result.append('N')
    elif i==14:
        result.append('O')
    elif i==15:
        result.append('P')
    elif i==16:
        result.append('Q')
    elif i==17:
        result.append('R')
    elif i==18:
        result.append('S')
    elif i==19:
        result.append('T')
    elif i==20:
        result.append('U')
    elif i==21:
        result.append('V')
    elif i==22:
        result.append('W')
    elif i==23:
        result.append('X')  
    elif i==24:
        result.append('Y')
    elif i==25:
        result.append('Z')
    elif i==26:
        result.append(' ')
        
write = ''.join(result)


s = os.getcwd()

os.remove('D:\Python Projects\asl-alphabet\Output.avi')
for file in files:
    os.remove('D:\Python Projects\asl-alphabet\predict\file')
    
import os
from gtts import gTTS
import pyglet
import time


test = 'I am Abhishek'
language = 'en'

obj = gTTS(text = test,lang=language,slow = False)
filename = 'first.mp3'
obj.save(filename)

word = pyglet.media.load('first.mp3', streaming=False)
word.play()
time.sleep(word.duration)
os.remove('filename')
    
    
    

