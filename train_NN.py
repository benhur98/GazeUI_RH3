from modelsNN import inceptionv3 as gnet
import numpy as np
import cv2
import time
import os


WIDTH = 250
HEIGHT = 250
LR = 1e-3
# EPOCHS = 17
EPOCHS = 2
LOAD_MODEL = False
i=3
MODEL_NAME="CNN-{}".format(i)
file_name ="train-data-{}.npy".format(i)
a = [1,0,0,0,0,0,0,0,0]
b = [0,1,0,0,0,0,0,0,0]
c = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
e = [0,0,0,0,1,0,0,0,0]
f = [0,0,0,0,0,1,0,0,0]
g = [0,0,0,0,0,0,1,0,0]
h = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]
model = gnet(WIDTH, HEIGHT, 3, LR, output=9, model_name=MODEL_NAME)
if LOAD_MODEL:
    model.load(MODEL_NAME)
    print('loaded previous model into inception_v3 as gnet!!')
train_data = np.load(file_name)
#np.random.shuffle(train_data)
train=train_data
test=train
for i in range(1,EPOCHS):
    try:
        #train = train_data[(i-1)*25:(i*25)]
        #train = train_data[(i - 1) * 100:(i * 100)]
##        train=train_data[-100:]
##        test=train_data[:-100]
##        
        #np.random.shuffle(test)
        
                
        X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 3)
        Y = [i[1] for i in train]
        
        T_X = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 3)
        T_Y = [i[1] for i in test]
##snap_step=200        
        model.fit({'input': X}, {'targets': Y}, n_epoch=30, validation_set=({'input': T_X}, {'targets': T_Y}),
                  snapshot_step=2500, show_metric=True, run_id=MODEL_NAME)
        if i == (EPOCHS-1):
            print('Training complete - SAVING MODEL_NN ....')
            model.save(MODEL_NAME)
    except Exception as e:
        print("Error occured",str(e))
print("Complete-END")
