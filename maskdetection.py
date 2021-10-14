try:
    import sys
    import numpy as np
    import os
    import keras
    import keras.backend as k
    from keras.layers import Conv2D,MaxPooling2D,SpatialDropout2D,Flatten,Dropout,Dense
    from keras.models import Sequential,load_model
    from tensorflow.keras.optimizers  import Adam
    from keras.preprocessing import image
    import cv2
    import datetime
    import pandas as pd
except Exception as e:
    print(e)

oldname = sys.argv[1]
newname=sys.argv[1]+".jpg"
os.rename(oldname,newname)

print("file renamed")
#newname=sys.argv[1]
model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D() )
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D() )
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D() )
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
mymodel=model.load_weights('mymodel.h5')
test_image=image.load_img(newname,
                          target_size=(150,150,3))

test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
pred=mymodel.predict(test_image)[0][0]
print("PRED is ",pred)
if pred==1:
    print("Not wearing mask")
else:
    print("Wearing mask")
# os.move(newname,"public/image.jpg")