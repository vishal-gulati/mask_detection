try:
    import sys
    import numpy as np
    import os
    from keras.models import load_model
    from keras.preprocessing import image
except Exception as e:
    print(e)

oldname = sys.argv[1]
newname=sys.argv[1]+".jpg"
os.rename(oldname,newname)

mymodel=load_model('mymodel.h5')
test_image=image.load_img(newname,
                          target_size=(150,150,3))
os.replace(newname,"public/image.jpg")
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
pred=mymodel.predict(test_image)[0][0]
if pred==1:
    print("Not wearing mask")
else:
    print("Wearing mask")