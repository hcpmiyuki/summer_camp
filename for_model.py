from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import argparse

model=load_model('fer2013_mini_XCEPTION.110-0.65.hdf5', compile=False)

def for_model(image_array):
    # X = []
    # X.append(image_array)
    # X = np.array(X)
    # X = X.astype("float") / 256
    classes = ({0:'angry',1:'disgust',2:'fear',3:'happy',
        4:'sad',5:'surprise',6:'neutral'})


    pre = model.predict(image_array)[0]
    top_indices = pre.argsort()[-5:][::-1]
    result = [(classes[i] , pre[i]) for i in top_indices]

    return result
