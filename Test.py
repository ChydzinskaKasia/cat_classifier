import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.preprocessing import image


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


##Sciezki do zdjec
Base_path = os.path.dirname(os.path.abspath(__file__))

#Test sieci

classifier = tf.keras.models.load_model(os.path.join(Base_path, r'PiesvsKot_model.h5'))

#test_image = image.load_img(os.path.join(Base_path, r'Baza\\Test\\Pieski\\dog.4087.jpg'))
#test_image = image.load_img(os.path.join(Base_path, r'Baza\\Training\\Pieski\\dog.13.jpg'))
test_image = image.load_img(os.path.join(Base_path, r'Baza\\tosia.jpg'))



test_image = test_image.resize((200, 200),3)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
results = classifier.predict(test_image)
#training_set.class_indices
if results[0][0] >= 0.8:
    prediction = 'pies'
else:
    prediction = 'kot'

print(prediction)
print(results[0][0])
