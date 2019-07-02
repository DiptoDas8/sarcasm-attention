from keras.optimizers import Adam
import keras.backend as K
from sklearn.model_selection import train_test_split
import dataset, model
import numpy as np
import os
from pprint import pprint
import tensorflow as tf

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess)

print("[info]: loading position data")
input_path = os.path.sep.join(["../images", "xy_00-49.csv"])
df = dataset.load_position(input_path)
# pprint(df)

print("[info]: loading image data")
images = dataset.load_images("../images/")
# images = images/255.0

split = train_test_split(df, images, test_size=0.20, random_state=42)
(train_posx, test_posx, train_images, test_images) = split
# train_images = train_images/255.0
# test_images = test_images/255.0
print(len(train_posx), len(test_posx), len(train_images), len(test_images))

def mean_pred(y_true, y_pred):
    print('hi')
    # print(y_true.eval())
    return K.mean(y_pred)

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

# reg_model = model.create_cnn(64, 64, 3, regress=True)
reg_model = model.transferlearning(regress=True)
optimizer = Adam(lr=1e-3, decay=1e-3/200)
reg_model.compile(optimizer=optimizer, loss='mean_absolute_percentage_error', metrics=[rmse])

print("[info]: start training")
reg_model.fit(train_images, train_posx, validation_data=(test_images, test_posx), epochs=20, batch_size=8)

print("[info]: prediction")
preds = reg_model.predict(test_images)

pprint(preds)
diff = preds - test_posx
diff_frac = np.abs(diff/test_posx)

mean = np.mean(diff_frac)
std = np.std(diff_frac)

print("[res]: mean: ", mean, " std: ", std)
reg_model_json = reg_model.to_json()
with open('model_struc.json', 'w') as jsonfile:
    jsonfile.write(reg_model_json)
reg_model.save_weights('model_weights.h5')
print('saved model')
