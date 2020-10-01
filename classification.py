import data_processing as dp
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt
import datetime

#model configuration setting
CLASS_WEIGHT = {0:0.015, 1:0.985}
LEARNING_RATE = 0.0001

#gpu configuration
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config = config)
set_session(sess)

#read csv file,
data = dp.InputData("Train_data.csv")
x_train, y_train, x_val, y_val, x_test, y_test = data.dataSplit(0.9, 0.03, 0.07)

model = Sequential()
model.add(Dense(100,input_shape=(x_train.shape[1],),activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy', 
              optimizer=Adam(LEARNING_RATE),
              metrics=['accuracy', tf.keras.metrics.Recall(name='recall'),tf.keras.metrics.Precision(name='precision')])
model.summary()

class_weight = CLASS_WEIGHT
hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size = 200, epochs=24, class_weight = class_weight)
loss, accuracy , recall, precision = model.evaluate(x_val, y_val)
print("Accuracy = {:.2f}".format(accuracy))

plt.figure(figsize=(12,8))
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.legend(['loss', 'val_loss', 'accuracy', 'val_accuracy'])
plt.grid()
plt.subplot(2, 1, 2)

plt.plot(hist.history['precision'])
plt.plot(hist.history['val_precision'])
plt.plot(hist.history['recall'])
plt.plot(hist.history['val_recall'])
plt.legend(['presicion', 'val_precision', 'recall', 'val_recall'])
plt.grid()
plt.show()
# y_pred = model.predict(x_test)
# y_test_class = np.argmax(y_test,axis=1)
# y_pred_class = np.argmax(y_pred,axis=1)
#
# print(classification_report(y_test_class,y_pred_class))
# print(confusion_matrix(y_test_class,y_pred_class))

y_pred = model.predict(x_test)
y_test_class = np.argmax(y_test,axis=1)
y_pred_class = np.argmax(y_pred,axis=1)
mat = confusion_matrix(y_test_class, y_pred_class)

co = mat[0, 1]*10 + mat[1,0]*500
print(mat)
print(co)
print(classification_report(y_test_class,y_pred_class))
model.save(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+".h5")
print(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+".h5" + " saved")
