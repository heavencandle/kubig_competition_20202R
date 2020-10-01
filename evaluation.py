import data_processing as dp
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.models import load_model
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
from keras.backend.tensorflow_backend import set_session


MODEL_NAME = "20201001_221051"
LEARNING_RATE = 0.0001

#configuration
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config = config)
set_session(sess)

#read csv file, use test data
data = dp.InputData("Train_data.csv")
x_train, y_train, x_val, y_val, x_test, y_test = data.dataSplit(0.0, 0.0, 1.0)

#get saved model
model = load_model(MODEL_NAME+".h5")

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(LEARNING_RATE),
              metrics=['accuracy', tf.keras.metrics.Recall(name='recall'),tf.keras.metrics.Precision(name='precision')])

#predict test data
y_pred = model.predict(x_test)
y_test_class = np.argmax(y_test,axis=1)
y_pred_class = np.argmax(y_pred,axis=1)
mat = confusion_matrix(y_test_class, y_pred_class)

co = mat[0, 1]*10 + mat[1,0]*500
print(mat)
print(co)
print(classification_report(y_test_class,y_pred_class))
