# Mengimpor library Keras dan turunannya
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

# Mengimpor dataset
dataset = pd.read_csv('DATA.csv')
X = dataset.iloc[:, 1:4].values
y = dataset.iloc[:, 4].values

# Resampling Data
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)
X, y = sm.fit_sample(X, y.ravel())

# Membagi data ke test dan training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Mendefenisikan keras model
model = Sequential()
model.add(Dense(3, kernel_initializer = 'uniform', input_dim=3, activation='relu'))
model.add(Dense(3, kernel_initializer = 'uniform', activation='relu'))
model.add(Dense(1, kernel_initializer = 'uniform', activation='sigmoid'))

# Kompilasi keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the keras model on the dataset
history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=100, batch_size=10, verbose = 1)

# Import plot
import matplotlib.pyplot as plt

# Summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Import library model
from tensorflow.python.keras.saving.save import load_model

# Simpan model
model.save("model.h5")
print("Model disimpan")

# Load model
loaded_model = load_model('model.h5')
print("Model diload")

# Evluasi keras model
_, accuracy = loaded_model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

# make class predictions with the model
predictions = loaded_model.predict_classes(X)

# summarize the first 10 cases
for i in range(10):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))

# Prediksi 1 data inputan
tes = [[48, 4, 3]]
prediksi = loaded_model.predict_classes(tes)
prediksi = (prediksi == 1)
print ("48,4,3 : ")
print (prediksi)

# Prediksi 1 data inputan
tes = [[1, 1, 1]]
prediksi = loaded_model.predict_classes(tes)
prediksi = (prediksi == 1)
print ("1,1,1 : ")
print (prediksi)

# Memprediksi hasil test set
y_pred = loaded_model.predict_classes(X_test)
y_pred = (y_pred > 0.5)

# Membuat confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print ("Confusion Matrix")
print (cm)

from sklearn.metrics import classification_report
print (classification_report(y_test, y_pred))