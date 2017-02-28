import datahandler as dh

from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D

X_train, y_train = dh.get()

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3), output_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Convolution2D(6, 5, 5, activation='relu', input_shape=(160, 320, 3)))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

# model.load_weights('model.h5')
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, verbose=1, nb_epoch=5)
model.save('model.h5')

exit()
