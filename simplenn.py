import datahandler

from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D

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
train_samples, validation_samples = datahandler.split_data()

train_generator = datahandler.generator(train_samples)
validation_generator = datahandler.generator(validation_samples)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples) * 4,
                    validation_data=validation_generator, nb_epoch=1, nb_val_samples=len(validation_samples) * 4)
model.save('model.h5')

