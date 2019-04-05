from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

import numpy as np

pokemon_train = []

start_string = "Spr_2g_"
data_ex = ".npy"

for i in range(251):
    #the + 1 is because we dont have a Spr_1b_000.png
    image_string = "./ml_conv_data/" + start_string + str(i + 1).zfill(3) + data_ex
    pokemon_train.append(np.load(image_string))

pokemon_train = np.array(pokemon_train)
pokemon_train = np.reshape(pokemon_train, (len(pokemon_train), 56, 56, 4))
input_img = Input(shape=(56, 56, 4))  # adapt this if using `channels_first` image data format

x = Conv2D(126, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(126, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(126, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(126, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(126, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)
#x = Conv2D(255, (3, 3), activation='relu')(x)
x = UpSampling2D((4, 4))(x)
decoded = Conv2D(4, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(pokemon_train, pokemon_train, epochs=100, batch_size=32, shuffle=True, validation_data=(pokemon_train, pokemon_train))
autoencoder.save("conv_auto")