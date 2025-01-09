from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, BatchNormalization, Dropout

def alexNet(num_classes):
    """
    The model takes an RGB image of fixed size (224x224 pixels) as input.
    """
    model = Sequential()
    # 1st Convolutional Layer
    model.add(Conv2D(filters=96,
                     kernel_size=(11,11),
                     strides=(4,4),
                     activation='relu'))
    # Max-Pooling
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Batch Normalisation
    model.add(BatchNormalization())
    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256,
                     kernel_size=(11,11),
                     activation='relu'))
    # Max-Pooling
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Batch Normalisation
    model.add(BatchNormalization())
    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384,
                     kernel_size=(3,3),
                     activation='relu'))
    # Batch Normalisation
    model.add(BatchNormalization())
    # 4th Convolutional Layer
    model.add(Conv2D(filters=384,
                     kernel_size=(3,3),
                     activation='relu'))
    # Batch Normalisation
    model.add(BatchNormalization())
    # 5th Convolutional Layer
    model.add(Conv2D(filters=256,
                     kernel_size=(3,3),
                     activation='relu'))
    # Max-Pooling
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Batch Normalisation
    model.add(BatchNormalization())
    # Flattening
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(units=4096,
                    input_shape=(224*224*3, ),
                    activation='relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())
    # 2nd Dense Layer
    model.add(Dense(units=4096,
              activation='relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())
    # Output Softmax Layer
    model.add(Dense(units=num_classes,
                    activation='softmax'))
    return model

def leNet5(num_classes):
    """
    Takes greyscale images 28x28 to 32x32 with all values normalised between 0 and 1
    """
    model = Sequential()
    # Layer C1
    model.add(Conv2D(filters=6,
                     kernel_size=(5,5),
                     activation='relu'))
    # Layer S2
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Layer C3
    model.add(Conv2D(filters=16,
                     kernel_size=(5,5),
                     activation='relu'))
    # Layer S4
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Flatten units
    model.add(Flatten())
    # Layer C5
    model.add(Dense(units=120,
                    activation='relu'))
    # Layer F6
    model.add(Dense(units=84,
                    activation='relu'))
    # Output layer
    model.add(Dense(units=num_classes,
                    activation='softmax'))
    return model

def vgg16C(num_classes):
    """
    VGG-16 configuration C
    """
    model = Sequential()
    # Block 1
    # Convolution Layer 1.1
    model.add(Conv2D(filters=64,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Convolution Layer 1.2
    model.add(Conv2D(filters=64,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Max-Pooling
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Block 2
    # Convolution Layer 2.1
    model.add(Conv2D(filters=128,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Convolution Layer 2.2
    model.add(Conv2D(filters=128,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Max-Pooling
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Block 3
    # Convolution Layer 3.1
    model.add(Conv2D(filters=256,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Convolution Layer 3.2
    model.add(Conv2D(filters=256,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Convolution Layer 3.3
    model.add(Conv2D(filters=256,
                     kernel_size=(1,1),
                     activation='relu',
                     padding='same'))
    # Max-Pooling
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Block 4
    # Convolution Layer 4.1
    model.add(Conv2D(filters=512,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Convolution Layer 4.2
    model.add(Conv2D(filters=512,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Convolution Layer 4.3
    model.add(Conv2D(filters=512,
                     kernel_size=(1,1),
                     activation='relu',
                     padding='same'))
    # Max-Pooling
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Block 5
    # Convolution Layer 5.1
    model.add(Conv2D(filters=512,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Convolution Layer 5.2
    model.add(Conv2D(filters=512,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Convolution Layer 5.3
    model.add(Conv2D(filters=512,
                     kernel_size=(1,1),
                     activation='relu',
                     padding='same'))
    # Max-Pooling
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Flattening
    model.add(Flatten())
    # Fully Connected Layers
    # Fully Connected 1
    model.add(Dense(units=4096,
                    activation='relu'))
    # Fully Connected 2
    model.add(Dense(units=4096,
                    activation='relu'))
    # Fully Connected 3 (OUTPUT)
    model.add(Dense(units=num_classes,
                    activation='softmax'))
    return model

def vgg16(num_classes):
    """
    VGG-16 configuration D
    """
    model = Sequential()
    # Block 1
    # Convolution Layer 1.1
    model.add(Conv2D(filters=64,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Convolution Layer 1.2
    model.add(Conv2D(filters=64,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Max-Pooling
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Block 2
    # Convolution Layer 2.1
    model.add(Conv2D(filters=128,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Convolution Layer 2.2
    model.add(Conv2D(filters=128,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Max-Pooling
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Block 3
    # Convolution Layer 3.1
    model.add(Conv2D(filters=256,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Convolution Layer 3.2
    model.add(Conv2D(filters=256,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Convolution Layer 3.3
    model.add(Conv2D(filters=256,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Max-Pooling
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Block 4
    # Convolution Layer 4.1
    model.add(Conv2D(filters=512,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Convolution Layer 4.2
    model.add(Conv2D(filters=512,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Convolution Layer 4.3
    model.add(Conv2D(filters=512,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Max-Pooling
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Block 5
    # Convolution Layer 5.1
    model.add(Conv2D(filters=512,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Convolution Layer 5.2
    model.add(Conv2D(filters=512,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Convolution Layer 5.3
    model.add(Conv2D(filters=512,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Max-Pooling
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Flattening
    model.add(Flatten())
    # Fully Connected Layers
    # Fully Connected 1
    model.add(Dense(units=4096,
                    activation='relu'))
    # Fully Connected 2
    model.add(Dense(units=4096,
                    activation='relu'))
    # Fully Connected 3 (OUTPUT)
    model.add(Dense(units=num_classes,
                    activation='softmax'))
    return model

def vgg19(num_classes):
    model = Sequential()
    # Block 1
    # Convolution Layer 1.1
    model.add(Conv2D(filters=64,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Convolution Layer 1.2
    model.add(Conv2D(filters=64,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Max-Pooling
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Block 2
    # Convolution Layer 2.1
    model.add(Conv2D(filters=128,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Convolution Layer 2.2
    model.add(Conv2D(filters=128,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Max-Pooling
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Block 3
    # Convolution Layer 3.1
    model.add(Conv2D(filters=256,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Convolution Layer 3.2
    model.add(Conv2D(filters=256,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Convolution Layer 3.3
    model.add(Conv2D(filters=256,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Convolution Layer 3.4
    model.add(Conv2D(filters=256,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Max-Pooling
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Block 4
    # Convolution Layer 4.1
    model.add(Conv2D(filters=512,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Convolution Layer 4.2
    model.add(Conv2D(filters=512,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Convolution Layer 4.3
    model.add(Conv2D(filters=512,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Convolution Layer 4.4
    model.add(Conv2D(filters=512,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Max-Pooling
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Block 5
    # Convolution Layer 5.1
    model.add(Conv2D(filters=512,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Convolution Layer 5.2
    model.add(Conv2D(filters=512,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Convolution Layer 5.3
    model.add(Conv2D(filters=512,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Convolution Layer 5.4
    model.add(Conv2D(filters=512,
                     kernel_size=(3,3),
                     activation='relu',
                     padding='same'))
    # Max-Pooling
    model.add(MaxPooling2D(pool_size=(2,2)))
    # Flattening
    model.add(Flatten())
    # Fully Connected Layers
    # Fully Connected 1
    model.add(Dense(units=4096,
                    activation='relu'))
    # Fully Connected 2
    model.add(Dense(units=4096,
                    activation='relu'))
    # Fully Connected 3 (OUTPUT)
    model.add(Dense(units=num_classes,
                    activation='softmax'))
    return model