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

import tensorflow.keras.datasets as datasets
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

def createModel(dataset, cnn_model, model_name, epoch=100, batch=2048):
    # Load dataset
    match dataset:
        case "numbers":
            (X_train,y_train),(X_test,y_test) = datasets.mnist.load_data()
            class_count = 10
        case "fashion_mnist":
            (X_train,y_train),(X_test,y_test) = datasets.fashion_mnist.load_data()
            class_count = 10
        case "cifar10":
            (X_train,y_train),(X_test,y_test) = datasets.cifar10.load_data()
            class_count = 10

    # Dataset to categorical
    y_train_cat = to_categorical(y_train, num_classes=class_count)
    y_test_cat = to_categorical(y_test, num_classes=class_count)

    if(cnn_model == "leNet5"):
        # Normalise data
        X_train_norm = X_train / 255
        X_test_norm = X_test / 255
        # Reshape data
        X_train_norm = X_train_norm.reshape(X_train_norm.shape[0], X_train_norm.shape[1], X_train_norm.shape[2], 1)
        X_test_norm = X_test_norm.reshape(X_test_norm.shape[0], X_test_norm.shape[1], X_test_norm.shape[2], 1)
    else:
        # Reshape data
        X_train_norm = X_train_norm.reshape(X_train_norm.shape[0], X_train_norm.shape[1], X_train_norm.shape[2], 3)
        X_test_norm = X_test_norm.reshape(X_test_norm.shape[0], X_test_norm.shape[1], X_test_norm.shape[2], 3)

    # Select model
    match cnn_model:
        case "leNet5":
            model = leNet5(class_count)
        case "alexNet":
            model = alexNet(class_count)
        case "vgg16C":
            model = vgg16C(class_count)
        case "vgg16":
            model = vgg16(class_count)
        case "vgg19":
            model = vgg19(class_count)
        case _:
            print("UNDEFINED CNN_MODEL\nThe supported models are LeNet-5 (leNet5), AlexNet (alexNet), VGG-16 Configuration C (vgg16C), VGG-16 Configuration D (vgg16) and VGG-19 (vgg19)")
            return
    
    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Define callback
    callback = [EarlyStopping(monitor='val_loss', patience=10)]

    # Fit the model
    model.fit(x=X_train_norm, y=y_train_cat, validation_data=(X_test_norm,y_test_cat), epochs=epoch, batch_size=batch, callbacks=callback)

    # Save model
    model.save(model_name + ".keras")

    # Get last accuracy scores
    batch_size = 1024
    y_pred_train = to_categorical(model.predict(X_train_norm,batch_size=batch_size).argmax(axis=1), num_classes=10)
    y_pred_test = to_categorical(model.predict(X_test_norm,batch_size=batch_size).argmax(axis=1), num_classes=10)

    print('Training Accuracy:', accuracy_score(y_pred_train, y_train_cat))
    print('Testing Accuracy:', accuracy_score(y_pred_test, y_test_cat))
    return

def loadModel(model_name):
    if(".keras" not in model_name):
        model_name += ".keras"
    # Load model
    model = load_model(model_name)
    return model

def predictImage(model_name, image):
    # Load model
    model = loadModel(model_name)
    # Use saved model to predict
    print(to_categorical(model.predict(image)))
    return