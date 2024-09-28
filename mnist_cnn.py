import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)
use_saved_model = True
model_file = "mnistmodel.keras"

if not use_saved_model:
    
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    # x_train shape: (60000, 28, 28, 1)
    # 60000 train samples
    # 10000 test samples
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    # Build the model
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    
    # Model: "sequential"
    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
    # ┃ Layer (type)                    ┃ Output Shape              ┃    Param # ┃
    # ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
    # │ conv2d (Conv2D)                 │ (None, 26, 26, 32)        │        320 │
    # ├─────────────────────────────────┼───────────────────────────┼────────────┤
    # │ max_pooling2d (MaxPooling2D)    │ (None, 13, 13, 32)        │          0 │
    # ├─────────────────────────────────┼───────────────────────────┼────────────┤
    # │ conv2d_1 (Conv2D)               │ (None, 11, 11, 64)        │     18,496 │
    # ├─────────────────────────────────┼───────────────────────────┼────────────┤
    # │ max_pooling2d_1 (MaxPooling2D)  │ (None, 5, 5, 64)          │          0 │
    # ├─────────────────────────────────┼───────────────────────────┼────────────┤
    # │ flatten (Flatten)               │ (None, 1600)              │          0 │
    # ├─────────────────────────────────┼───────────────────────────┼────────────┤
    # │ dropout (Dropout)               │ (None, 1600)              │          0 │
    # ├─────────────────────────────────┼───────────────────────────┼────────────┤
    # │ dense (Dense)                   │ (None, 10)                │     16,010 │
    # └─────────────────────────────────┴───────────────────────────┴────────────┘
    #  Total params: 34,826 (136.04 KB)
    #  Trainable params: 34,826 (136.04 KB)
    #  Non-trainable params: 0 (0.00 B)
    model.summary()
    
    # Train the model
    batch_size = 128
    epochs = 15
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    
    
    # show model loss and accuracy
    plt.figure(figsize=(13, 5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Model Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'])
    plt.grid()
    plt.show()
    
    
    plt.figure(figsize=(13, 5))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train','Test'])
    plt.grid()
    plt.show()
    
    
    # Evaluate the trained model
    score = model.evaluate(x_test, y_test, verbose=0)
    
    # Test loss: 0.02499214932322502
    # Test accuracy: 0.9919000267982483
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    model.save(model_file)

else:
    model = keras.models.load_model(model_file)


# process image for prediction - test_img2 is '1'
file='/books/MachineLearning/FastAPI/FastAPI-ml-demo/test_img2.png'

# img = []
# temp = np.array(Image.open(file).resize((256, 256), Image.ANTIALIAS))
# temp.shape = temp.shape + (1,) # now its (256, 256, 1)
# img.append(temp)
# test = np.array(img) # (1, 1024, 1024, 1)
# prediction = model.predict(test) 


# gray scale is mode ('L') see site-packages/PIL/Image.py line 940
img = Image.open(file).convert('L')
img = np.array(img).astype("float32") / 255
img = np.expand_dims(img, (0, -1))

# predict the result
predict = model.predict(img)
result = predict.argmax()
print("result", str(result))
