from fastapi import FastAPI, UploadFile, BackgroundTasks
from contextlib import asynccontextmanager

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import os
import pickle
import logging
import numpy as np
import keras
from keras import layers

from PIL import Image
from io import BytesIO

from zipfile import ZipFile

# log to console
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)


logger.debug('init models dict and models params')    

ml_models = {}

# MNIST model meta parameters
num_classes = 10
input_shape = (28, 28, 1)
batch_size = 128
epochs = 15

def penguins_pipeline():
    # a classification prediction model that uses the Nearest Neighbors algorithm 
    # to predict the species of various penguins based on their bill and flipper length         
    logger.debug('penguins_pipeline starting...')        
    
    data = pd.read_csv('penguins.csv')
    data = data.dropna()

    # encode target labels (y) with value between 0 and n_classes-1
    le = preprocessing.LabelEncoder()
    
    # drop columns (features) we dont need
    X = data[["bill_length_mm", "flipper_length_mm"]]
    
    # extract labels from the dataset
    le.fit(data["species"])

    # transform labels to normalized encoding (encode values into [0, n_uniques - 1])
    y = le.transform(data["species"])
    
    # create data sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
    
    # create our pipeline: a sequentially-applied set of transformers to preprocess 
    # the data and (optionally) a final `predictor` for predictive modeling
    clf = Pipeline(
        steps=[("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=11))]
    )
    
    # set_params returns the pipeline class instance,then fit the model
    clf.set_params().fit(X_train, y_train)

    logger.debug('penguins_pipeline finished...')

    # return pipeline and labels
    return clf, le

def mnist_pipeline():
    logger.debug('mnist_pipeline starting...')        

    # Load the data and split it between train and test sets
    (x_train, y_train), _ = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255

    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)

    # create our convolutional neural net (CNN) model for image recognition, 
    # compile and train
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

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    logger.debug('mnist_pipeline finished...')        

    return model


LEMODELFILE = "lemodel.pickle"
CLFMODELFILE = "clfmodel.pickle"
CNNMODELFILE = "cnnmodel.keras"


# A `Lifespan` context manager handler. This replaces `startup` and `shutdown` functions
# with a single context manager. Gets called when FastAPI application starts (passed as 
# arg to the FastAPI constructor). REST API is not available until lifespan completes 
# (thanks to it being a context)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.debug('lifespan starting...')        
    
    if os.path.isfile(LEMODELFILE) and os.path.isfile(CLFMODELFILE):
        # use saved models
        logger.debug('using saved LE and CLF models')                
        le = pickle.load(open(LEMODELFILE, "rb"))
        clf = pickle.load(open(CLFMODELFILE, "rb"))

    else:
        # setup the ML model here
        clf, le = penguins_pipeline()
    
        # and save the models
        logger.debug('saving LE and CLF models')                
        pickle.dump(le, open(LEMODELFILE, "wb"))
        pickle.dump(clf, open(CLFMODELFILE, "wb"))

    ml_models["clf"] = clf
    ml_models["le"] = le

    if os.path.isfile(CNNMODELFILE):
        logger.debug('using saved CNN model')                        
        cnn = keras.models.load_model(CNNMODELFILE)
        
    else:
        cnn = mnist_pipeline()
        logger.debug('saving CNN model')                
        cnn.save(CNNMODELFILE)      
        
    ml_models["cnn"] = cnn

    # will resume after yield when called second time (at shutdown)
    yield
    
    # Clean up the models and release resources
    ml_models.clear()


# main, this will callback to lifespan() above, which will train the models
logger.debug('starting FASTAPI app main...')    
app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    logger.debug('root starting...')        
    
    return {
        "Name": "Penguins Prediction",
        "description": "This is a penguins prediction model based on the bill length and flipper length of the bird.",
    }

@app.get("/predict/")
async def predict(bill_length_mm: float = 0.0, flipper_length_mm: float = 0.0):
    logger.debug('predict starting...')        

    param = {
                "bill_length_mm": bill_length_mm,
                "flipper_length_mm": flipper_length_mm
            }
    if bill_length_mm <=0.0 or flipper_length_mm <=0.0:
        return {
            "parameters": param,
            "error message": "Invalid input values",
        }
    else:
        result = ml_models["clf"].predict([[bill_length_mm, flipper_length_mm]])
        return {
            "parameters": param,
            "result": ml_models["le"].inverse_transform(result)[0],
        }

@app.post("/predict-image/")
async def predict_upload_file(file: UploadFile):
    logger.debug('predict_upload_file starting...')        

    img = await file.read()

    # process image for prediction
    img = Image.open(BytesIO(img)).convert('L')
    img = np.array(img).astype("float32") / 255
    img = np.expand_dims(img, (0, -1))

    # predict the result
    result = ml_models["cnn"].predict(img).argmax(axis=-1)[0]
    return {"filename": file.filename,
            "result": str(result)}

@app.post("/upload-images/")
async def retrain_upload_file(file: UploadFile, background_tasks: BackgroundTasks):
    global batch_size

    logger.debug('retrain_upload_file starting... file:{} size:{}'
                 .format(file.filename, file.size))

    # debug URL routing...
    # return {"filename": file.filename}

    img_files = []
    labels_file = None
    train_img = None

    with ZipFile(BytesIO(await file.read()), 'r') as zfile:
        for fname in zfile.namelist():
            if fname[-4:] == '.txt' and fname[:2] != '__':
                labels_file = fname
            elif fname[-4:] == '.png':
                img_files.append(fname)

        if len(img_files) == 0:
            return {"error": "No training images (png files) found."}
        else:
            for fname in sorted(img_files):
                with zfile.open(fname) as img_file:
                    img = img_file.read()

                    # process image
                    img = Image.open(BytesIO(img)).convert('L')
                    img = np.array(img).astype("float32") / 255
                    img = np.expand_dims(img, (0, -1))

                    if train_img is None:
                        train_img = img
                    else:
                        train_img = np.vstack((train_img, img))

        if labels_file is None:
            return {"error": "No training labels file (txt file) found."}
        else:
            with zfile.open(labels_file) as labels:
                labels_data = labels.read()
                labels_data = labels_data.decode("utf-8").split()
                labels_data = np.array(labels_data).astype("int")
                labels_data = keras.utils.to_categorical(labels_data, num_classes)

    # schedule model retraining (we could also schedule periodic retraining)
    background_tasks.add_task(
        ml_models["cnn"].fit,
        train_img,
        labels_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1
    )

    return {"message": "Data received successfully, model training has started."}

@app.get("/hello/{name}")
async def say_hello(name: str):
    logger.debug('say_hello starting...')        
    return {"message": f"Hello {name}"}
