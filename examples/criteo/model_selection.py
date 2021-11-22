import os
import numpy as np
import pandas as pd
import time
import ray
import tensorflow.keras as keras

from cerebro.backend.ray.backend import RayBackend
from cerebro.storage import LocalStore
from cerebro.keras.ray.estimator import RayEstimator
from cerebro.tune import RandomSearch, GridSearch, hp_choice

import random
random.seed(2021)

def estimator_gen_fn(params): # lr, lambda_value

    lr = params["lr"]
    lambda_regularizer = params["lambda_value"]

    INPUT_SHAPE_CRITEO = (7306, )
    NUM_CLASSES_CRITEO = 2
    SEED = 2021

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1000, activation='relu',
                                    input_shape=INPUT_SHAPE_CRITEO))
    model.add(keras.layers.Dense(500, activation='relu'))
    model.add(keras.layers.Dense(NUM_CLASSES_CRITEO, activation='softmax'))

    regularizer = keras.regularizers.l2(lambda_regularizer)

    for layer in model.layers:
        for attr in ['kernel_regularizer', 'bias_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)
        for attr in ['kernel_initializer', 'bias_initializer']:
            if hasattr(layer, attr):
                layer_initializer = getattr(layer, attr)
                if hasattr(layer_initializer, 'seed'):
                    setattr(layer_initializer, 'seed', SEED)

    optimizer = keras.optimizers.Adam(lr=lr)
    loss = 'categorical_crossentropy'


    keras_estimator = RayEstimator(
        model=model,
        optimizer=optimizer,
        loss=loss,
        metrics=['loss'],
        batch_size=128,
        transformation_fn=None)

    return keras_estimator
    

def main():

    OUTPUT_PATH = "/proj/orion-PG0/rayCriteoDataset/"

    print("STARTING BACKEND NOW")
    backend = RayBackend(num_workers = 4)
    store = LocalStore(OUTPUT_PATH, train_path=os.path.join(OUTPUT_PATH, 'train_0.parquet'), val_path=os.path.join(OUTPUT_PATH, 'valid_0.parquet'))
    backend.initialize_data_loaders(store)
    print("Initialization done")

    param_grid_criteo = {
    "lr": hp_choice([1e-3, 1e-4]),
    "lambda_value": hp_choice([1e-4, 1e-5]),
    # "batch_size": hp_choice([32, 64, 256, 512]),
    }

    model_selection = GridSearch(backend, store, estimator_gen_fn, search_space, 10, evaluation_metric='loss',
                        feature_columns=['features'], label_columns=['labels'])
    model = model_selection.fit_on_prepared_data()

if __name__ == "__main__":
    main()




