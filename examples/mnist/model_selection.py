# Copyright 2022 Abhishek Gupta, Rishikesh Ingale and Arun Kumar. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import numpy as np
import pandas as pd
import time
import ray
import tensorflow.keras as keras

from cerebro.backend.ray.backend import RayBackend
from cerebro.storage import LocalStore
from cerebro.keras import RayEstimator
from cerebro.tune import RandomSearch, GridSearch, hp_choice

import random
random.seed(2021)

# Change the model/loss/optimizer definitions in this function.
def estimator_gen_fn(params): # params used: lr, lambda_value (for regularization)

    lr = params["lr"]
    lambda_regularizer = params["lambda_value"]

    INPUT_SHAPE = (784, )
    NUM_CLASSES = 10
    SEED = 2021

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1000, activation='relu',
                                    input_shape=INPUT_SHAPE))
    model.add(keras.layers.Dense(500, activation='relu'))
    model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))

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
    loss = keras.losses.CategoricalCrossentropy()

    keras_estimator = RayEstimator(
        model=model,
        optimizer=optimizer,
        loss=loss,
        metrics=['acc'],
        batch_size=128,
        transformation_fn=None)

    return keras_estimator
    

def main():

    # Replace this root folder with the folder where you place your data
    OUTPUT_PATH = "/proj/orion-PG0/rayMnistDataset/"
    NUM_PARTITIONS = 4

    print("STARTING BACKEND NOW")
    backend = RayBackend(num_workers=NUM_PARTITIONS)

    # You can change train_data.parquet and val_data.parquet with your data files, but make sure they are parquet files.
    store = LocalStore(OUTPUT_PATH, train_path=os.path.join(OUTPUT_PATH, 'train_data.parquet'), \
        val_path=os.path.join(OUTPUT_PATH, 'val_data.parquet'))
    
    # You can change the hyperparameter search space over here.
    param_grid_criteo = {
        "lr": hp_choice([1e-3, 1e-4]),
        "lambda_value": hp_choice([1e-4, 1e-5]),
    }

    model_selection = GridSearch(backend, store, estimator_gen_fn, param_grid_criteo, 5, evaluation_metric='acc',
                        feature_columns=['features'], label_columns=['label'], verbose=0)

    begin_time = time.time()
    model = model_selection.fit_on_prepared_data()
    time_taken = time.time() - begin_time

    print(model.get_best_model_history())
    
    return time_taken

if __name__ == "__main__":
    
    time_taken = main()
    print("Time for Cerebro Ray:")
    print(time_taken)
