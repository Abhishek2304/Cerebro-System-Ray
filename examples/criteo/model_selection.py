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

def estimator_gen_fn(params): # lr, lambda_value

    lr = params["lr"]
    lambda_regularizer = params["lambda_value"]

    INPUT_SHAPE_CRITEO = (13, )
    NUM_CLASSES_CRITEO = 1
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
    loss = keras.losses.BinaryCrossentropy()


    keras_estimator = RayEstimator(
        model=model,
        optimizer=optimizer,
        loss=loss,
        metrics=['acc'],
        batch_size=128,
        transformation_fn=None)

    return keras_estimator
    

def main():

    # data_dir = "/proj/orion-PG0/rayCriteoDataset/valid_0.tsv"
    OUTPUT_PATH = "/proj/orion-PG0/rayCriteoDataset/"
    # TRAIN_FRACTION = 0.7

    NUM_PARTITIONS = 2

    # header_list = ['label']
    # for i in range(13):
    #     label = 'n' + str(i)
    #     header_list.append(label)
    # for i in range(26):
    #     label = 'c' + str(i)
    #     header_list.append(label)

    # df = pd.read_csv(data_dir, sep = '\t', names = header_list, header = None)
    # print("Reading Done")

    # print("Preparing dataset")
    # for i in range(26):
    #     label = 'c' + str(i)
    #     df = df.drop(columns = [label])
    # df = df.sample(frac = 0.1)
    # df.fillna(0, inplace=True)

    # df = (df - df.min())/(df.max() - df.min())
    # df['features'] = df.apply(lambda x: list([x['n0'], x['n1'],x['n2'], x['n3'],x['n4'], x['n5'],x['n6'], x['n7'],
    #                                             x['n8'], x['n9'],x['n10'], x['n11'],x['n12']]), axis = 1)
    # for i in range(13):
    #     label = 'n' + str(i)
    #     df = df.drop(columns = [label])
    

    # print("STARTING BACKEND NOW")
    # backend = RayBackend(num_workers = 4)
    # store = LocalStore(OUTPUT_PATH, train_path=os.path.join(OUTPUT_PATH, 'train_data.parquet'), val_path=os.path.join(OUTPUT_PATH, 'val_data.parquet'))

    # train_rows, val_rows, metadata, _ = backend.prepare_data(store, df, 0.2)
    # backend.teardown_workers()

    print("STARTING BACKEND NOW")
    backend = RayBackend(num_workers = NUM_PARTITIONS)
    store = LocalStore(OUTPUT_PATH, train_path=os.path.join(OUTPUT_PATH, 'train_data.parquet'), val_path=os.path.join(OUTPUT_PATH, 'val_data.parquet'))

    param_grid_criteo = {
    "lr": hp_choice([1e-3, 1e-4]),
    "lambda_value": hp_choice([1e-4, 1e-5]),
    # "batch_size": hp_choice([32, 64, 256, 512]),
    }

    model_selection = GridSearch(backend, store, estimator_gen_fn, param_grid_criteo, 5, evaluation_metric='acc',
                        feature_columns=['features'], label_columns=['label'], verbose = 1)
    model = model_selection.fit_on_prepared_data()
    print(model.get_all_model_history())

if __name__ == "__main__":
    begin_time = time.time()
    main()
    print("Time for Cerebro Ray:")
    print(time.time() - begin_time)



