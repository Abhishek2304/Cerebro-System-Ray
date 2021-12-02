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

    OUTPUT_PATH = "/proj/orion-PG0/rayMnistDataset/"
    NUM_PARTITIONS = 4

    print("STARTING BACKEND NOW")
    backend = RayBackend(num_workers=NUM_PARTITIONS)
    store = LocalStore(OUTPUT_PATH, train_path=os.path.join(OUTPUT_PATH, 'train_data.parquet'), \
        val_path=os.path.join(OUTPUT_PATH, 'val_data.parquet'))

    param_grid_criteo = {
        "lr": hp_choice([1e-3, 1e-4]),
        "lambda_value": hp_choice([1e-4, 1e-5]),
        # "batch_size": hp_choice([32, 64, 256, 512]),
    }

    model_selection = GridSearch(backend, store, estimator_gen_fn, param_grid_criteo, 5, evaluation_metric='acc',
                        feature_columns=['features'], label_columns=['label'], verbose=0)

    begin_time = time.time()
    model = model_selection.fit_on_prepared_data()
    time_taken = time.time() - begin_time

    print(model.get_best_model_history())
    
    return time_taken

    # OUTPUT_PATH = "/proj/orion-PG0/rayMnistDataset/"
    # data_dir = "/proj/orion-PG0/rayMnistDataset/mnist_train.csv"
    # val_dir = "/proj/orion-PG0/rayMnistDataset/mnist_test.csv"
    
    # header_list = ['label']
    # for i in range(784):
    #     label = 'n' + str(i)
    #     header_list.append(label)
    
    # df1 = pd.read_csv(data_dir, names = header_list, header = None)
    # df2 = pd.read_csv(val_dir, names = header_list, header = None)
    # df = pd.concat([df1, df2]*10).sort_index()
    # print(len(df))
    # print("Reading Done")

    # print("Preparing dataset")
    # final_col_list = list()
    # for i in range(784):
    #     final_col_list.append('n' + str(i))
    
    # df['features'] = df[final_col_list].values.tolist()
    
    # for i in range(784):
    #     label = 'n' + str(i)
    #     df = df.drop(columns = [label])

    # df_tar = df.pop('label')
    # df_tar_one_hot = np.zeros((df_tar.size, df_tar.max()+1))
    # df_tar_one_hot[np.arange(df_tar.size),df_tar] = 1
    # df['label'] = df_tar_one_hot.tolist()

    # print("STARTING BACKEND NOW")
    # backend = RayBackend(num_workers = 4)
    # store = LocalStore(OUTPUT_PATH, train_path=os.path.join(OUTPUT_PATH, 'train_data.parquet'), val_path=os.path.join(OUTPUT_PATH, 'val_data.parquet'))

    # train_rows, val_rows, metadata, _ = backend.prepare_data(store, df, 0.2)
    # backend.teardown_workers()

if __name__ == "__main__":
    
    time_taken = main()
    print("Time for Cerebro Ray:")
    print(time_taken)
