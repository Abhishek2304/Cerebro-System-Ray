import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd
import numpy as np
import time

INPUT_SHAPE_CRITEO = (13, )
NUM_CLASSES_CRITEO = 1
SEED = 2021

def define_model(lr, lambda_regularizer):        
    
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
    # loss = keras.losses.CategoricalCrossentropy()
    loss = 'categorical_crossentropy'
    metrics = ['acc']

    return model, loss, optimizer, metrics


def main():

    lrs = [1e-3, 1e-4]
    lambdas = [1e-4, 1e-5]

    train_df = pd.read_parquet("/proj/orion-PG0/rayCriteoDataset/train_data.parquet")
    train_tar = train_df.pop('label')
    np_train = np.array([arr.tolist().pop() for arr in np.asarray(train_df)]).astype('float64')
    train_data = tf.convert_to_tensor(np_train)
    train_tar = tf.convert_to_tensor(np.asarray(train_tar))

    val_df = pd.read_parquet("/proj/orion-PG0/rayCriteoDataset/val_data.parquet")
    val_tar = val_df.pop('label')
    np_val = np.array([arr.tolist().pop() for arr in np.asarray(val_df)]).astype('float64')
    val_data = tf.convert_to_tensor(np_val)
    val_tar = tf.convert_to_tensor(np.asarray(val_tar))

    for lr in lrs:
        for lambda_regularizer in lambdas:

            model, loss, optimizer, metrics = define_model(lr, lambda_regularizer)
            model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
            history = model.fit(train_data, train_tar, batch_size = 64, epochs = 10, validation_data = (val_data, val_tar))

            print(history.history)

if __name__ == "__main__":
    
    begin_time = time.time()
    main()
    print("Total time for sequential Keras:")
    print(time.time() - begin_time)