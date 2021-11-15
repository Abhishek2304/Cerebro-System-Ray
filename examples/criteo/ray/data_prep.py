import os
import numpy as np
import pandas as pd
import time
import ray

from cerebro.backend.ray.backend import RayBackend
from cerebro.storage import LocalStore

import random
random.seed(2021)

def main():

    data_dir = "/proj/orion-PG0/rayCriteoDataset/valid_0.tsv"
    OUTPUT_PATH = "/proj/orion-PG0/rayCriteoDataset/"
    TRAIN_FRACTION = 0.7

    NUM_PARTITIONS = 4

    header_list = ['label']
    for i in range(13):
        label = 'n' + str(i)
        header_list.append(label)
    for i in range(26):
        label = 'c' + str(i)
        header_list.append(label)

    df = pd.read_csv(data_dir, sep = '\t', names = header_list, header = None)
    #df = pd.read_csv(data_dir, sep = '\t')
    print("Reading Done")
    # df.to_csv("/proj/orion-PG0/rayCriteoDataset/train_data.csv")
    # print("Writing Done")

    # df1 = pd.read_csv("/proj/orion-PG0/rayCriteoDataset/train_data.csv")
    # print(df1.head())
    # print(len(df1))

    # df2 = ray.data.read_csv("/proj/orion-PG0/rayCriteoDataset/train_data.csv")
    # print(df2.show(5))

    print("Removing categorical features for now")
    for i in range(26):
        label = 'c' + str(i)
        df = df.drop(columns = [label])
    print("FILLING NAs")
    df.fillna(0, inplace=True)
    print("SAMPLING")
    df = df.sample(frac = 0.1)

    print("STARTING BACKEND NOW")
    backend = RayBackend(num_workers = 4)
    store = LocalStore(OUTPUT_PATH, train_path=os.path.join(OUTPUT_PATH, 'train_data.parquet'), val_path=os.path.join(OUTPUT_PATH, 'val_data.parquet'))

    train_rows, val_rows, metadata, _ = backend.prepare_data(store, df, 0.2)
    backend.initialize_data_loaders(store)
    print("Initialization done")
    print()

    for shard, worker in zip(backend.train_shards, backend.workers):
        ref = worker.testing_function.remote(shard)
        print(ray.get(ref))
        print()
    backend.teardown_workers()

if __name__ == "__main__":
    main()

