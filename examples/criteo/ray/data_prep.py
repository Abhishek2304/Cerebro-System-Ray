import os
import numpy as np
import pandas as pd

from cerebro.backend.ray.backend import RayBackend
from cerebro.storage import LocalStore

import random
random.seed(2021)

def main():

    data_dir = "/proj/orion-PG0/rayCriteoDataset/valid_0.tsv"
    output_path = "/proj/orion-PG0/rayCriteoDataset/"
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

    print(data.head())
    print()
    print(data.columns)
    raise NotImplementedError
    print("Removing categorical features for now")
    for i in range(26):
        label = 'c' + str(i)
        df = df.drop(label)
    print("FILLING NAs")
    df.fillna(0, inplace=True)
    print("SAMPLING")
    df = df.sample(frac = 0.1)

    print(df.head())
    print()
    print(df.columns)
    print()
    print(df.dtypes)
    print()
    print(len(df))

if __name__ == "__main__":
    main()

