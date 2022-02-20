
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

from __future__ import absolute_import

import datetime
import numpy as np
import pyarrow as pa
import pandas as pd

from .. import constants

def get_simple_meta_from_parquet(store, schema_cols, dataset_idx=None):
    """ Gets metadata from the parquet train/val files
    """ 

    train_data_path = store.get_train_data_path(dataset_idx)
    validation_data_path = store.get_val_data_path(dataset_idx)

    if not store.exists(train_data_path):
        raise ValueError("{} path does not exist in the store".format(train_data_path))

    train_df = pd.read_parquet(train_data_path)
    train_rows = len(train_df)

    val_rows = 0
    if store.exists(validation_data_path):
        val_df = pd.read_parquet(validation_data_path)
        val_rows = len(val_df)

    metadata = {}
    for col in train_df.columns:
        col_info = {'dtype': train_df.dtypes[col].name, 'size': len(train_df[col])}
        metadata[col] = col_info

    avg_row_size = None

    return train_rows, val_rows, metadata, avg_row_size

def _train_val_split(df, validation):
    """ Splits the dataframe into the train and val data according to ratio provided
    """
    validation_ratio = 0.0

    if isinstance(validation, float) and validation > 0:
        msk = np.random.rand(len(df)) < validation
        train_df = df[~msk]
        val_df = df[msk]
        validation_ratio = validation
    
    elif isinstance(validation, str):
        dtype = df.dtypes['validation']
        
        bool_dtype = (dtype == 'bool')
        if bool_dtype:
            val_df = (df[df['validation']==True]).drop(columns = ['validation'])
            train_df = (df[df['validation']==False]).drop(columns = ['validation'])

        else:
            val_df = (df[df['validation']>0]).drop(columns = ['validation'])
            train_df = (df[df['validation']==0]).drop(columns = ['validation'])

        train_rows = len(train_df)
        val_rows = len(val_df)
        validation_ratio = val_rows / (val_rows + train_rows)
    elif validation:
        raise ValueError('Unrecognized validation type: {}'.format(type(validation)))

    return train_df, val_df, validation_ratio


def _create_dataset(store, df, validation, compress_sparse,
                    num_partitions, num_workers, dataset_idx, parquet_row_group_size_mb, verbose):
    """ Creates the train/val set from the dataframe provided, does some validation and saves it
        as parquet files to the given paths
    """

    train_data_path = store.get_train_data_path(dataset_idx)
    val_data_path = store.get_val_data_path(dataset_idx)
    if verbose >= 1:
        print('CEREBRO => Time: {}, Writing DataFrames'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        print('CEREBRO => Time: {}, Train Data Path: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                               train_data_path))
        print('CEREBRO => Time: {}, Val Data Path: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                             val_data_path))
        
    schema_cols = df.columns

    if isinstance(validation, str):
        schema_cols.append(validation)
    df = df[schema_cols]

    train_df, val_df, validation_ratio = _train_val_split(df, validation)

    train_df.to_parquet(train_data_path, compression = None, index = False)
    if val_df is not None:
        val_df.to_parquet(val_data_path, compression = None, index = False)

    train_rows = len(train_df)
    val_rows = len(val_df)

    metadata = {}
    for col in train_df.columns:
        col_info = {'dtype': train_df.dtypes[col].name, 'size': len(train_df[col])}
        metadata[col] = col_info

    avg_row_size = None

    if verbose:
        print(
        'CEREBRO => Time: {}, Train Rows: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), train_rows))
    if val_df is not None:
        if val_rows == 0:
            raise ValueError(
                'Validation DataFrame does not any samples with validation param {}'
                    .format(validation))
        if verbose:
            print(
            'CEREBRO => Time: {}, Val Rows: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), val_rows))

    return train_rows, val_rows, metadata, avg_row_size

    

def check_validation(validation, df = None):
    """ Check if the given validation value is legal
    """
    if validation:
        if isinstance(validation, float):
            if validation < 0 or validation >= 1:
                raise ValueError('Validation split {} must be in the range: [0, 1)'
                                 .format(validation))
        elif isinstance(validation, str):
            if df is not None and validation not in df.columns:
                raise ValueError('Validation column {} does not exist in the DataFrame'
                                 .format(validation))
        else:
            raise ValueError('Param validation must be of type "float" or "str", found: {}'
                             .format(type(validation)))

def prepare_data(num_workers, store, df,
                 validation=None, compress_sparse=False,
                 num_partitions=None, parquet_row_group_size_mb=8, dataset_idx=None, verbose=0):
    """ Checks validation and sends the dataframe to the create dataset function
    """
    check_validation(validation, df=df)
    if num_workers <= 0:
        raise ValueError('num_workers={} must be > 0'
                         .format(num_workers))

    if verbose >= 1:
        print('CEREBRO => Time: {}, Num Partitions = Num Workers: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                              num_partitions))

    return _create_dataset(store, df, validation, compress_sparse,
                           num_partitions, num_workers, dataset_idx, parquet_row_group_size_mb, verbose)
