from __future__ import absolute_import

import datetime
import numpy as np
import pyarrow as pa
import pandas as pd

from .. import constants

def _train_val_split(df, validation):
    
    validation_ratio = 0.0

    if isinstance(validation, float) and validation > 0:
        msk = np.random.rand(len(df)) < validation
        train_df = df[~msk]
        val_df = df[msk]
        validation_ratio = validation
    
    elif isinstance(validation, str):
        dtype = df.dtypes['validation']
        
        bool_dtype = (df.a.dtypes.name == 'bool')
        if bool_dtype:
            val_df = (df[df['validation']==True]).drop(columns = ['validation'])
            train_df = (df[df['validation']==False]).drop(columns = ['validation'])

        else:
            val_df = (df[df['validation']>0]).drop(columns = ['validation'])
            train_df = (df[df['validation']==0]).drop(columns = ['validation'])

        # Approximate ratio of validation data to training data for proportionate scale
        # of partitions
        train_rows = len(train_df)
        val_rows = len(val_df)
        validation_ratio = val_rows / (val_rows + train_rows)
    elif validation:
        raise ValueError('Unrecognized validation type: {}'.format(type(validation)))

    return train_df, val_df, validation_ratio


def _create_dataset(store, df, validation, compress_sparse,
                    num_partitions, num_workers, dataset_idx, parquet_row_group_size_mb, verbose):
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
    if val_df:
        val_df.to_parquet(val_data_path, compression = None, index = False)

    # Implement get simple metadata from parquet, return those values here.
    return

    

def check_validation(validation, df = None):
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
    
    check_validation(validation, df=df)
    if num_workers <= 0:
        raise ValueError('num_workers={} must be > 0'
                         .format(num_workers))

    if verbose >= 1:
        print('CEREBRO => Time: {}, Num Partitions = Num Workers: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                              num_partitions))

    return _create_dataset(store, df, validation, compress_sparse,
                           num_partitions, num_workers, dataset_idx, parquet_row_group_size_mb, verbose)
