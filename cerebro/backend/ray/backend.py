from __future__ import absolute_import
import random
import numpy as np
import tensorflow as tf
import datetime

from . import util
from .. import constants
from .. import timeout, settings as ray_settings, secret, host_hash, job_id
from ..backend import Backend
from ...commons.util import patch_hugginface_layer_methods
from ...commons.constants import exit_event

import ray
import psutil
import pandas as pd
import pyarrow.csv as csv

# Not specifying the number of CPUs in ray.remote (@ray.remote(num_cpus=1)) as we are doing a single core computation right now.
# But may see if we want to specify it later or we dont want to keep it that dynamic. Also find a way to dynamically provide 
# resources (num_cpus = 1 OR num_gpus = 1)
@ray.remote
class Worker(object):
    def __init__(self):
        self.completion_status = True

    def get_completion_status(self):
        return self.completion_status
    
    def train_subepoch(self, model, weights):
        self.completion_status = False
        print(model, weights)
        self.completion_status = True
        return weights

    def testing_function(data):
        pass

class RayBackend(Backend):

    def __init__(self, num_workers = None, start_timeout = 600, verbose = 1, 
                data_readers_pool_type = 'thread'):

        # Putting ray.init() here, hoping it will not go out of scope once the __init__ function exits, but only when the
        # class is destroyed. This may not be true, and may have to invoke ray.init() globally somehow.
        ray.init(namespace="exp1")

        tmout = timeout.Timeout(start_timeout,
                                message='Timed out waiting for {activity}. Please check that you have '
                                        'enough resources to run all Cerebro processes. Each Cerebro '
                                        'process runs on a Ray Actor. You may need to increase the '
                                        'start_timeout parameter to a larger value if your Ray resources '
                                        'are allocated on-demand.')
        
        # data_readers_pool_type specifies process or thread (for one/many nodes). May not need this as ray is automatic.
        # Putting nics as none as we do not need to explicitly do TCP communication. Troubleshoot on nics and disk_cache_size_bytes.
        settings = ray_settings.Settings(verbose = verbose, 
                                        timeout = tmout,
                                        disk_cache_size_bytes = None,
                                        data_readers_pool_type = data_readers_pool_type,
                                        nics = None) 
        
        # If num_workers not given, use psutil to set the workers to cores - 1.
        if num_workers is None:
            num_workers = psutil.cpu_count() - 1
            if settings.verbose >= 1:
                print('CEREBRO => Time: {}, Running {} Workers (set a default of cores - 1)'.format(
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), num_workers))
        else:
            if settings.verbose >= 1:
                print('CEREBRO => Time: {}, Running {} Workers'.format(datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"), num_workers))
        settings.num_workers = num_workers
        self.settings = settings

        self.workers_initialized = False
        self.workers = None
        self.data_loaders_initialized = False
        self.train_shards = None
        self.val_shards = None

        # Add self.num_data_readers if it is different from num_workers
        
        # May not need the below attributes, remove if not needed for Ray
        self.rand = np.random.RandomState(constants.RANDOM_SEED)

        # Check this again, since these initializations are never called.
        self.initialize_workers()
        # self.initialize_data_loaders() # Need to provide the store to initialize_data_loaders()

    def _num_workers(self):
        return self.settings.num_workers

    def initialize_workers(self):

        num_workers = self._num_workers()
        self.workers = [Worker.options(name = str(i), lifetime = "detached").remote() for i in range(num_workers)]
        self.workers_initialized = True

    def initialize_data_loaders(self, store, schema_Fields = None, dataset_idx = None):
        ### Assume data is in parquet format in train/val_data_path Initialize data loaders to read this parquet format and shard automatically

        if self.workers_initialized:
            shard_count = self._num_workers()
            train_dataset = ray.data.read_csv('/proj/orion-PG0/rayCriteoDataset/train_0.tsv', arrow_csv_args={'parse_options':
                                                csv.ParseOptions(delimiter="\t")}) 
            # train_dataset = ray.data.read_parquet(store.get_train_data_path(dataset_idx)) 
            print(train_dataset)
            self.train_shards = train_dataset.split(n=shard_count, locality_hints = self.workers)
            print()
            # print(self.train_shards)
            # val_dataset = ray.data.read_parquet(store.get_val_data_path(dataset_idx)) 
            # print()
            # print(val_dataset)
            # self.val_shards = val_dataset.split(n=shard_count, locality_hints = self.workers)
            # print(self.val_shards)

            self.data_loaders_initialized = True
            
            
        else:
            raise Exception('Spark tasks not initialized for Cerebro. Please run SparkBackend.initialize_workers() '
                            'first!')

    def teardown_workers(self):
        # Need to reimplement, probably not related to killing of the workers.
        
        # Consider, instead of forcefully killing it, to remove the detached lifetime and set it to null.
        # Hence, when all references to actor handle are removed, it is automatically garbage collected
        # and the process is stopped.
        
        # Consider explicitly shutting down ray here, but before that make sure that all data is 
        # written to persistent storage

        # Simply shutting down ray right now, may have to do more later
        ray.shutdown()

    def prepare_data(self, store, dataset, validation, num_partitions=None, parquet_row_group_size_mb=8, dataset_idx=None):

        # IMP - Takes the number of partitions as equal to the number of workers here. DOES NOT USE THE num_partitions SUPPLIED.
        return util.prepare_data(self._num_workers(), store, dataset, validation, 
                            num_partitions=self._num_workers(), dataset_idx=dataset_idx, 
                            parquet_row_group_size_mb = parquet_row_group_size_mb, verbose = self.settings.verbose)
                            
    def get_metadata_from_parquet(self, store, label_columns=['label'], feature_columns=['features'], dataset_idx = None):
        
        return util.get_simple_meta_from_parquet(store, label_columns + feature_columns, dataset_idx = dataset_idx)

    def train_for_one_epoch(self, models, store, feature_col, label_col, is_train=True):
        
        mode = "Training"
        if not is_train:
            mode = "Validation"
        if self.settings.verbose >= 1:
            print('CEREBRO => Time: {}, Starting EPOCH {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), mode))
        
        sub_epoch_trainers = []
        for model in models:
            if type(store) == dict:
                a_store = store[model.getRunId()]
            else:
                a_store = store
            
            if type(feature_cols) == dict:
                a_feature_col = feature_cols[model.getRunId()]
            else:
                a_feature_col = feature_cols
            
            if type(label_cols) == dict:
                a_label_col = label_cols[model.getRunId()]
            else:
                a_label_col = label_cols
            
            sub_epoch_trainers.append(_get_remote_trainer(model, self, a_store, None, a_feature_col, a_label_col, is_train, self.settings.verbose))

        Q = [(i, j) for i in range(len(models)) for j in range(self.settings.num_workers)]
        random.shuffle(Q)

        # Implement the store here
        # Need to use a _get_remote_trainer function to initially checkpoint and put model in store, etc.
        
        model_idle = [True for _ in range(len(models))]
        model_weights = [initial_weights(model) for model in models]

        worker_idle = [True for _ in range(self.settings.num_workers)]
        model_on_worker = [-1 for _ in range(self.settings.num_workers)]

        def place_model_on_worker(j):
            random.shuffle(Q)
            for idx, s in enumerate(Q):
                i, j_prime = s
                if j_prime == j and model_idle[i]:
                    model_idle[i] = False
                    worker_idle[j] = False
                    model_on_worker[j] = i
                    model_weights[i] = self.workers[j].sub_epoch_train.remote(models[i], model_weights[i])
                    break

        while not exit_event.is_set() and len(Q) > 0:
            for j in range(self.settings.num_workers):
                if worker_idle[j]:
                    place_model_on_worker(j)
                elif ray.get(self.workers[j].get_completion_status.remote()):
                    i = model_on_worker[j]
                    Q.remove((i, j))
                    model_idle[i] = True
                    model_weights[i] = ray.get(model_weights[i])
                    worker_idle[j] = True
                    model_on_worker[j] = -1
                    place_model_on_worker(j)
                
            exit_event.wait(self.settings.polling_period)

def initial_weights(model):
    return None
