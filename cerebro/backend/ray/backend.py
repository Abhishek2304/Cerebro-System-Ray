from __future__ import absolute_import

import gc
import h5py
import io
import inspect
import random
import numpy as np
import tensorflow as tf
import datetime
import os
import time
import traceback
import tensorflow_io.arrow as arrow_io

from . import util
from .. import constants
from .. import timeout, settings as ray_settings, secret, host_hash, job_id
from ..backend import Backend
from ...commons.util import patch_hugginface_layer_methods
from ...commons.constants import exit_event

import ray
import psutil
import pandas as pd

### Ray actor class to define the workers that will keep the sharded data and perform the sub-epoch training
@ray.remote
class Worker(object):
    def __init__(self):
        self.completion_status = True

    def get_completion_status(self):
        """ Tells master whether the current sub epoch has been executed
        """
        return self.completion_status
    
    def accept_data(self, data_shard, is_train):
        """ Takes data from the initialize_data_loaders() function, transforms it to make it loadable in keras
            and stores the data
        """
        data_shard = data_shard.to_pandas(limit = data_shard.count())
        target = data_shard.pop('label')
        data_np = np.array([arr.tolist().pop() for arr in np.asarray(data_shard)]).astype('float64')
        tar_np = np.array([arr.tolist() for arr in np.asarray(target)]).astype('float64')
        if is_train:
            self.train_data = tf.convert_to_tensor(data_np)
            self.train_target = tf.convert_to_tensor(tar_np)
        else:
            self.val_data = tf.convert_to_tensor(data_np)
            self.val_target = tf.convert_to_tensor(tar_np)
    
    def execute_subepoch(self, fn, is_train, initial_epoch):
        """Takes the appropriate data shard and sends it to the train function to perform sub epoch training.
            Sends results back to master
        """

        if is_train:
            data = self.train_data
            target = self.train_target
        else:
            data = self.val_data
            target = self.val_target

        try:
            self.completion_status = False
            result, _ = fn(data, target, is_train, initial_epoch)
            self.completion_status = True
            return result

        except Exception as e:
            self.completion_status = True
            print(str(e) + "\n" + traceback.format_exc())

    def testing_function(self, data):
        """ Unimportant, testing function to verify workers are working as intended
        """
        time.sleep(0.01)
        return ray._private.services.get_node_ip_address()

class KerasStepCounter(tf.keras.callbacks.Callback):
    """Helper callback to count the number of step in sub-epoch training"""

    def __init__(self):
        self.counter = 0

    def on_train_batch_begin(self, batch, logs={}):
        self.counter += 1

    def on_test_batch_begin(self, batch, logs={}):
        self.counter += 1
    
    def get_step_count(self):
        return self.counter

class RayBackend(Backend):

    def __init__(self, num_workers = None, start_timeout = 600, verbose = 1, 
                data_readers_pool_type = 'thread'):
        """ Initialization function. Connects to a ray cluster, assigns the required parameters and the 
            number of resources per worker
        """

        # First try to detect cluster for multi-machine and connect to it. Else, run plain ray.init() for a single node.
        try:
            ray.init(address = "auto")
            print("Running on a Ray Cluster")
            num_cpus = ray.available_resources()['CPU']
            num_machines = 0
            for key in ray.available_resources().keys():
                if key[:5] == 'node:':
                    num_machines += 1
            num_machines = float(num_machines)
            self.cpus_per_worker = num_cpus/num_machines
            
        except ConnectionError:
            ray.init()
            print("No cluster found, running on a single Ray instance")
            self.cpus_per_worker = 4

        tmout = timeout.Timeout(start_timeout,
                                message='Timed out waiting for {activity}. Please check that you have '
                                        'enough resources to run all Cerebro processes. Each Cerebro '
                                        'process runs on a Ray Actor. You may need to increase the '
                                        'start_timeout parameter to a larger value if your Ray resources '
                                        'are allocated on-demand.')
        
        settings = ray_settings.Settings(verbose = verbose, 
                                        timeout = tmout,
                                        disk_cache_size_bytes = None,
                                        data_readers_pool_type = data_readers_pool_type,
                                        nics = None) 

        
        # If num_workers not given, set the workers to the number of machines detected by Ray
        if num_workers is None:
            num_workers = 0
            for key in ray.available_resources().keys():
                if key[:5] == 'node:':
                    num_workers += 1
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

    def _num_workers(self):
        """ Helper function to get the number of workers.
        """
        return self.settings.num_workers

    def initialize_workers(self):
        """ Initializes the number of workers from self._num_workers() with a detached lifetime (infinite lifetime till shutdown)
            and allocates the number of CPUs to each worker.
        """

        num_workers = self._num_workers()
        print(self.cpus_per_worker)
        self.workers = [Worker.options(name = str(i), lifetime = "detached", num_cpus = self.cpus_per_worker).remote() \
                        for i in range(num_workers)]
        self.workers_initialized = True

    def initialize_data_loaders(self, store, schema_Fields=None, dataset_idx=None):
        """ Assumes that the train/val data is loaded in the local store in parquet format
            Splits both the data across the workers.
            initialize_workers needs to be called before calling this function
        """

        if self.workers_initialized:
            shard_count = self._num_workers()
            
            train_dataset = ray.data.read_parquet(store.get_train_data_path(dataset_idx), parallelism=1000) 
            self.train_shards = train_dataset.split(n=shard_count, equal=True, locality_hints=self.workers)
            for i, s in enumerate(self.train_shards): self.workers[i].accept_data.remote(s, True)
            
            val_dataset = ray.data.read_parquet(store.get_val_data_path(dataset_idx))
            self.val_shards = val_dataset.split(n=shard_count, equal=True, locality_hints=self.workers)
            for i, s in enumerate(self.val_shards): self.workers[i].accept_data.remote(s, False)

            self.data_loaders_initialized = True

        else:
            raise Exception('Ray tasks not initialized for Cerebro. Please run RayBackend.initialize_workers() first!')

    def teardown_workers(self):
        """ Final call for killing all the respective workers and shutting down the ray cluster
        """
        
        ray.shutdown()

    def prepare_data(self, store, dataset, validation, num_partitions=None, parquet_row_group_size_mb=8, dataset_idx=None):
        """ This function is called by the main script if the data is provided as a single dataframe. Calls the
            relevant function to load the data, split it into train-val sets and save it as parquet files from where
            it can be picked up for sharding to dataloaders.
        """
        # IMP - Takes the number of partitions as equal to the number of workers here. DOES NOT USE THE num_partitions SUPPLIED.
        return util.prepare_data(self._num_workers(), store, dataset, validation, 
                            num_partitions=self._num_workers(), dataset_idx=dataset_idx, 
                            parquet_row_group_size_mb = parquet_row_group_size_mb, verbose = self.settings.verbose)
                            
    def get_metadata_from_parquet(self, store, label_columns=['label'], feature_columns=['features'], dataset_idx = None):
        """ Helper function to get metadata of stored data from parquet
        """
        return util.get_simple_meta_from_parquet(store, label_columns + feature_columns, dataset_idx = dataset_idx)

    def train_for_one_epoch(self, models, store, feature_cols, label_cols, is_train=True):
        """
        Takes a set of Keras models and trains for one epoch. If is_train is False, validation is performed
        instead of training.
        """

        mode = "Training"
        if not is_train:
            mode = "Validation"
        if self.settings.verbose >= 1:
            print('CEREBRO => Time: {}, Starting EPOCH {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), mode))
        
        # First get the data and send it to _get_remote_trainer to get the trainers for each model
        sub_epoch_trainers = []
        epoch_results = {}
        result_refs = {}
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

            epoch_results[model.getRunId()] = {}
            result_refs[model.getRunId()] = []
            
            sub_epoch_trainers.append(_get_remote_trainer(model, a_store, None, a_feature_col, a_label_col))

        # Make a queue and the model and worker availability lists for the scheduling
        Q = [(i, j) for i in range(len(models)) for j in range(self.settings.num_workers)]
        random.shuffle(Q)
        model_idle = [True for _ in range(len(models))]
        worker_idle = [True for _ in range(self.settings.num_workers)]
        model_on_worker = [-1 for _ in range(self.settings.num_workers)]

        def place_model_on_worker(j):
            ### Checks which model from the model-worker pairs for worker j is idle and assigns model to that worker.
            ### Calls the execute subepoch function for that worker and gets the result Object reference, returns it.
            for idx, s in enumerate(Q):
                i, j_prime = s
                if j_prime == j and model_idle[i]:
                    model_idle[i] = False
                    worker_idle[j] = False
                    model_on_worker[j] = i
                    if is_train:
                        result_ref = self.workers[j].execute_subepoch.remote(sub_epoch_trainers[i], is_train, models[i].epoch)
                    else:
                        result_ref = self.workers[j].execute_subepoch.remote(sub_epoch_trainers[i], is_train, models[i].epoch)
                    return result_ref

        ### Till the queue is not empty/exit event is not called, keep assigning idle models to idle workers using place_model_on_worker
        ### Append the subepoch result object references for each model to their dictionary result_refs
        while not exit_event.is_set() and len(Q) > 0:
            for j in range(self.settings.num_workers):
                if worker_idle[j]:
                    result_ref = place_model_on_worker(j)
                    if result_ref is not None:
                        result_refs[models[model_on_worker[j]].getRunId()].append(result_ref)
                    
                elif ray.get(self.workers[j].get_completion_status.remote()):
                    i = model_on_worker[j]
                    Q.remove((i, j))
                    
                    model_idle[i] = True
                    worker_idle[j] = True
                    model_on_worker[j] = -1
                    result_ref = place_model_on_worker(j)
                    if result_ref is not None:
                        result_refs[models[model_on_worker[j]].getRunId()].append(result_ref)
                    
        ### Iterate over the subepoch result object references of each model, get the results (loss, acc etc.) Average them out
        ### and return these statistics as epoch results to be used by the callee tune functions
        for model_id in result_refs.keys():
            refs_this = result_refs[model_id]
            for ref in refs_this:
                result_this = ray.get(ref)
                for k in result_this.keys():
                    if k in epoch_results[model_id]:
                        epoch_results[model_id][k].extend(result_this[k])
                    else:
                        epoch_results[model_id][k] = result_this[k]
            for k in epoch_results[model_id]:
                epoch_results[model_id][k] = np.average(epoch_results[model_id][k])
        
        ### If training, increment the epoch number of each model
        if is_train:
            for model in models:
                if model.getEpochs() is None:
                    model.setEpoch(1)
                else:
                    model.setEpoch(model.getEpochs() + 1)

        return epoch_results


def _get_remote_trainer(estimator, store, dataset_idx, feature_columns, label_columns):
    """ Function to load the required values from Ray Estimator, compile and checkpoint the initial model
        and return the trainer that can be sent to the Ray workers by master
    """
    
    run_id = estimator.getRunId()
    keras_utils = estimator._get_keras_utils()

    # Checkpointing the model if it does not exist.
    if not estimator._has_checkpoint(run_id): 
        remote_store = store.to_remote(run_id, dataset_idx) 

        with remote_store.get_local_output_dir() as run_output_dir:
            model = estimator._compile_model(keras_utils) 
            ckpt_file = os.path.join(run_output_dir, remote_store.checkpoint_filename)
            model.save(ckpt_file)
            remote_store.sync(run_output_dir)

    return sub_epoch_trainer(estimator, keras_utils, run_id, dataset_idx)


def sub_epoch_trainer(estimator, keras_utils, run_id, dataset_idx):
    """ Trainer keeping the required values for each estimator object for model loading, checkpointing etc.
        Returns the subepoch training function (train() - defined below) that can be called by the workers
    """ 

    # Estimator parameters
    user_callbacks = estimator.callbacks
    custom_objects = estimator.custom_objects
    metrics_names = [name.__name__ if callable(name) else name for name in estimator.metrics]
    user_verbose = estimator.verbose

    floatx = tf.keras.backend.floatx()
    fit_sub_epoch_fn = keras_utils.fit_sub_epoch_fn()
    eval_sub_epoch_fn = keras_utils.eval_sub_epoch_fn()

    # Utility functions
    deserialize_keras_model = _deserialize_keras_model_fn()
    pin_gpu = _pin_gpu_fn() 

    # Storage
    store = estimator.store
    remote_store = store.to_remote(run_id, dataset_idx)

    def train(x_data, y_data, is_train, starting_epoch, local_task_index=0):
        ### Subepoch training function

        begin_time = time.time()

        # Workaround for the issue with huggingface layers needing a python
        # object as config (not a dict) and explicit definition of get_config method.
        # We monkey patch the __init__ method get_config methods of such layers.
        if custom_objects is not None:
            for k in custom_objects:
                if issubclass(custom_objects[k], tf.keras.layers.Layer) and inspect.getmodule(custom_objects[k]).__name__.startswith('transformers.'):
                    patch_hugginface_layer_methods(custom_objects[k])

        tf.keras.backend.set_floatx(floatx)
        pin_gpu(local_task_index)

        # Verbose mode 1 will print a progress bar.
        verbose = user_verbose

        with remote_store.get_local_output_dir() as run_output_dir:
            step_counter_callback = KerasStepCounter() 
            callbacks = [step_counter_callback]
            if user_callbacks is not None:
                callbacks = callbacks + user_callbacks
            ckpt_file = os.path.join(run_output_dir, remote_store.checkpoint_filename)
      
            # restoring the model from the previous checkpoint
            # with tf.keras.utils.custom_object_scope(custom_objects):
            #     model = deserialize_keras_model(
            #         remote_store.get_last_checkpoint(), lambda x: tf.keras.models.load_model(x))
            
            # print(remote_store.get_last_checkpoint)

            if custom_objects is None:
                model = deserialize_keras_model(
                    remote_store.get_last_checkpoint(), lambda x: tf.keras.models.load_model(x))
            else:
                with tf.keras.utils.custom_object_scope(custom_objects):
                    model = deserialize_keras_model(
                        remote_store.get_last_checkpoint(), lambda x: tf.keras.models.load_model(x))
            
            if starting_epoch is None:
                starting_epoch = 0

            if is_train:
                ### Load the model from checkpoing train and save
                initialization_time = time.time() - begin_time
                begin_time = time.time()
                result = fit_sub_epoch_fn(starting_epoch, model, x_data, y_data, callbacks, verbose).history
                training_time = time.time() - begin_time
                begin_time = time.time()
                result = {'train_' + name: result[name] for name in result}
                model.save(ckpt_file)
            else:
                ### Load the model from checkpoint and validate
                initialization_time = time.time() - begin_time
                begin_time = time.time()
                result = eval_sub_epoch_fn(starting_epoch, model, x_data, y_data, callbacks, verbose)
                training_time = time.time() - begin_time
                begin_time = time.time()
                result = [[x] for x in result]
                result = {k: v for k, v in zip(['val_loss'] + ['val_' + name for name in metrics_names], result)}

            del model
            gc.collect()
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()

            remote_store.sync(run_output_dir)
            finalization_time = time.time() - begin_time

            if verbose >= 1:
                print('CEREBRO => Time: {}, Model: {}, Mode: {}, Initialization Time: {}, Training Time: {}, '
                      'Finalization Time: {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        run_id, 'TRAIN' if is_train else 'VALID', initialization_time, training_time, finalization_time))

            return result, step_counter_callback.get_step_count()

    return train


def _deserialize_keras_model_fn():
    def deserialize_keras_model(model_bytes, load_model_fn):
        """Deserialize model from byte array encoded in base 64."""
        bio = io.BytesIO(model_bytes)
        with h5py.File(bio, 'r') as f:
            return load_model_fn(f)

    return deserialize_keras_model


def _pin_gpu_fn():
    def fn(local_task_index):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[local_task_index], 'GPU')

    return fn
