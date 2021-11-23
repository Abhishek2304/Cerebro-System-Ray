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

# Not specifying the number of CPUs in ray.remote (@ray.remote(num_cpus=1)) as we are doing a single core computation right now.
# But may see if we want to specify it later or we dont want to keep it that dynamic. Also find a way to dynamically provide 
# resources (num_cpus = 1 OR num_gpus = 1)
@ray.remote
class Worker(object):
    def __init__(self):
        self.completion_status = True

    def get_completion_status(self):
        return self.completion_status
    
    def execute_subepoch(self, fn, data_shard, is_train, initial_epoch):

        data_shard = data_shard.to_pandas()
        target = data_shard.pop('label')
        data_np = np.array([arr.tolist().pop() for arr in np.asarray(data_shard)]).astype('float64')
        data = tf.convert_to_tensor(data_np)
        target = tf.convert_to_tensor(np.asarray(target))

        try:
            self.completion_status = False
            result, _ = fn(data, target, is_train, initial_epoch)
            self.completion_status = True
            return result

        except Exception as e:
            self.completion_status = True
            print(str(e) + "\n" + traceback.format_exc())

    def testing_function(self, data):
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

        # First try to detect cluster and connect to it. Else, run plain ray.init() for a single node.
        try:
            ray.init(address = "auto")
            print("Running on a Ray Cluster")
        except ConnectionError:
            ray.init()
            print("No cluster found, running on a single Ray instance")

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

    def _num_workers(self):
        return self.settings.num_workers

    def initialize_workers(self):

        num_workers = self._num_workers()
        self.workers = [Worker.options(name = str(i), lifetime = "detached").remote() for i in range(num_workers)]
        self.workers_initialized = True

    def initialize_data_loaders(self, store, schema_Fields=None, dataset_idx=None):
        ### Assume data is in parquet format in train/val_data_path Initialize data loaders to read this parquet format and shard automatically

        if self.workers_initialized:
            shard_count = self._num_workers()
            
            train_dataset = ray.data.read_parquet(store.get_train_data_path(dataset_idx), parallelism=1000) 
            self.train_shards = train_dataset.split(n=shard_count, equal=True, locality_hints=self.workers)
            
            val_dataset = ray.data.read_parquet(store.get_val_data_path(dataset_idx)) 
            self.val_shards = val_dataset.split(n=shard_count, equal=True, locality_hints=self.workers)

            self.data_loaders_initialized = True

        else:
            raise Exception('Ray tasks not initialized for Cerebro. Please run RayBackend.initialize_workers() first!')

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

    def train_for_one_epoch(self, models, store, feature_cols, label_cols, is_train=True):
        
        mode = "Training"
        if not is_train:
            mode = "Validation"
        if self.settings.verbose >= 1:
            print('CEREBRO => Time: {}, Starting EPOCH {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), mode))
        
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

        
        Q = [(i, j) for i in range(len(models)) for j in range(self.settings.num_workers)]
        random.shuffle(Q)
        # print()
        # print()
        # print(Q)
        # print()
        # print()
        model_idle = [True for _ in range(len(models))]
        worker_idle = [True for _ in range(self.settings.num_workers)]
        model_on_worker = [-1 for _ in range(self.settings.num_workers)]

        def place_model_on_worker(j):
            for idx, s in enumerate(Q):
                i, j_prime = s
                if j_prime == j and model_idle[i]:
                    model_idle[i] = False
                    worker_idle[j] = False
                    model_on_worker[j] = i
                    if is_train: data_shard = self.train_shards[j]
                    else: data_shard = self.val_shards[j]
                    result_ref = self.workers[j].execute_subepoch.remote(sub_epoch_trainers[i], data_shard, is_train, models[i].epoch)
                    return result_ref

        while not exit_event.is_set() and len(Q) > 0:
            for j in range(self.settings.num_workers):
                if worker_idle[j]:
                    result_ref = place_model_on_worker(j)
                    if result_ref is not None:
                        result_refs[models[model_on_worker[j]].getRunId()].append(result_ref)
                    # result_this = ray.get(result_ref)
                    # print("result_this:")
                    # print(result_this)
                    # model_id = models[model_on_worker[j]].getRunId()
                    # for k in result_this.keys():
                    #     if k in epoch_results[model_id]:
                    #         epoch_results[model_id][k] = epoch_results[model_id][k].append(result_this[k])
                    #     else:
                    #         epoch_results[model_id][k] = result_this[k]
                    
                elif ray.get(self.workers[j].get_completion_status.remote()):
                    i = model_on_worker[j]
                    Q.remove((i, j))
                    # print()
                    # print(Q)
                    # print()
                    model_idle[i] = True
                    worker_idle[j] = True
                    model_on_worker[j] = -1
                    result_ref = place_model_on_worker(j)
                    if result_ref is not None:
                        result_refs[models[model_on_worker[j]].getRunId()].append(result_ref)
                    
            exit_event.wait(self.settings.polling_period)
        
        for model_id in result_refs.keys():
            refs_this = result_refs[model_id]
            for ref in refs_this:
                result_this = ray.get(ref)
                print(result_this)
                for k in result_this.keys():
                    if k in epoch_results[model_id]:
                        epoch_results[model_id][k] = epoch_results[model_id][k].append(result_this[k])
                    else:
                        epoch_results[model_id][k] = result_this[k]

        print(epoch_results)

        return epoch_results


def _get_remote_trainer(estimator, store, dataset_idx, feature_columns, label_columns):
    run_id = estimator.getRunId()
    
    _, _, metadata, _ = \
        util.get_simple_meta_from_parquet(store,
                                          schema_cols=label_columns + feature_columns,
                                          dataset_idx=dataset_idx)

    estimator._check_params(metadata)
    keras_utils = estimator._get_keras_utils()

    # Checkpointing the model if it does not exist.
    if not estimator._has_checkpoint(run_id): # TODO: Check if you need to use the Ray store instead (WHOLE BLOCK)
        remote_store = store.to_remote(run_id, dataset_idx) 

        with remote_store.get_local_output_dir() as run_output_dir:
            model = estimator._compile_model(keras_utils) 
            ckpt_file = os.path.join(run_output_dir, remote_store.checkpoint_filename)
            model.save(ckpt_file)
            remote_store.sync(run_output_dir)
    return sub_epoch_trainer(estimator, keras_utils, run_id, dataset_idx)


def sub_epoch_trainer(estimator, keras_utils, run_id, dataset_idx):
    # Estimator parameters
    user_callbacks = estimator.callbacks
    custom_objects = estimator.custom_objects
    metrics_names = [name.__name__ if callable(name) else name for name in estimator.metrics]
    user_verbose = estimator.verbose

    floatx = tf.keras.backend.floatx()
    fit_sub_epoch_fn = keras_utils.fit_sub_epoch_fn()
    eval_sub_epoch_fn = keras_utils.eval_sub_epoch_fn()

    # Utility functions
    deserialize_keras_model = _deserialize_keras_model_fn() #TODO: If implementing ray store, do we need this function?
    pin_gpu = _pin_gpu_fn() 

    # Storage
    store = estimator.store # TODO:Check if you need to use the Ray store instead
    remote_store = store.to_remote(run_id, dataset_idx) # TODO: Check if you need to use the Ray store instead

    def train(x_data, y_data, is_train, starting_epoch, local_task_index=0):

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
            step_counter_callback = KerasStepCounter() # Is it making a new Step Counter everytime, so the counter is always 1?
            callbacks = [step_counter_callback]
            if user_callbacks is not None:
                callbacks = callbacks + user_callbacks
            ckpt_file = os.path.join(run_output_dir, remote_store.checkpoint_filename) ## TODO: Check if using Ray store instead of physical store.
            # print(ckpt_file)
            # restoring the model from the previous checkpoint #TODO: Check what to do for Ray Store
            # with tf.keras.utils.custom_object_scope(custom_objects):
            #     model = deserialize_keras_model(
            #         remote_store.get_last_checkpoint(), lambda x: tf.keras.models.load_model(x))
            
            # print(remote_store.get_last_checkpoint)
            model = deserialize_keras_model(
                remote_store.get_last_checkpoint(), lambda x: tf.keras.models.load_model(x))
            
            if starting_epoch is None:
                starting_epoch = 0

            if is_train:
                initialization_time = time.time() - begin_time
                begin_time = time.time()
                result = fit_sub_epoch_fn(starting_epoch, model, x_data, y_data, callbacks, verbose).history
                training_time = time.time() - begin_time
                begin_time = time.time()
                result = {'train_' + name: result[name] for name in result}
                model.save(ckpt_file) # TODO: Check how to work with Ray store, passing to and from checkpoint, and a final saving of model.
            else:
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
