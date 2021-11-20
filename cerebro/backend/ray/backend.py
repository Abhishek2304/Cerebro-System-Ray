from __future__ import absolute_import

import random
import numpy as np
import tensorflow as tf
import datetime
import time

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
    
    def execute_subepoch(fn, data_shard, is_train, initial_epoch):
        try:
            self.completion_status = False
            func_result = fn(data_shard, is_train, initial_epoch)
            self.completion_status = True
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

        # Putting ray.init() here, hoping it will not go out of scope once the __init__ function exits, but only when the
        # class is destroyed. This may not be true, and may have to invoke ray.init() globally somehow.
        ray.init(address = "auto")

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
            
            train_dataset = ray.data.read_parquet(store.get_train_data_path(dataset_idx)) 
            self.train_shards = train_dataset.split(n=shard_count, equal=True, locality_hints=self.workers)
            
            val_dataset = ray.data.read_parquet(store.get_val_data_path(dataset_idx)) 
            self.val_shards = val_dataset.split(n=shard_count, equal=True, locality_hints=self.workers)

            self.data_loaders_initialized = True
        else:
            raise Exception('Ray tasks not initialized for Cerebro. Please run RayBackend.initialize_workers() '
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
        
        model_idle = [True for _ in range(len(models))]
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
                    if is_train: data_shard = self.train_shards[j]
                    else: data_shard = self.val_shards[j]
                    self.workers[j].execute_subepoch.remote(sub_epoch_trainers[i], data_shard, is_train, models[m].epoch)
                    break

        while not exit_event.is_set() and len(Q) > 0:
            for j in range(self.settings.num_workers):
                if worker_idle[j]:
                    place_model_on_worker(j)
                elif ray.get(self.workers[j].get_completion_status.remote()):
                    i = model_on_worker[j]
                    Q.remove((i, j))
                    model_idle[i] = True
                    worker_idle[j] = True
                    model_on_worker[j] = -1
                    place_model_on_worker(j)
                
            exit_event.wait(self.settings.polling_period)


def _get_remote_trainer(estimator, backend, store, dataset_idx, feature_columns, label_columns):
    run_id = estimator.getRunId()
    
    train_rows, val_rows, metadata, avg_row_size = \
        util.get_simple_meta_from_parquet(store,
                                          schema_cols=label_columns + feature_columns,
                                          dataset_idx=dataset_idx)
    estimator._check_params(metadata)
    keras_utils = estimator._get_keras_utils()

    # Checkpointing the model if it does not exist.
    if not estimator._has_checkpoint(run_id):
        remote_store = store.to_remote(run_id, dataset_idx)

        with remote_store.get_local_output_dir() as run_output_dir:
            model = estimator._compile_model(keras_utils)
            ckpt_file = os.path.join(run_output_dir, remote_store.checkpoint_filename)
            model.save(ckpt_file)
            remote_store.sync(run_output_dir)
    return sub_epoch_trainer(estimator, metadata, keras_utils, run_id, dataset_idx,
                                train_rows, val_rows, backend._num_workers())


def sub_epoch_trainer(estimator, metadata, keras_utils, run_id, dataset_idx, train_rows, val_rows,
                      num_workers):
    # Estimator parameters
    label_columns = estimator.label_cols
    feature_columns = estimator.feature_cols
    user_callbacks = estimator.callbacks
    batch_size = estimator.batch_size
    custom_objects = estimator.custom_objects
    metrics_names = [name.__name__ if callable(name) else name for name in estimator.metrics]
    user_verbose = estimator.verbose

    # Model parameters
    input_shapes, output_shapes = estimator.get_model_shapes()
    output_names = estimator.model.output_names
    input_names = estimator.model.input_names

    floatx = tf.keras.backend.floatx()
    fit_sub_epoch_fn = keras_utils.fit_sub_epoch_fn()
    eval_sub_epoch_fn = keras_utils.eval_sub_epoch_fn()
    transformation_fn = estimator.transformation_fn

    # Utility functions
    deserialize_keras_model = _deserialize_keras_model_fn()
    pin_gpu = _pin_gpu_fn()

    # Storage
    store = estimator.store
    remote_store = store.to_remote(run_id, dataset_idx)

    def train(data_shard, is_train, starting_epoch, local_task_index=0):

        begin_time = time.time()

        # Workaround for the issue with huggingface layers needing a python
        # object as config (not a dict) and explicit definition of get_config method.
        # We monkey patch the __init__ method get_config methods of such layers.
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
            callbacks = callbacks + user_callbacks
            ckpt_file = os.path.join(run_output_dir, remote_store.checkpoint_filename)

            # restoring the model from the previous checkpoint
            with tf.keras.utils.custom_object_scope(custom_objects):
                model = deserialize_keras_model(
                    remote_store.get_last_checkpoint(), lambda x: tf.keras.models.load_model(x))

            schema_fields = feature_columns + label_columns

            if is_train:
                initialization_time = time.time() - begin_time
                begin_time = time.time()
                result = fit_sub_epoch_fn(starting_epoch, model, data_shard, callbacks, verbose).history
                training_time = time.time() - begin_time
                begin_time = time.time()
                result = {'train_' + name: result[name] for name in result}
                model.save(ckpt_file)
            else:
                initialization_time = time.time() - begin_time
                begin_time = time.time()
                result = eval_sub_epoch_fn(starting_epoch, model, data_shard, callbacks, verbose)
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

            data_reader.reset()
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
