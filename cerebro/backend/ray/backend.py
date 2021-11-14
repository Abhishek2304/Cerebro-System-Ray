from __future__ import absolute_import
import random
import numpy as np
import tensorflow as tf
import datetime

from . import service_driver, service_task, util
from .. import constants
from .. import timeout, settings as ray_settings, secret, host_hash, job_id
from ..backend import Backend
from ...commons.util import patch_hugginface_layer_methods
from ...commons.constants import exit_event

import ray
import psutil

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

class RayBackend(Backend):

    def __init__(self, num_workers = None, start_timeout = 600, num_data_readers = 10, verbose = 1, 
                data_readers_pool_type = 'thread', ):

        # Putting ray.init() here, hoping it will not go out of scope once the __init__ function exits, but only when the
        # class is destroyed. This may not be true, and may have to invoke ray.init() globally somehow.
        ray.init()

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

        # Spark defines the spark context here, to specify the connection to the spark cluster. Can consider doing ray init here,
        # but need to make sure __init__ will not go out of scope or Ray will be shut down.
        
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
        
        # May not need the below attributes, remove if not needed for Ray
        self.driver = None
        self.driver_client = None
        self.spark_job_group = None
        self.rand = np.random.RandomState(constants.RANDOM_SEED)

    def _num_workers(self):
        return self.settings.num_workers

    def initialize_workers(self):

        num_workers = self._num_workers()
        self.workers = [Worker.options(name=str(i), lifetime = "detached").remote() for i in range(num_workers)]
        self.workers_initialized = True

    def initialize_data_loaders(self, store, schema_fields):
        pass

    def teardown_workers(self):
        
        # Consider, instead of forcefully killing it, to remove the detached lifetime and set it to null.
        # Hence, when all references to actor handle are removed, it is automatically garbage collected
        # and the process is stopped.
        for worker in self.workers:
            ray.kill(worker)

        self.workers = None
        self.workers_initialized = False
        self.data_loaders_initialized = False

        # Consider explicitly shutting down ray here, but before that make sure that all data is 
        # written to persistent storage
        # ray.shutdown()

    def prepare_data(self, store, dataset, validation, compress_sparse=False, verbose=2):
        pass

    def get_metadata_from_parquet(self, store, label_columns=['label'], feature_columns=['features']):
        pass

    def train_for_one_epoch(self, models, store, feature_col, label_col, is_train=True):
        Q = [(i, j) for i in range(len(models)) for j in range(self.settings.num_workers)]
        random.shuffle(Q)
        
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
                    model_weights[i] = self.workers[j].train_subepoch.remote(models[i], model_weights[i])
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