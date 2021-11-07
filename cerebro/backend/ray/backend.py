from __future__ import absolute_import
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


class RayBackend(Backend):

    def __init__(self, num_workers = None, start_timeout = 600, num_data_readers = 10, verbose = 1, 
                data_readers_pool_type = 'thread', ):

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
        self.task_clients = None
        self.data_loaders_initialized = False
        
        # May not need the below attributes, remove if not needed for Ray
        self.driver = None
        self.driver_client = None
        self.spark_job_group = None
        self.rand = np.random.RandomState(constants.RANDOM_SEED)

    def _num_workers(self):
        
        return self.settings.num_workers

    def initialize_workers(self):
        pass

    def initialize_data_loaders(self, store, schema_fields):
        pass

    def teardown_workers(self):
        pass

    def prepare_data(self, store, dataset, validation, compress_sparse=False, verbose=2):
        pass

    def get_metadata_from_parquet(self, store, label_columns=['label'], feature_columns=['features']):
        pass

    def train_for_one_epoch(self, models, store, feature_col, label_col, is_train=True):
        pass

@ray.remote
class Worker(object):
    def __init__(self):
        pass
