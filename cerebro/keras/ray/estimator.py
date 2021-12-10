from __future__ import absolute_import

import time

from ...backend import codec
from .util import TF_KERAS, TFKerasUtil
from ..estimator import CerebroEstimator, CerebroModel

import threading

LOCK = threading.Lock()
MODEL_ID = -1

def next_model_id():
    global LOCK, MODEL_ID
    with LOCK:
        MODEL_ID += 1
        return MODEL_ID

def _check_validation(validation):
    if validation:
        if isinstance(validation, float):
            if validation < 0 or validation >= 1:
                raise ValueError('Validation split {} must be in the range: [0, 1)'
                                 .format(validation))
        else:
            raise ValueError('Param validation must be of type "float", found: {}'
                             .format(type(validation)))

class RayEstimator(CerebroEstimator):

    """Cerebro Ray Estimator for fitting Keras model to a DataFrame.

    Supports ``tf.keras >= 2.3``.

    Args:
        model: Keras model to train.
        store: The object store storing train and val paths
        epoch: The current epoch number that has been trained
        batch_size: Batch size of the data
        feature_cols/label_cols: The column names of the features/labels
        custom_objects: Optional dictionary mapping names (strings) to custom classes or functions to be considered
                        during serialization/deserialization.
        optimizer: Keras optimizer.
        loss: Keras loss or list of losses.
        batch_size: Number of rows from the DataFrame per batch.
        loss_weights: (Optional) List of float weight values to assign each loss.
        metrics: (Optional) List of Keras metrics to record.
        callbacks: (Optional) List of Keras callbacks.
        transformation_fn: (Optional) Function that takes a TensorFlow Dataset as its parameter
                       and returns a modified Dataset that is then fed into the
                       train or validation step. This transformation is applied before batching.
    """
    def __init__(self,
                 model=None,
                 store=None,
                 custom_objects=None,
                 epoch=None,
                 optimizer=None,
                 loss=None,
                 batch_size=None,
                 loss_weights=None,
                 feature_cols=None,
                 label_cols=None,
                 validation=None,
                 metrics=None,
                 callbacks=None,
                 transformation_fn=None,
                 verbose=None
                 ):
        
        super(RayEstimator, self).__init__()
        self.model = model
        self.store = store
        self.custom_objects = custom_objects
        self.epoch=epoch
        self.optimizer = optimizer
        self.loss = loss
        self.batch_size = batch_size
        self.loss_weights = loss_weights
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.validation = validation
        self.metrics = metrics
        self.callbacks = callbacks
        self.transformation_fn = transformation_fn
        self.verbose = verbose
        self.hparams = None

        run_id = 'model_' + str(next_model_id()) + '_' + str(int(time.time()))
        self.run_id = run_id

    def setModel(self,model):
        self.model = model

    def setCustomObjects(self,custom_objects):
        self.custom_objects = custom_objects

    def setOptimizer(self,optimizer):
        self.optimizer = optimizer

    def setEpoch(self, epoch):
        self.epoch = epoch

    def setLossWeights(self, weights):
        self.loss_weights = weights

    def setFeatureCols(self,feature_cols):
        self.feature_cols = feature_cols

    def setLabelCols(self,label_cols):
        self.label_cols = label_cols

    def setStore(self,store):
        self.store = store
    
    def setVerbose(self, verbose):
        self.verbose = verbose

    def setHyperParams(self, params):
        self.hparams = params

    def getEpochs(self):
        return self.epoch

    def getFeatureCols(self):
        return self.feature_cols

    def getLabelCols(self):
        return self.label_cols
    
    def getRunId(self):
        return self.run_id

    def getModel(self):
        return self.model
    
    def getEpoch(self):
        return self.epoch

    def getCustomObjects(self):
        return self.custom_objects
    
    def _get_keras_utils(self):
        return TFKerasUtil
    
    def _check_params(self, metadata):
        model = self.model
        if not model:
            raise ValueError('Model parameter is required')

        _check_validation(self.validation)

        feature_columns = self.feature_cols
        missing_features = [col for col in feature_columns if col not in metadata]
        if missing_features:
            raise ValueError('Feature columns {} not found in training DataFrame metadata'
                             .format(missing_features))

        label_columns = self.label_cols
        missing_labels = [col for col in label_columns if col not in metadata]
        if missing_labels:
            raise ValueError('Label columns {} not found in training DataFrame metadata'
                             .format(missing_labels))
    
    def get_model_shapes(self):
        input_shapes = [[dim if dim else -1 for dim in input.shape.as_list()]
                        for input in self.model.inputs]
        output_shapes = [[dim if dim else -1 for dim in output.shape.as_list()]
                         for output in self.model.outputs]
        return input_shapes, output_shapes

    def _load_model_from_checkpoint(self, run_id):
        last_ckpt_path = self.store.get_checkpoint_path(run_id)

        model_bytes = self.store.read(last_ckpt_path)
        return codec.dumps_base64(model_bytes)
    
    def _compile_model(self, keras_utils):
        # Compile the model with all the parameters
        model = self.model

        loss = self.loss
        loss_weights = self.loss_weights

        if not loss:
            raise ValueError('Loss parameter is required for the model to compile')

        optimizer = self.optimizer
        if not optimizer:
            optimizer = model.optimizer

        if not optimizer:
            raise ValueError('Optimizer must be provided either as a parameter or as part of a '
                             'compiled model')

        metrics = self.metrics
        optimizer_weight_values = optimizer.get_weights()

        model.compile(optimizer=optimizer,
                      loss=loss,
                      loss_weights=loss_weights,
                      metrics=metrics)

        if optimizer_weight_values:
            model.optimizer.set_weights(optimizer_weight_values)

        return model
    
    def create_model(self, history, run_id, metadata = None):
        keras_utils = TFKerasUtil
        keras_module = keras_utils.keras()
        floatx = keras_module.backend.floatx()
        custom_objects = self.custom_objects
        serialized_model = self._load_model_from_checkpoint(run_id)

        def load_model_fn(x):
            if custom_objects is None:
                return keras_module.models.load_model(x)
            with keras_module.utils.custom_object_scope(custom_objects):
                return keras_module.models.load_model(x)

        model = keras_utils.deserialize_model(serialized_model, load_model_fn=load_model_fn)
        return self.get_model_class()(**self._get_model_kwargs(model, history, run_id, metadata, floatx))

    def get_model_class(self):
        return RayModel

    def _get_model_kwargs(self, model, history, run_id, floatx, metadata = None):
        return dict(history=history,
                    model=model,
                    feature_columns=self.getFeatureCols(),
                    label_columns=self.getLabelCols(),
                    custom_objects=self.getCustomObjects(),
                    run_id=run_id,
                    _metadata = metadata,
                    _floatx=floatx)
    
    def _has_checkpoint(self, run_id):
        store = self.store
        last_ckpt_path = store.get_checkpoint_path(run_id)
        return last_ckpt_path is not None and store.exists(last_ckpt_path)


class RayModel(CerebroModel):
    """Ray Transformer wrapping a Keras model, used for making predictions on a DataFrame.

    Args:
        history: List of metrics, one entry per epoch during training.
        model: Trained Keras model.
        feature_columns: List of feature column names.
        label_columns: List of label column names.
        custom_objects: Keras custom objects.
        metadata: The metadata of the underlying data the model is trained on
        run_id: ID of the run used to train the model.
    """

    def __init__(self,
                 history=None,
                 model=None,
                 feature_columns=None,
                 label_columns=None,
                 custom_objects=None,
                 run_id=None,
                 _metadata=None,
                 _floatx=None):

        super(RayModel, self).__init__()
        self.history = history
        self.model = model
        self.feature_columns = feature_columns
        self.label_columns = label_columns
        self.custom_objects = custom_objects
        self.run_id = run_id
        self._metadata = _metadata
        self._floatx = _floatx

        if label_columns:
            self.output_cols = [col + '__output' for col in label_columns]

    def setCustomObjects(self, value):
        self.custom_objects = value

    def getCustomObjects(self):
        return self.custom_objects

    def getHistory(self):
        return self.history

    def keras(self):
        """ Returns the trained model in Keras format.

            :return: TensorFlow Keras Model
        """
        return self.model