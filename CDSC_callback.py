# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=g-import-not-at-top
# pylint: disable=g-classes-have-attributes
"""Callbacks: utilities called at certain points during model training."""

import collections
import copy
import csv
import json
import os
import re
import sys
import time

import numpy as np

from tensorflow.core.framework import summary_pb2
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.checkpoint import checkpoint_options as checkpoint_options_lib
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.keras import backend
from tensorflow.python.keras.distribute import distributed_file_utils
from tensorflow.python.keras.distribute import worker_training_state
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.keras.utils.io_utils import path_to_string
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.python.saved_model import save_options as save_options_lib
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls

try:
  import requests
except ImportError:
  requests = None


# Note: `configure_callbacks` is only used in TF1.
def configure_callbacks(callbacks,
                        model,
                        do_validation=False,
                        batch_size=None,
                        epochs=None,
                        steps_per_epoch=None,
                        samples=None,
                        verbose=1,
                        count_mode='steps',
                        mode=ModeKeys.TRAIN):
  """Configures callbacks for use in various training loops.

  Args:
      callbacks: List of Callbacks.
      model: Model being trained.
      do_validation: Whether or not validation loop will be run.
      batch_size: Number of samples per batch.
      epochs: Number of epoch to train.
      steps_per_epoch: Number of batches to run per training epoch.
      samples: Number of training samples.
      verbose: int, 0 or 1. Keras logging verbosity to pass to ProgbarLogger.
      count_mode: One of 'steps' or 'samples'. Per-batch or per-sample count.
      mode: String. One of ModeKeys.TRAIN, ModeKeys.TEST, or ModeKeys.PREDICT.
        Which loop mode to configure callbacks for.

  Returns:
      Instance of CallbackList used to control all Callbacks.
  """
  # Check if callbacks have already been configured.
  if isinstance(callbacks, CallbackList):
    return callbacks

  if not callbacks:
    callbacks = []

  # Add additional callbacks during training.
  if mode == ModeKeys.TRAIN:
    model.history = History()
    callbacks = [BaseLogger()] + (callbacks or []) + [model.history]
    if verbose:
      callbacks.append(ProgbarLogger(count_mode))
  callback_list = CallbackList(callbacks)

  # Set callback model
  callback_model = model._get_callback_model()  # pylint: disable=protected-access
  callback_list.set_model(callback_model)

  set_callback_parameters(
      callback_list,
      model,
      do_validation=do_validation,
      batch_size=batch_size,
      epochs=epochs,
      steps_per_epoch=steps_per_epoch,
      samples=samples,
      verbose=verbose,
      mode=mode)

  callback_list.model.stop_training = False
  return callback_list


def set_callback_parameters(callback_list,
                            model,
                            do_validation=False,
                            batch_size=None,
                            epochs=None,
                            steps_per_epoch=None,
                            samples=None,
                            verbose=1,
                            mode=ModeKeys.TRAIN):
  """Sets callback parameters.

  Args:
      callback_list: CallbackList instance.
      model: Model being trained.
      do_validation: Whether or not validation loop will be run.
      batch_size: Number of samples per batch.
      epochs: Number of epoch to train.
      steps_per_epoch: Number of batches to run per training epoch.
      samples: Number of training samples.
      verbose: int, 0 or 1. Keras logging verbosity to pass to ProgbarLogger.
      mode: String. One of ModeKeys.TRAIN, ModeKeys.TEST, or ModeKeys.PREDICT.
        Which loop mode to configure callbacks for.
  """
  metric_names = model.metrics_names
  for cbk in callback_list:
    if isinstance(cbk, (BaseLogger, ProgbarLogger)):
      cbk.stateful_metrics = metric_names[1:]  # Exclude `loss`

  # Set callback parameters
  callback_metrics = []
  # When we have deferred build scenario with iterator input, we will compile
  # when we standardize first batch of data.
  if mode != ModeKeys.PREDICT:
    callback_metrics = copy.copy(metric_names)
    if do_validation:
      callback_metrics += ['val_' + n for n in metric_names]
  callback_params = {
      'batch_size': batch_size,
      'epochs': epochs,
      'steps': steps_per_epoch,
      'samples': samples,
      'verbose': verbose,
      'do_validation': do_validation,
      'metrics': callback_metrics,
  }
  callback_list.set_params(callback_params)


def _is_generator_like(data):
  """Checks if data is a generator, Sequence, or Iterator."""
  return (hasattr(data, '__next__') or hasattr(data, 'next') or isinstance(
      data, (Sequence, iterator_ops.Iterator, iterator_ops.IteratorBase)))


def make_logs(model, logs, outputs, mode, prefix=''):
  """Computes logs for sending to `on_batch_end` methods."""
  metric_names = model.metrics_names
  if mode in {ModeKeys.TRAIN, ModeKeys.TEST} and metric_names:
    for label, output in zip(metric_names, outputs):
      logs[prefix + label] = output
  else:
    logs['outputs'] = outputs
  return logs


class CallbackList:
  """Container abstracting a list of callbacks."""

  def __init__(self,
               callbacks=None,
               add_history=False,
               add_progbar=False,
               model=None,
               **params):
    """Container for `Callback` instances.

    This object wraps a list of `Callback` instances, making it possible
    to call them all at once via a single endpoint
    (e.g. `callback_list.on_epoch_end(...)`).

    Args:
      callbacks: List of `Callback` instances.
      add_history: Whether a `History` callback should be added, if one does not
        already exist in the `callbacks` list.
      add_progbar: Whether a `ProgbarLogger` callback should be added, if one
        does not already exist in the `callbacks` list.
      model: The `Model` these callbacks are used with.
      **params: If provided, parameters will be passed to each `Callback` via
        `Callback.set_params`.
    """
    self.callbacks = nest.flatten(callbacks) if callbacks else []
    self._add_default_callbacks(add_history, add_progbar)

    if model:
      self.set_model(model)
    if params:
      self.set_params(params)

    # Performance optimization: determines if batch hooks need to be called.
    # pylint: disable=protected-access
    self._supports_tf_logs = all(
        getattr(cb, '_supports_tf_logs', False) for cb in self.callbacks)
    self._batch_hooks_support_tf_logs = all(
        getattr(cb, '_supports_tf_logs', False)
        for cb in self.callbacks
        if cb._implements_train_batch_hooks() or cb
        ._implements_test_batch_hooks() or cb._implements_predict_batch_hooks())

    self._should_call_train_batch_hooks = any(
        cb._implements_train_batch_hooks() for cb in self.callbacks)
    self._should_call_test_batch_hooks = any(
        cb._implements_test_batch_hooks() for cb in self.callbacks)
    self._should_call_predict_batch_hooks = any(
        cb._implements_predict_batch_hooks() for cb in self.callbacks)
    # pylint: enable=protected-access

    self._disallow_batch_hooks_in_ps_strategy()

    # Performance check: Check batch hooks for slowness compared to batch time.
    # Only run check for custom callbacks (i.e. not present in this file).
    self._check_timing = any(
        cbk.__class__.__name__ not in globals() for cbk in self.callbacks)
    self._num_batches_for_timing_check = 5
    self._hook_times = {}
    self._batch_start_time = None
    self._batch_times = []

  def _add_default_callbacks(self, add_history, add_progbar):
    """Adds `Callback`s that are always present."""
    self._progbar = None
    self._history = None

    for cb in self.callbacks:
      if isinstance(cb, ProgbarLogger):
        self._progbar = cb
      elif isinstance(cb, History):
        self._history = cb

    if self._progbar is None and add_progbar:
      self._progbar = ProgbarLogger(count_mode='steps')
      self.callbacks.insert(0, self._progbar)

    if self._history is None and add_history:
      self._history = History()
      self.callbacks.append(self._history)

  def _process_logs(self, logs, is_batch_hook=False):
    """Turns tensors into numpy arrays or Python scalars if necessary."""
    if logs is None:
      return {}
    if self._supports_tf_logs:
      return logs
    if is_batch_hook and self._batch_hooks_support_tf_logs:
      return logs
    return tf_utils.sync_to_numpy_or_python_type(logs)

  def append(self, callback):
    self.callbacks.append(callback)

  def set_params(self, params):
    self.params = params
    for callback in self.callbacks:
      callback.set_params(params)

  def set_model(self, model):
    self.model = model
    if self._history:
      model.history = self._history
    for callback in self.callbacks:
      callback.set_model(model)

  def _call_batch_hook(self, mode, hook, batch, logs=None):
    """Helper function for all batch_{begin | end} methods."""
    if not self.callbacks:
      return

    if hook == 'begin':
      self._call_batch_begin_hook(mode, batch, logs)
    elif hook == 'end':
      self._call_batch_end_hook(mode, batch, logs)
    else:
      raise ValueError('Unrecognized hook: {}'.format(hook))

  def _call_batch_begin_hook(self, mode, batch, logs):
    """Helper function for `on_*_batch_begin` methods."""
    hook_name = 'on_{mode}_batch_begin'.format(mode=mode)
    self._call_batch_hook_helper(hook_name, batch, logs)

    if self._check_timing:
      self._batch_start_time = time.time()

  def _call_batch_end_hook(self, mode, batch, logs):
    """Helper function for `on_*_batch_end` methods."""
    hook_name = 'on_{mode}_batch_end'.format(mode=mode)

    if self._check_timing and batch >= 1:
      batch_time = time.time() - self._batch_start_time
      self._batch_times.append(batch_time)

    self._call_batch_hook_helper(hook_name, batch, logs)

    if len(self._batch_times) >= self._num_batches_for_timing_check:
      end_hook_name = hook_name
      begin_hook_name = 'on_{mode}_batch_begin'.format(mode=mode)
      avg_batch_time = sum(self._batch_times) / len(self._batch_times)
      avg_end_hook_time = sum(self._hook_times[end_hook_name]) / len(
          self._hook_times[end_hook_name])
      avg_begin_hook_time = sum(self._hook_times[begin_hook_name]) / len(
          self._hook_times[begin_hook_name])

      threshold_time = 1.0 * avg_batch_time
      warning_msg = ('Callback method `{hook}` is slow compared to '
                     'the batch time (batch time: {batch_time:.4f}s vs '
                     '`{hook}` time: {hook_time:.4f}s). Check your callbacks.')
      if avg_begin_hook_time > threshold_time:
        logging.warning(warning_msg.format(
            hook=begin_hook_name,
            batch_time=avg_batch_time,
            hook_time=avg_begin_hook_time))
      if avg_end_hook_time > threshold_time:
        logging.warning(warning_msg.format(
            hook=end_hook_name,
            batch_time=avg_batch_time,
            hook_time=avg_end_hook_time))
      self._check_timing = False
      self._batch_start_time = None
      self._batch_times = []
      self._hook_times = {}

  def _call_batch_hook_helper(self, hook_name, batch, logs):
    """Helper function for `on_*_batch_*` methods."""
    if self._check_timing:
      start_time = time.time()

    logs = self._process_logs(logs, is_batch_hook=True)
    for callback in self.callbacks:
      hook = getattr(callback, hook_name)
      hook(batch, logs)

    if self._check_timing:
      if hook_name not in self._hook_times:
        self._hook_times[hook_name] = []
      self._hook_times[hook_name].append(time.time() - start_time)

  def _call_begin_hook(self, mode):
    """Helper function for on_{train|test|predict}_begin methods."""
    if mode == ModeKeys.TRAIN:
      self.on_train_begin()
    elif mode == ModeKeys.TEST:
      self.on_test_begin()
    else:
      self.on_predict_begin()

  def _call_end_hook(self, mode):
    """Helper function for on_{train|test|predict}_end methods."""
    if mode == ModeKeys.TRAIN:
      self.on_train_end()
    elif mode == ModeKeys.TEST:
      self.on_test_end()
    else:
      self.on_predict_end()

  def on_batch_begin(self, batch, logs=None):
    if self._should_call_train_batch_hooks:
      self._call_batch_hook(ModeKeys.TRAIN, 'begin', batch, logs=logs)

  def on_batch_end(self, batch, logs=None):
    if self._should_call_train_batch_hooks:
      self._call_batch_hook(ModeKeys.TRAIN, 'end', batch, logs=logs)

  def on_epoch_begin(self, epoch, logs=None):
    """Calls the `on_epoch_begin` methods of its callbacks.

    This function should only be called during TRAIN mode.

    Args:
        epoch: Integer, index of epoch.
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    logs = self._process_logs(logs)
    for callback in self.callbacks:
      callback.on_epoch_begin(epoch, logs)

  def on_epoch_end(self, epoch, logs=None):
    """Calls the `on_epoch_end` methods of its callbacks.

    This function should only be called during TRAIN mode.

    Args:
        epoch: Integer, index of epoch.
        logs: Dict, metric results for this training epoch, and for the
          validation epoch if validation is performed. Validation result keys
          are prefixed with `val_`.
    """
    logs = self._process_logs(logs)
    for callback in self.callbacks:
      callback.on_epoch_end(epoch, logs)

  def on_train_batch_begin(self, batch, logs=None):
    """Calls the `on_train_batch_begin` methods of its callbacks.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict, contains the return value of `model.train_step`. Typically,
          the values of the `Model`'s metrics are returned.  Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
    """
    if self._should_call_train_batch_hooks:
      self._call_batch_hook(ModeKeys.TRAIN, 'begin', batch, logs=logs)

  def on_train_batch_end(self, batch, logs=None):
    """Calls the `on_train_batch_end` methods of its callbacks.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Aggregated metric results up until this batch.
    """
    if self._should_call_train_batch_hooks:
      self._call_batch_hook(ModeKeys.TRAIN, 'end', batch, logs=logs)

  def on_test_batch_begin(self, batch, logs=None):
    """Calls the `on_test_batch_begin` methods of its callbacks.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict, contains the return value of `model.test_step`. Typically,
          the values of the `Model`'s metrics are returned.  Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
    """
    if self._should_call_test_batch_hooks:
      self._call_batch_hook(ModeKeys.TEST, 'begin', batch, logs=logs)

  def on_test_batch_end(self, batch, logs=None):
    """Calls the `on_test_batch_end` methods of its callbacks.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Aggregated metric results up until this batch.
    """
    if self._should_call_test_batch_hooks:
      self._call_batch_hook(ModeKeys.TEST, 'end', batch, logs=logs)

  def on_predict_batch_begin(self, batch, logs=None):
    """Calls the `on_predict_batch_begin` methods of its callbacks.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict, contains the return value of `model.predict_step`,
          it typically returns a dict with a key 'outputs' containing
          the model's outputs.
    """
    if self._should_call_predict_batch_hooks:
      self._call_batch_hook(ModeKeys.PREDICT, 'begin', batch, logs=logs)

  def on_predict_batch_end(self, batch, logs=None):
    """Calls the `on_predict_batch_end` methods of its callbacks.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Aggregated metric results up until this batch.
    """
    if self._should_call_predict_batch_hooks:
      self._call_batch_hook(ModeKeys.PREDICT, 'end', batch, logs=logs)

  def on_train_begin(self, logs=None):
    """Calls the `on_train_begin` methods of its callbacks.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    logs = self._process_logs(logs)
    for callback in self.callbacks:
      callback.on_train_begin(logs)

  def on_train_end(self, logs=None):
    """Calls the `on_train_end` methods of its callbacks.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    logs = self._process_logs(logs)
    for callback in self.callbacks:
      callback.on_train_end(logs)

  def on_test_begin(self, logs=None):
    """Calls the `on_test_begin` methods of its callbacks.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    logs = self._process_logs(logs)
    for callback in self.callbacks:
      callback.on_test_begin(logs)

  def on_test_end(self, logs=None):
    """Calls the `on_test_end` methods of its callbacks.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    logs = self._process_logs(logs)
    for callback in self.callbacks:
      callback.on_test_end(logs)

  def on_predict_begin(self, logs=None):
    """Calls the 'on_predict_begin` methods of its callbacks.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    logs = self._process_logs(logs)
    for callback in self.callbacks:
      callback.on_predict_begin(logs)

  def on_predict_end(self, logs=None):
    """Calls the `on_predict_end` methods of its callbacks.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    logs = self._process_logs(logs)
    for callback in self.callbacks:
      callback.on_predict_end(logs)

  def __iter__(self):
    return iter(self.callbacks)

  def _disallow_batch_hooks_in_ps_strategy(self):
    """Error out if batch-level callbacks are passed with PSStrategy."""
    # pylint: disable=protected-access
    strategy = distribute_lib.get_strategy()
    if strategy._should_use_with_coordinator:
      unsupported_callbacks = []
      for cb in self.callbacks:
        # These Callbacks can accept RemoteValues directly.
        if getattr(cb, '_supports_tf_logs', False):
          continue
        if (cb._implements_train_batch_hooks() or
            cb._implements_test_batch_hooks() or
            cb._implements_predict_batch_hooks()):
          unsupported_callbacks.append(cb)
      if unsupported_callbacks:
        raise ValueError('Batch-level `Callback`s are not supported with '
                         '`ParameterServerStrategy`. Found unsupported '
                         'callbacks: {}'.format(unsupported_callbacks))
    # pylint: enable=protected-access


class Callback:
  """Abstract base class used to build new callbacks.

  Callbacks can be passed to keras methods such as `fit`, `evaluate`, and
  `predict` in order to hook into the various stages of the model training and
  inference lifecycle.

  To create a custom callback, subclass `keras.callbacks.Callback` and override
  the method associated with the stage of interest. See
  https://www.tensorflow.org/guide/keras/custom_callback for more information.

  Example:

  >>> training_finished = False
  >>> class MyCallback(tf.keras.callbacks.Callback):
  ...   def on_train_end(self, logs=None):
  ...     global training_finished
  ...     training_finished = True
  >>> model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
  >>> model.compile(loss='mean_squared_error')
  >>> model.fit(tf.constant([[1.0]]), tf.constant([[1.0]]),
  ...           callbacks=[MyCallback()])
  >>> assert training_finished == True

  If you want to use `Callback` objects in a custom training loop:

  1. You should pack all your callbacks into a single `callbacks.CallbackList`
     so they can all be called together.
  2. You will need to manually call all the `on_*` methods at the apropriate
     locations in your loop. Like this:

     ```
     callbacks =  tf.keras.callbacks.CallbackList([...])
     callbacks.append(...)

     callbacks.on_train_begin(...)
     for epoch in range(EPOCHS):
       callbacks.on_epoch_begin(epoch)
       for i, data in dataset.enumerate():
         callbacks.on_train_batch_begin(i)
         batch_logs = model.train_step(data)
         callbacks.on_train_batch_end(i, batch_logs)
       epoch_logs = ...
       callbacks.on_epoch_end(epoch, epoch_logs)
     final_logs=...
     callbacks.on_train_end(final_logs)
     ```

  Attributes:
      params: Dict. Training parameters
          (eg. verbosity, batch size, number of epochs...).
      model: Instance of `keras.models.Model`.
          Reference of the model being trained.

  The `logs` dictionary that callback methods
  take as argument will contain keys for quantities relevant to
  the current batch or epoch (see method-specific docstrings).
  """

  def __init__(self):
    self.validation_data = None  # pylint: disable=g-missing-from-attributes
    self.model = None
    # Whether this Callback should only run on the chief worker in a
    # Multi-Worker setting.
    # TODO(omalleyt): Make this attr public once solution is stable.
    self._chief_worker_only = None
    self._supports_tf_logs = False

  def set_params(self, params):
    self.params = params

  def set_model(self, model):
    self.model = model

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_batch_begin(self, batch, logs=None):
    """A backwards compatibility alias for `on_train_batch_begin`."""

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_batch_end(self, batch, logs=None):
    """A backwards compatibility alias for `on_train_batch_end`."""

  @doc_controls.for_subclass_implementers
  def on_epoch_begin(self, epoch, logs=None):
    """Called at the start of an epoch.

    Subclasses should override for any actions to run. This function should only
    be called during TRAIN mode.

    Args:
        epoch: Integer, index of epoch.
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """

  @doc_controls.for_subclass_implementers
  def on_epoch_end(self, epoch, logs=None):
    """Called at the end of an epoch.

    Subclasses should override for any actions to run. This function should only
    be called during TRAIN mode.

    Args:
        epoch: Integer, index of epoch.
        logs: Dict, metric results for this training epoch, and for the
          validation epoch if validation is performed. Validation result keys
          are prefixed with `val_`. For training epoch, the values of the
         `Model`'s metrics are returned. Example : `{'loss': 0.2, 'accuracy':
           0.7}`.
    """

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_train_batch_begin(self, batch, logs=None):
    """Called at the beginning of a training batch in `fit` methods.

    Subclasses should override for any actions to run.

    Note that if the `steps_per_execution` argument to `compile` in
    `tf.keras.Model` is set to `N`, this method will only be called every `N`
    batches.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict, contains the return value of `model.train_step`. Typically,
          the values of the `Model`'s metrics are returned.  Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
    """
    # For backwards compatibility.
    self.on_batch_begin(batch, logs=logs)

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_train_batch_end(self, batch, logs=None):
    """Called at the end of a training batch in `fit` methods.

    Subclasses should override for any actions to run.

    Note that if the `steps_per_execution` argument to `compile` in
    `tf.keras.Model` is set to `N`, this method will only be called every `N`
    batches.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Aggregated metric results up until this batch.
    """
    # For backwards compatibility.
    self.on_batch_end(batch, logs=logs)

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_test_batch_begin(self, batch, logs=None):
    """Called at the beginning of a batch in `evaluate` methods.

    Also called at the beginning of a validation batch in the `fit`
    methods, if validation data is provided.

    Subclasses should override for any actions to run.

    Note that if the `steps_per_execution` argument to `compile` in
    `tf.keras.Model` is set to `N`, this method will only be called every `N`
    batches.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict, contains the return value of `model.test_step`. Typically,
          the values of the `Model`'s metrics are returned.  Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
    """

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_test_batch_end(self, batch, logs=None):
    """Called at the end of a batch in `evaluate` methods.

    Also called at the end of a validation batch in the `fit`
    methods, if validation data is provided.

    Subclasses should override for any actions to run.

    Note that if the `steps_per_execution` argument to `compile` in
    `tf.keras.Model` is set to `N`, this method will only be called every `N`
    batches.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Aggregated metric results up until this batch.
    """

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_predict_batch_begin(self, batch, logs=None):
    """Called at the beginning of a batch in `predict` methods.

    Subclasses should override for any actions to run.

    Note that if the `steps_per_execution` argument to `compile` in
    `tf.keras.Model` is set to `N`, this method will only be called every `N`
    batches.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict, contains the return value of `model.predict_step`,
          it typically returns a dict with a key 'outputs' containing
          the model's outputs.
    """

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_predict_batch_end(self, batch, logs=None):
    """Called at the end of a batch in `predict` methods.

    Subclasses should override for any actions to run.

    Note that if the `steps_per_execution` argument to `compile` in
    `tf.keras.Model` is set to `N`, this method will only be called every `N`
    batches.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Aggregated metric results up until this batch.
    """

  @doc_controls.for_subclass_implementers
  def on_train_begin(self, logs=None):
    """Called at the beginning of training.

    Subclasses should override for any actions to run.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """

  @doc_controls.for_subclass_implementers
  def on_train_end(self, logs=None):
    """Called at the end of training.

    Subclasses should override for any actions to run.

    Args:
        logs: Dict. Currently the output of the last call to `on_epoch_end()`
          is passed to this argument for this method but that may change in
          the future.
    """

  @doc_controls.for_subclass_implementers
  def on_test_begin(self, logs=None):
    """Called at the beginning of evaluation or validation.

    Subclasses should override for any actions to run.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """

  @doc_controls.for_subclass_implementers
  def on_test_end(self, logs=None):
    """Called at the end of evaluation or validation.

    Subclasses should override for any actions to run.

    Args:
        logs: Dict. Currently the output of the last call to
          `on_test_batch_end()` is passed to this argument for this method
          but that may change in the future.
    """

  @doc_controls.for_subclass_implementers
  def on_predict_begin(self, logs=None):
    """Called at the beginning of prediction.

    Subclasses should override for any actions to run.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """

  @doc_controls.for_subclass_implementers
  def on_predict_end(self, logs=None):
    """Called at the end of prediction.

    Subclasses should override for any actions to run.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """

  def _implements_train_batch_hooks(self):
    """Determines if this Callback should be called for each train batch."""
    return (not generic_utils.is_default(self.on_batch_begin) or
            not generic_utils.is_default(self.on_batch_end) or
            not generic_utils.is_default(self.on_train_batch_begin) or
            not generic_utils.is_default(self.on_train_batch_end))

  def _implements_test_batch_hooks(self):
    """Determines if this Callback should be called for each test batch."""
    return (not generic_utils.is_default(self.on_test_batch_begin) or
            not generic_utils.is_default(self.on_test_batch_end))

  def _implements_predict_batch_hooks(self):
    """Determines if this Callback should be called for each predict batch."""
    return (not generic_utils.is_default(self.on_predict_batch_begin) or
            not generic_utils.is_default(self.on_predict_batch_end))


class BaseLogger(Callback):
  """Callback that accumulates epoch averages of metrics.

  This callback is automatically applied to every Keras model.

  Args:
      stateful_metrics: Iterable of string names of metrics that
          should *not* be averaged over an epoch.
          Metrics in this list will be logged as-is in `on_epoch_end`.
          All others will be averaged in `on_epoch_end`.
  """

  def __init__(self, stateful_metrics=None):
    super(BaseLogger, self).__init__()
    self.stateful_metrics = set(stateful_metrics or [])

  def on_epoch_begin(self, epoch, logs=None):
    self.seen = 0
    self.totals = {}

  def on_batch_end(self, batch, logs=None):
    logs = logs or {}
    batch_size = logs.get('size', 0)
    # In case of distribution strategy we can potentially run multiple steps
    # at the same time, we should account for that in the `seen` calculation.
    num_steps = logs.get('num_steps', 1)
    self.seen += batch_size * num_steps

    for k, v in logs.items():
      if k in self.stateful_metrics:
        self.totals[k] = v
      else:
        if k in self.totals:
          self.totals[k] += v * batch_size
        else:
          self.totals[k] = v * batch_size

  def on_epoch_end(self, epoch, logs=None):
    if logs is not None:
      for k in self.params['metrics']:
        if k in self.totals:
          # Make value available to next callbacks.
          if k in self.stateful_metrics:
            logs[k] = self.totals[k]
          else:
            logs[k] = self.totals[k] / self.seen




class CDSC(Callback):
  """Callback to stop the training of a model when the rolling Pearson Correlation coefficient decreases below a pre-defined threshold and stays below for a pre-defined number of epochs.

  `CDSC` callback is used in conjunction with training using
  `model.fit()` to save a model or weights (in a checkpoint file) that has the best validation loss, so the model or weights can be loaded later to continue the training
  from the state saved. If the training is stopped, the model with the best validation loss is saved at `filepath`.

  A few options this callback provides include:
  - Parametrizable window-size, patience and threshold
  - Notification if the parameters are not set right or the model is formulated wrong.
  - Whether to only keep the model that has achieved the "best performance" so
    far, or whether to save the model at the end of every epoch regardless of
    performance.
  - Definition of 'best'; which quantity to monitor and whether it should be
    maximized or minimized.
  - Whether only weights are saved, or the whole model is saved.

  Args:
      filepath: string or `PathLike`, path to save the model file. e.g.
        filepath = os.path.join(working_dir, 'ckpt', file_name). `filepath`
        can contain named formatting options, which will be filled the value of
        `epoch` and keys in `logs` (passed in `on_epoch_end`). For example: if
        `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`, then the model
        checkpoints will be saved with the epoch number and the validation loss
        in the filename. The directory of the filepath should not be reused by
        any other callbacks to avoid conflicts.
      window_size: The size of the window to compute the rolling Pearson Correlation coefficient.
      threshold: The rolling Prearson correlation coefficient below which the patience counter is decreased
      patience: The number of epochs where the rolling Pearson correlation coefficient is allowed to be below the threshold before training is stopped.
      monitor: The metric name to monitor. Typically the metrics are set by the
        `Model.compile` method. Note:

        * Prefix the name with `"val_`" to monitor validation metrics.
        * Use `"loss"` or "`val_loss`" to monitor the model's total loss.
        * If you specify metrics as strings, like `"accuracy"`, pass the same
          string (with or without the `"val_"` prefix).
        * If you pass `metrics.Metric` objects, `monitor` should be set to
          `metric.name`
        * If you're not sure about the metric names you can check the contents
          of the `history.history` dictionary returned by
          `history = model.fit()`
        * Multi-output models set additional prefixes on the metric names.

      verbose: verbosity mode, 0 or 1.
      save_best_only: if `save_best_only=True`, it only saves when the model
        is considered the "best" and the latest best model according to the
        quantity monitored will not be overwritten. If `filepath` doesn't
        contain formatting options like `{epoch}` then `filepath` will be
        overwritten by each new better model.
      mode: one of {'auto', 'min', 'max'}. If `save_best_only=True`, the
        decision to overwrite the current save file is made based on either
        the maximization or the minimization of the monitored quantity.
        For `val_acc`, this should be `max`, for `val_loss` this should be
        `min`, etc. In `auto` mode, the mode is set to `max` if the quantities
        monitored are 'acc' or start with 'fmeasure' and are set to `min` for
        the rest of the quantities.
      save_weights_only: if True, then only the model's weights will be saved
        (`model.save_weights(filepath)`), else the full model is saved
        (`model.save(filepath)`).
      options: Optional `tf.train.CheckpointOptions` object if
        `save_weights_only` is true or optional `tf.saved_model.SaveOptions`
        object if `save_weights_only` is false.
      **kwargs: Additional arguments for backwards compatibility. Possible key
        is `period`.
  """

  def __init__(self,
               filepath,
               window_size = 5,
               threshold = 0.5,
               patience = 5,
               monitor='val_loss',
               verbose=1,
               save_best_only=False,
               save_weights_only=False,
               mode='auto',
               options=None,
               **kwargs):
    super(CDSC, self).__init__()
    self._supports_tf_logs = True
    self._current_epoch = 0
    self.window_size = window_size
    self.threshold = threshold
    self.patience = patience
    self.monitor = monitor
    self.verbose = verbose
    self.filepath = path_to_string(filepath)
    self.save_best_only = save_best_only
    self.save_weights_only = save_weights_only
    self.epochs_since_last_save = 0
    self._batches_seen_since_last_saving = 0
    self._last_batch_seen = 0

    if save_weights_only:
      if options is None or isinstance(
          options, checkpoint_options_lib.CheckpointOptions):
        self._options = options or checkpoint_options_lib.CheckpointOptions()
      else:
        raise TypeError('If save_weights_only is True, then `options` must be '
                        'either None or a tf.train.CheckpointOptions')
    else:
      if options is None or isinstance(options, save_options_lib.SaveOptions):
        self._options = options or save_options_lib.SaveOptions()
      else:
        raise TypeError('If save_weights_only is False, then `options` must be'
                        'either None or a tf.saved_model.SaveOptions')

    if mode not in ['auto', 'min', 'max']:
      logging.warning('ModelCheckpoint mode %s is unknown, '
                      'fallback to auto mode.', mode)
      mode = 'auto'

    if mode == 'min':
      self.monitor_op = np.less
      self.best = np.Inf
    elif mode == 'max':
      self.monitor_op = np.greater
      self.best = -np.Inf
    else:
      if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
        self.monitor_op = np.greater
        self.best = -np.Inf
      else:
        self.monitor_op = np.less
        self.best = np.Inf

    # Only the chief worker writes model checkpoints, but all workers
    # restore checkpoint at on_train_begin().
    self._chief_worker_only = False

  def on_train_begin(self, logs=None):
    if False:
      filepath_to_load = (
          self._get_most_recently_modified_file_matching_pattern(self.filepath))
      if (filepath_to_load is not None and
          self._checkpoint_exists(filepath_to_load)):
        try:
          # `filepath` may contain placeholders such as `{epoch:02d}`, and
          # thus it attempts to load the most recently modified file with file
          # name matching the pattern.
          self.model.load_weights(filepath_to_load)
        except (IOError, ValueError) as e:
          raise ValueError('Error loading file from {}. Reason: {}'.format(
              filepath_to_load, e))

  def _implements_train_batch_hooks(self):
    # Only call batch hooks when saving on batch
    return False

  def on_train_batch_end(self, batch, logs=None):
    if self._should_save_on_batch(batch):
      self._save_model(epoch=self._current_epoch, logs=logs)

  def on_epoch_begin(self, epoch, logs=None):
    self._current_epoch += 1

  def on_epoch_end(self, epoch, logs=None):
      self._save_model(epoch=epoch, logs=logs)
      self._check_if_training_should_be_stopped(epoch, logs)

  def _should_save_on_batch(self, batch):
      return False
  
  def _check_if_training_should_be_stopped(self, logs, epoch):
        if self._current_epoch >= self.window_size:
            rolling_correlation = np.corrcoef(self.model.history.history[self.monitor][-self.window_size:],self.model.history.history["loss"][-self.window_size:])[0][1]
            if self._current_epoch == self.window_size and rolling_correlation < 0:
              print("The strating rolling pearson rolling correlation is negative. Please check your model and/or data.")
            if rolling_correlation < self.threshold:
                self.patience -= 1
                if self.patience == 0:
                    self.model.stop_training = True
                    print("Training stopped due to CDSC callback")
            else:
                self.patience = self.patience
            if self.verbose > 0:
                print("Training loss: ", self.model.history.history["loss"][-1], " Validation loss: ", self.model.history.history["val_loss"][-1], " Rolling correlation: ", rolling_correlation, " Patience: ", self.patience)
        return True

  def _save_model(self, epoch, logs):
    """Saves the model.

    Args:
        epoch: the epoch this iteration is in.
        logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
    """
    logs = logs or {}
    # Block only when saving interval is reached.
    logs = tf_utils.sync_to_numpy_or_python_type(logs)
    self.epochs_since_last_save = 0
    filepath = self._get_file_path(epoch, logs)

    try:
        if self.save_best_only:
          current = logs.get(self.monitor)
          if current is None:
            logging.warning('Can save best model only with %s available, '
                            'skipping.', self.monitor)
          else:
            if self.monitor_op(current, self.best):
              if self.verbose > 0:
                print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                      ' saving model to %s' % (epoch + 1, self.monitor,
                                               self.best, current, filepath))
              self.best = current
              if self.save_weights_only:
                self.model.save_weights(
                    filepath, overwrite=True, options=self._options)
              else:
                self.model.save(filepath, overwrite=True, options=self._options)
            else:
              if self.verbose > 0:
                print('\nEpoch %05d: %s did not improve from %0.5f' %
                      (epoch + 1, self.monitor, self.best))
        else:
          if self.verbose > 0:
            print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
          if self.save_weights_only:
            self.model.save_weights(
                filepath, overwrite=True, options=self._options)
          else:
            self.model.save(filepath, overwrite=True, options=self._options)

        self._maybe_remove_file()
    except IsADirectoryError as e:  # h5py 3.x
        raise IOError('Please specify a non-directory filepath for '
                      'ModelCheckpoint. Filepath used is an existing '
                      'directory: {}'.format(filepath))
    except IOError as e:  # h5py 2.x
        # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
        if 'is a directory' in str(e.args[0]).lower():
          raise IOError('Please specify a non-directory filepath for '
                        'ModelCheckpoint. Filepath used is an existing '
                        'directory: {}'.format(filepath))
        # Re-throw the error for any other causes.
        raise e

  def _get_file_path(self, epoch, logs):
    """Returns the file path for checkpoint."""
    # pylint: disable=protected-access
    try:
      # `filepath` may contain placeholders such as `{epoch:02d}` and
      # `{mape:.2f}`. A mismatch between logged metrics and the path's
      # placeholders can cause formatting to fail.
      file_path = self.filepath.format(epoch=epoch + 1, **logs)
    except KeyError as e:
      raise KeyError('Failed to format this callback filepath: "{}". '
                     'Reason: {}'.format(self.filepath, e))
    self._write_filepath = distributed_file_utils.write_filepath(
        file_path, self.model.distribute_strategy)
    return self._write_filepath

  def _maybe_remove_file(self):
    # Remove the checkpoint directory in multi-worker training where this worker
    # should not checkpoint. It is a dummy directory previously saved for sync
    # distributed training.
    distributed_file_utils.remove_temp_dir_with_filepath(
        self._write_filepath, self.model.distribute_strategy)

  def _checkpoint_exists(self, filepath):
    """Returns whether the checkpoint `filepath` refers to exists."""
    if filepath.endswith('.h5'):
      return file_io.file_exists_v2(filepath)
    tf_saved_model_exists = file_io.file_exists_v2(filepath)
    tf_weights_only_checkpoint_exists = file_io.file_exists_v2(
        filepath + '.index')
    return tf_saved_model_exists or tf_weights_only_checkpoint_exists

  def _get_most_recently_modified_file_matching_pattern(self, pattern):
    """Returns the most recently modified filepath matching pattern.

    Pattern may contain python formatting placeholder. If
    `tf.train.latest_checkpoint()` does not return None, use that; otherwise,
    check for most recently modified one that matches the pattern.

    In the rare case where there are more than one pattern-matching file having
    the same modified time that is most recent among all, return the filepath
    that is largest (by `>` operator, lexicographically using the numeric
    equivalents). This provides a tie-breaker when multiple files are most
    recent. Note that a larger `filepath` can sometimes indicate a later time of
    modification (for instance, when epoch/batch is used as formatting option),
    but not necessarily (when accuracy or loss is used). The tie-breaker is
    put in the logic as best effort to return the most recent, and to avoid
    undeterministic result.

    Modified time of a file is obtained with `os.path.getmtime()`.

    This utility function is best demonstrated via an example:

    ```python
    file_pattern = 'f.batch{batch:02d}epoch{epoch:02d}.h5'
    test_dir = self.get_temp_dir()
    path_pattern = os.path.join(test_dir, file_pattern)
    file_paths = [
        os.path.join(test_dir, file_name) for file_name in
        ['f.batch03epoch02.h5', 'f.batch02epoch02.h5', 'f.batch01epoch01.h5']
    ]
    for file_path in file_paths:
      # Write something to each of the files
    self.assertEqual(
        _get_most_recently_modified_file_matching_pattern(path_pattern),
        file_paths[-1])
    ```

    Args:
        pattern: The file pattern that may optionally contain python placeholder
            such as `{epoch:02d}`.

    Returns:
        The most recently modified file's full filepath matching `pattern`. If
        `pattern` does not contain any placeholder, this returns the filepath
        that
        exactly matches `pattern`. Returns `None` if no match is found.
    """
    dir_name = os.path.dirname(pattern)
    base_name = os.path.basename(pattern)
    base_name_regex = '^' + re.sub(r'{.*}', r'.*', base_name) + '$'

    # If tf.train.latest_checkpoint tells us there exists a latest checkpoint,
    # use that as it is more robust than `os.path.getmtime()`.
    latest_tf_checkpoint = checkpoint_management.latest_checkpoint(dir_name)
    if latest_tf_checkpoint is not None and re.match(
        base_name_regex, os.path.basename(latest_tf_checkpoint)):
      return latest_tf_checkpoint

    latest_mod_time = 0
    file_path_with_latest_mod_time = None
    n_file_with_latest_mod_time = 0
    file_path_with_largest_file_name = None

    if file_io.file_exists_v2(dir_name):
      for file_name in os.listdir(dir_name):
        # Only consider if `file_name` matches the pattern.
        if re.match(base_name_regex, file_name):
          file_path = os.path.join(dir_name, file_name)
          mod_time = os.path.getmtime(file_path)
          if (file_path_with_largest_file_name is None or
              file_path > file_path_with_largest_file_name):
            file_path_with_largest_file_name = file_path
          if mod_time > latest_mod_time:
            latest_mod_time = mod_time
            file_path_with_latest_mod_time = file_path
            # In the case a file with later modified time is found, reset
            # the counter for the number of files with latest modified time.
            n_file_with_latest_mod_time = 1
          elif mod_time == latest_mod_time:
            # In the case a file has modified time tied with the most recent,
            # increment the counter for the number of files with latest modified
            # time by 1.
            n_file_with_latest_mod_time += 1

    if n_file_with_latest_mod_time == 1:
      # Return the sole file that has most recent modified time.
      return file_path_with_latest_mod_time
    else:
      # If there are more than one file having latest modified time, return
      # the file path with the largest file name.
      return file_path_with_largest_file_name