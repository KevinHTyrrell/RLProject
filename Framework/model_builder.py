import numpy as np
import pandas as pd
from warnings import warn as WARN
from tensorflow import keras
from Misc.dict_fns import clean_dictionary


class ModelBuilder:
    """
    Builds keras functional model from dictionary 
    """
    def __init__(self):
        self._input_layer_flag      = 'input'
        self._activation_flag       = 'activation'
        self._dropout_flag          = 'dropout'
        self._flatten_flag          = 'flatten'
        self._initializer_flag      = 'initializer'
        self._layer_flag            = 'layer'
        self._regularizer_flag      = 'regularizer'
        self._type_flag             = 'type'
        self._value_flag            = 'value'
        self._kwargs                = dict()

        self._populate_keras_dict('activation')
        self._populate_keras_dict('layer')

    def _populate_keras_dict(self, dict_name):
        keras_dict = dict()
        keras_module = getattr(keras, f'{dict_name}s')
        for act_name, act_fn in keras_module.__dict__.items():
            keras_dict[act_name.lower()] = act_fn
        self._kwargs[dict_name] = keras_dict

    def layer_builder(self, layer_type: str, layer_args: dict, previous_layer=None):
        tensor_type = self._get_layer(layer_type)
        custom_tensor = self._handle_layer(tensor_type, layer_args, previous_layer)
        return custom_tensor

    def _get_layer(self, layer_type: str):
        layer_fn = self._kwargs[self._layer_flag].get(layer_type.lower())
        return layer_fn

    def _handle_layer_input(self, layer_to_config: keras.layers.Layer, layer_args):
        layer_defined = layer_to_config(**layer_args)
        return layer_defined

    def _handle_layer(self, layer_to_config: keras.layers.Layer, layer_args, previous_layer):
        if self._input_layer_flag in layer_to_config.__name__.lower():
            return self._handle_layer_input(layer_to_config, layer_args)

        activation_args = layer_args.get(self._activation_flag)
        dropout_args = layer_args.get(self._dropout_flag)
        flatten_args = layer_args.get(self._flatten_flag)
        layer_args[self._activation_flag] = None
        layer_args[self._dropout_flag] = None
        layer_args[self._flatten_flag] = None
        layer_args = clean_dictionary(layer_args)

        reg_list = [arg for arg in layer_args if self._regularizer_flag in arg]
        init_list = [arg for arg in layer_args if self._initializer_flag in arg]
        for reg in reg_list:
            regularizer = self._handle_special_args(self._regularizer_flag, layer_args[reg])
            layer_args[reg] = regularizer
        for intl in init_list:
            initializer = self._handle_special_args(self._initializer_flag, layer_args[intl])
            layer_args[intl] = initializer
        layer_built = layer_to_config(**layer_args)(previous_layer)
        if activation_args is not None:
            layer_built = self._handle_activations(activation_args, layer_built)
        if dropout_args is not None:
            layer_built = keras.layers.Dropout(**dropout_args)(layer_built)
        if flatten_args is not None and flatten_args.get(self._value_flag):
            layer_built = keras.layers.Flatten()(layer_built)
        return layer_built

    def _handle_special_args(self, arg_type: str, args):
        reg_type = args.get(self._type_flag)
        reg_val = args.get(self._value_flag)
        arg_type_module = getattr(keras, f'{arg_type}s')
        reg_fn = getattr(arg_type_module, reg_type)
        if type(reg_val) == list and len(reg_val) > 1:
            built_arg = reg_fn(*reg_val)
        else:
            built_arg = reg_fn(reg_val)
        return built_arg

    def _handle_activations(self, activation_args, previous_layer: keras.layers.Layer):
        if type(activation_args) == str:
            warning_msg = f'ACTIVATION {activation_args} IN {previous_layer.name} NOT PROPERLY PROVIDED, ' + \
                          'MAY NOT CONFIGURE CORRECTLY'
            WARN(warning_msg)
            custom_tensor = self._kwargs[self._activation_flag].get(activation_args)
            return custom_tensor(previous_layer)
        activation_type = activation_args[self._type_flag]
        activation_args[self._type_flag] = None
        activation_args = clean_dictionary(activation_args)
        if activation_type in self._kwargs[self._activation_flag]:
            activation_args['x'] = previous_layer
            tensor_fn = self._kwargs[self._activation_flag].get(activation_type)
            custom_tensor = tensor_fn(**activation_args)
            return custom_tensor
        elif activation_type in self._kwargs[self._layer_flag]:
            tensor_fn = self._kwargs[self._layer_flag].get(activation_type)
            custom_tensor = tensor_fn(**activation_args)
            return custom_tensor(previous_layer)
        else:
            TypeError(f'ARG {activation_type} NOT HANDLED, PLEASE CHECK INPUT')

    def build_tensor_model(self, input_dims, model_config):
        """
        :param input_dims: dimensions of train data
        :param model_config: yaml containing model configuration
        :return: input_layer, last tensor, and list of linked tensors
        """
        input_layer = keras.layers.Input(shape=input_dims)
        built_layer_list = [input_layer]
        current_layer = input_layer
        for layer_name, layer_config in model_config.items():
            layer_type = layer_config.get('type')
            del layer_config['type']
            current_layer = self.layer_builder(layer_type, layer_config, current_layer)
            built_layer_list.append(current_layer)
        return input_layer, current_layer, built_layer_list
