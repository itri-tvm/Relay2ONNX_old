# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, import-self, len-as-condition, unused-argument, too-many-lines
# pylint: disable=import-outside-toplevel
import logging
import numpy as np
import tvm
from tvm.relay.ty import TensorType, TupleType
from tvm.relay.expr import Function, Var, Call, Constant, TupleGetItem, Tuple
from tvm.relay.op import Op
from tvm.ir import IRModule

from ... import nd as _nd
from .. import analysis
from .. import expr as _expr
from .. import op as _op
from ..expr_functor import ExprVisitor
from onnx import numpy_helper, helper, NodeProto
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
import onnx
from tvm.relay.transform import PartitionGraphInOrder, PartitionGraphInUnorder, PartitionGraphByExpr
from .common import infer_type
from torch._jit_internal import ignore
__all__ = ['to_onnx']

class AttrCvt(object):
    """Common attribute converter. An AttrConverter instance is a callable:
    ```
    attr_converter = AttrConverter(op_name, transforms={'a':'b', 'c':('d', 1)})
    new_op_name, new_attr = attr_converter(attrs)
    ```

    Parameters
    ----------
    op_name : str or callable
        If set as str, returned operator name is the str.
        If set as callable, returned operator is the str returned by calling:
        `op_name = func(attr)`

    transforms : dict of `new_name, or (new_name, default_value, transform function)`
        If only a new_name is provided, it's like renaming the attribute name.
        If default_value if provided, then the attribute is considered as optional.
        If transform function is provided, the original attribute value is handled
        by transform function.

    excludes : list
        A list of excluded attributes that should `NOT` appear.
        Raise NotImplementedError if occurred.

    disables : list
        A list of attributes that is disabled in relay. Log warnings.

    ignores : list
        A list of attributes that is ignoresd in relay. Debug level logging.

    extras : dict
        A series of additional attributes should be added anyway to the returned
        attribute dict.

    custom_check : callable
        A custom function takes attribute, and return True/False.
        Raise RuntimeError if not bool(True) returned.
    """
    def __init__(self, op_name, transforms=None,
                 excludes=None, disables=None, ignores=None,
                 extras=None, custom_check=None):      
        self._op_name = op_name
        self._transforms = transforms if transforms else {}
        self._excludes = excludes if excludes else []
        self._disables = disables if disables else []
        self._ignores = ignores if ignores else []
        self._extras = extras if extras else {}
        self._custom_check = custom_check

    def __call__(self, inputs, outputs, **attrs):
        # self._ignores.append('_output_shapes')
        # self._ignores.append('_input_shapes')
        # self._ignores.append('T')
        # self._ignores.append('use_cudnn_on_gpu')
        # self._ignores.append('_node_name')
        # self._ignores.append('is_training')
        # self._ignores.append('_target_layout')
        # apply custom check
        if self._custom_check:
            func, msg = self._custom_check
            if not func(attrs):
                raise RuntimeError("Check failed: {}".format(msg))
        # get new op_name
        if isinstance(self._op_name, str):
            op_name = self._op_name
        else:
            assert callable(self._op_name), "op_name can either be string or callable"
            op_name = self._op_name(attrs)
        
        # ignores 'tvm_custom' always
        # self._ignores.append('tvm_custom')
        # convert attributes
        new_attrs = {}
        for k in attrs.keys():
            if k in self._excludes:
                raise NotImplementedError('Attribute %s in operator %s is not' +
                                          ' supported.', k, op_name)
            if k in self._disables:
                logging.warning("Attribute %s is disabled in operator %s", k, op_name)
            elif k in self._ignores:
                if k != 'tvm_custom':
                    logging.warning("Attribute %s is ignoresd in operator %s", k, op_name)
            elif k in self._transforms:
                new_name, defaults, transform = self._parse_default(self._transforms[k])
                if defaults is None:
                    new_attr = self._required_attr(attrs, k)
                else:
                    new_attr = attrs.get(k, None)
                if new_attr is None:
                    new_attrs[new_name] = defaults
                else:
                    new_attrs[new_name] = transform(new_attr)
            else:
                # copy
                new_attrs[k] = attrs[k]
        # add extras
        new_attrs.update(self._extras)
        return helper.make_node(op_name, inputs, outputs, **new_attrs)
    def _parse_default(self, target):
        """Helper function to parse default values."""
        if not isinstance(target, (list, tuple)):
            k, v, t = target, None, lambda x: x
        elif len(target) == 1:
            k, v, t = target[0], None, lambda x: x
        elif len(target) == 2:
            k, v, t = target[0], target[1], lambda x: x
        elif len(target) > 2:
            k, v, t = target[0], target[1], target[2]
        else:
            k = None  # should raise
        if not isinstance(k, str):
            msg = "{} is not a valid target, (name, default) expected.".format(target)
            raise ValueError(msg)
        return k, v, t

    def _parse_bool(self, value):
        """Helper function to parse default boolean values."""
        if isinstance(value, str):
            return value.strip().lower() in ['true', '1', 't', 'y', 'yes']
        return bool(value)

    def _required_attr(self, attr, key):
        """Wrapper for getting required attributes."""
        assert isinstance(attr, dict)
        if key not in attr:
            raise AttributeError("Required attribute {} not found.".format(key))
        return attr[key]

   
class ToOnnxOpConverter(object):
    """ A helper class for holding onnx op converters.
    """
    opset = 0
    @classmethod
    def get_converter(cls, opset):
        """ Get converter matches given opset.

        Parameters
        ----------
        opset: int
            opset from model.

        Returns
        -------
        converter, which should be `_impl_vx`. Number x is the biggest
            number smaller than or equal to opset belongs to all support versions.
        """
        cls.opset=opset
        versions = [
            int(d.replace('_impl_v', '')) for d in dir(cls) if '_impl_v' in d
        ]
        versions = sorted(versions + [opset])
        version = versions[
            max([i for i, v in enumerate(versions) if v == opset]) - 1]
        if hasattr(cls, '_impl_v{}'.format(version)):
            return getattr(cls, '_impl_v{}'.format(version))
        raise NotImplementedError(
            'opset version {} of {} not implemented'.format(
                version, cls.__name__))
def convert_tvm_object_for_py(object):
    """ Convert tvm object for Python."""
    if isinstance(object, tvm.ir.attrs.Attrs):
        new_map={}
        for key in object.keys():  
            new_value = tvm_array_to_list(object[key])
            new_map[key] = new_value
        return new_map
    elif isinstance(object, tvm.ir.container.Array):
        return tvm_array_to_list(object)
    else:
        return object
def tvm_array_to_list(object):
    """ Convert tvm.ir.container.Array to List."""
    if isinstance(object, tvm.ir.container.Array) or isinstance(object, list):
        new_list=[]
        for value in object:
            new_list.append(tvm_array_to_list(value))
        return new_list
    elif isinstance(object, tvm.ir.PrimExpr):
        return object.value
    else:
        return object
class Constant(ToOnnxOpConverter):
    @classmethod
    def _impl_v1(cls, name, data):
        if type(data) in [list,tuple]:
            dtype = type(data[0]).__name__
            array = np.array(data, dtype=dtype)
        elif type(data) is np.ndarray:
            dtype = data.dtype
            array = data
        else:
            dtype = type(data).__name__
            array = np.array(data, dtype=dtype)
        assert dtype not in (np.float16, np.float, np.double), 'Constant with dtype \"{}\" is not supported in opset=1.'.format(dtype)
        const_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=[name],
            value = onnx.helper.make_tensor(
                name=name,
                data_type=NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)],
                dims=array.shape,
                vals=array.flatten().astype(dtype)
            )
        )
        return const_node
    @classmethod
    def _impl_v9(cls, name, data):
        if type(data) in [list,tuple]:
            dtype = type(data[0]).__name__
            array = np.array(data, dtype=dtype)
        elif type(data) is np.ndarray:
            dtype = data.dtype
            array = data
        else:
            dtype = type(data).__name__
            array = np.array(data, dtype=dtype)
        const_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=[name],
            value = onnx.helper.make_tensor(
                name=name,
                data_type=NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)],
                dims=array.shape,
                vals=array.flatten().astype(dtype)
            )
        )
        return const_node
    @classmethod
    def _impl_v11(cls, name, data):
        if type(data) in [list,tuple]:
            dtype = type(data[0]).__name__
            array = np.array(data, dtype=dtype)
        elif type(data) is np.ndarray:
            dtype = data.dtype
            array = data
        else:
            dtype = type(data).__name__
            array = np.array(data, dtype=dtype)
        shape = [1] if array.size==1 else array.shape
        const_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=[name],
            value = onnx.helper.make_tensor(
                name=name,
                data_type=NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)],
                dims=array.shape,
                vals=array.flatten().astype(dtype)
            )
        )
        return const_node
    @classmethod
    def _impl_v12(cls, name, data):
        if type(data) in [list,tuple] and len(data)>0:
            dtype = type(data[0]).__name__
            value = {'value_{}s'.format(dtype): data}
        else:
            dtype = type(data).__name__
            value = {'value_{}'.format(dtype): data}
        const_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=[name],
            **value
        )
        return const_node
class NNBatchNorm(ToOnnxOpConverter):
    """ Operator converter for BatchNorm.
    """
    @classmethod
    def _impl_v12(cls, inputs, outputs, **attrs):
        op_type='BatchNormalization'
        #extras={'momentum':1}
        transforms={
            'epsilon':'epsilon'
        }
        ignores=['axis', 'center', 'scale']
        return AttrCvt(op_name = op_type, transforms=transforms, ignores=ignores)(inputs, outputs, **attrs)


class NNBatchFlatten(ToOnnxOpConverter):
    """ Operator converter for Flatten.
    """
    @classmethod
    def _impl_v11(cls, inputs, outputs, **attrs):
        op_type= 'Flatten'        
        extras={'axis': 1}
        return AttrCvt(op_name = op_type, extras=extras)(inputs, outputs, **attrs)
    
class NNMaxPool(ToOnnxOpConverter):
    """ Operator converter for MaxPool
    """
    @classmethod
    def _impl_v12(cls, inputs, outputs, **attrs):
        op_type = 'MaxPool'
        transforms ={
            'dilation':'dilations',
            'pool_size': 'kernel_shape',
            'padding':'pads'
        }
        ignores=['layout']
        return AttrCvt(op_name = op_type, transforms = transforms ,ignores = ignores)(inputs, outputs, **attrs)

class NNConviDTranspose(ToOnnxOpConverter):
    """ Operator converter for nn.convid_transpose.
    """
    @classmethod
    def _impl_v11(cls, inputs, outputs, **attrs):
        op_type = 'ConvTranspose'
        transforms ={
            'dilation':'dilations',
            'groups':'group',
            'kernel_size': 'kernel_shape',
            'padding':'pads'
        }
        ignores=['channels', 'data_layout','kernel_layout','out_layout','out_dtype']
        extras={'output_shape': c.checked_type.concrete_shape()}
        return AttrCvt(op_name = op_type, transforms = transforms ,ignores = ignores, extras=extras)(inputs, outputs, **attrs)
class NNConviD(ToOnnxOpConverter):
    """ Operator converter for nn.convid.
    """
    @classmethod
    def _impl_v11(cls, inputs, outputs, **attrs):
        op_type = 'Conv'
        transforms ={
            'dilation':'dilations',
            'groups':'group',
            'kernel_size': 'kernel_shape',
            'padding':'pads'
        }
        ignores=['channels','data_layout','kernel_layout','out_layout','out_dtype']
        return AttrCvt(op_name = op_type, transforms = transforms ,ignores = ignores)(inputs, outputs, **attrs)
class NNDropout(ToOnnxOpConverter):
    """ Operator converter for nn.dropout.
    """
    @classmethod
    def _impl_v10(cls, inputs, outputs, **attrs):
        transforms ={
            'rate':'ratio'
        }
        return AttrCvt('Dropout',transforms = transforms)(inputs, outputs, **attrs)
    @classmethod
    def _impl_v12(cls, inputs, outputs, **attrs):
        rate_name = '{}_rate'.format(outputs[0])
        rate_data = attrs['rate']
        inputs.append(const_name)
        ignores=['rate']
        make_const_node = Constant.get_converter(cls.opset)
        return [make_const_node(rate_name, rate_data),
                    AttrCvt('Dropout',ignores = ignores)(inputs, outputs, **attrs)]
class Reshape(ToOnnxOpConverter):
    """ Operator converter for reshape.
    """
    @classmethod
    def _impl_v1(cls, inputs, outputs, **attrs):
        op_type = 'Reshape'
        transforms ={
            'newshape':'shape'
        }
        ignores=['reverse']
        return AttrCvt(op_name = op_type, ignores=ignores, transforms=transforms)(inputs, outputs, **attrs)
    @classmethod
    def _impl_v5(cls, inputs, outputs, **attrs): 
        shape_name = '{}_shape'.format(outputs[0])
        shape_data = attrs['newshape']
        inputs.append(shape_name)
        make_const_node = Constant.get_converter(cls.opset)
        return [make_const_node(shape_name, shape_data),
                AttrCvt('Reshape', ignores=['newshape', 'reverse'])(inputs, outputs, **attrs)]
class NNAvgPooliD(ToOnnxOpConverter):
    """ Operator converter for nn.avg_poolid.
    """
    @classmethod
    def _impl_v11(cls, inputs, outputs, **attrs):
        op_type = 'AveragePool'
        transforms ={
            'pool_size': 'kernel_shape',
            'padding':'pads'
        }
        ignores=['layout']
        return AttrCvt(op_name = op_type, transforms = transforms ,ignores = ignores)(inputs, outputs, **attrs)

class NNPad(ToOnnxOpConverter):
    """ Operator converter for nn.pad.
    """
    @classmethod
    def _convert_pads(cls, pads):
        l = len(pads)
        new_pads = [None]*(2*l)
        for i, pad in enumerate(pads):
            new_pads[i], new_pads[i+l] = pad[0], pad[1]
        return new_pads
    @classmethod
    def _impl_v1(cls, inputs, outputs, **attrs):
        op_type = 'Pad'
        transforms ={
            'pad_value': 'value',
            'pad_mode': 'mode',
            'pad_width': ('paddings', None, cls._convert_pads)
        }
        return AttrCvt(op_name = op_type, transforms = transforms)(inputs, outputs, **attrs)

    @classmethod
    def _impl_v2(cls, inputs, outputs, **attrs):
        op_type = 'Pad'
        transforms ={
            'pad_value': 'value',
            'pad_mode': 'mode',
            'pad_width': ('pads', None, cls._convert_pads)
        }
        return AttrCvt(op_name = op_type, transforms = transforms)(inputs, outputs, **attrs)
    @classmethod
    def _impl_v11(cls, inputs, outputs, **attrs):
        assert attrs['pad_mode']=='constant', 'For opset=11, \"pad_mode\" must be \"constant\".'
        pads_name = '{}_pads'.format(outputs[0])
        pads_data = cls._convert_pads(attrs['pad_width'])
        constant_value_name = '{}_constant_value'.format(outputs[0])
        constant_value_data = attrs['pad_value']
        inputs.append(pads_name)
        inputs.append(constant_value_name)     
        ignores = ['pad_mode', 'pad_width', 'pad_value']
        make_const_node = Constant.get_converter(cls.opset)
        return [make_const_node(pads_name, pads_data),
                make_const_node(constant_value_name, constant_value_data),
                AttrCvt('Pad', ignores=ignores)(inputs, outputs, **attrs)]

class BroadcastTo(ToOnnxOpConverter):
    """ Operator converter for broadcast_to.
    """
    @classmethod
    def _impl_v8(cls, inputs, outputs, **attrs):
        shape_name = '{}_shape'.format(outputs[0])
        shape_data = attrs['shape']
        inputs.append(shape_name)
        ignores = ['shape', 'dtype']
        make_const_node = Constant.get_converter(cls.opset)
        return [make_const_node(shape_name, shape_data),
                AttrCvt('Expand', ignores=ignores)(inputs, outputs, **attrs)]
class Split(ToOnnxOpConverter):
    """ Operator converter for split.
    """
    @classmethod
    def _impl_v11(cls, inputs, outputs, **attrs):
        assert 'size_axis' in attrs, 'Additional parameter \"size_axis\" is not defined.'
        def convert_split(indices, size_axis):
            split = [None]*(len(indices)+1)
            split[0] = indices[0] - 0
            for i in range(len(indices)-1):
                split[i+1] = indices[i+1] - indices[i]
            split[-1] = size_axis - indices[-1]
            return split
        extras={'split': convert_split(attrs['indices_or_sections'], attrs['size_axis'])}
        return AttrCvt('Split', extras=extras, ignores=['size_axis', 'indices_or_sections'])(inputs, outputs, **attrs)
class Tile(ToOnnxOpConverter):
    """ Operator converter for tile.
    """
    @classmethod
    def _impl_v1(cls, inputs, outputs, **attrs):
        return AttrCvt('Tile', {'reps':'repeats'})(inputs, outputs, **attrs)
    @classmethod
    def _impl_v6(cls, inputs, outputs, **attrs):
        repeats_name = '{}_repeats'.format(outputs[0])
        repeats_data = attrs['reps']
        inputs.append(repeats_name)
        make_const_node = Constant.get_converter(cls.opset)
        return [make_const_node(repeats_name, repeats_data),
                AttrCvt('Tile', ignores = ['reps'])(inputs, outputs, **attrs)]
class Resize(ToOnnxOpConverter):
    """Operator converter for image.resize
    """
    @classmethod
    def _convert_mode(cls, method):
        if method== 'nearest_neighbor':
            return 'nearest'
        elif method == 'bilinear':
            return 'linear'
        else:
            raise tvm.error.OpAttributeInvalid(
                'Value {} in attribute "method" of operator image.resize is not valid.'.format(method))
    @classmethod
    def _impl_v11(cls, inputs, outputs, **attrs):
        roi_name = '{}_roi'.format(outputs[0])
        roi_data = np.array([],dtype='int8')
        inputs.append(roi_name)
        scales_name = '{}_scales'.format(outputs[0])
        scales_data = np.array([],dtype='float32')
        inputs.append(scales_name)
        sizes_name = '{}_sizes'.format(outputs[0])
        sizes_data = attrs['sizes']
        inputs.append(sizes_name)
        make_const_node = Constant.get_converter(cls.opset)
        return [make_const_node(roi_name, roi_data),
                make_const_node(scales_name, scales_data),
                make_const_node(sizes_name, sizes_data),
                AttrCvt('Resize', {'method':('mode', None, cls._convert_mode)},ignores = ['size', 'layout', 'out_dtype'])(inputs, outputs, **attrs)]        
class Reduce(ToOnnxOpConverter):
    """ Operator converter for reduce ops.
    """
    name = ''
    @classmethod
    def _impl_v1(cls, inputs, outputs, **attrs):
        assert exclude==False, 'Value \"exclude\" can not be True.'
        return AttrCvt(cls.name,transforms={'axis':'axes'}, ignores=[exclude])(inputs, outputs, **attrs)

class Max(Reduce):
    """ Operator converter for max.
    """
    name = 'ReduceMax'
class Min(Reduce):
    """ Operator converter for min.
    """
    name = 'ReduceMin'
class Sum(Reduce):
    """ Operator converter for sum.
    """
    name = 'ReduceSum'
class Prod(Reduce):
    """ Operator converter for prod.
    """
    name = 'ReduceProd'
class NNUpsampling(ToOnnxOpConverter):
    """ Operator converter for upsample (nearest mode).
    """
    @classmethod
    def _convert_mode(cls, method):
        if method == 'nearest_neighbor':
            return 'nearest'
        elif method == 'bilinear':
            return 'linear'
        else:
            raise tvm.error.OpAttributeInvalid(
                'Value {} in attribute "method" of operator Upsample is not valid.'.format(method))
    @classmethod
    def _impl_v9(cls, inputs, outputs, **attrs):
        assert attrs['align_corners']==True, 'Value \"align_corners\" need to be True.'
        scales_name = '{}_scales'.format(outputs[0])
        scales_data = [1.0, 1.0, attrs['scale_h'],attrs['scale_w']]
        inputs.append(scales_name)
        make_const_node = Constant.get_converter(cls.opset)
        return [make_const_node(scales_name, scales_data),
                AttrCvt('Upsample',
                        transforms={'method':('mode',None,cls._convert_mode)},
                        ignores=['scale_h', 'scale_w', 'layout', 'align_corners']
                        )(inputs, outputs, **attrs)]
class Mean(ToOnnxOpConverter):
    """ Operator converter for mean.
    """
    @classmethod
    def _impl_v6(cls,  inputs, outputs, **attrs):
        return AttrCvt('Mean',extras={'consumed_inputs':[]},ignores=['axis','keepdims'])(inputs, outputs, **attrs)
    @classmethod
    def _impl_v6(cls,  inputs, outputs, **attrs):
        return AttrCvt('Mean',ignores=['axis','keepdims'])(inputs, outputs, **attrs)
    @classmethod
    def _impl_v8(cls,  inputs, outputs, **attrs):
        return AttrCvt('Mean',ignores=['axis','keepdims'])(inputs, outputs, **attrs)
class ONNXRenamer(object):
    """A simply renamer for operators.

    Parameters
    ----------
    new_name : str
        The new name for the operator
    """
    def __init__(self, new_name):
        self._new_name = new_name

    def __call__(self, **params):
        return helper.make_node(self._new_name, **params)

# compatible operators that do NOT require any conversion.
_identity_list = []


# _convert_map defines maps of name to converter functor(callable)
# for 1 to 1 mapping, use Renamer if nothing but name is different
# use AttrCvt if attributes need to be converted
# for 1 to N mapping(composed), use custom callable functions
# for N to 1 mapping, currently not supported(?)
def _get_convert_map(opset):
    return {
        # defs/experimental
        'copy': ONNXRenamer('Identity'),
        #'Affine': has been simplified.
        #'ThresholdedRelu': ThresholdedRelu.get_converter(opset),
        #'ScaledTanh': has been simplified.
        #'ParametricSoftplus': has been simplified.
        #'ConstantOfShape': ConstantOfShape.get_converter(opset),
        #'GivenTensorFill'
        'nn.dense':AttrCvt(op_name = 'FC', ignores=['units', 'out_dtype']),
        #'Scale': has been simplified.
        # 'GRUUnit'
        # 'ATen'
        # 'ImageScaler'
        # 'MeanVarianceNormalization'
        # 'Crop'
        # 'Embedding'
        'nn.upsampling': NNUpsampling.get_converter(opset),
        #'SpatialBN': has been simplified.

        # defs/generator
        'Constant': Constant.get_converter(opset), # Implemented
        # 'RandomUniform'
        # 'RandomNormal'
        # 'RandomUniformLike'
        # 'RandomNormalLike'

        # defs/logical

        # defs/math
        'add': ONNXRenamer('Add'),
        'sub':ONNXRenamer('Sub'),
        'multiply': ONNXRenamer('Mul'),
        'divide': ONNXRenamer('Div'),
        'negative': ONNXRenamer('Neg'),
        'abs': ONNXRenamer('Abs'),
        #'Reciprocal': has been simplified.
        'floor': ONNXRenamer('Floor'),
        'ceil': ONNXRenamer('Ceil'),
        'sqrt': ONNXRenamer('Sqrt'),
        'nn.relu': ONNXRenamer('Relu'),
        'nn.leaky_relu': ONNXRenamer('LeakyRelu'),
        #'Selu': has been simplified.
        #'Elu': has been simplified.
        'exp': ONNXRenamer('Exp'),
        'greater': ONNXRenamer('Greater'),
        'less': ONNXRenamer('Less'),
        'log': ONNXRenamer('Log'),
        'tanh': ONNXRenamer('Tanh'),
        'power': ONNXRenamer('Pow'),
        'nn.prelu': ONNXRenamer('PRelu'),
        'sigmoid': ONNXRenamer('Sigmoid'),
        #'HardSigmoid': has been simplified.
        'maximum': ONNXRenamer('Max'),
        'minimum':  ONNXRenamer('Min'),
        #'Sum': has been simplified.
        'clip': AttrCvt('Clip', {'a_min': 'min', 'a_max': 'max'}),
        # softmax default axis is different in onnx
        'nn.softmax': ONNXRenamer('Softmax'),
        'nn.log_softmax': ONNXRenamer('LogSoftmax'),
        #'OneHot': OneHot.get_converter(opset),
        # 'Hardmax'
        #'Softsign': has been simplified.
        #'SoftPlus': has been simplified.
        'fused_gemm': ONNXRenamer('Gemm'),
        #'MatMul': MatMul.get_converter(opset),

        # defs/nn
        'nn.avg_pool1d':NNAvgPooliD.get_converter(opset),
        'nn.avg_pool2d':NNAvgPooliD.get_converter(opset),
        'nn.avg_pool3d':NNAvgPooliD.get_converter(opset),
        'nn.max_pool2d': NNMaxPool.get_converter(opset),
        'nn.conv1d': NNConviD.get_converter(opset),
        'nn.conv2d': NNConviD.get_converter(opset),
        'nn.conv3d': NNConviD.get_converter(opset),
        'nn.conv2d_transpose': NNConviDTranspose.get_converter(opset),
        'nn.conv1d_transpose': NNConviDTranspose.get_converter(opset),
        'nn.global_avg_pool2d': AttrCvt(op_name = 'GlobalAveragePool', ignores=['layout']),
        'nn.global_max_pool2d': ONNXRenamer('GlobalMaxPool'),
        'nn.batch_norm': NNBatchNorm.get_converter(opset),
        'nn.instance_norm': ONNXRenamer('InstanceNormalization'), # _impl_v1
        # 'LpNormalization'
        'nn.dropout': NNDropout.get_converter(opset),
        'nn.batch_flatten': NNBatchFlatten.get_converter(opset),
        'nn.lrn': AttrCvt(op_name = 'LRN', ignores=['axis']),
        # Recurrent Layers
        #'LSTM': LSTM.get_converter(opset),

        # defs/reduction
        'max': Max.get_converter(opset),
        'min': Min.get_converter(opset),
        'sum': Sum.get_converter(opset),
        'mean': Mean.get_converter(opset),
        'prod': Prod.get_converter(opset),
        # 'ReduceLogSumExp'
        'argmax':ONNXRenamer('ArgMax'),
        'argmin':ONNXRenamer('ArgMin'),

        # defs/tensor
        'cast': AttrCvt('Cast', {'dtype': 'to'}), # _impl_v1
        'reshape': Reshape.get_converter(opset),
        'broadcast_to': BroadcastTo.get_converter(opset),  # Expand
        'concatenate': ONNXRenamer('Concat'),
        'split': Split.get_converter(opset),
        #'Slice': Slice.get_converter(opset),
        'transpose': AttrCvt('Transpose', {'axes': 'perm'}),
        #'DepthToSpace': DepthToSpace.get_converter(opset),
        #'SpaceToDepth': SpaceToDepth.get_converter(opset),
        'take': ONNXRenamer('Gather'), # Gather
        'squeeze': AttrCvt('Squeeze', {'axis':'axes'}),
        'fused_unsqueeze': ONNXRenamer('Unsqueeze'),
        'nn.pad': NNPad.get_converter(opset),
        'shape_of': AttrCvt('Shape', ignores=['dtype']),
        'sign': ONNXRenamer('Sign'),
        'equal': ONNXRenamer('Equal'),
        'logical_not':ONNXRenamer('Not'),
        'logical_and':ONNXRenamer('And'),
        'tile': Tile.get_converter(opset),
        'erf':ONNXRenamer('Erf'),
        'where': ONNXRenamer('Where'),
        'logical_or':ONNXRenamer('Or'),
        'image.resize': Resize.get_converter(opset),
    }

def get_func_name_attrs(name, func_attrs):
    new_attrs = {}
    if name == 'fused_gemm':
        for call, attr in func_attrs.items():
            if call.op.name == 'nn.dense' and call.args[0].op.name == 'multiply':
                new_attrs['alpha'] = np.asscalar(call.args[0].args[0].data.asnumpy())
                if isinstance( call.args[0].args[1], Call) and  call.args[0].args[1].op.name is 'transpose':
                    new_attrs['transB'] = False
                else:
                    new_attrs['transB'] = True
            elif call.op.name == 'nn.batch_flatten':
                if isinstance(call.args[0], Call) and call.args[0].op.name == 'transpose':
                    new_attrs['transA'] = True
                else:
                    new_attrs['transA'] = False
            elif call.op.name == 'nn.bias_add' and call.args[1].op.name == 'multiply':
                new_attrs['beta'] = np.asscalar(call.args[1].args[0].data.asnumpy())
    elif name == 'fused_unsqueeze':
        new_attrs['axes'] = [] 
        for call, attrs in func_attrs.items():
            new_attrs['axes'].append(attrs['axis'])
        new_attrs['axes'].reverse()
    else:
        def get_func_major_op(func_name):
            return func_name.replace('_nn_', '_nn.').split('_')[1]
        name = get_func_major_op(name)
        if func_attrs is None:
            new_attrs = attr 
        else:    
            for call, attrs in func_attrs.items():
                if call.op.name == name:
                    new_attrs = attrs
                    break
    return name, new_attrs

class ExprCountor(ExprVisitor):
    def __init__(self):
        super(ExprVisitor, self).__init__()
        self.visit_count = {}
    def visit(self, expr):
        self.visit_count[expr] = self.visit_count[expr] + 1 if expr in self.visit_count else 1
        super(ExprCountor, self).visit(expr)
class GraphVisitor(ExprVisitor):
    """A visitor to traverse Relay Function and record the parameters for ONNX graph."""
    def __init__(self, param_names, visit_count):
        super(GraphVisitor, self).__init__()
        self.param_names=param_names
        self.node_map = {}
        self.name_map = {}
        self.input_map = {}
        self.counts_name = 0
        self.output_map = {}
        self.in_func = False
        self.func_attrs = {}
        self.constant_map = {}
        self.parent_map = {}
        self.visit_count = visit_count
    def visit_constant(self, const):
        if not self.in_func:
            data = const.data.asnumpy()
            name = '{}_const'.format(self.counts_name)
            self.counts_name += 1
            self.constant_map[name] = data
            self.name_map[const] = [name]
    def visit_var(self, v):
        if not self.in_func:
            if v.name_hint not in self.param_names:
                self.input_map[v] = {
                    'name':v.name_hint,
                    'elem_type':NP_TYPE_TO_TENSOR_TYPE[np.dtype(v.checked_type.dtype)],
                    'shape':tvm_array_to_list(v.checked_type.shape)}       
            self.name_map[v] = [v.name_hint]
    def visit_tuple(self, t):
        self.name_map[t] = []
        if not self.in_func:
            for x in t.fields:
                self.visit(x)
                self.name_map[t].extend(self.name_map[x])
        else:
            super(GraphVisitor, self).visit_tuple(t)
    def visit_tuple_getitem(self, op):
        if not self.in_func:
            self.visit(op.tuple_value)
            self.name_map[op] = [self.name_map[op.tuple_value][op.index]]
        else:
            super(GraphVisitor, self).visit_tuple_getitem(t)    
    def visit_call(self, c):
        if not self.in_func:
            self.node_map[c]={'inputs':[]}
            name = ''
            if isinstance(c.op, Op):
                name = c.op.name
                self.node_map[c]['op_name'] = name
                if c.attrs is not None:
                    self.node_map[c].update(convert_tvm_object_for_py(c.attrs))
            elif isinstance(c.op, Function):
                func_name = c.op.get_attribute('Name').value
                self.in_func = True
                self.visit(c.op)
                self.in_func = False
                name, attrs = get_func_name_attrs(func_name, self.func_attrs)
                self.node_map[c]['op_name'] = name
                if self.func_attrs:
                    self.node_map[c].update(attrs)
                self.func_attrs.clear()
                name = func_name
            for a in c.args:
                self.visit(a)
                self.node_map[c]['inputs'].extend(self.name_map[a])
            if isinstance(c.checked_type, TensorType) or self.visit_count[c]==1:
                if isinstance(c.checked_type, TensorType):
                    dtype = c.checked_type.dtype
                    shape = c.checked_type.concrete_shape
                else: 
                    dtype = c.checked_type.fields[0].dtype
                    shape = c.checked_type.fields[0].concrete_shape
                output_name = '{}_{}'.format(self.counts_name, name.replace('.', '_'))
                self.name_map[c] = [output_name]
                self.output_map[c]=[{
                    'name':output_name,
                    'elem_type':NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)],
                    'shape':shape
                }]
                self.node_map[c]['outputs']=[output_name]
            else:
                self.name_map[c] = [None]*len(c.checked_type.fields)
                self.output_map[c] = [None]*len(c.checked_type.fields)
                self.node_map[c]['outputs'] = [None]*len(c.checked_type.fields)
                for i, tensortype in enumerate(c.checked_type.fields):
                    output_name = '{}_{}_{}'.format(self.counts_name, i, name.replace('.', '_'))
                    self.name_map[c][i] = output_name
                    output = {
                        'name':output_name,
                        'elem_type':NP_TYPE_TO_TENSOR_TYPE[np.dtype(tensortype.dtype)],
                        'shape':tensortype.concrete_shape
                    }
                    self.output_map[c][i] = output
                    self.node_map[c]['outputs'][i] = output_name
            self.counts_name += 1
        else:
            self.func_attrs[c]=convert_tvm_object_for_py(c.attrs)
            for a in c.args:
                self.visit(a)

class GraphProto(object):
    """A helper class for handling Relay expression copying from pb2.GraphProto.
    Definition: https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

        Parameters
    ----------
    shape : dict of str to tuple, optional
        The input shape to the graph

    dtype : str or dict of str to str
        The input types to the graph
    """

    def __init__(self, opset):
        self._nodes = []
        self._inputs= []
        self._outputs = []
        self._initializer = []
        self._value_info = []
        self._opset = opset

    def _to_onnx(self, model, params, graph_name, doc_string):
        """Construct ONNX graph from Relay Function.
        Parameters
        ----------
        model : tvm.IRModule or tvm.relay.Function
            The relay module for compilation
        params : dict of str to tvm.nd.NDArray
            The parameter dict to be used by relay
        graph_name : str
            The name of the model's graph.
        doc_string :str
            The description of the model's graph
        Returns
        -------
        onnx_model : onnx protobuf object
            The returned onnx graph
        """
        # Fuse operator
        model = self._fuse_operator(model)
        func = model['main']
        # Traverse the Relay function and record the nodes.
        visitor = ExprCountor()
        visitor.visit(func)
        visitor = GraphVisitor(params.keys(),visitor.visit_count)
        visitor.visit(func)
        
        
        # Deal with constant (Must create constant earlierly than other nodes.)
        for constant_name, data  in visitor.constant_map.items():
            node = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=[constant_name],
                value=onnx.helper.make_tensor(
                    name=constant_name,
                    data_type=NP_TYPE_TO_TENSOR_TYPE[data.dtype],
                    dims=data.shape,
                    vals=data.flatten().astype(data.dtype),
                    ),
            )
            self._nodes.append(node)
        # Make nodes, ps. the order of making node must be sorted by the visiting order.
        for call, node_params in sorted(visitor.node_map.items(), key=lambda d: int(d[1]['outputs'][0].split('_')[0])):
            if isinstance(call.op, Op):
                if call.op.name=='split' and self._opset >= 10:
                    node_params['size_axis']=call.args[0].checked_type.concrete_shape[node_params['axis']]
                elif call.op.name=='image.resize' and self._opset >= 11:
                    node_params['sizes']=call.checked_type.concrete_shape
            node = self._convert_operator(**node_params)
            if isinstance(node, NodeProto):
                self._nodes.append(node)
            else:
                self._nodes.extend(node)
        # Make input info
        for input_params in visitor.input_map.values():
            input_info = helper.make_tensor_value_info(**input_params)
            self._inputs.append(input_info)
        # Make TensorProto
        for name, value in params.items():
            tensor = numpy_helper.from_array(value.asnumpy(), name)
            self._initializer.append(tensor)           
        # Make middel output info        
        for expr, outputs in visitor.output_map.items():
            for output_params in outputs:
                output_info = helper.make_tensor_value_info(**output_params)
                self._value_info.append(output_info)
        # Make output info
        output_name = self._nodes[-1].output[0]
        dtype = NP_TYPE_TO_TENSOR_TYPE[np.dtype(func.ret_type.dtype)]
        shape = tvm_array_to_list(func.ret_type.shape)
        self._outputs.append(helper.make_tensor_value_info(output_name, dtype, shape))
        graph = helper.make_graph(self._nodes, graph_name, self._inputs, self._outputs, self._initializer, doc_string, self._value_info)
        onnx_id = helper.make_opsetid("", self._opset)
        onnx_model = helper.make_model(graph, opset_imports=[onnx_id])
        return onnx_model
       
    def _convert_operator(self,
                          op_name,
                          **params):
        """Convert Relay operator into a ONNX operator.
        Parameters
        ----------
        op_name : str
            Operator name, such as nn.conv2d, nn.dense
        params : dict
            Dict of operator parameters
        Returns
        -------
        sym : onnx.NodeProto
            Converted ONNX node.
        """
        convert_map = _get_convert_map(self._opset)
        if op_name in convert_map:
            sym = convert_map[op_name](**params)
        else:
            raise NotImplementedError(
                "Operator {} not implemented.".format(op_name))
        return sym
    
    def _fused_xx_nn_bias(self, model):
        """Fuse the series of convolution operators and bias add operators"""
        model = PartitionGraphInOrder(
            op_names = ['nn.bias_add', None],
            func_name = True
        )(model)
        return model

    def _fused_gemm(self, model):
        """Fuse operatprs to gemm operator"""
        def gemm(transA, transB):
            A = tvm.relay.var('A', shape=(1,1))
            B = tvm.relay.var('B', shape=(1,1))
            C = tvm.relay.var('C', shape=(1,1))
            alpha = tvm.relay.const(1.0)
            beta = tvm.relay.const(1.0)
            _A, _B = A, B
            if transA:
                _A = tvm.relay.transpose(_A, axes=(1, 0))
            if transB:
                _B = tvm.relay.transpose(_B, axes=(1, 0))
            _A = tvm.relay.nn.batch_flatten(_A)
            aA_B = tvm.relay.nn.dense(alpha * _A, _B, units=1)
            out = tvm.relay.nn.bias_add(aA_B,beta*C)
            return tvm.relay.Function([A,B,C],out)
        for i in range(0,2):
            model = PartitionGraphByExpr(
                subexpr = gemm(i,1-i),
                func_name = "fused_gemm"
            )(model)
            model = PartitionGraphByExpr(
                subexpr=gemm(i,i),
                func_name = 'fused_gemm'
            )(model)
        return model

    def _fused_unsqueeze(self, model):
        """Fuse expand_dims operators to unsqueeze operator."""
        model = PartitionGraphInUnorder(
            op_names = ['expand_dims'],
            func_name = 'fused_unsqueeze',
        )(model)
        return model

    def _fuse_operator(self, model):
        "Fuse the Relay operators to fit ONNX model"
        model = self._fused_gemm(model)
        model = self._fused_xx_nn_bias(model)
        model = self._fused_unsqueeze(model)
        return model

def to_onnx(model, params, graph_name="", doc_string=None, opset = None):
    """Convert a Relay Function into ONNX model.
    Parameters
    ----------
    model : tvm.IRModule or tvm.relay.Function
        The relay module for compilation
    params : dict of str to tvm.nd.NDArray
        The parameter dict to be used by relay
    graph_name : str
        The name of the model's graph.
    doc_string :str
        The description of the model's graph
    Returns
    -------
    onnx_graph : protobuf object
        The returned ONNX ModelProto
    """
    if opset is None:
        try:
            opset = onnx.defs.onnx_opset_version() # get the supported opset version
        except AttributeError:
            opset = 11
    if isinstance(model, tvm.IRModule):
        pass
    elif isinstance(model, tvm.relay.Function):
        model = IRModule.from_expr(model)
    else:
        raise TypeError(
                'The paramter "model" must be tvm.IRModule or tvm.relay.Function.')
    g = GraphProto(opset)
    onnx_model = g._to_onnx(model, params, graph_name, doc_string)
    return onnx_model
