<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

Relay2ONNX
----------
There are 66 operator conversion written in /python/tvm/relay/frontend/to_onnx.py file. For some special ONNX operators, Gemm, Conv and Unsqueeze, I create three new partition graph Relay pass (in /src/relay/pass folder) to fuse their converted Relay operators back to original ONNX operator.In some case, if nn.bias_add is not simplified by TVM optimization, these operators can be fused back to original ONNX operator. For the same reason, fuse expand_dims operators to unsqueeze operator.

Test ONNX Model list
--------------------
|Model|
-------
|MobileNet|
|ResNet|
|SqueezeNet|
|VGG|
|AlexNet|
|GoogleNet|
|CaffeNet|
|RCNN_ILSVRC13|
|DenseNet-121|
|Inception_V1|
|Inception_V2|
|ShuffleNet_V1|
|ShuffleNet_V2|
|ZFNet-512|
|EfficientNet|


ONNX to Relay OP list
---------------------
|ONNX|Relay|
------------
|Abs|abs|
|Add|add|
|ArgMax|argmax|
|ArgMin|argmin|
|Expand|broadcast_to|
|Cast|cast|
|Ceil|ceil|
|Clip|clip|
|Concat|concatenate|
|Constant|Constant|
|Identity|copy|
|Div|divide|
|Equal|equal|
|Erf|erf|
|Exp|exp|
|Floor|floor|
|Gemm|fused_gemm|
|Unsqueeze|fused_unsqueeze|
|Greater|greater|
|Resize|image.resize|
|Less|less|
|Log|log|
|And|logical_and|
|Not|logical_not|
|Or|logical_or|
|ReduceMax|max|
|Max|maximun|
|Mean|mean|
|ReduceMean|mean|
|Min|min|
|ReduceMin|min|
|Mul|multiply|
|Neg|negative|
|AveragePool|nn.avg_poolid|
|Flatten|nn.batch_flatten|
|BatchNormalization|nn.batch_norm|
|Conv|nn.convid|
|ConvTranspose|nn.convid_tranpose|
|FC|nn.dense|
|Dropout|nn.dropout|
|GlobalAveragePool|nn.global_avg_poolid|
|GlobalMaxPool|nn.global_max_poolid|
|InstanceNormalization|nn.instance_norm|
|LeakyRelu|nn.leaky_relu|
|LogSoftmax|nn.log_softmax|
|LRN|nn.lrn|
|MaxPool|nn.max_pool|
|Pad|nn.pad|
|PRelu|nn.prelu|
|Relu|nn.relu|
|Pow|power|
|ReduceProd|prod|
|Reshape|reshape|
|Shape|shape_of|
|Sigmoid|sigmoid|
|Sign|sign|
|Split|split|
|Sqrt|sqrt|
|Squeeze|squeeze|
|Sub|sub|
|ReduceSum|sum|
|Gather|take|
|Tile|tile|
|Transpose|transpose|
|Upsample|upsampling|
|Where|where|
