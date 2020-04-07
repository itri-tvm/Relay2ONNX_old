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

Introduction
----------
There are 66 operator conversion written in /python/tvm/relay/frontend/to_onnx.py file. For some special ONNX operators, Gemm, Conv and Unsqueeze, we create three new partition graph Relay pass (in /src/relay/pass folder) to fuse their converted Relay operators back to original ONNX operator.In some case, if nn.bias_add is not simplified by TVM optimization, these operators can be fused back to original ONNX operator. For the same reason, fuse expand_dims operators to unsqueeze operator.

Test ONNX Model List
--------------------
|Model|
|-----|
|MobileNet_V2|
|ResNet50|
|SqueezeNet|
|VGG19|
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


Relay to ONNX OP List
---------------------
|Relay|ONNX|
|-----|----|
|abs|Abs|
|add|Add|
|argmax|ArgMax|
|argmin|ArgMin|
|broadcast_to|Expand|
|cast|Cast|
|ceil|Ceil|
|clip|Clip|
|concatenate|Concat|
|Constant|Constant|
|copy|Identity|
|divide|Div|
|equal|Equal|
|erf|Erf|
|exp|Exp|
|floor|Floor|
|fused_gemm|Gemm|
|fused_unsqueeze|Unsqueeze|
|greater|Greater|
|image.resize|Resize|
|less|Less|
|log|Log|
|logical_and|And|
|logical_not|Not|
|logical_or|Or|
|max|ReduceMax|
|maximun|Max|
|mean|Mean|
|mean|ReduceMean|
|min|Min|
|min|ReduceMin|
|multiply|Mul|
|negative|Neg|
|nn.avg_poolid|AveragePool|
|nn.batch_flatten|Flatten|
|nn.batch_norm|BatchNormalization|
|nn.convid|Conv|
|nn.convid_tranpose|ConvTranspose|
|nn.dense|FC|
|nn.dropout|Dropout|
|nn.global_avg_poolid|GlobalAveragePool|
|nn.global_max_poolid|GlobalMaxPool|
|nn.instance_norm|InstanceNormalization|
|nn.leaky_relu|LeakyRelu|
|nn.log_softmax|LogSoftmax|
|nn.lrn|LRN|
|nn.max_pool|MaxPool|
|nn.pad|Pad|
|nn.prelu|PRelu|
|nn.relu|Relu|
|power|Pow|
|prod|ReduceProd|
|reshape|Reshape|
|shape_of|Shape|
|sigmoid|Sigmoid|
|sign|Sign|
|split|Split|
|sqrt|Sqrt|
|squeeze|Squeeze|
|sub|Sub|
|sum|ReduceSum|
|take|Gather|
|tile|Tile|
|transpose|Transpose|
|upsampling|Upsample|
|where|Where|
