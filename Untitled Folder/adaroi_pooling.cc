/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!\base a lot code on
 * Copyright (c) 2015 by Contributors
 * \file roi_pooling.cc
 * \brief roi pooling operator
 * \author Ross Girshick, Kye-Hyeon Kim, Jian Guo, Xinyu Chen
 * \modified by Zhong XingYu
*/

#include "./adaroi_pooling-inl.h"
#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mshadow/packet-inl.h>
#include <mshadow/dot_engine-inl.h>
#include <cassert>
#include <cmath>

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace mshadow {
template<typename Dtype>
inline void ADAROIPoolForward(const Tensor<cpu, 4, Dtype> &out,
                           const Tensor<cpu, 4, Dtype> &data,
                           const Tensor<cpu, 2, Dtype> &bbox,
                           const Tensor<cpu, 2, Dtype> &offset,
                           const float spatial_scale_) {
  return;
}

template<typename Dtype>
inline void ADAROIPoolBackwardAcc(const Tensor<cpu, 4, Dtype> &in_grad,
                               const Tensor<cpu, 4, Dtype> &out_grad,
                               const Tensor<cpu, 2, Dtype> &bbox,
                               const Tensor<cpu, 2, Dtype> &offset,
                               const float spatial_scale_) {
  return;
}
}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(ADAROIPoolingParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ADAROIPoolingOp<cpu, DType>(param);
  });
  return op;
}

Operator *ADAROIPoolingProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                           std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(ADAROIPoolingParam);

MXNET_REGISTER_OP_PROPERTY(ADAROIPooling, ADAROIPoolingProp)
.describe("Perform ROIPooling Operator with a little modified, first give the power to change the region area,"
"second, use highst frequent interval instead of max or avg")
.add_argument("data", "NDArray-or-Symbol", "The input array to the pooling operator")
.add_argument("rois", "NDArray-or-Symbol", "Bounding box coordinates, a 2D array of "
"[[batch_index, x1, y1, x2, y2]], where (x1, y1) and (x2, y2) are top left and bottom right "
"corners of designated region of interest. `batch_index` indicates the index of corresponding "
"image in the input array")
.add_argument("offset", "NDArray-or-Symbol", "Input offset [ration, x1, y1, x2, y2]")
.add_arguments(ADAROIPoolingParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
