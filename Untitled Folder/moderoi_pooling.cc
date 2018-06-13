#include "./moderoi_pooling-inl.h"
#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mshadow/packet-inl.h>
#include <mshadow/dot_engine-inl.h>
#include <cassert>

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace mshadow{
template<typename DType>
inline void MODEROIPoolForward(const Tensor<cpu, 4, DType> &out,
                           const Tensor<cpu, 4, DType> &data,
                           const Tensor<cpu, 2, DType> &bbox,
                           const Tensor<cpu, 2, DType> &offset,
                           const Tensor<cpu, 4, DType> &max_idx,
                           const float spatial_scale){
  return;
}

template<typename DType>
inline void MODEROIPoolBackwardAcc(const Tensor<cpu, 4, DType> &in_grad,
                            const Tensor<cpu, 4, DType> &out_grad,
                            const Tensor<cpu, 2, DType> &bbox,
                            const Tensor<cpu, 2, DType> &offset,
                            const Tensor<cpu, 4, DType> &max_idx,
                            const float spatial_scale){
  return;
}
}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(MODEROIPoolingParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new MODEROIPoolingOp<cpu, DType>(param);
  });
  return op;
}

Operator *MODEROIPoolingProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                           std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(MODEROIPoolingParam);

MXNET_REGISTER_OP_PROPERTY(MODEROIPooling, MODEROIPoolingProp)
.describe("Performs region-of-interest pooling on inputs. Resize bounding box coordinates by "
"spatial_scale and crop input feature maps accordingly. The cropped feature maps are pooled "
"by max pooling to a fixed size output indicated by pooled_size. batch_size will change to "
"the number of region bounding boxes after MODEROIPooling")
.add_argument("data", "Symbol", "Input data to the pooling operator, a 4D Feature maps")
.add_argument("rois", "Symbol", "Bounding box coordinates, a 2D array of "
"[[batch_index, x1, y1, x2, y2]]. (x1, y1) and (x2, y2) are top left and down right corners "
"of designated region of interest. batch_index indicates the index of corresponding image "
"in the input data")
.add_argument("offset", "Symbol","same as bbox, but the last present  position offset, and and one ratio")
.add_arguments(MODEROIPoolingParam::__FIELDS__());
}  // namespace op
} // namespace mxnet