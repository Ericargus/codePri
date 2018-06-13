#ifndef MXNET_OPERATOR_MODEROI_POOLING_INL_H_
#define MXNET_OPERATOR_MODEROI_POOLING_INL_H_



#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./mshadow_op.h"
#include "./operator_common.h"


namespace mxnet{
	namespace op{
		namespace  moderoipool{
			enum MODEROIPoolingOpInputs {kData, kBox, kOffset};
			enum MODEROIPoolingOpOutputs {kOut, kMaxIdx};
			} //end for moderoipooling namespace
		struct MODEROIPoolingParam: public dmlc::Parameter<MODEROIPoolingParam>{
			float spatial_scale;
			//int output_dim;
			TShape pooled_size;
			//int group_size;
			DMLC_DECLARE_PARAMETER(MODEROIPoolingParam){
				DMLC_DECLARE_FIELD(spatial_scale).set_range(0.0, 1.0);
				//DMLC_DECLARE_FIELD(output_dim).describle("fix output dim")
				DMLC_DECLARE_FIELD(pooled_size).describe("fix pooled size");
				}
			}; //end for struct
		template<typename xpu, typename DType>
		class MODEROIPoolingOp: public Operator{
			public:
			explicit MODEROIPoolingOp(MODEROIPoolingParam p){
				this->param_ = p;
			}
			virtual void Forward(
				const OpContext &ctx,
				const std::vector<TBlob> &in_data,
				const std::vector<OpReqType> &req,
				const std::vector<TBlob> &out_data,
				const std::vector<TBlob> &aux_args){
				using namespace mshadow;
				CHECK_EQ(in_data.size(), 3U);
				CHECK_EQ(out_data.size(), 2U);
				CHECK_EQ(out_data[moderoipool::kOut].shape_[0], in_data[moderoipool::kBox].shape_[0]);
				CHECK_EQ(out_data[moderoipool::kMaxIdx].shape_[0], in_data[moderoipool::kBox].shape_[0]);
				CHECK_EQ(in_data[moderoipool::kBox].shape_[0], in_data[moderoipool::kOffset].shape_[0]);
				Stream<xpu> *s = ctx.get_stream<xpu>();

				Tensor<xpu, 4, DType> data = in_data[moderoipool::kData].get<xpu, 4, DType>(s);
				Tensor<xpu, 2, DType> bbox = in_data[moderoipool::kBox].get<xpu, 2, DType>(s);
				Tensor<xpu, 2, DType> offset = in_data[moderoipool::kOffset].get<xpu, 2, DType>(s);
				Tensor<xpu, 4, DType> out = out_data[moderoipool::kOut].get<xpu, 4, DType>(s);
				Tensor<xpu, 4, DType> max_idx = out_data[moderoipool::kMaxIdx].get<xpu, 4, DType>(s);

				CHECK_EQ(data.CheckContiguous(), true);
				CHECK_EQ(bbox.CheckContiguous(), true);
				CHECK_EQ(offset.CheckContiguous(), true);
				CHECK_EQ(out.CheckContiguous(), true);
				CHECK_EQ(max_idx.CheckContiguous(), true);
				out = -FLT_MAX;
				max_idx = -1.0f;
				MODEROIPoolForward(out, data, bbox, offset, max_idx, param_.spatial_scale); 
			}
			virtual void Backward(
				const OpContext &ctx,
				const std::vector<TBlob> &out_grad,
				const std::vector<TBlob> &in_data,
				const std::vector<OpReqType> &req,
				const std::vector<TBlob> &out_data,
				const std::vector<TBlob> &in_grad,
				const std::vector<TBlob> &aux_args){
				using namespace mshadow;
				CHECK_EQ(in_data.size(), 3U);
				CHECK_EQ(out_data.size(), 2U);
				CHECK_EQ(out_grad[moderoipool::kOut].shape_[0], in_data[moderoipool::kBox].shape_[0]);
				CHECK_EQ(out_data[moderoipool::kMaxIdx].shape_[0], in_data[moderoipool::kBox].shape_[0]);
				CHECK_NE(req[moderoipool::kData], kWriteInplace)<<
				"Don't support kWriteInplace.";
				CHECK_NE(req[moderoipool::kBox], kWriteInplace)<<
				"Don't support kWriteInplace.";
				CHECK_NE(req[moderoipool::kOffset], kWriteInplace)<<
				"Don't support kWriteInplace.";
				Stream<xpu> *s = ctx.get_stream<xpu>();

				//Tensor<xpu, 4, DType> data = in_data[moderoipool::kData].get<xpu, 4, DType>(s);
				Tensor<xpu, 2, DType> bbox = in_data[moderoipool::kBox].get<xpu, 2, DType>(s);
				Tensor<xpu, 2, DType> offset = in_data[moderoipool::kOffset].get<xpu, 2, DType>(s);
				//Tensor<xpu, 4, DType> out = out_data[moderoipool::kOut].get<xpu, 4, DType>(s);
				Tensor<xpu, 4, DType> grad_out = out_grad[moderoipool::kOut].get<xpu, 4, DType>(s);
				Tensor<xpu, 4, DType> grad_in = in_grad[moderoipool::kData].get<xpu, 4, DType>(s);
				Tensor<xpu, 2, DType> grad_roi = in_grad[moderoipool::kBox].get<xpu, 2, DType>(s);
				Tensor<xpu, 2, DType> grad_offset = in_grad[moderoipool::kOffset].get<xpu, 2, DType>(s);
				Tensor<xpu, 4, DType> max_idx = out_data[moderoipool::kMaxIdx].get<xpu, 4, DType>(s);

				CHECK_EQ(grad_out.CheckContiguous(), true);
				CHECK_EQ(bbox.CheckContiguous(), true);
				CHECK_EQ(max_idx.CheckContiguous(), true);
				CHECK_EQ(grad_in.CheckContiguous(), true);

				if (kAddTo == req[moderoipool::kData] || kWriteTo == req[moderoipool::kData]) {
					if (kWriteTo == req[moderoipool::kData]) {
						grad_in = 0.0f;
					}
					MODEROIPoolBackwardAcc(grad_in, grad_out, bbox, offset, max_idx, param_.spatial_scale);
				}
				if (kWriteTo == req[moderoipool::kBox]||kWriteTo == req[moderoipool::kOffset]){
					grad_roi = 0.0f;
					grad_offset = 0.0f;
				}
			}
			private:
			MODEROIPoolingParam param_;
		};
		
		template<typename xpu>
		Operator* CreateOp(MODEROIPoolingParam param, int dtype);

		#if DMLC_USE_CXX11
		class MODEROIPoolingProp: public OperatorProperty{
			public:
			std::vector<std::string> ListArguments() const override{
				return {"data", "rois", "offset"};
				}
			std::vector<std::string> ListOutputs() const override {
				return {"output", "maxidx"};
				}
			int NumOutputs() const override {
				return 2;
				}
			int NumVisibleOutputs() const override {
				return 1;
				}
			void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
				param_.Init(kwargs);
				//if (param_.group_size == 0) {
				//	param_.group_size = param_.pooled_size;
				//}
				}
			std::map<std::string, std::string> GetParams() const override {
				return param_.__DICT__();
			}
			bool InferShape(std::vector<TShape> *in_shape,
				std::vector<TShape> *out_shape,
				std::vector<TShape> *aux_shape) const override {
				using namespace mshadow;
				CHECK_EQ(in_shape->size(), 3U)<< "input:[data, rois, offset]";
				TShape dshape = in_shape->at(moderoipool::kData);
				CHECK_EQ(dshape.ndim(), 4)<< "data should be a 4D tensor";
				TShape bshape = in_shape->at(moderoipool::kBox);
				CHECK_EQ(bshape.ndim(), 2) << "bbox should be a 2D tensor of shape [batch, 5]";
				CHECK_EQ(bshape[1], 5) << "bbox should be a 2D tensor of shape [batch, 5]";
				TShape oshape = in_shape->at(moderoipool::kOffset);
				CHECK_EQ(oshape.ndim(), 2) << "bbox should be a 2D tensor of shape [batch, 5]";
				CHECK_EQ(oshape[1], 5) << "bbox should be a 2D tensor of shape [batch, 5]";
				out_shape->clear();
				out_shape->push_back(
					Shape4(bshape[0], dshape[1], param_.pooled_size[0], param_.pooled_size[1]));
				out_shape->push_back(
					Shape4(bshape[0], dshape[1], param_.pooled_size[0], param_.pooled_size[1]));
				return true;
			}
			bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
				CHECK_EQ(in_type->size(), 3U);
				int DType = (*in_type)[0];
				CHECK_EQ(DType, (*in_type)[1]);
				CHECK_NE(DType, -1) << "Input must have specified type";
				out_type->clear();
				out_type->push_back(DType);
				out_type->push_back(DType);
				return true;
			}
			OperatorProperty* Copy() const override {
				MODEROIPoolingProp* moderoi_pooling_sym = new MODEROIPoolingProp();
				moderoi_pooling_sym->param_ = this->param_;
				return moderoi_pooling_sym;
			}
			std::string TypeString() const override {
				return "MODEROIPooling";
			}
			std::vector<int> DeclareBackwardDependency(
			 	const std::vector<int> &out_grad,
			 	const std::vector<int> &in_data,
			 	const std::vector<int> &out_data) const override {
			 	return {out_grad[moderoipool::kOut], in_data[moderoipool::kBox], out_data[moderoipool::kMaxIdx]};
				 }
			Operator* CreateOperator(Context ctx) const override {
				LOG(FATAL) << "Not Implemented.";
				return NULL;
				}
			Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;
		private:
			MODEROIPoolingParam param_;
		};
		#endif
	}
}
#endif
