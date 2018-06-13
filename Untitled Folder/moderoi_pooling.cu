#include "./moderoi_pooling-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include "../common/cuda_utils.h"
#include "./mxnet_op.h"

#define MODEROIPOOLING_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)
#define CUDA_KERNEL_LOOP(i, n) \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)

namespace mshadow {
	namespace cuda {
	template <typename Dtype>
	__global__ void MODEROIPoolForwardKernel(
		const int count, const Dtype* bottom_data,
		const float spatial_scale, const int channels,
		const int height, const int width, const Dtype* offset,
        const int pooled_height, const int pooled_width,
        const Dtype* bottom_rois, Dtype* top_data, Dtype* argmax_data){
        CUDA_KERNEL_LOOP(index, count){
		int pw = index % pooled_width;
		int ph = index / pooled_width % pooled_height;
		int c = index / pooled_width /pooled_height % channels;
		int n = index / pooled_width / pooled_height / channels;

		bottom_rois += n*5;
		offset += n*5;
		int roi_batch_ind = bottom_rois[0];
		float roi_ratio = offset[0];
		int roi_start_w = round(bottom_rois[1] * spatial_scale + offset[1]);
		int roi_start_h = round(bottom_rois[2] * spatial_scale + offset[2]);
		int roi_end_w = round(bottom_rois[3] * spatial_scale + offset[3]);
		int roi_end_h = round(bottom_rois[4] * spatial_scale + offset[4]);


		int roi_width = max(roi_end_w - roi_start_w + 1,1);
		int roi_height = max(roi_end_h - roi_start_h +1,1);

		Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height);
		Dtype bin_size_w = static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width);

		int hstart = static_cast<int>(floor(static_cast<Dtype>(ph) * bin_size_h));
		int wstart = static_cast<int>(floor(static_cast<Dtype>(pw) * bin_size_w));
		int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1) * bin_size_h));
		int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1) * bin_size_w));


		hstart = min(max(hstart + roi_start_h, 0), height);
		hend = min(max(hend + roi_start_h, 0), height);
		wstart = min(max(wstart +roi_start_w, 0), width);
		wend = min(max(wend + roi_start_w, 0), width);
		bool is_empty = (hstart >= hend)||(wstart >= wend);

		const Dtype* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;
		Dtype output_avg = 0;
		int max = 0;
		int max_mode_idx = -1;
		int num_point = (hend - hstart) * (wend - wstart);
		int *s = new int [num_point]();
		int i = 0;
		int ind_required = 0;
		int *index_list = new int[num_point]();
		Dtype *avgsum = new Dtype[num_point]();
		for (int h = hstart; h < hend; ++h){
			for (int w = wstart; w < wend; ++w){
				int bottom_data_index = h * width + w;
				Dtype data = offset_bottom_data[bottom_data_index];
				float interval = static_cast<float>(data * roi_ratio);
				float interval_min = static_cast<float>(data) - interval;
				float interval_max = static_cast<float>(data) + interval;
				for (int h_new = hstart; h_new < hend; ++h_new){
					for(int w_new = wstart; w_new < wend; ++w_new){
						int indx = h_new * width + w_new;
						Dtype data_new = offset_bottom_data[indx];
						if ((data_new >= interval_min) && (data_new <= interval_max)){
							s[i]++;
							avgsum[i] += data_new;
							index_list[i] = bottom_data_index;
						}
					}
				}
				i++;
			}
		}
		for(int k = 0; k<num_point; ++k){
			if (s[k] > max){
				max = s[k];
				ind_required = k;
				max_mode_idx = index_list[k];
			}
		}

		output_avg = is_empty ? static_cast<Dtype>(0):(avgsum[ind_required]/max) ;

		top_data[index] = output_avg;
		argmax_data[index] = static_cast<Dtype>(max_mode_idx);
		delete[] s;
		delete[] index_list;
		delete[] avgsum;
	}
}

template<typename Dtype>
inline void MODEROIPoolForward(const Tensor<gpu, 4, Dtype>& out,
	                           const Tensor<gpu, 4, Dtype>& data,
	                           const Tensor<gpu, 2, Dtype>& bbox,
	                           const Tensor<gpu, 2, Dtype>& bbox_diff,
	                           const Tensor<gpu, 4, Dtype>& max_idx,
	                           const float spatial_scale){
	const Dtype *bottom_data = data.dptr_;
	const Dtype *bottom_rois = bbox.dptr_;
	const Dtype *offset = bbox_diff.dptr_;
	Dtype *top_data = out.dptr_;
	Dtype *argmax_data = max_idx.dptr_;
	const int count = out.shape_.Size();
	const int channels = data.size(1);
	const int height = data.size(2);
	const int width = data.size(3);
	const int pooled_height = out.size(2);
	const int pooled_width = out.size(3);
	const int gridSize = (count + kMaxThreadsPerBlock -1) / kMaxThreadsPerBlock;
	dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim -1)/ kMaxGridDim);
	dim3 dimBlock(kMaxThreadsPerBlock);
	CheckLaunchParam(dimGrid, dimBlock, "MODEPOOLING FORWARD");
	cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
	MODEROIPoolForwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
		count, bottom_data, spatial_scale, channels, height, width, offset, 
		pooled_height, pooled_width, bottom_rois, top_data, argmax_data);
		MSHADOW_CUDA_POST_KERNEL_CHECK(MODEROIPoolForwardKernel);
}


template<typename Dtype>
__global__ void MODEROIPoolBackwardAccKernel(const int count, const Dtype *top_diff,
	                                const Dtype *argmax_data, const int num_rois,
	                                const float spatial_scale, const int channels,
	                                const int height, const int width,
	                                const int pooled_height, const int pooled_width,
	                                Dtype *bottom_diff, const Dtype *bottom_rois, const Dtype *offset){
	CUDA_KERNEL_LOOP(index, count){
		int w = index / width;
		int h = index / width % height;
		int c = index / width / height % channels;
		int n = index / width / height / channels ;

		Dtype gradient = 0;
		for (int roi_n = 0; roi_n < num_rois; ++roi_n){
			const Dtype *offset_bottom_rois = bottom_rois + roi_n * 5;
			int roi_batch_ind = offset_bottom_rois[0];

			if (n != roi_batch_ind){
				continue;
			}

			//float roi_ratio = offset[0];
			int roi_start_w = round(bottom_rois[1] * spatial_scale + offset[1]);
			int roi_start_h = round(bottom_rois[2] * spatial_scale + offset[2]);
			int roi_end_w = round(bottom_rois[3] * spatial_scale + offset[3]);
			int roi_end_h = round(bottom_rois[4] * spatial_scale + offset[4]);

			const bool in_roi = ( w >= roi_start_w && w <= roi_end_w && h >= roi_start_h && h <= roi_end_h);
			if(!in_roi){
				continue;
			}
			int bottom_offset = (roi_n*channels + c) * pooled_height * pooled_width;
			const Dtype* offset_top_diff = top_diff + bottom_offset;
			const Dtype* offset_argmax_data = argmax_data + bottom_offset;

			int roi_width = max((roi_end_w - roi_start_w + 1), 1);
			int roi_height = max((roi_end_h - roi_start_h+ 1), 1);

			
			Dtype  bin_size_h = static_cast <Dtype>(roi_height)/static_cast<Dtype>(pooled_height);
			Dtype  bin_size_w = static_cast <Dtype>(roi_width)/static_cast<Dtype>(pooled_width);

			int phstart = floor(static_cast<Dtype>(h - roi_start_h) / bin_size_h);
			int phend = ceil(static_cast<Dtype>(h - roi_end_h + 1) / bin_size_h);
			int pwstart = floor(static_cast<Dtype>(w - roi_start_w) / bin_size_w);
			int pwend = ceil(static_cast<Dtype>(w - roi_end_h + 1) / bin_size_w);

			phstart = min(max(phstart, 0), pooled_height);
			phend = min(max(phend, 0), pooled_height);
			pwstart = min(max(pwstart, 0), pooled_width);
			pwend = min(max(pwend, 0), pooled_width);

			for(int ph = phstart; ph < phend; ++ph){
				for(int pw = pwstart; pw < pwend; ++pw){
					if(static_cast<Dtype>(offset_argmax_data[ph*pooled_width + pw]) == (h * width + w)){
						gradient += offset_top_diff[ph*pooled_width + pw];
					}
				}
			}

		}
		bottom_diff[index] += gradient; 
	}
	
}

template<typename Dtype>
inline void MODEROIPoolBackwardAcc(const Tensor<gpu, 4, Dtype> &in_grad,
	                               const Tensor<gpu, 4, Dtype> &out_grad,
	                               const Tensor<gpu, 2, Dtype> &bbox,
	                               const Tensor<gpu, 2, Dtype> &bbox_diff,
	                               const Tensor<gpu, 4, Dtype> &max_idx,
	                               const float spatial_scale){
	const Dtype *top_diff = out_grad.dptr_;
	const Dtype *bottom_rois = bbox.dptr_;
	Dtype *bottom_diff = in_grad.dptr_;
	const Dtype *argmax_data = max_idx.dptr_;
	const Dtype *offset = bbox_diff.dptr_;
	const int count = in_grad.shape_.Size();
	const int num_rois = bbox.size(0);
	const int channels = in_grad.size(1);
	const int height = in_grad.size(2);
	const int width = in_grad.size(3);
	const int pooled_height = out_grad.size(2);
	const int pooled_width = out_grad.size(3);
	const int gridSize = (count + kMaxThreadsPerBlock -1) / kMaxThreadsPerBlock;
	dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim -1) / kMaxGridDim);
	dim3 dimBlock(kMaxThreadsPerBlock);
	CheckLaunchParam(dimGrid, dimBlock, "MODEPOOLING BACKWARD");
	cudaStream_t stream = Stream<gpu>::GetStream(in_grad.stream_);
	MODEROIPoolBackwardAccKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
		count, top_diff, argmax_data, num_rois, spatial_scale, channels, height, width, pooled_height, pooled_width, bottom_diff, bottom_rois, offset);
	MSHADOW_CUDA_POST_KERNEL_CHECK(MODEROIPoolBackwardAccKernel);
}
} // namespace cuda

template<typename Dtype>
inline void MODEROIPoolForward(const Tensor<gpu, 4, Dtype>& out,
	                           const Tensor<gpu, 4, Dtype>& data,
	                           const Tensor<gpu, 2, Dtype>& bbox,
	                           const Tensor<gpu, 2, Dtype>& offset,
	                           const Tensor<gpu, 4, Dtype>& max_idx,
	                           const float spatial_scale){
	cuda::MODEROIPoolForward(out, data, bbox, offset, max_idx, spatial_scale);
}
template<typename Dtype>
inline void MODEROIPoolBackwardAcc(const Tensor<gpu, 4, Dtype> &in_grad,
	                               const Tensor<gpu, 4, Dtype> &out_grad,
	                               const Tensor<gpu, 2, Dtype> &bbox,
	                               const Tensor<gpu, 2, Dtype> &offset,
	                               const Tensor<gpu, 4, Dtype> &max_idx,
	                               const float spatial_scale){
	cuda::MODEROIPoolBackwardAcc(in_grad, out_grad, bbox, offset, max_idx, spatial_scale);
}
} //namespace mshadow

namespace mxnet{
	namespace op{
		template<>
		Operator *CreateOp<gpu>(MODEROIPoolingParam param, int dtype) {
			Operator* op = NULL;
			MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
				op = new MODEROIPoolingOp<gpu, DType>(param);
				});
				return op;
		}
	} //namespace op
} // namespace mxnet
