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
 * \file roi_pooling.cu
 * \brief roi pooling operator
 * \author Ross Girshick, Kye-Hyeon Kim, Jian Guo
 * \modifed by Zhong XingYu
*/
#include "./adaroi_pooling-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include <cmath>
//#include <cstdlib>

namespace mshadow {
namespace cuda {

/*
template<typename T>
__device__ T* cal_interval( const T* bottom_data,
                           const int hstart,
                           const int hend,
                           const int wstart,
                           const int wend,
                           const float interval_left,
                           const float interval_right,
                           T* avgsum,
                           int i,
                           int* s){
  //T data_new = bottom_data[indx];
  for (int h = hstart; h < hend; ++hstart){
    for (int w = wstart; w < wend; ++ wstart){
      T data_new = bottom_data[h * width + w];
      if ((data_new >= interval_left)&&(data_new<=interval_right)){
        s[i]++;
        avgsum[i] += data_new;
      }
      return s, avgsum;
    }
  }
}
*/
template<typename T>
__device__ T cal_random( T distant,
                         T start,
                         T num){
  return start + distant/(num+1);
}





template<typename Dtype>
__global__ void ADAROIPoolForwardKernel(const int count, const Dtype* bottom_data,
                                     const float spatial_scale, const int channels,
                                     const int height, const int width,
                                     const int pooled_height, const int pooled_width,
                                     const Dtype* bottom_rois,const Dtype* bottom_offset ,
                                     Dtype* top_data) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    bottom_offset += n * 5; 
    int roi_batch_ind = bottom_rois[0];

    if (roi_batch_ind < 0) {
      top_data[index] = 0;
      //argmax_data[index] = 0;
      continue;
    }

    int rois_ratio = fabs(bottom_offset[0]);

    int roi_start_w_ = round(bottom_rois[1] * spatial_scale + bottom_offset[1]);
    int roi_start_h_ = round(bottom_rois[2] * spatial_scale + bottom_offset[2]);
    int roi_end_w_ = round(bottom_rois[3] * spatial_scale + bottom_offset[3]);
    int roi_end_h_ = round(bottom_rois[4] * spatial_scale + bottom_offset[4]);
    int roi_start_w = min( max(roi_start_w_, static_cast<int>(0)), width );
    int roi_start_h = min( max(roi_start_h_, static_cast<int>(0)), height);
    int roi_end_w = min(max(roi_end_w_, static_cast<int>(0)), width);
    int roi_end_h = min(max(roi_end_h_, static_cast<int>(0)), height);
    if((roi_start_w >= roi_end_w)||(roi_start_h >= roi_end_h)){
      continue;
    }

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    Dtype bin_size_h = static_cast<Dtype>(roi_height)
                       / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)
                       / static_cast<Dtype>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
                                        * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
                                        * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    //bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    //Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    //int maxidx = -1;
    
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    Dtype output_avg = 0;
    int num_pixels = (hend-hstart) * (wend - wstart);
    int start_v = (hstart*width + wstart);
    int distance = ((hend*width + wend) - (hstart*width + wstart));

    //int *s = new int [num_pixels]();
    //int i = 0;
    //int ind_required = -1;
    //int max = -INT_MAX;
    //Dtype *avgsum = new Dtype[num_pixels]();
    if((rois_ratio <=1)||(rois_ratio >=num_pixels)){
      rois_ratio = num_pixels;
      for(int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          int bottom_index = h * width + w;
          output_avg += bottom_data[bottom_index];
        }
      }
    }
    else{
      for(int i = 1; i <=rois_ratio; ++i){
        int rand_ind = cal_random(distance, start_v, i);
        output_avg += bottom_data[rand_ind];     
      }
    }
    output_avg /= rois_ratio;
    top_data[index] = output_avg;
  }
}


    /*
    
    / for another;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h * width + w;
        if((rois_ratio < 1)||(rois_ratio >num_pixels)){
          rois_ratio = num_pixels;
        }
    
        
        if (bottom_data[bottom_index] > maxval) {
          maxval = bottom_data[bottom_index];
          maxidx = bottom_index;
          }
        Dtype data = bottom_data[bottom_index];
        float diff_interval = static_cast<float>(data * rois_ratio);
        float interval_left = static_cast<float>(data) - diff_interval;
        float interval_right = static_cast<float>(data) + diff_interval;
        for(int h_new = hstart; h_new < hend; ++h_new){
          for(int w_new = wstart; w_new < wend; ++w_new){
            int indx = h_new * width + w_new;
            Dtype data_new = bottom_data[indx];
            if ((data_new >=interval_left)&&(data_new <=interval_right)){
              s[i]++;
              avgsum[i] += data_new;
            }
          }
        }
        i++;
      
      }
    }
    for(int k = 0; k<num_pixels; k++){
      if(s[k] > max){
        max = s[k];
        ind_required = k;
      }
    }
    output_avg = is_empty ? static_cast<Dtype>(0):(avgsum[ind_required]/( max));
    top_data[index] = output_avg;
    delete[] s;
    delete[] avgsum;
    //argmax_data[index] = (Dtype)maxidx;
    */

template<typename Dtype>
inline void ADAROIPoolForward(const Tensor<gpu, 4, Dtype> &out,
                           const Tensor<gpu, 4, Dtype> &data,
                           const Tensor<gpu, 2, Dtype> &bbox,
                           /*const Tensor<gpu, 4, Dtype> &max_idx,*/
                           const Tensor<gpu, 2, Dtype> &offset,
                           const float spatial_scale) {
  const Dtype *bottom_data = data.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;
  Dtype *top_data = out.dptr_;
  //Dtype *argmax_data = max_idx.dptr_;
  const Dtype *bottom_offset = offset.dptr_;
  const int count = out.shape_.Size();
  const int channels = data.size(1);
  const int height = data.size(2);
  const int width = data.size(3);
  const int pooled_height = out.size(2);
  const int pooled_width = out.size(3);
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "ADAROIPooling Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
  ADAROIPoolForwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, bottom_data, spatial_scale, channels, height, width,
      pooled_height, pooled_width, bottom_rois, bottom_offset ,top_data);
  MSHADOW_CUDA_POST_KERNEL_CHECK(ADAROIPoolForwardKernel);
}

template<typename Dtype>
__global__ void ADAROIPoolBackwardAccKernel(
  const int count,
  const Dtype* top_diff,
  const int num_rois,
  const float spatial_scale, 
  const int channels,
  const int height, const int width,
  const int pooled_height, const int pooled_width,
  Dtype* bottom_diff, 
  const Dtype* bottom_rois, 
  const Dtype* bottom_offset) {
  for (int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;index < count;index += blockDim.x * gridDim.x * gridDim.y){
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    // [start, end) interval for spatial sampling
    const Dtype* offset_bottom_rois = bottom_rois + n * 5;
    const Dtype* offset_bottom_offset = bottom_offset + n*5;
    int roi_batch_ind = offset_bottom_rois[0];
    int roi_start_w_ = round(offset_bottom_rois[1] * spatial_scale + offset_bottom_offset[1]);
    int roi_start_h_ = round(offset_bottom_rois[2] * spatial_scale + offset_bottom_offset[2]);
    int roi_end_w_ = round(offset_bottom_rois[3] * spatial_scale + offset_bottom_offset[3]);
    int roi_end_h_ = round(offset_bottom_rois[4] * spatial_scale + offset_bottom_offset[4]);
    int roi_start_w = min( max(roi_start_w_, static_cast<int>(0)), width);
    int roi_start_h = min( max(roi_start_h_, static_cast<int>(0)), height);
    int roi_end_w = min(max(roi_end_w_, static_cast<int>(0)), width);
    int roi_end_h = min(max(roi_end_h_, static_cast<int>(0)), height);
    if((roi_start_w >= roi_end_w)||(roi_start_h >= roi_end_h)){
      continue;
    }
    Dtype* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;

    // Force too small ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);  // avoid 0
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);

    // Compute w and h at bottom
    Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width);

    int hstart = floor(static_cast<Dtype>(ph)* bin_size_h
      + roi_start_h);
    int wstart = floor(static_cast<Dtype>(pw)* bin_size_w
      + roi_start_w);
    int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h
      + roi_start_h);
    int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w
      + roi_start_w);
    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart, 0), height);
    hend = min(max(hend, 0), height);
    wstart = min(max(wstart, 0), width);
    wend = min(max(wend, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);
    Dtype bin_area = (hend - hstart)*(wend - wstart);
    Dtype diff_val = is_empty ? (Dtype)0. : top_diff[index] / bin_area;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h*width + w;
        atomicAdd(offset_bottom_diff + bottom_index, diff_val);
      }
    }
  }
 }
template<typename Dtype>
inline void ADAROIPoolBackwardAcc(const Tensor<gpu, 4, Dtype> &in_grad,
                               const Tensor<gpu, 4, Dtype> &out_grad,
                               const Tensor<gpu, 2, Dtype> &bbox,
                               const Tensor<gpu, 2, Dtype> &offset,
                               /*const Tensor<gpu, 4, Dtype> &max_idx,*/
                               const float spatial_scale) {
  const Dtype *top_diff = out_grad.dptr_;
  const Dtype *bottom_rois = bbox.dptr_;
  Dtype *bottom_diff = in_grad.dptr_;
  const Dtype *bottom_offset = offset.dptr_;
  //Dtype *argmax_data = max_idx.dptr_;
  const int count = in_grad.shape_.Size();
  const int num_rois = bbox.size(0);
  const int channels = in_grad.size(1);
  const int height = in_grad.size(2);
  const int width = in_grad.size(3);
  const int pooled_height = out_grad.size(2);
  const int pooled_width = out_grad.size(3);
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "ADAROIPooling Backward");
  cudaStream_t stream = Stream<gpu>::GetStream(in_grad.stream_);
  ADAROIPoolBackwardAccKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
      count, top_diff, num_rois, spatial_scale, channels, height, width,
      pooled_height, pooled_width, bottom_diff, bottom_rois, bottom_offset);
  MSHADOW_CUDA_POST_KERNEL_CHECK(ADAROIPoolBackwardAccKernel);
}

}  // namespace cuda

template<typename Dtype>
inline void ADAROIPoolForward(const Tensor<gpu, 4, Dtype> &out,
                           const Tensor<gpu, 4, Dtype> &data,
                           const Tensor<gpu, 2, Dtype> &bbox,
                           const Tensor<gpu, 2, Dtype> &offset,
                           const float spatial_scale) {
  cuda::ADAROIPoolForward(out, data, bbox, offset, spatial_scale);
}

template<typename Dtype>
inline void ADAROIPoolBackwardAcc(const Tensor<gpu, 4, Dtype> &in_grad,
                               const Tensor<gpu, 4, Dtype> &out_grad,
                               const Tensor<gpu, 2, Dtype> &bbox,
                               const Tensor<gpu, 2, Dtype> &offset,
                               const float spatial_scale) {
  cuda::ADAROIPoolBackwardAcc(in_grad, out_grad, bbox, offset, spatial_scale);
}

}  // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(ADAROIPoolingParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, Dtype, {
    op = new ADAROIPoolingOp<gpu, Dtype>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
