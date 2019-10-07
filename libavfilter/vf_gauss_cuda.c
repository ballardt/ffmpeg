/*
* Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "libavutil/avstring.h"
#include "libavutil/common.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/internal.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"

#include "avfilter.h"
#include "formats.h"
#include "internal.h"
#include "scale.h"
#include "video.h"

static const enum AVPixelFormat supported_formats[] = {
    AV_PIX_FMT_YUV420P
    // AV_PIX_FMT_NV12,
    // AV_PIX_FMT_YUV444P,
    // AV_PIX_FMT_P010,
    // AV_PIX_FMT_P016
};

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )
#define ALIGN_UP(a, b) (((a) + (b) - 1) & ~((b) - 1))
#define NUM_BUFFERS 2

typedef struct CUDAGaussContext CUDAGaussContext;

struct CUDAGaussContext
{
    const AVClass *class;
    enum AVPixelFormat in_fmt;
    enum AVPixelFormat out_fmt;

    struct {
        int width;
        int height;
    } planes[3];

    AVBufferRef* frames_ctx;
    AVFrame*     frame;
    AVFrame*     new_frame;

    AVBufferRef* tmp_frames_ctx;
    AVFrame*     tmp_frame;

    /**
     * Output sw format. AV_PIX_FMT_NONE for no conversion.
     */
    enum AVPixelFormat format;

    int64_t edge_expr;   ///< %percentage to filter
    int64_t pole_expr;   ///< %percentage to filter
    int64_t right_expr;
    int64_t left_expr;
    int64_t top_expr;
    int64_t bottom_expr;
    double  sigma_expr;  ///< %sigma for computing Gaussian filter
    int64_t loop_expr;

    // CUcontext   cu_ctx;
    // CUevent     cu_event;
    CUmodule    cu_module;
    CUfunction  cu_func_horiz;
    CUfunction  cu_func_vert;
    // CUtexref    cu_tex_uchar;
    CUdeviceptr cu_constant;

    // CUdeviceptr srcBuffer;
    // CUdeviceptr tmpBuffer;
    // CUdeviceptr dstBuffer;
};

typedef struct Filter Filter;

struct Filter
{
    float value[20];
    int   span;
};

static float sigma_on_gpu = 1.6f;
static float sigma        = 1.6f;
static Filter filter = { { },
                         8 };

static av_cold int cudagauss_init(AVFilterContext *ctx)
{
    CUDAGaussContext *s = ctx->priv;

    s->format = AV_PIX_FMT_NONE;
    s->frame = av_frame_alloc();
    if (!s->frame)
        return AVERROR(ENOMEM);

    s->new_frame = av_frame_alloc();
    if (!s->frame)
        return AVERROR(ENOMEM);

    s->tmp_frame = av_frame_alloc();
    if (!s->tmp_frame)
        return AVERROR(ENOMEM);

    return 0;
}

static av_cold void cudagauss_uninit(AVFilterContext *ctx)
{
    CUDAGaussContext *s = ctx->priv;

    av_frame_free(&s->frame);
    av_frame_free(&s->new_frame);
    av_frame_free(&s->tmp_frame);
    av_buffer_unref(&s->frames_ctx);
    av_buffer_unref(&s->tmp_frames_ctx);
}

static int cudagauss_query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pixel_formats[] = {
        AV_PIX_FMT_CUDA, AV_PIX_FMT_NONE,
    };
    AVFilterFormats *pix_fmts = ff_make_format_list(pixel_formats);

    return ff_set_common_formats(ctx, pix_fmts);
}

static av_cold int init_stage(CUDAGaussContext *s, AVBufferRef *device_ctx)
{
    int in_sw, in_sh;
    int ret, i;
    CUresult err;
    AVBufferRef*       out_ref = NULL;
    AVBufferRef*       tmp_ref = NULL;

    av_pix_fmt_get_chroma_sub_sample(s->in_fmt,  &in_sw,  &in_sh);
    if (!s->planes[0].width) {
        s->planes[0].width  = s->planes[0].width;
        s->planes[0].height = s->planes[0].height;
    }

    for (i = 1; i < FF_ARRAY_ELEMS(s->planes); i++) {
        s->planes[i].width   = s->planes[0].width   >> in_sw;
        s->planes[i].height  = s->planes[0].height  >> in_sh;
    }

    {
        // block to allocate out frame
        AVHWFramesContext* out_ctx;

        out_ref = av_hwframe_ctx_alloc(device_ctx);
        if (!out_ref)
            return AVERROR(ENOMEM);
        out_ctx = (AVHWFramesContext*)out_ref->data;

        out_ctx->format    = AV_PIX_FMT_CUDA;
        out_ctx->sw_format = s->out_fmt;
        out_ctx->width     = FFALIGN(s->planes[0].width,  32);
        out_ctx->height    = FFALIGN(s->planes[0].height, 32);

        ret = av_hwframe_ctx_init(out_ref);
        if (ret < 0)
            goto fail;

        av_frame_unref(s->frame);
        ret = av_hwframe_get_buffer(out_ref, s->frame, 0);
        if (ret < 0)
            goto fail;

        s->frame->width  = s->planes[0].width;
        s->frame->height = s->planes[0].height;

        av_buffer_unref(&s->frames_ctx); // if an old frame_ctx existed, delete it
        s->frames_ctx = out_ref;         // store new ctx in frame_ctx
    }

    {
        // block to allocate tmp frame
        AVHWFramesContext* tmp_ctx;

        tmp_ref = av_hwframe_ctx_alloc(device_ctx);
        if (!tmp_ref)
            return AVERROR(ENOMEM);
        tmp_ctx = (AVHWFramesContext*)tmp_ref->data;

        tmp_ctx->format    = AV_PIX_FMT_CUDA;
        tmp_ctx->sw_format = s->out_fmt;
        tmp_ctx->width     = FFALIGN(s->planes[0].width,  32);
        tmp_ctx->height    = FFALIGN(s->planes[0].height, 32);

        ret = av_hwframe_ctx_init(tmp_ref);
        if (ret < 0)
            goto fail;

        av_frame_unref(s->tmp_frame);
        ret = av_hwframe_get_buffer(tmp_ref, s->tmp_frame, 0);
        if (ret < 0)
            goto fail;

        s->frame->width  = s->planes[0].width;
        s->frame->height = s->planes[0].height;

        av_buffer_unref(&s->tmp_frames_ctx);
        s->tmp_frames_ctx = tmp_ref;
    }

    sigma = s->sigma_expr;

    if( sigma != sigma_on_gpu )
    {
        float sum;
        int x;

        // The total span of the filter is span+span-1, but it is symmetrical on both
        // side of the center element. We compute the Gauss table only for the center
        // and one side.
        filter.span = (int)ceilf( 4.0f * sigma ) + 1; // this cutoff is used in vlfeat
        if( filter.span > 20 ) filter.span = 20;

        filter.value[0] = 1.0f;

        sum = 1.0f;
        for( x = 1; x<filter.span; x++ )
        {
            const float val = expf( -0.5f * powf( x/sigma, 2.0f ) );
            filter.value[x] = val;
            sum += 2.0f * val; // this filter cell counts double, left and right side
        }

        for( x=0; x<filter.span; x++ )  filter.value[x] /= sum;
        for( x=filter.span; x<20; x++ ) filter.value[x] = 0.0f;

        err = cuMemcpyHtoD( s->cu_constant, &filter, sizeof(Filter) );
        if( err != CUDA_SUCCESS )
        {
            fprintf(stderr,"Failed to copy new Gauss filter values to CUDA device\n");
        }

        sigma_on_gpu = sigma;
    }

    return 0;
fail:
    av_buffer_unref(&out_ref);
    av_buffer_unref(&tmp_ref);
    return ret;
}

static int format_is_supported(enum AVPixelFormat fmt)
{
    int i;

    for (i = 0; i < FF_ARRAY_ELEMS(supported_formats); i++)
        if (supported_formats[i] == fmt)
        {
            return 1;
        }
    return 0;
}

static av_cold int init_processing_chain(AVFilterContext *ctx, int in_width, int in_height )
{
    CUDAGaussContext *s = ctx->priv;

    AVHWFramesContext *in_frames_ctx;

    enum AVPixelFormat in_format;
    enum AVPixelFormat out_format;
    int ret;

    /* check that we have a hw context */
    if (!ctx->inputs[0]->hw_frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on input\n");
        return AVERROR(EINVAL);
    }
    in_frames_ctx = (AVHWFramesContext*)ctx->inputs[0]->hw_frames_ctx->data;
    in_format     = in_frames_ctx->sw_format;
    out_format    = (s->format == AV_PIX_FMT_NONE) ? in_format : s->format;

    if (!format_is_supported(in_format)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported input format: %s\n",
               av_get_pix_fmt_name(in_format));
        return AVERROR(ENOSYS);
    }
    if (!format_is_supported(out_format)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported output format: %s\n",
               av_get_pix_fmt_name(out_format));
        return AVERROR(ENOSYS);
    }

    s->in_fmt = in_format;
    s->out_fmt = out_format;

    s->planes[0].width   = in_width;
    s->planes[0].height  = in_height;

    ret = init_stage(s, in_frames_ctx->device_ref);
    if (ret < 0)
        return ret;

    ctx->outputs[0]->hw_frames_ctx = av_buffer_ref(s->frames_ctx);
    if (!ctx->outputs[0]->hw_frames_ctx)
        return AVERROR(ENOMEM);

    return 0;
}

static av_cold int cudagauss_config_props(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = outlink->src->inputs[0];
    CUDAGaussContext *s  = ctx->priv;
    AVHWFramesContext     *frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVCUDADeviceContext *device_hwctx = frames_ctx->device_ctx->hwctx;
    CUcontext dummy, cuda_ctx = device_hwctx->cuda_ctx;
    CUresult err;
    size_t bytes;
    int ret;

    extern char vf_gauss_cuda_ptx[];

    err = cuCtxPushCurrent(cuda_ctx);
    if (err != CUDA_SUCCESS) {
        av_log(ctx, AV_LOG_ERROR, "Error pushing cuda context\n");
        ret = AVERROR_UNKNOWN;
        goto fail;
    }

    err = cuModuleLoadData(&s->cu_module, vf_gauss_cuda_ptx);
    if (err != CUDA_SUCCESS) {
        av_log(ctx, AV_LOG_ERROR, "Error loading module data\n");
        ret = AVERROR_UNKNOWN;
        goto fail;
    }

    if( s->pole_expr < 50 || s->edge_expr < 50 || s->right_expr < 100 || s->left_expr < 100 || s->top_expr < 100 || s->bottom_expr < 100 )
    {
        err = cuModuleGetFunction(&s->cu_func_horiz, s->cu_module, "Gauss_horiz_constrained");
        if( err != CUDA_SUCCESS ) fprintf(stderr,"Kernel Gauss_horiz_constrained not found\n");
        err = cuModuleGetFunction(&s->cu_func_vert,  s->cu_module, "Gauss_vert_constrained");
        if( err != CUDA_SUCCESS ) fprintf(stderr,"Kernel Gauss_vert_constrained not found\n");
    }
    else
    {
        err = cuModuleGetFunction(&s->cu_func_horiz, s->cu_module, "Gauss_horiz");
        if( err != CUDA_SUCCESS ) fprintf(stderr,"Kernel Gauss_horiz not found\n");
        err = cuModuleGetFunction(&s->cu_func_vert,  s->cu_module, "Gauss_vert");
        if( err != CUDA_SUCCESS ) fprintf(stderr,"Kernel Gauss_vert not found\n");
    }

    // cuModuleGetTexRef(&s->cu_tex_uchar, s->cu_module, "uchar_tex");

    err = cuModuleGetGlobal(&s->cu_constant, &bytes, s->cu_module, "filter");
    if( err == CUDA_ERROR_NOT_FOUND )
    {
        fprintf(stderr,"Cannot find constant memory named filter in CUDA device\n");
    }
    else if( err != CUDA_SUCCESS )
    {
        fprintf(stderr,"Some error finding constant memory named filter\n");
    }
    else if( bytes != sizeof(Filter) )
    {
        fprintf(stderr,"Size %d of constant memory on CUDA device is unexpected, should be %d\n", (int)bytes, (int)sizeof(Filter));
    }

    // read values as float: no flag, but must add cudaReadModeNormalizedFloat to
    //                       the texture specification
    // read values as integer: CU_TRSF_READ_AS_INTEGER
    //
    // address cells by integer index: no flag
    // address cells relative to width and height 1: CU_TRSF_NORMALIZED_COORDINATES
    //
    // filter mode can be CU_TR_FILTER_MODE_POINT or CU_TR_FILTER_MODE_LINEAR

    // cuTexRefSetFlags(s->cu_tex_uchar, CU_TRSF_NORMALIZED_COORDINATES);

    // cuTexRefSetFilterMode(s->cu_tex_uchar, CU_TR_FILTER_MODE_LINEAR);
    cuCtxPopCurrent(&dummy);

    outlink->w = inlink->w;
    outlink->h = inlink->h;
    outlink->sample_aspect_ratio = inlink->sample_aspect_ratio;

    ret = init_processing_chain(ctx, inlink->w, inlink->h);
    if (ret < 0)
        return ret;

    av_log(ctx, AV_LOG_VERBOSE, "w:%d h:%d -> filter\n",
           inlink->w, inlink->h);

    return 0;

fail:
    return ret;
}

static int call_gauss_kernel( CUDAGaussContext *s,
                              CUfunction func_horiz,
                              CUfunction func_vert,
                              // CUtexref tex,
                              uint8_t *src_dptr,
                              uint8_t *tmp_dptr,
                              uint8_t *dst_dptr,
                              int w, int h,
                              int limit_up, int limit_down, int limit_left, int limit_right,
                              int src_pitch, int dst_pitch,
                              int loop )
{
    CUdeviceptr src_devptr = (CUdeviceptr)src_dptr;
    CUdeviceptr tmp_devptr = (CUdeviceptr)tmp_dptr;
    CUdeviceptr dst_devptr = (CUdeviceptr)dst_dptr;
    void *args_horiz[] = { &src_devptr,
                           &tmp_devptr,
                           &w, &h,
                           &limit_up, &limit_down, &limit_left, &limit_right,
                           &src_pitch, &dst_pitch };
    void *args_vert[]  = { &tmp_devptr,
                           &dst_devptr,
                           &w, &h,
                           &limit_up, &limit_down, &limit_left, &limit_right,
                           &dst_pitch, &dst_pitch };
    void *args_horiz_loop[] = { &dst_devptr,
                                &tmp_devptr,
                                &w, &h,
                                &limit_up, &limit_down, &limit_left, &limit_right,
                                &dst_pitch, &dst_pitch };
    CUDA_ARRAY_DESCRIPTOR desc;

    const int block_x = 32;
    const int block_y = 16;
    const int block_z = 1;
    const int grid_x  = DIV_UP(w,  block_x);
    const int grid_y  = DIV_UP(h, block_y);
    const int grid_z  = 1;

    desc.Width       = w;
    desc.Height      = h;
    desc.NumChannels = 1;
    desc.Format      = CU_AD_FORMAT_UNSIGNED_INT8;

    // cuTexRefSetAddress2D_v3(tex, &desc, src_devptr, src_pitch);
    cuLaunchKernel(func_horiz,
                   grid_x, grid_y, grid_z,
                   block_x, block_y, block_z,
                   0, // shared mem bytes
                   0, // stream ID
                   args_horiz,
                   NULL);
    // cuTexRefSetAddress2D_v3(tex, &desc, tmp_devptr, dst_pitch);
    cuLaunchKernel(func_vert,
                   grid_x, grid_y, grid_z,
                   block_x, block_y, block_z,
                   0, // shared mem bytes
                   0, // stream ID
                   args_vert,
                   NULL);
    loop--;
    if( loop == 0 )
    {
        return 0;
    }

    while( loop > 0 )
    {
        // cuTexRefSetAddress2D_v3(tex, &desc, dst_devptr, dst_pitch);
        cuLaunchKernel(func_horiz,
                       grid_x, grid_y, grid_z,
                          block_x, block_y, block_z,
                       0, // shared mem bytes
                       0, // stream ID
                       args_horiz_loop,
                       NULL);
        // cuTexRefSetAddress2D_v3(tex, &desc, tmp_devptr, dst_pitch);
        cuLaunchKernel(func_vert,
                       grid_x, grid_y, grid_z,
                       block_x, block_y, block_z,
                       0, // shared mem bytes
                       0, // stream ID
                       args_vert,
                       NULL);
        loop--;
    }

    return 0;
}

static int gauss_frame_kernel_calls(AVFilterContext *ctx,
                            AVFrame *out, AVFrame* tmp, AVFrame *in)
{
    AVHWFramesContext *in_frames_ctx = (AVHWFramesContext*)in->hw_frames_ctx->data;
    CUDAGaussContext *s = ctx->priv;
    int      w;
    int      h;
    int      limit_up, limit_down, limit_left, limit_right;
    int      src_pitch, dst_pitch, tmp_pitch;
    uint8_t* src_data;
    uint8_t* dst_data;
    uint8_t* tmp_data;

    if( in_frames_ctx->sw_format != AV_PIX_FMT_YUV420P )
    {
        av_log(ctx, AV_LOG_ERROR,
               "Init check failed, received pixel format %s but accepting only %s\n",
               av_get_pix_fmt_name(in_frames_ctx->sw_format),
               av_get_pix_fmt_name(AV_PIX_FMT_YUV420P));
        return AVERROR(ENOSYS);
    }

    w           = in->width;
    h           = in->height;
    src_pitch   = in->linesize[0];
    src_data    = in->data[0];
    dst_pitch   = out->linesize[0];
    dst_data    = out->data[0];
    tmp_pitch   = tmp->linesize[0];
    tmp_data    = tmp->data[0];
    limit_up    = 0;
    limit_down  = h;
    limit_left  = 0;
    limit_right = w;

    if( s->edge_expr < 50 )
    {
        limit_left  = (int)( (int64_t)(w * s->edge_expr)/100 );
        limit_up    = (int)( (int64_t)(h * s->edge_expr)/100 );
        limit_right = w - limit_left;
        limit_down  = h - limit_up;
    }
    if( s->pole_expr < 50 )
    {
        limit_up    = (int)( (int64_t)(h * s->pole_expr)/100 );
        limit_down  = h - limit_up;
    }
    if( s->left_expr < 100 )
    {
        limit_left = (int)( (int64_t)(w * s->left_expr)/100 );
    }
    if( s->right_expr < 100 )
    {
        limit_right = w - (int)( (int64_t)(w * s->right_expr)/100 );
    }
    if( s->top_expr < 100 )
    {
        limit_up = (int)( (int64_t)(h * s->top_expr)/100 );
    }
    if( s->bottom_expr < 100 )
    {
        limit_down = h - (int)( (int64_t)(h * s->bottom_expr)/100 );
    }

    call_gauss_kernel(s,
                      s->cu_func_horiz,
                      s->cu_func_vert,
                      // s->cu_tex_uchar,
                      src_data,
                      tmp_data,
                      dst_data,
                      w, h,
                      limit_up, limit_down, limit_left, limit_right,
                      src_pitch,
                      dst_pitch,
                      s->loop_expr );

    src_data = src_data + src_pitch * h;
    dst_data = dst_data + dst_pitch * h;
    tmp_data = tmp_data + tmp_pitch * h;
    w /= 2;
    h /= 2;
    limit_up /= 2;
    limit_down /= 2;
    limit_left /= 2;
    limit_right /= 2;
    call_gauss_kernel(s,
                      s->cu_func_horiz,
                      s->cu_func_vert,
                      // s->cu_tex_uchar,
                      src_data,
                      tmp_data,
                      dst_data,
                      w, h,
                      limit_up, limit_down, limit_left, limit_right,
                      src_pitch/2,
                      dst_pitch/2,
                      s->loop_expr );

    src_data = src_data + src_pitch/2 * h;
    dst_data = dst_data + dst_pitch/2 * h;
    tmp_data = tmp_data + tmp_pitch/2 * h;
    call_gauss_kernel(s,
                      s->cu_func_horiz,
                      s->cu_func_vert,
                      // s->cu_tex_uchar,
                      src_data,
                      tmp_data,
                      dst_data,
                      w, h,
                      limit_up, limit_down, limit_left, limit_right,
                      src_pitch/2,
                      dst_pitch/2,
                      s->loop_expr );

    return 0;
}

static int gauss_frame(AVFilterContext *ctx, AVFrame *out, AVFrame *in)
{
    CUDAGaussContext *s = ctx->priv;
    AVFrame *src = in;
    int ret;

    ret = gauss_frame_kernel_calls(ctx, s->frame, s->tmp_frame, src);
    if (ret < 0)
        return ret;

    src = s->frame;
#if 1
    ret = av_hwframe_get_buffer(src->hw_frames_ctx, s->new_frame, 0);
    if (ret < 0)
        return ret;

    av_frame_move_ref(out, s->frame);
    av_frame_move_ref(s->frame, s->new_frame);
#else
    av_frame_move_ref(out, s->frame);

    ret = av_hwframe_get_buffer(src->hw_frames_ctx, s->frame, 0);
    if (ret < 0)
        return ret;
#endif

    ret = av_frame_copy_props(out, in);
    if (ret < 0)
        return ret;

    return 0;
}

static int cudagauss_filter_frame(AVFilterLink *link, AVFrame *in)
{
    AVFilterContext*     ctx          = link->dst;
    CUDAGaussContext*    s            = ctx->priv;
    AVFilterLink*        outlink      = ctx->outputs[0];
    AVHWFramesContext*   frames_ctx   = (AVHWFramesContext*)s->frames_ctx->data;
    AVCUDADeviceContext* device_hwctx = frames_ctx->device_ctx->hwctx;

    AVFrame *out = NULL;
    CUresult err;
    CUcontext dummy;
    int ret = 0;

    out = av_frame_alloc();
    if (!out) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    err = cuCtxPushCurrent(device_hwctx->cuda_ctx);
    if (err != CUDA_SUCCESS) {
        ret = AVERROR_UNKNOWN;
        goto fail;
    }

    ret = gauss_frame(ctx, out, in);

    cuCtxPopCurrent(&dummy);
    if (ret < 0)
        goto fail;

    // griff: I have no idea what this is all about. Pixel stretch?
    out->sample_aspect_ratio.num = in->sample_aspect_ratio.num;
    out->sample_aspect_ratio.den = in->sample_aspect_ratio.den;

    av_frame_free(&in);
    return ff_filter_frame(outlink, out);
fail:
    av_frame_free(&in);
    av_frame_free(&out);
    return ret;
}

#define OFFSET(x) offsetof(CUDAGaussContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption options[] = {
    { "edge",
      "Percentage of frame from the edges that is filtered",
      OFFSET(edge_expr),     AV_OPT_TYPE_INT64, { .i64 = 100 }, 0, 100, .flags = FLAGS },
    { "poles",
      "Percentage of frame height from the poles that are filtered (can be used on top of edge)",
      OFFSET(pole_expr),     AV_OPT_TYPE_INT64, { .i64 = 100 }, 0, 100, .flags = FLAGS },
    { "right",
      "Percentage of frame from the right side that is filtered (can be used on top of poles)",
      OFFSET(right_expr),     AV_OPT_TYPE_INT64, { .i64 = 100 }, 0, 100, .flags = FLAGS },
    { "left",
      "Percentage of frame from the left side that is filtered",
      OFFSET(left_expr),     AV_OPT_TYPE_INT64, { .i64 = 100 }, 0, 100, .flags = FLAGS },
    { "top",
      "Percentage of frame from the top that is filtered",
      OFFSET(top_expr),     AV_OPT_TYPE_INT64, { .i64 = 100 }, 0, 100, .flags = FLAGS },
    { "bottom",
      "Percentage of frame from the bottom that is filtered",
      OFFSET(bottom_expr),     AV_OPT_TYPE_INT64, { .i64 = 100 }, 0, 100, .flags = FLAGS },
    { "sigma",
      "Sigma for Gaussian filter (0-5, default 1.6)",
      OFFSET(sigma_expr),     AV_OPT_TYPE_DOUBLE, { .dbl = 1.6 }, 0.0, 5.0, .flags = FLAGS },
    { "loop",
      "Repeat the same Gaussian filtering N times (default 1)",
      OFFSET(loop_expr),      AV_OPT_TYPE_INT64 , { .i64 = 1 }, 1, 100, .flags = FLAGS },
    { NULL },
};

static const AVClass cudagauss_class = {
    .class_name = "cudagauss",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT,
};

static const AVFilterPad cudagauss_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = cudagauss_filter_frame,
    },
    { NULL }
};

static const AVFilterPad cudagauss_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = cudagauss_config_props,
    },
    { NULL }
};

AVFilter ff_vf_gauss_cuda = {
    .name      = "gauss_cuda",
    .description = NULL_IF_CONFIG_SMALL("GPU-accelerated video Gaussian filtering"),

    .init          = cudagauss_init,
    .uninit        = cudagauss_uninit,
    .query_formats = cudagauss_query_formats,

    .priv_size = sizeof(CUDAGaussContext),
    .priv_class = &cudagauss_class,

    .inputs    = cudagauss_inputs,
    .outputs   = cudagauss_outputs,

    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};

