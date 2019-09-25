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

typedef struct CUDAReprojectContext CUDAReprojectContext;

struct CUDAReprojectContext
{
    const AVClass *class;
    enum AVPixelFormat in_fmt;
    enum AVPixelFormat out_fmt;

    struct {
        int width;
        int height;
    } planes_in[3], planes_out[3];

    AVBufferRef *frames_ctx;
    AVFrame     *frame;

    AVFrame *tmp_frame;

    /**
     * Output sw format. AV_PIX_FMT_NONE for no conversion.
     */
    enum AVPixelFormat format;

    char *w_expr;               ///< width  expression string
    char *h_expr;               ///< height expression string
    char *type_expr;            ///< type   epxression string: floor, ceil or face

    CUcontext   cu_ctx;
    CUevent     cu_event;
    CUmodule    cu_module;
    CUfunction  cu_func_uchar;
    CUtexref    cu_tex_uchar;

    CUdeviceptr srcBuffer;
    CUdeviceptr dstBuffer;
};

static av_cold int cudareproject_init(AVFilterContext *ctx)
{
    CUDAReprojectContext *s = ctx->priv;

    s->format = AV_PIX_FMT_NONE;
    s->frame = av_frame_alloc();
    if (!s->frame)
        return AVERROR(ENOMEM);

    s->tmp_frame = av_frame_alloc();
    if (!s->tmp_frame)
        return AVERROR(ENOMEM);

    return 0;
}

static av_cold void cudareproject_uninit(AVFilterContext *ctx)
{
    CUDAReprojectContext *s = ctx->priv;

    av_frame_free(&s->frame);
    av_buffer_unref(&s->frames_ctx);
    av_frame_free(&s->tmp_frame);
}

static int cudareproject_query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pixel_formats[] = {
        AV_PIX_FMT_CUDA, AV_PIX_FMT_NONE,
    };
    AVFilterFormats *pix_fmts = ff_make_format_list(pixel_formats);

    return ff_set_common_formats(ctx, pix_fmts);
}

static av_cold int init_stage(CUDAReprojectContext *s, AVBufferRef *device_ctx)
{
    AVBufferRef *out_ref = NULL;
    AVHWFramesContext *out_ctx;
    int in_sw, in_sh, out_sw, out_sh;
    int ret, i;

    av_pix_fmt_get_chroma_sub_sample(s->in_fmt,  &in_sw,  &in_sh);
    av_pix_fmt_get_chroma_sub_sample(s->out_fmt, &out_sw, &out_sh);
    if (!s->planes_out[0].width) {
        s->planes_out[0].width  = s->planes_in[0].width;
        s->planes_out[0].height = s->planes_in[0].height;
    }

    for (i = 1; i < FF_ARRAY_ELEMS(s->planes_in); i++) {
        s->planes_in[i].width   = s->planes_in[0].width   >> in_sw;
        s->planes_in[i].height  = s->planes_in[0].height  >> in_sh;
        s->planes_out[i].width  = s->planes_out[0].width  >> out_sw;
        s->planes_out[i].height = s->planes_out[0].height >> out_sh;
    }

    out_ref = av_hwframe_ctx_alloc(device_ctx);
    if (!out_ref)
        return AVERROR(ENOMEM);
    out_ctx = (AVHWFramesContext*)out_ref->data;

    out_ctx->format    = AV_PIX_FMT_CUDA;
    out_ctx->sw_format = s->out_fmt;
    out_ctx->width     = FFALIGN(s->planes_out[0].width,  32);
    out_ctx->height    = FFALIGN(s->planes_out[0].height, 32);

    ret = av_hwframe_ctx_init(out_ref);
    if (ret < 0)
        goto fail;

    av_frame_unref(s->frame);
    ret = av_hwframe_get_buffer(out_ref, s->frame, 0);
    if (ret < 0)
        goto fail;

    s->frame->width  = s->planes_out[0].width;
    s->frame->height = s->planes_out[0].height;

    av_buffer_unref(&s->frames_ctx);
    s->frames_ctx = out_ref;

    return 0;
fail:
    av_buffer_unref(&out_ref);
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

static av_cold int init_processing_chain(AVFilterContext *ctx, int in_width, int in_height,
                                         int out_width, int out_height)
{
    CUDAReprojectContext *s = ctx->priv;

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

    s->planes_in[0].width   = in_width;
    s->planes_in[0].height  = in_height;
    s->planes_out[0].width  = out_width;
    s->planes_out[0].height = out_height;

    ret = init_stage(s, in_frames_ctx->device_ref);
    if (ret < 0)
        return ret;

    ctx->outputs[0]->hw_frames_ctx = av_buffer_ref(s->frames_ctx);
    if (!ctx->outputs[0]->hw_frames_ctx)
        return AVERROR(ENOMEM);

    return 0;
}

static av_cold int cudareproject_config_props(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = outlink->src->inputs[0];
    CUDAReprojectContext *s  = ctx->priv;
    AVHWFramesContext     *frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVCUDADeviceContext *device_hwctx = frames_ctx->device_ctx->hwctx;
    CUcontext dummy, cuda_ctx = device_hwctx->cuda_ctx;
    CUresult err;
    int w, h;
    int ret;

    extern char vf_reproject_cuda_ptx[];

    err = cuCtxPushCurrent(cuda_ctx);
    if (err != CUDA_SUCCESS) {
        av_log(ctx, AV_LOG_ERROR, "Error pushing cuda context\n");
        ret = AVERROR_UNKNOWN;
        goto fail;
    }

    err = cuModuleLoadData(&s->cu_module, vf_reproject_cuda_ptx);
    if (err != CUDA_SUCCESS) {
        av_log(ctx, AV_LOG_ERROR, "Error loading module data\n");
        ret = AVERROR_UNKNOWN;
        goto fail;
    }

    if( strncmp( s->type_expr, "face", 5 ) == 0 )
    {
        cuModuleGetFunction(&s->cu_func_uchar, s->cu_module, "Reproject_Fisheye_Equirect_Face_uchar");
    }
    else
    {
        cuModuleGetFunction(&s->cu_func_uchar, s->cu_module, "Reproject_Fisheye_Equirect_Floor_uchar");
    }

    cuModuleGetTexRef(&s->cu_tex_uchar, s->cu_module, "uchar_tex");

    // read values as float: no flag, but must add cudaReadModeNormalizedFloat to
    //                       the texture specification
    // read values as integer: CU_TRSF_READ_AS_INTEGER
    //
    // address cells by integer index: no flag
    // address cells relative to width and height 1: CU_TRSF_NORMALIZED_COORDINATES
    //
    // filter mode can be CU_TR_FILTER_MODE_POINT or CU_TR_FILTER_MODE_LINEAR

    cuTexRefSetFlags(s->cu_tex_uchar, CU_TRSF_NORMALIZED_COORDINATES);

    cuTexRefSetFilterMode(s->cu_tex_uchar, CU_TR_FILTER_MODE_LINEAR);
    cuCtxPopCurrent(&dummy);

    if ((ret = ff_scale_eval_dimensions(s,
                                        s->w_expr, s->h_expr,
                                        inlink, outlink,
                                        &w, &h)) < 0)
        goto fail;

    if (((int64_t)h * inlink->w) > INT_MAX  ||
        ((int64_t)w * inlink->h) > INT_MAX)
        av_log(ctx, AV_LOG_ERROR, "Rescaled value for width or height is too big.\n");

    outlink->w = w;
    outlink->h = h;

    ret = init_processing_chain(ctx, inlink->w, inlink->h, w, h);
    if (ret < 0)
        return ret;

    av_log(ctx, AV_LOG_VERBOSE, "w:%d h:%d -> w:%d h:%d\n",
           inlink->w, inlink->h, outlink->w, outlink->h);

    if (inlink->sample_aspect_ratio.num) {
        outlink->sample_aspect_ratio = av_mul_q((AVRational){outlink->h*inlink->w,
                                                             outlink->w*inlink->h},
                                                inlink->sample_aspect_ratio);
    } else {
        outlink->sample_aspect_ratio = inlink->sample_aspect_ratio;
    }

    return 0;

fail:
    return ret;
}

static int call_reproject_kernel(CUDAReprojectContext *s,
                                 CUfunction func,
                                 CUtexref tex,
                                 uint8_t *src_dptr,
                                 int src_width, int src_height, int src_pitch,
                                 uint8_t *dst_dptr,
                                 int dst_width, int dst_height, int dst_pitch )
{
    CUdeviceptr src_devptr = (CUdeviceptr)src_dptr;
    CUdeviceptr dst_devptr = (CUdeviceptr)dst_dptr;
    void *args_uchar[] = { &dst_devptr,
                           &dst_width, &dst_height, &dst_pitch,
                           &src_width, &src_height };
    CUDA_ARRAY_DESCRIPTOR desc;

    const int block_x = 32;
    const int block_y = 16;
    const int block_z = 1;
    const int grid_x  = DIV_UP(dst_width,  block_x);
    const int grid_y  = DIV_UP(dst_height, block_y);
    const int grid_z  = 1;

    desc.Width       = src_width;
    desc.Height      = src_height;
    desc.NumChannels = 1;
    desc.Format      = CU_AD_FORMAT_UNSIGNED_INT8;

    cuTexRefSetAddress2D_v3(tex, &desc, src_devptr, src_pitch);
    cuLaunchKernel(func,
                   grid_x, grid_y, grid_z,
                   block_x, block_y, block_z,
                   0, // shared mem bytes
                   0, // stream ID
                   args_uchar,
                   NULL);

    return 0;
}

static int reproject_frame_kernel_calls(AVFilterContext *ctx,
                            AVFrame *out, AVFrame *in)
{
    AVHWFramesContext *in_frames_ctx = (AVHWFramesContext*)in->hw_frames_ctx->data;
    CUDAReprojectContext *s = ctx->priv;

    if( in_frames_ctx->sw_format != AV_PIX_FMT_YUV420P )
    {
        av_log(ctx, AV_LOG_ERROR,
               "Init check failed, received pixel format %s but accepting only %s\n",
               av_get_pix_fmt_name(in_frames_ctx->sw_format),
               av_get_pix_fmt_name(AV_PIX_FMT_YUV420P));
        return AVERROR(ENOSYS);
    }

    int      src_w     = in->width;
    int      src_h     = in->height;
    int      src_pitch = in->linesize[0];
    uint8_t* src_data  = in->data[0];
    int      dst_w     = out->width;
    int      dst_h     = out->height;
    int      dst_pitch = out->linesize[0];
    uint8_t* dst_data  = out->data[0];

    call_reproject_kernel(s,
                          s->cu_func_uchar,
                          s->cu_tex_uchar,
                          src_data, src_w, src_h, src_pitch,
                          dst_data, dst_w, dst_h, dst_pitch );

    src_data = src_data + src_pitch * src_h;
    dst_data = dst_data + dst_pitch * dst_h;
    src_w /= 2;
    src_h /= 2;
    dst_w /= 2;
    dst_h /= 2;
    call_reproject_kernel(s,
                          s->cu_func_uchar,
                          s->cu_tex_uchar,
                          src_data, src_w, src_h, src_pitch/2,
                          dst_data, dst_w, dst_h, dst_pitch/2 );

    src_data = src_data + src_pitch/2 * src_h;
    dst_data = dst_data + dst_pitch/2 * dst_h;
    call_reproject_kernel(s,
                          s->cu_func_uchar,
                          s->cu_tex_uchar,
                          src_data, src_w, src_h, src_pitch/2,
                          dst_data, dst_w, dst_h, dst_pitch/2 );

    return 0;
}

static int reproject_frame(AVFilterContext *ctx, AVFrame *out, AVFrame *in)
{
    CUDAReprojectContext *s = ctx->priv;
    AVFrame *src = in;
    int ret;

    ret = reproject_frame_kernel_calls(ctx, s->frame, src);
    if (ret < 0)
        return ret;

    src = s->frame;
    ret = av_hwframe_get_buffer(src->hw_frames_ctx, s->tmp_frame, 0);
    if (ret < 0)
        return ret;

    av_frame_move_ref(out, s->frame);
    av_frame_move_ref(s->frame, s->tmp_frame);

    ret = av_frame_copy_props(out, in);
    if (ret < 0)
        return ret;

    return 0;
}

static int cudareproject_filter_frame(AVFilterLink *link, AVFrame *in)
{
    AVFilterContext              *ctx = link->dst;
    CUDAReprojectContext           *s = ctx->priv;
    AVFilterLink             *outlink = ctx->outputs[0];
    AVHWFramesContext     *frames_ctx = (AVHWFramesContext*)s->frames_ctx->data;
    AVCUDADeviceContext *device_hwctx = frames_ctx->device_ctx->hwctx;

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

    ret = reproject_frame(ctx, out, in);

    cuCtxPopCurrent(&dummy);
    if (ret < 0)
        goto fail;

    // griff: I have no idea what this is all about. Pixel stretch?
    av_reduce(&out->sample_aspect_ratio.num, &out->sample_aspect_ratio.den,
              (int64_t)in->sample_aspect_ratio.num * outlink->h * link->w,
              (int64_t)in->sample_aspect_ratio.den * outlink->w * link->h,
              INT_MAX);

    av_frame_free(&in);
    return ff_filter_frame(outlink, out);
fail:
    av_frame_free(&in);
    av_frame_free(&out);
    return ret;
}

#define OFFSET(x) offsetof(CUDAReprojectContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM)
static const AVOption options[] = {
    { "w",      "Output video width",  OFFSET(w_expr),     AV_OPT_TYPE_STRING, { .str = "iw"   }, .flags = FLAGS },
    { "h",      "Output video height", OFFSET(h_expr),     AV_OPT_TYPE_STRING, { .str = "ih"   }, .flags = FLAGS },
    { "type",   "Values: floor, ceil or face", OFFSET(type_expr), AV_OPT_TYPE_STRING, { .str = "floor"   }, .flags = FLAGS },
    { NULL },
};

static const AVClass cudareproject_class = {
    .class_name = "cudareproject",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT,
};

static const AVFilterPad cudareproject_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = cudareproject_filter_frame,
    },
    { NULL }
};

static const AVFilterPad cudareproject_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = cudareproject_config_props,
    },
    { NULL }
};

AVFilter ff_vf_reproject_cuda = {
    .name      = "reproject_cuda",
    .description = NULL_IF_CONFIG_SMALL("GPU-accelerated video reprojection"),

    .init          = cudareproject_init,
    .uninit        = cudareproject_uninit,
    .query_formats = cudareproject_query_formats,

    .priv_size = sizeof(CUDAReprojectContext),
    .priv_class = &cudareproject_class,

    .inputs    = cudareproject_inputs,
    .outputs   = cudareproject_outputs,

    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};

