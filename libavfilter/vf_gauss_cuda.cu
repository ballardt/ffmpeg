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

extern "C" {

struct Filter
{
    float value[20];
    int   span;
};

__constant__ Filter filter = { { 0.2493394166,
                                 0.2051010132,
                                 0.1141559035,
                                 0.0429915078,
                                 0.0109552098,
                                 0.0018889150,
                                 0.0002203727,
                                 0.0000173963 },
                               8 };

__global__ void Gauss_horiz(unsigned char* src,
                            unsigned char* dst,
                            int w, int h,
                            int,int,int,int, // unused boundaries
                            int src_pitch,
                            int dst_pitch )
{
    int xo = blockIdx.x * blockDim.x + threadIdx.x;
    int yo = blockIdx.y * blockDim.y + threadIdx.y;
#if 0
    if (yo < h && xo < w)
    {
        dst[yo*dst_pitch+xo] = src[yo*dst_pitch+xo];
    }
#else
    if (yo < h && xo < w)
    {
        float sum = 0.0f;
        for( int i=filter.span-1; i>0; i-- )
        {
            int l = max( xo-i, 0 );
            int r = min( xo+i, w-1 );
            int val  = src[yo*src_pitch+l] + src[yo*src_pitch+r];
            sum += val * filter.value[i];
        }

        sum += src[yo*src_pitch+xo] * filter.value[0];

        dst[yo*dst_pitch+xo] = (unsigned char)roundf(sum);
    }
#endif
}

__global__ void Gauss_vert( unsigned char* src,
                            unsigned char* dst,
                            int w, int h,
                            int,int,int,int, // unused boundaries
                            int src_pitch,
                            int dst_pitch )
{
    int xo = blockIdx.x * blockDim.x + threadIdx.x;
    int yo = blockIdx.y * blockDim.y + threadIdx.y;

#if 0
    if (yo < h && xo < w)
    {
        dst[yo*dst_pitch+xo] = src[yo*dst_pitch+xo];
    }
#else
    if (yo < h && xo < w)
    {
        float sum = 0.0f;
        for( int i=filter.span-1; i>0; i-- )
        {
            int u = max( yo-i, 0 );
            int d = min( yo+i, h-1 );
            int val  = src[u*src_pitch+xo] + src[d*src_pitch+xo];
            sum += val * filter.value[i];
        }

        sum += src[yo*src_pitch+xo] * filter.value[0];

        dst[yo*dst_pitch+xo] = (unsigned char)roundf(sum);
    }
#endif
}

__global__ void Gauss_horiz_constrained( unsigned char* src,
                                         unsigned char* dst,
                                         int w, int h,
                                         int limit_up,   int limit_down,
                                         int limit_left, int limit_right,
                                         int src_pitch,
                                         int dst_pitch )
{
    int xo = blockIdx.x * blockDim.x + threadIdx.x;
    int yo = blockIdx.y * blockDim.y + threadIdx.y;

    if (yo < h && xo < w)
    {
        unsigned char result;
        bool do_filter = ( yo < limit_up || yo > limit_down || xo < limit_left || xo > limit_right );

        if( do_filter )
        {
            float sum = 0.0f;
            for( int i=filter.span-1; i>0; i-- )
            {
                int l = max( xo-i, 0 );
                int r = min( xo+i, w-1 );
                int val  = src[yo*src_pitch+l] + src[yo*src_pitch+r];
                sum += val * filter.value[i];
            }

            sum += src[yo*src_pitch+xo] * filter.value[0];

            result = (unsigned char)roundf(sum);
        }
        else
        {
            result = src[yo*dst_pitch+xo];
        }
        dst[yo*dst_pitch+xo] = result;
    }
}

__global__ void Gauss_vert_constrained( unsigned char* src,
                                        unsigned char* dst,
                                        int w, int h,
                                        int limit_up,   int limit_down,
                                        int limit_left, int limit_right,
                                        int src_pitch,
                                        int dst_pitch )
{
    int xo = blockIdx.x * blockDim.x + threadIdx.x;
    int yo = blockIdx.y * blockDim.y + threadIdx.y;

    if (yo < h && xo < w)
    {
        unsigned char result;
        bool do_filter = ( yo < limit_up || yo > limit_down || xo < limit_left || xo > limit_right );

        if( do_filter )
        {
            float sum = 0.0f;
            for( int i=filter.span-1; i>0; i-- )
            {
                int u = max( yo-i, 0 );
                int d = min( yo+i, h-1 );
                int val  = src[u*src_pitch+xo] + src[d*src_pitch+xo];
                sum += val * filter.value[i];
            }

            sum += src[yo*src_pitch+xo] * filter.value[0];

            result = (unsigned char)roundf(sum);
        }
        else
        {
            result = src[yo*dst_pitch+xo];
        }
        dst[yo*dst_pitch+xo] = result;
    }
}

}

