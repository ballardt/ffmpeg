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
#include <math_constants.h>

extern "C" {

texture<unsigned char, cudaTextureType2D, cudaReadModeNormalizedFloat> uchar_tex;

__device__ inline void panomorpth_stretch( float& x, float& y )
{
    float len = sqrtf(x*x+y*y);

    float a      = 1.0f;
    float base   = sinf( CUDART_PI_F/2.0f * len );
    float newlen = powf( base, 3.0f );
    newlen = a * newlen - ( 1.0f - a ) * base;

    x *= newlen / len;
    y *= newlen / len;
}

__device__ inline void pix_from_angle( float& xi, float& yi )
{
    float angle = xi;
    float radius = yi;
    xi = radius * sinf( angle );
    yi = radius * cosf( angle );
}

__global__ void Reproject_Fisheye_Equirect_Floor_uchar(unsigned char *dst,
                                    int dst_width, int dst_height, int dst_pitch,
                                    int src_width, int src_height)
{
    int xo = blockIdx.x * blockDim.x + threadIdx.x;
    int yo = blockIdx.y * blockDim.y + threadIdx.y;

    if (yo < dst_height && xo < dst_width)
    {
        float xi = xo / (float)dst_width;
        float yi = yo / (float)dst_height;

        // flip the direction for xi
        xi = 1.0f - xi;

        // we keep Y unchanged
        // we stretch X
        xi = 2.0f * CUDART_PI_F * xi; // xi is now between [-PI/2 and PI/2]
        // yi unchanged - yi is between [0 and 1], where 0 is the top

        pix_from_angle( xi, yi );
        
        // This seems to be the perfect stretch factor for our lens, implying
        // our lens cuts through unit sphere at a height above the equator that
        // has only 90% of the equator's radius. That is 0.4358, which appears
        // very far above the equator. We are probably not stretching correctly ...

        // float stretch = sqrtf( 0.90f - yi*yi );

        // xi *= stretch;

        xi = ( xi + 1.0f ) / 2.0f; // xf now between [ 0 and 1 ]
        yi = ( yi + 1.0f ) / 2.0f; // xf now between [ 0 and 1 ]

        // panomorpth_stretch(xi,yi);

        float y = tex2D(uchar_tex, xi, yi);
        dst[yo*dst_pitch+xo] = (unsigned char)(y*255.0f);
    }
}

__global__ void Reproject_Fisheye_Equirect_Face_uchar(unsigned char *dst,
                                    int dst_width, int dst_height, int dst_pitch,
                                    int src_width, int src_height)
{
    int xo = blockIdx.x * blockDim.x + threadIdx.x;
    int yo = blockIdx.y * blockDim.y + threadIdx.y;

    if (yo < dst_height && xo < dst_width)
    {
        float xi = xo / (float)dst_width;
        float yi = yo / (float)dst_height;

        // we keep Y unchanged
        // we stretch X
        xi = xi * 2.0f - 1.0f; // xi is now between [-1 and 1]
        yi = yi * 2.0f - 1.0f; // yi is now between [-1 and 1]

        // stretch for a perfect half sphere covering 180 degrees
        // float stretch = sqrtf( 1.0f - yi*yi );

        // This seems to be the perfect stretch factor for our lens, implying
        // our lens cuts through unit sphere at a height above the equator that
        // has only 90% of the equator's radius. That is 0.4358, which appears
        // very far above the equator. We are probably not stretching correctly ...
        float stretch = sqrtf( 0.90f - yi*yi );

        xi *= stretch;

        // panomorpth_stretch(xi,yi);

        xi = ( xi + 1.0f ) / 2.0f; // xf now between [ 0 and 1 ]
        yi = ( yi + 1.0f ) / 2.0f; // xf now between [ 0 and 1 ]

        float y = tex2D(uchar_tex, xi, yi);
        dst[yo*dst_pitch+xo] = (unsigned char)(y*255.0f);
    }
}

}
