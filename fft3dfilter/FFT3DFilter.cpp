/*
  FFT3DFilter plugin for Avisynth 2.6 - 3D Frequency Domain filter

  Copyright(C)2004-2006 A.G.Balakhnin aka Fizick, bag@hotmail.ru, http://avisynth.org.ru

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License version 2 as published by
  the Free Software Foundation.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

  Plugin uses external FFTW library version 3 (http://www.fftw.org)
  as Windows binary DLL (compiled with gcc under MinGW by Alessio Massaro),
  which support for threads and have AMD K7 (3dNow!) support in addition to SSE/SSE2.
  It may be downloaded from ftp://ftp.fftw.org/pub/fftw/fftw3win32mingw.zip
  You must put FFTW3.DLL file from this package to some directory in path
  (for example, C:\WINNT\System32).

  The algorithm is based on the 3D IIR/3D Frequency Domain Filter from:
  MOTION PICTURE RESTORATION. by Anil Christopher Kokaram. Ph.D. Thesis. May 1993.
  http://www.mee.tcd.ie/~ack/papers/a4ackphd.ps.gz

  Version 0.1, 23 November 2004 - initial
  Version 0.2, 3 December 2004 - add beta parameter of noise margin
  Version 0.3, 21 December 2004 - add bt parameter of temporal size
  Version 0.4, 16 January 2005 - algorithm optimized for speed for bt=2 (now default),
    mode bt=3 is temporary disabled, changed default bw=bh=32, filtered region now centered.
  Version 0.5, 28 January 2005 - added YUY2 support
  Version 0.6, 29 January 2005 - added Kalman filter mode for bt=0, ratio parameter
  Version 0.7, 30 January 2005 - re-enabled Wiener filter mode with 3 frames (bt=3)
  Version 0.8, 05 February2005 - added option to sharpen, and bt=-1
  Version 0.8.1, 6 February2005 - skip sharpening of the lowest frequencies to prevent parasitic lines near border
  Version 0.8.2,  February 15, 2005 - added internal buffer to process whole frame (borders included) for any bw, bh (a little slower)
  Version 0.8.3, March 16, 2005 - fixed sharpen mode (bt=-1) for YUY2
  Version 0.8.4, April 3, 2005 - delayed FFTW3.DLL loading
  Version 0.9 - April 3,2005 - variable overlapping size
  Version 0.9.1 - April 7,2005 - some assembler 3DNow! optimization for mode bt=3
  Version 0.9.2 - April 10,2005 - some assembler 3DNow! optimization for mode bt=2,
    option measure=true is now default as more fast
  Version 0.9.3 - April 24,2005 - bug fixed for bt=2 with 3DNow;	bt=3 now default;
  modifyed sharpen to horizontal only (still experimental)
  Version 1.0 - June 22, 2005 - improved edges processing (by padding);
      added svr parameter to control vertical sharpening
  Version 1.0.1 - July 05, 2005 - fixed bug for YUY2 chroma planes
  Version 1.1 - July 8,2005 - improved sharpen mode to prevent grid artifactes and to limit sharpening,
    added parameters smin, smax; renamed parameter ratio to kratio.
  Version 1.2 - July 12, 2005 - changed parameters defaults (bw=bh=48, ow=bw/3, oh=bh/3) to prevent grid artifactes
  Version 1.3 - July 20, 2005 - added interlaced mode
  Version 1.3.1 - July 21, 2005 - fixed bug for YUY2 interlaced
  Version 1.4 - July 23, 2005 - corrected neutral level for chroma processing, added wintype to decrease grid artefactes
  Version 1.5 - July 26, 2005 - added noise pattern method and its parameters pframe, px, py, pshow, pcutoff, pfactor
  Version 1.5.1 - July 29, 2005 - fixed bug with pshow
  Version 1.5.2 - July 31, 2005 - fixed bug with Kalman mode (bt=0) for Athlon (introduced in v1.5)
  Version 1.6 - August 01, 2005 - added mode bt=4; optimized SSE version for bt=2,3
  Version 1.7 - August 29, 2005 - added SSE version for for sharpen and pattern modes bt=2,3 ; restuctured code, GPL v2
  Version 1.8 - September 6, 2005 - improved internal fft cache; added degrid=0; changed wintype=0
  Version 1.8.1 - October 26, 2005 - fixed bug with sharpen>0 AND degrid>0 for bt not equal 1.
  Version 1.8.2 - November 04, 2005 - really set default degrid=1.0 (was = 0)
  Version 1.8.3 - November 28, 2005 - fixed bug with first frame for Kalman YV12 (thanks to Tsp)
  Version 1.8.4 - November 29, 2005 - added multiplane modes plane=3,4
  Version 1.8.5 - 4 December 2005 - fixed bug with memory leakage (thanks to tsp).
  Version 1.9 - April 25, 2006 - added dehalo options; corrected sharpen mode a little;
    re-enabled 3DNow and SSE optimization for degrid=0;  added SSE optimization for bt=3,-1 with degrid>0 (faster by 15%)
  Version 1.9.1 - May 10, 2006 - added SSE optimization for bt=4 with degrid>0 (faster by 30%)
  Version 1.9.2 - September 6, 2006 - added new mode bt=5
  Version 2.0.0 - november 6, 2006 - added motion compensation mc parameter, window reorganized, multi-cpu
  Version 2.1.0 - January 17, 2007 - removed motion compensation mc parameter
  Version 2.1.1 - February 19, 2007 - fixed bug with bw not mod 4 (restored v1.9.2 window method)
  Version 2.2   - February 25, 2015 - martin53: made AviSynth 2.6 ready, FFT3dFilter2_VersionNumber function, error msgs

  Version 2.3     February 21, 2017 pinterf
                    - apply current avs+ headers
                    - 10-16 bits and 32 bit float colorspace support in AVS+
                    - Planar RGB support
                    - look for libfftw3f-3.dll first, then fftw3.dll
                    - inline asm ignored on x64 builds
                    - pre-check: if plane to process for greyscale is U and/or V return original clip
                    - auto register MT mode for avs+: MT_SERIALIZED

*/
#define VERSION_NUMBER 2.3

//#include "windows.h"
#include <avisynth.h>
#include <stdio.h>
#include <stdint.h>
#include <Windows.h>
#include "math.h"
//#include "fftw3.h" // replaced by fftwlite.h for dynamic load
#include "fftwlite.h" // added in v.0.8.4
#include "info.h"
#include <emmintrin.h>
#include <mmintrin.h> // _mm_empty
#include <algorithm>
#include <atomic>

// declarations of filtering functions:
#ifndef X86_64
// 3DNow
void ApplyWiener3D2_3DNow(fftwf_complex *outcur, fftwf_complex *outprev, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta);
void ApplyWiener3D3_3DNow(fftwf_complex *outcur, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta);
void ApplyWiener3D4_3DNow(fftwf_complex *outcur, fftwf_complex *outprev2, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta);
void ApplyKalman_3DNow(fftwf_complex *outcur, fftwf_complex *outLast, fftwf_complex *covar, fftwf_complex *covarProcess, int outwidth, int outpitch, int bh, int howmanyblocks, float covarNoiseNormed, float kratio2);
#endif
void ApplyKalman_SSE2_simd(fftwf_complex *outcur, fftwf_complex *outLast, fftwf_complex *covar, fftwf_complex *covarProcess, int outwidth, int outpitch, int bh, int howmanyblocks, float covarNoiseNormed, float kratio2);
// SSE
void ApplyWiener3D2_SSE(fftwf_complex *outcur, fftwf_complex *outprev, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta);
void ApplyWiener3D2_SSE_simd(fftwf_complex *outcur, fftwf_complex *outprev, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta);
void ApplyPattern3D2_SSE(fftwf_complex *outcur, fftwf_complex *outprev, int outwidth, int outpitch, int bh, int howmanyblocks, float * pattern3d, float beta);
void ApplyWiener3D3_SSE(fftwf_complex *outcur, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta);
void ApplyPattern3D3_SSE(fftwf_complex *outcur, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float *pattern3d, float beta);
void Sharpen_SSE(fftwf_complex *outcur, int outwidth, int outpitch, int bh, int howmanyblocks, float sharpen, float sigmaSquaredSharpenMin, float sigmaSquaredSharpenMax, float *wsharpen);
// C
void ApplyWiener2D_C(fftwf_complex *out, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta, float sharpen, float sigmaSquaredSharpenMin, float sigmaSquaredSharpenMax, float *wsharpen, float dehalo, float *wdehalo, float ht2n);
void ApplyPattern2D_C(fftwf_complex *outcur, int outwidth, int outpitch, int bh, int howmanyblocks, float pfactor, float *pattern2d0, float beta);
void ApplyWiener3D2_C(fftwf_complex *outcur, fftwf_complex *outprev, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta);
void ApplyPattern3D2_C(fftwf_complex *outcur, fftwf_complex *outprev, int outwidth, int outpitch, int bh, int howmanyblocks, float *pattern3d, float beta);
void ApplyWiener3D3_C(fftwf_complex *out, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta);
void ApplyPattern3D3_C(fftwf_complex *out, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float *pattern3d, float beta);
void ApplyWiener3D4_C(fftwf_complex *out, fftwf_complex *outprev2, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta);
void ApplyPattern3D4_C(fftwf_complex *out, fftwf_complex *outprev2, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float* pattern3d, float beta);
void ApplyWiener3D5_C(fftwf_complex *out, fftwf_complex *outprev2, fftwf_complex *outprev, fftwf_complex *outnext, fftwf_complex *outnext2, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta);
void ApplyPattern3D5_C(fftwf_complex *out, fftwf_complex *outprev2, fftwf_complex *outprev, fftwf_complex *outnext, fftwf_complex *outnext2, int outwidth, int outpitch, int bh, int howmanyblocks, float* pattern3d, float beta);
void ApplyKalmanPattern_C(fftwf_complex *outcur, fftwf_complex *outLast, fftwf_complex *covar, fftwf_complex *covarProcess, int outwidth, int outpitch, int bh, int howmanyblocks, float *covarNoiseNormed, float kratio2);
void ApplyKalman_C(fftwf_complex *outcur, fftwf_complex *outLast, fftwf_complex *covar, fftwf_complex *covarProcess, int outwidth, int outpitch, int bh, int howmanyblocks, float covarNoiseNormed, float kratio2);
void Sharpen_C(fftwf_complex *outcur, int outwidth, int outpitch, int bh, int howmanyblocks, float sharpen, float sigmaSquaredSharpenMin, float sigmaSquaredSharpenMax, float *wsharpen, float dehalo, float *wdehalo, float ht2n);
// degrid_C
void ApplyWiener2D_degrid_C(fftwf_complex *out, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta, float sharpen, float sigmaSquaredSharpenMin, float sigmaSquaredSharpenMax, float *wsharpen, float degrid, fftwf_complex *gridsample, float dehalo, float *wdehalo, float ht2n);
void ApplyWiener3D2_degrid_C(fftwf_complex *outcur, fftwf_complex *outprev, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta, float degrid, fftwf_complex *gridsample);
void ApplyWiener3D3_degrid_C(fftwf_complex *outcur, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta, float degrid, fftwf_complex *gridsample);
void ApplyWiener3D4_degrid_C(fftwf_complex *out, fftwf_complex *outprev2, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta, float degrid, fftwf_complex *gridsample);
void ApplyWiener3D5_degrid_C(fftwf_complex *out, fftwf_complex *outprev2, fftwf_complex *outprev, fftwf_complex *outnext, fftwf_complex *outnext2, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta, float degrid, fftwf_complex *gridsample);
void Sharpen_degrid_C(fftwf_complex *outcur, int outwidth, int outpitch, int bh, int howmanyblocks, float sharpen, float sigmaSquaredSharpenMin, float sigmaSquaredSharpenMax, float *wsharpen, float degrid, fftwf_complex *gridsample, float dehalo, float *wdehalo, float ht2n);
void ApplyPattern2D_degrid_C(fftwf_complex *outcur, int outwidth, int outpitch, int bh, int howmanyblocks, float pfactor, float *pattern2d0, float beta, float degrid, fftwf_complex *gridsample);
void ApplyPattern3D2_degrid_C(fftwf_complex *outcur, fftwf_complex *outprev, int outwidth, int outpitch, int bh, int howmanyblocks, float *pattern3d, float beta, float degrid, fftwf_complex *gridsample);
void ApplyPattern3D3_degrid_C(fftwf_complex *out, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float *pattern3d, float beta, float degrid, fftwf_complex *gridsample);
void ApplyPattern3D4_degrid_C(fftwf_complex *out, fftwf_complex *outprev2, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float* pattern3d, float beta, float degrid, fftwf_complex *gridsample);
void ApplyPattern3D5_degrid_C(fftwf_complex *out, fftwf_complex *outprev2, fftwf_complex *outprev, fftwf_complex *outnext, fftwf_complex *outnext2, int outwidth, int outpitch, int bh, int howmanyblocks, float* pattern3d, float beta, float degrid, fftwf_complex *gridsample);
// degrid_SSE
void ApplyWiener3D3_degrid_SSE(fftwf_complex *outcur, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta, float degrid, fftwf_complex *gridsample);
void ApplyWiener3D3_degrid_SSE_simd(fftwf_complex *outcur, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta, float degrid, fftwf_complex *gridsample);
void ApplyPattern3D3_degrid_SSE(fftwf_complex *out, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float *pattern3d, float beta, float degrid, fftwf_complex *gridsample);
void Sharpen_degrid_SSE(fftwf_complex *outcur, int outwidth, int outpitch, int bh, int howmanyblocks, float sharpen, float sigmaSquaredSharpenMin, float sigmaSquaredSharpenMax, float *wsharpen, float degrid, fftwf_complex *gridsample, float dehalo, float *wdehalo, float ht2n);
void ApplyWiener3D4_degrid_SSE(fftwf_complex *outcur, fftwf_complex *outprev2, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta, float degrid, fftwf_complex *gridsample);
void ApplyPattern3D4_degrid_SSE(fftwf_complex *outcur, fftwf_complex *outprev2, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float *pattern3d, float beta, float degrid, fftwf_complex *gridsample);
//-------------------------------------------------------------------------------------------
void ApplyWiener2D(fftwf_complex *out, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed,
  float beta, float sharpen, float sigmaSquaredSharpenMin, float sigmaSquaredSharpenMax, float *wsharpen, float dehalo, float *wdehalo, float ht2n, int CPUFlags)
{
  ApplyWiener2D_C(out, outwidth, outpitch, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, sharpen, sigmaSquaredSharpenMin, sigmaSquaredSharpenMax, wsharpen, dehalo, wdehalo, ht2n);
}
//-------------------------------------------------------------------------------------------
void ApplyWiener3D2(fftwf_complex *outcur, fftwf_complex *outprev, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta, int CPUFlags)
{
  /*
#ifndef X86_64
  if (CPUFlags & CPUF_3DNOW_EXT)
    ApplyWiener3D2_3DNow(outcur, outprev, outwidth, outpitch, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta);
  else
  if (CPUFlags & CPUF_SSE)
    ApplyWiener3D2_SSE(outcur, outprev, outwidth, outpitch, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta);
  else
#endif
  */
  if (CPUFlags & CPUF_SSE2) // 170302 simd, SSE2
    ApplyWiener3D2_SSE_simd(outcur, outprev, outwidth, outpitch, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta);
  else
    ApplyWiener3D2_C(outcur, outprev, outwidth, outpitch, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta);
}
//-------------------------------------------------------------------------------------------
void ApplyPattern3D2(fftwf_complex *outcur, fftwf_complex *outprev, int outwidth, int outpitch, int bh, int howmanyblocks, float *pattern3d, float beta, int CPUFlags)
{
#ifndef X86_64
  if (CPUFlags & CPUF_SSE)
    ApplyPattern3D2_SSE(outcur, outprev, outwidth, outpitch, bh, howmanyblocks, pattern3d, beta);
  else
#endif
    ApplyPattern3D2_C(outcur, outprev, outwidth, outpitch, bh, howmanyblocks, pattern3d, beta);
}
//-------------------------------------------------------------------------------------------
void ApplyWiener3D3(fftwf_complex *out, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta, int CPUFlags)
{
#ifndef X86_64
  if (CPUFlags & CPUF_3DNOW_EXT)
    ApplyWiener3D3_3DNow(out, outprev, outnext, outwidth, outpitch, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta);
  else 
    if (CPUFlags & CPUF_SSE)
    ApplyWiener3D3_SSE(out, outprev, outnext, outwidth, outpitch, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta);
  else
#endif
    ApplyWiener3D3_C(out, outprev, outnext, outwidth, outpitch, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta);
}
//-------------------------------------------------------------------------------------------
void ApplyWiener3D3_degrid(fftwf_complex *out, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta, float degrid, fftwf_complex *gridsample, int CPUFlags)
{
#ifndef X86_64
  if (CPUFlags & CPUF_SSE)
    ApplyWiener3D3_degrid_SSE_simd(out, outprev, outnext, outwidth, outpitch, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, degrid, gridsample);
  else
#else
  if (CPUFlags & CPUF_SSE2)
    ApplyWiener3D3_degrid_SSE_simd(out, outprev, outnext, outwidth, outpitch, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, degrid, gridsample);
  else
#endif
    ApplyWiener3D3_degrid_C(out, outprev, outnext, outwidth, outpitch, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, degrid, gridsample);
}
//-------------------------------------------------------------------------------------------
void ApplyWiener3D4_degrid(fftwf_complex *out, fftwf_complex *outprev2, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta, float degrid, fftwf_complex *gridsample, int CPUFlags)
{
#ifndef X86_64
  if (CPUFlags & CPUF_SSE)
    ApplyWiener3D4_degrid_SSE(out, outprev2, outprev, outnext, outwidth, outpitch, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, degrid, gridsample);
  else
#endif
    ApplyWiener3D4_degrid_C(out, outprev2, outprev, outnext, outwidth, outpitch, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, degrid, gridsample);
}
//-------------------------------------------------------------------------------------------
void ApplyPattern2D(fftwf_complex *outcur, int outwidth, int outpitch, int bh, int howmanyblocks, float pfactor, float *pattern2d0, float beta, int CPUFlags)
{
  ApplyPattern2D_C(outcur, outwidth, outpitch, bh, howmanyblocks, pfactor, pattern2d0, beta);
}
//-------------------------------------------------------------------------------------------
void ApplyPattern3D3(fftwf_complex *out, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float *pattern3d, float beta, int CPUFlags)
{
#ifndef X86_64
  if (CPUFlags & CPUF_SSE)
    ApplyPattern3D3_SSE(out, outprev, outnext, outwidth, outpitch, bh, howmanyblocks, pattern3d, beta);
  else
#endif
    ApplyPattern3D3_C(out, outprev, outnext, outwidth, outpitch, bh, howmanyblocks, pattern3d, beta);
}
//-------------------------------------------------------------------------------------------
void ApplyPattern3D3_degrid(fftwf_complex *out, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float *pattern3d, float beta, float degrid, fftwf_complex *gridsample, int CPUFlags)
{
#ifndef X86_64
  if (CPUFlags & CPUF_SSE)
    ApplyPattern3D3_degrid_SSE(out, outprev, outnext, outwidth, outpitch, bh, howmanyblocks, pattern3d, beta, degrid, gridsample);
  else
#endif
    ApplyPattern3D3_degrid_C(out, outprev, outnext, outwidth, outpitch, bh, howmanyblocks, pattern3d, beta, degrid, gridsample);
}
//-------------------------------------------------------------------------------------------
void ApplyPattern3D4_degrid(fftwf_complex *out, fftwf_complex *outprev2, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float *pattern3d, float beta, float degrid, fftwf_complex *gridsample, int CPUFlags)
{
#ifndef X86_64
  if (CPUFlags & CPUF_SSE)
    ApplyPattern3D4_degrid_SSE(out, outprev2, outprev, outnext, outwidth, outpitch, bh, howmanyblocks, pattern3d, beta, degrid, gridsample);
  else
#endif
    ApplyPattern3D4_degrid_C(out, outprev2, outprev, outnext, outwidth, outpitch, bh, howmanyblocks, pattern3d, beta, degrid, gridsample);
}
//-------------------------------------------------------------------------------------------
void ApplyWiener3D4(fftwf_complex *out, fftwf_complex *outprev2, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta, int CPUFlags)
{
#ifndef X86_64
  if (CPUFlags & CPUF_3DNOW_EXT)
    ApplyWiener3D4_3DNow(out, outprev2, outprev, outnext, outwidth, outpitch, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta);
  else
#endif
    ApplyWiener3D4_C(out, outprev2, outprev, outnext, outwidth, outpitch, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta);
}
//-------------------------------------------------------------------------------------------
void ApplyPattern3D4(fftwf_complex *out, fftwf_complex *outprev2, fftwf_complex *outprev, fftwf_complex *outnext, int outwidth, int outpitch, int bh, int howmanyblocks, float* pattern3d, float beta, int CPUFlags)
{
  ApplyPattern3D4_C(out, outprev2, outprev, outnext, outwidth, outpitch, bh, howmanyblocks, pattern3d, beta);
}
//-------------------------------------------------------------------------------------------
void ApplyKalmanPattern(fftwf_complex *outcur, fftwf_complex *outLast, fftwf_complex *covar, fftwf_complex *covarProcess, int outwidth, int outpitch, int bh, int howmanyblocks, float *covarNoiseNormed, float kratio2, int CPUFlags)
{
  ApplyKalmanPattern_C(outcur, outLast, covar, covarProcess, outwidth, outpitch, bh, howmanyblocks, covarNoiseNormed, kratio2);
}
//-------------------------------------------------------------------------------------------
void ApplyKalman(fftwf_complex *outcur, fftwf_complex *outLast, fftwf_complex *covar, fftwf_complex *covarProcess, int outwidth, int outpitch, int bh, int howmanyblocks, float covarNoiseNormed, float kratio2, int CPUFlags)
{
  // bt=0
  // moved to SSE2 simd (though only 8 bytes internal working mode)
  if (CPUFlags & CPUF_SSE2)
    ApplyKalman_SSE2_simd(outcur, outLast, covar, covarProcess, outwidth, outpitch, bh, howmanyblocks, covarNoiseNormed, kratio2);
  else
    ApplyKalman_C(outcur, outLast, covar, covarProcess, outwidth, outpitch, bh, howmanyblocks, covarNoiseNormed, kratio2);
}
//-------------------------------------------------------------------------------------------
void Sharpen(fftwf_complex *outcur, int outwidth, int outpitch, int bh, int howmanyblocks, float sharpen, float sigmaSquaredSharpenMin, float sigmaSquaredSharpenMax, float *wsharpen, float dehalo, float *wdehalo, float ht2n, int CPUFlags)
{
#ifndef X86_64
  if ((CPUFlags & CPUF_SSE) && dehalo == 0)
    Sharpen_SSE(outcur, outwidth, outpitch, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMin, sigmaSquaredSharpenMax, wsharpen);
  else
#endif
    Sharpen_C(outcur, outwidth, outpitch, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMin, sigmaSquaredSharpenMax, wsharpen, dehalo, wdehalo, ht2n);
}
//-------------------------------------------------------------------------------------------
void Sharpen_degrid(fftwf_complex *outcur, int outwidth, int outpitch, int bh, int howmanyblocks, float sharpen, float sigmaSquaredSharpenMin, float sigmaSquaredSharpenMax, float *wsharpen, float degrid, fftwf_complex *gridsample, float dehalo, float *wdehalo, float ht2n, int CPUFlags)
{
#ifndef X86_64
  if ((CPUFlags & CPUF_SSE))
    Sharpen_degrid_SSE(outcur, outwidth, outpitch, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMin, sigmaSquaredSharpenMax, wsharpen, degrid, gridsample, dehalo, wdehalo, ht2n);
  else
#endif
    Sharpen_degrid_C(outcur, outwidth, outpitch, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMin, sigmaSquaredSharpenMax, wsharpen, degrid, gridsample, dehalo, wdehalo, ht2n);
}
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------
void fill_complex(fftwf_complex *plane, int outsize, float realvalue, float imgvalue)
{
  // it is not fast, but called only in constructor
  int w;
  for (w = 0; w < outsize; w++) {
    plane[w][0] = realvalue;
    plane[w][1] = imgvalue;
  }
}
//-------------------------------------------------------------------
void SigmasToPattern(float sigma, float sigma2, float sigma3, float sigma4, int bh, int outwidth, int outpitch, float norm, float *pattern2d)
{
  // it is not fast, but called only in constructor
  float sigmacur;
  float ft2 = sqrt(0.5f) / 2; // frequency for sigma2
  float ft3 = sqrt(0.5f) / 4; // frequency for sigma3
  for (int h = 0; h < bh; h++)
  {
    for (int w = 0; w < outwidth; w++)
    {
      float fy = (bh - 2.0f*abs(h - bh / 2)) / bh; // normalized to 1
      float fx = (w*1.0f) / outwidth;  // normalized to 1
      float f = sqrt((fx*fx + fy*fy)*0.5f); // normalized to 1
      if (f < ft3)
      { // low frequencies
        sigmacur = sigma4 + (sigma3 - sigma4)*f / ft3;
      }
      else if (f < ft2)
      { // middle frequencies
        sigmacur = sigma3 + (sigma2 - sigma3)*(f - ft3) / (ft2 - ft3);
      }
      else
      {// high frequencies
        sigmacur = sigma + (sigma2 - sigma)*(1 - f) / (1 - ft2);
      }
      pattern2d[w] = sigmacur*sigmacur / norm;
    }
    pattern2d += outpitch;
  }
}


//-------------------------------------------------------------------------------------------
class FFT3DFilter : public GenericVideoFilter {
  // FFT3DFilter defines the name of your filter class.
  // This name is only used internally, and does not affect the name of your filter or similar.
  // This filter extends GenericVideoFilter, which incorporates basic functionality.
  // All functions present in the filter must also be present here.

  //  parameters
  float sigma; // noise level (std deviation) for high frequncies ***
  float beta; // relative noise margin for Wiener filter
  int plane; // color plane
  int bw;// block width
  int bh;// block height
  int bt;// block size  along time (mumber of frames), =0 for Kalman, >0 for Wiener
  int ow; // overlap width - v.0.9
  int oh; // overlap height - v.0.9
  float kratio; // threshold to sigma ratio for Kalman filter
  float sharpen; // sharpen factor (0 to 1 and above)
  float scutoff; // sharpen cufoff frequency (relative to max) - v1.7
  float svr; // sharpen vertical ratio (0 to 1 and above) - v.1.0
  float smin; // minimum limit for sharpen (prevent noise amplifying) - v.1.1  ***
  float smax; // maximum limit for sharpen (prevent oversharping) - v.1.1      ***
  bool measure; // fft optimal method
  bool interlaced;
  int wintype; // window type
  int pframe; // noise pattern frame number
  int px; // noise pattern window x-position
  int py; // noise pattern window y-position
  bool pshow; // show noise pattern
  float pcutoff; // pattern cutoff frequency (relative to max)
  float pfactor; // noise pattern denoise strength
  float sigma2; // noise level for middle frequencies           ***
  float sigma3; // noise level for low frequencies              ***
  float sigma4; // noise level for lowest (zero) frequencies    ***
  float degrid; // decrease grid
  float dehalo; // remove halo strength - v.1.9
  float hr; // halo radius - v1.9
  float ht; // halo threshold - v1.9
//	bool mc; // motion compensation - v2.0
  int ncpu; // number of threads - v2.0

  int multiplane; // multiplane value

  // additional parameterss
  float *in;
  fftwf_complex *out, *outprev, *outnext, *outtemp, *outprev2, *outnext2;
  fftwf_complex *outrez, *gridsample; //v1.8
  fftwf_plan plan, planinv, plan1;
  int nox, noy;
  int outwidth;
  int outpitch; //v.1.7

  int outsize;
  int howmanyblocks;

  int ndim[2];
  int inembed[2];
  int onembed[2];

  float *wanxl; // analysis
  float *wanxr;
  float *wanyl;
  float *wanyr;

  float *wsynxl; // synthesis
  float *wsynxr;
  float *wsynyl;
  float *wsynyr;

  float *wsharpen;
  float *wdehalo;

  int nlast;// frame number at last step, PF: multithread warning, used for cacheing when sequential access detected
  int btcurlast;  //v1.7 to prevent multiple Pattern2Dto3D for the same btcurrent. btcurrent can change and may differ from bt for e.g. first/last frame

  fftwf_complex *outLast, *covar, *covarProcess;
  float sigmaSquaredNoiseNormed;
  float sigmaSquaredNoiseNormed2D;
  float sigmaNoiseNormed2D;
  float sigmaMotionNormed;
  float sigmaSquaredSharpenMinNormed;
  float sigmaSquaredSharpenMaxNormed;
  float ht2n; // halo threshold squared normed
  float norm; // normalization factor

  BYTE *coverbuf; //  block buffer covering the frame without remainders (with sufficient width and heigth)
  int coverwidth;
  int coverheight;
  int coverpitch;

  int mirw; // mirror width for padding
  int mirh; // mirror height for padding

  int planeBase; // color base value (0 for luma, 128 for chroma)
  float planeBase_f; // for float

  float *mean;

  float *pwin;
  float *pattern2d;
  float *pattern3d;
  bool isPatternSet;
  float psigma;
  char *messagebuf;

  // added in v.0.9 for delayed FFTW3.DLL loading
  HINSTANCE hinstLib;
  fftwf_malloc_proc fftwf_malloc;
  fftwf_free_proc fftwf_free;
  fftwf_plan_many_dft_r2c_proc fftwf_plan_many_dft_r2c;
  fftwf_plan_many_dft_c2r_proc fftwf_plan_many_dft_c2r;
  fftwf_destroy_plan_proc fftwf_destroy_plan;
  fftwf_execute_dft_r2c_proc fftwf_execute_dft_r2c;
  fftwf_execute_dft_c2r_proc fftwf_execute_dft_c2r;
  fftwf_init_threads_proc fftwf_init_threads;
  fftwf_plan_with_nthreads_proc fftwf_plan_with_nthreads;

  int CPUFlags;

  // avs+
  int pixelsize;
  int bits_per_pixel;
  int planes[4]; // prefilled PLANAR_Y/PLANAR_U/PLANAR_V/PLANAR_A or PLANAR_G/PLANAR_B/PLANAR_R

  fftwf_complex ** cachefft; //v1.8
  int * cachewhat;//v1.8
  int cachesize;//v1.8

  int _instance_id; // debug unique id
  std::atomic<bool> reentrancy_check;

//	float *fullwinan; // disabled in v2.2.1, return to v1.9.2 method
//	float *fullwinsyn;

  //void FFT3DFilter::InitOverlapPlane(float * inp, const BYTE *srcp, int src_pitch, int planeBase);
  template<typename pixel_t>
  void do_InitOverlapPlane(float * inp, const BYTE *srcp, int src_pitch, int planeBase, float planebase_f);

  void InitOverlapPlane(float * inp, const BYTE *srcp, int src_pitch, int planeBase, float planebase_f);

  template<typename pixel_t, int bits_per_pixel>
  void do_DecodeOverlapPlane(float *in, float norm, BYTE *dstp, int dst_pitch, int planeBase, float planebase_f);

  void DecodeOverlapPlane(float *in, float norm, BYTE *dstp, int dst_pitch, int planeBase, float planebase_f);
  //	void FFT3DFilter::InitFullWin(float * inp0, float *wanxl, float *wanxr, float *wanyl, float *wanyr);
  //	void FFT3DFilter::InitOverlapPlaneWin(float * inp0, const BYTE *srcp0, int src_pitch, int planeBase, float * fullwin);

public:
  // This defines that these functions are present in your class.
  // These functions must be that same as those actually implemented.
  // Since the functions are "public" they are accessible to other classes.
  // Otherwise they can only be called from functions within the class itself.

  FFT3DFilter(PClip _child, float _sigma, float _beta, int _plane, int _bw, int _bh, int _bt, int _ow, int _oh,
    float _kratio, float _sharpen, float _scutoff, float _svr, float _smin, float _smax,
    bool _measure, bool _interlaced, int _wintype,
    int _pframe, int _px, int _py, bool _pshow, float _pcutoff, float _pfactor,
    float _sigma2, float _sigma3, float _sigma4, float _degrid,
    float _dehalo, float _hr, float _ht, int _ncpu, int _multiplane, IScriptEnvironment* env);
  // This is the constructor. It does not return any value, and is always used,
  //  when an instance of the class is created.
  // Since there is no code in this, this is the definition.

  ~FFT3DFilter();
  // The is the destructor definition. This is called when the filter is destroyed.


  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
  // This is the function that AviSynth calls to get a given frame.
  // So when this functions gets called, the filter is supposed to return frame n.

  // Auto register AVS+ mode: serialized
  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    return cachehints == CACHE_GET_MTMODE ? MT_SERIALIZED : 0;
  }

};


//-------------------------------------------------------------------

// The following is the implementation
// of the defined functions.

//Here is the acutal constructor code used
FFT3DFilter::FFT3DFilter(PClip _child, float _sigma, float _beta, int _plane, int _bw, int _bh, int _bt, int _ow, int _oh,
  float _kratio, float _sharpen, float _scutoff, float _svr, float _smin, float _smax,
  bool _measure, bool _interlaced, int _wintype,
  int _pframe, int _px, int _py, bool _pshow, float _pcutoff, float _pfactor,
  float _sigma2, float _sigma3, float _sigma4, float _degrid,
  float _dehalo, float _hr, float _ht, int _ncpu, int _multiplane, IScriptEnvironment* env) :

  GenericVideoFilter(_child), sigma(_sigma), beta(_beta), plane(_plane), bw(_bw), bh(_bh), bt(_bt), ow(_ow), oh(_oh),
  kratio(_kratio), sharpen(_sharpen), scutoff(_scutoff), svr(_svr), smin(_smin), smax(_smax),
  measure(_measure), interlaced(_interlaced), wintype(_wintype),
  pframe(_pframe), px(_px), py(_py), pshow(_pshow), pcutoff(_pcutoff), pfactor(_pfactor),
  sigma2(_sigma2), sigma3(_sigma3), sigma4(_sigma4), degrid(_degrid),
  dehalo(_dehalo), hr(_hr), ht(_ht), ncpu(_ncpu), multiplane(_multiplane) {
  // This is the implementation of the constructor.
  // The child clip (source clip) is inherited by the GenericVideoFilter,
  //  where the following variables gets defined:
  //   PClip child;   // Contains the source clip.
  //   VideoInfo vi;  // Contains videoinfo on the source clip.

  static int id = 0; _instance_id = id++;
  reentrancy_check = false;
  _RPT1(0, "FFT3DFilter.Create instance_id=%d\n", _instance_id);

  int i, j;

  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent();

  float factor;
  if (pixelsize == 1) factor = 1.0f;
  else if (pixelsize == 2) factor = float(1 << (bits_per_pixel-8));
  else // float
    factor = 1 / 255.0f;

  sigma = sigma * factor;
  sigma2 = sigma2 * factor;
  sigma3 = sigma3 * factor;
  sigma4 = sigma4 * factor;
  smin = smin * factor;
  smax = smax * factor;

  int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
  int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
  int *current_planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;
  for (i = 0; i < 4; i++)
    planes[i] = current_planes[i];

#ifndef X86_64
  _mm_empty(); // _asm emms;
#endif

  //		if (bw%2 !=0 ) env->ThrowError("FFT3DFilter: Block width must be even"); //  I forget why even, so removed in v 1.2
  //		if (bh%2 !=0 ) env->ThrowError("FFT3DFilter: Block height must be even");
  if (ow * 2 > bw) env->ThrowError("FFT3DFilter: Must not be 2*ow > bw");
  if (oh * 2 > bh) env->ThrowError("FFT3DFilter: Must not be 2*oh > bh");
  if (ow < 0) ow = bw / 3; // changed from bw/4 to bw/3 in v.1.2
  if (oh < 0) oh = bh / 3; // changed from bh/4 to bh/3 in v.1.2

  if (bt < -1 || bt >5) env->ThrowError("FFT3DFilter: bt must be -1(Sharpen), 0(Kalman), 1,2,3,4,5(Wiener)");

/*
    (Parameter bt = 1) 
      2D(spatial) Wiener filter for spectrum data.Use current frame data only.
      Reduce weak frequencies(with small power spectral density) by optimal Wiener filter with some 
      given noise value. Sharpening and denoising are simultaneous in this mode.
    (Parameter bt = 2) 
      3D Wiener filter for spectrum data.
      Add third dimension to FFT by using previous and current frame data.Reduce weak frequencies
      (with small power spectral density) by optimal Wiener filter with some given noise value.
    (Parameter bt = 3) 
      Also 3D Wiener filter for spectrum data with previous, current and next frame data.
    (Parameter bt = 4) 
      Also 3D Wiener filter for spectrum data with two previous, current and next frame data.
    (Parameter bt = 5) 
      Also 3D Wiener filter for spectrum data with two previous, current and two next frames data.
    (Parameter bt = 0) 
      Temporal Kalman filter for spectrum data.
      Use all previous frames data to get estimation of cleaned current data with optimal recursive 
      data process algorithm.
      The filter starts work with small(= 1) gain(degree of noise reducing), and than gradually(in frames sequence) increases the gain 
      if inter - frame local spectrum(noise) variation is small.
      So, Kalman filter can provide stronger noise reduction than Wiener filter.
      The Kalman filter gain is limited by some given noise value.
      The local gain(and filter work) is reset to 1 when local variation exceeds the given threshold
      (due to motion, scene change, etc).
      So, the Kalman filter output is history - dependent(on frame taken as a start filtered frame).
*/

  // plane: 0 - luma(Y), 1 - chroma U, 2 - chroma V
  // multiplanes are handled in FFT3DFilterMulti constructor: 3 - chroma planes U and V, 4 - both luma and chroma(default = 0)
  if (vi.IsPlanar()) // also for grey
  {
    int avs_plane = planes[plane];
    bool greyOrRgb = vi.IsY() || vi.IsRGB();
    int xRatioShift = (greyOrRgb) ? 0 : vi.GetPlaneWidthSubsampling(avs_plane);
    int yRatioShift = (greyOrRgb) ? 0 : vi.GetPlaneHeightSubsampling(avs_plane);

    nox = ((vi.width >> xRatioShift) - ow + (bw - ow - 1)) / (bw - ow);
    noy = ((vi.height >> yRatioShift) - oh + (bh - oh - 1)) / (bh - oh);
    /*
    if (plane == 0)
    { // Y
      nox = (vi.width - ow + (bw - ow - 1)) / (bw - ow); //removed mirrors (added below) in v.1.2
      noy = (vi.height - oh + (bh - oh - 1)) / (bh - oh);
    }
    else if (plane == 1 || plane == 2) // U,V
    {
      nox = ((vi.width >> vi.GetPlaneWidthSubsampling(1 << plane)) - ow + (bw - ow - 1)) / (bw - ow);
      noy = ((vi.height >> vi.GetPlaneHeightSubsampling(1 << plane)) - oh + (bh - oh - 1)) / (bh - oh);
    }
    */
  }
  /* handled in planar
  else if (vi.IsY8())
  {
    if (plane == 0)
    { // Y
      nox = (vi.width - ow + (bw - ow - 1)) / (bw - ow);
      noy = (vi.height - oh + (bh - oh - 1)) / (bh - oh);
    }
  }*/
  else if (vi.IsYUY2())
  {
    if (plane == 0)
    { // Y
      nox = (vi.width - ow + (bw - ow - 1)) / (bw - ow);
      noy = (vi.height - oh + (bh - oh - 1)) / (bh - oh);
    }
    else if (plane == 1 || plane == 2) // U,V
    {
      nox = (vi.width / 2 - ow + (bw - ow - 1)) / (bw - ow);
      noy = (vi.height - oh + (bh - oh - 1)) / (bh - oh);
    }
    else
      env->ThrowError("FFT3DFilter: internal plane must be 0,1,2");
  }
  else
    env->ThrowError("FFT3DFilter: video must be planar or YUY2");


  // padding by 1 block per side
  nox += 2;
  noy += 2;
  mirw = bw - ow; // set mirror size as block interval
  mirh = bh - oh;

  if (beta < 1)
    env->ThrowError("FFT3DFilter: beta must be not less 1.0");

  int istat;

  hinstLib = LoadLibrary("libfftw3f-3.dll"); // PF full original name
  if (hinstLib == NULL)
    hinstLib = LoadLibrary("fftw3.dll"); // added in v 0.8.4 for delayed loading
  if (hinstLib != NULL)
  {
    fftwf_free = (fftwf_free_proc)GetProcAddress(hinstLib, "fftwf_free");
    fftwf_malloc = (fftwf_malloc_proc)GetProcAddress(hinstLib, "fftwf_malloc");
    fftwf_plan_many_dft_r2c = (fftwf_plan_many_dft_r2c_proc)GetProcAddress(hinstLib, "fftwf_plan_many_dft_r2c");
    fftwf_plan_many_dft_c2r = (fftwf_plan_many_dft_c2r_proc)GetProcAddress(hinstLib, "fftwf_plan_many_dft_c2r");
    fftwf_destroy_plan = (fftwf_destroy_plan_proc)GetProcAddress(hinstLib, "fftwf_destroy_plan");
    fftwf_execute_dft_r2c = (fftwf_execute_dft_r2c_proc)GetProcAddress(hinstLib, "fftwf_execute_dft_r2c");
    fftwf_execute_dft_c2r = (fftwf_execute_dft_c2r_proc)GetProcAddress(hinstLib, "fftwf_execute_dft_c2r");
    fftwf_init_threads = (fftwf_init_threads_proc)GetProcAddress(hinstLib, "fftwf_init_threads");
    fftwf_plan_with_nthreads = (fftwf_plan_with_nthreads_proc)GetProcAddress(hinstLib, "fftwf_plan_with_nthreads");
    istat = fftwf_init_threads();
  }
  if (istat == 0 || hinstLib == NULL || fftwf_free == NULL || fftwf_malloc == NULL || fftwf_plan_many_dft_r2c == NULL ||
    fftwf_plan_many_dft_c2r == NULL || fftwf_destroy_plan == NULL || fftwf_execute_dft_r2c == NULL || fftwf_execute_dft_c2r == NULL)
    env->ThrowError("FFT3DFilter: libfftw3f-3.dll or fftw3.dll not found. Please put in PATH or use LoadDll() plugin");


  coverwidth = nox*(bw - ow) + ow;
  coverheight = noy*(bh - oh) + oh;
  coverpitch = ((coverwidth + 7) / 8) * 8; // align to 8 elements. Pitch is element-granularity. For byte pitch, multiply is by pixelsize
  coverbuf = (BYTE*)malloc(coverheight*coverpitch*pixelsize);

  int insize = bw*bh*nox*noy;
  in = (float *)fftwf_malloc(sizeof(float) * insize);
  outwidth = bw / 2 + 1; // width (pitch) of complex fft block
  outpitch = ((outwidth + 1) / 2) * 2; // must be even for SSE - v1.7
  outsize = outpitch*bh*nox*noy; // replace outwidth to outpitch here and below in v1.7
//	out = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * outsize);
//	if (bt >= 2)
//		outprev = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * outsize);
//	if (bt >= 3)
//		outnext = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * outsize);
//	if (bt >= 4)
//		outprev2 = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * outsize);
  if (bt == 0) // Kalman
  {
    outLast = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * outsize);
    covar = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * outsize);
    covarProcess = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * outsize);
  }
  outrez = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * outsize); //v1.8
  gridsample = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * outsize); //v1.8

  // fft cache - added in v1.8
  cachesize = bt + 2;
  cachewhat = (int *)malloc(sizeof(int)*cachesize);
  cachefft = (fftwf_complex **)fftwf_malloc(sizeof(fftwf_complex *)*cachesize);
  for (i = 0; i < cachesize; i++)
  {
    cachefft[i] = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * outsize);
    cachewhat[i] = -1; // init as notexistant
  }


  int planFlags;
  // use FFTW_ESTIMATE or FFTW_MEASURE (more optimal plan, but with time calculation at load stage)
  if (measure)
    planFlags = FFTW_MEASURE;
  else
    planFlags = FFTW_ESTIMATE;

  int rank = 2; // 2d
  ndim[0] = bh; // size of block along height
  ndim[1] = bw; // size of block along width
  int istride = 1;
  int ostride = 1;
  int idist = bw*bh;
  int odist = outpitch*bh;//  v1.7 (was outwidth)
  inembed[0] = bh;
  inembed[1] = bw;
  onembed[0] = bh;
  onembed[1] = outpitch;//  v1.7 (was outwidth)
  howmanyblocks = nox*noy;

  //	*inembed = NULL;
  //	*onembed = NULL;

  fftwf_plan_with_nthreads(ncpu);

  plan = fftwf_plan_many_dft_r2c(rank, ndim, howmanyblocks,
    in, inembed, istride, idist, outrez, onembed, ostride, odist, planFlags);
  if (plan == NULL)
    env->ThrowError("FFT3DFilter: FFTW plan error");

  planinv = fftwf_plan_many_dft_c2r(rank, ndim, howmanyblocks,
    outrez, onembed, ostride, odist, in, inembed, istride, idist, planFlags);
  if (planinv == NULL)
    env->ThrowError("FFT3DFilter: FFTW plan error");

  fftwf_plan_with_nthreads(1);

  wanxl = (float*)malloc(ow * sizeof(float));
  wanxr = (float*)malloc(ow * sizeof(float));
  wanyl = (float*)malloc(oh * sizeof(float));
  wanyr = (float*)malloc(oh * sizeof(float));

  wsynxl = (float*)malloc(ow * sizeof(float));
  wsynxr = (float*)malloc(ow * sizeof(float));
  wsynyl = (float*)malloc(oh * sizeof(float));
  wsynyr = (float*)malloc(oh * sizeof(float));

  wsharpen = (float*)fftwf_malloc(bh*outpitch * sizeof(float));
  wdehalo = (float*)fftwf_malloc(bh*outpitch * sizeof(float));

  // define analysis and synthesis windows
  // combining window (analize mult by synthesis) is raised cosine (Hanning)

  float pi = 3.1415926535897932384626433832795f;
  if (wintype == 0) // window type
  { // , used in all version up to 1.3
    // half-cosine, the same for analysis and synthesis
    // define analysis windows
    for (i = 0; i < ow; i++)
    {
      wanxl[i] = cosf(pi*(i - ow + 0.5f) / (ow * 2)); // left analize window (half-cosine)
      wanxr[i] = cosf(pi*(i + 0.5f) / (ow * 2)); // right analize window (half-cosine)
    }
    for (i = 0; i < oh; i++)
    {
      wanyl[i] = cosf(pi*(i - oh + 0.5f) / (oh * 2));
      wanyr[i] = cosf(pi*(i + 0.5f) / (oh * 2));
    }
    // use the same windows for synthesis too.
    for (i = 0; i < ow; i++)
    {
      wsynxl[i] = wanxl[i]; // left  window (half-cosine)

      wsynxr[i] = wanxr[i]; // right  window (half-cosine)
    }
    for (i = 0; i < oh; i++)
    {
      wsynyl[i] = wanyl[i];
      wsynyr[i] = wanyr[i];
    }
  }
  else if (wintype == 1) // added in v.1.4
  {
    // define analysis windows as more flat (to decrease grid)
    for (i = 0; i < ow; i++)
    {
      wanxl[i] = sqrt(cosf(pi*(i - ow + 0.5f) / (ow * 2)));
      wanxr[i] = sqrt(cosf(pi*(i + 0.5f) / (oh * 2)));
    }
    for (i = 0; i < oh; i++)
    {
      wanyl[i] = sqrt(cosf(pi*(i - oh + 0.5f) / (oh * 2)));
      wanyr[i] = sqrt(cosf(pi*(i + 0.5f) / (oh * 2)));
    }
    // define synthesis as supplenent to rised cosine (Hanning)
    for (i = 0; i < ow; i++)
    {
      wsynxl[i] = wanxl[i] * wanxl[i] * wanxl[i]; // left window
      wsynxr[i] = wanxr[i] * wanxr[i] * wanxr[i]; // right window
    }
    for (i = 0; i < oh; i++)
    {
      wsynyl[i] = wanyl[i] * wanyl[i] * wanyl[i];
      wsynyr[i] = wanyr[i] * wanyr[i] * wanyr[i];
    }
  }
  else //  (wintype==2) - added in v.1.4
  {
    // define analysis windows as flat (to prevent grid)
    for (i = 0; i < ow; i++)
    {
      wanxl[i] = 1;
      wanxr[i] = 1;
    }
    for (i = 0; i < oh; i++)
    {
      wanyl[i] = 1;
      wanyr[i] = 1;
    }
    // define synthesis as rised cosine (Hanning)
    for (i = 0; i < ow; i++)
    {
      wsynxl[i] = cosf(pi*(i - ow + 0.5f) / (ow * 2));
      wsynxl[i] = wsynxl[i] * wsynxl[i];// left window (rised cosine)
      wsynxr[i] = cosf(pi*(i + 0.5f) / (ow * 2));
      wsynxr[i] = wsynxr[i] * wsynxr[i]; // right window (falled cosine)
    }
    for (i = 0; i < oh; i++)
    {
      wsynyl[i] = cosf(pi*(i - oh + 0.5f) / (oh * 2));
      wsynyl[i] = wsynyl[i] * wsynyl[i];
      wsynyr[i] = cosf(pi*(i + 0.5f) / (oh * 2));
      wsynyr[i] = wsynyr[i] * wsynyr[i];
    }
  }

  // window for sharpen
  for (j = 0; j < bh; j++)
  {
    int dj = j;
    if (j >= bh / 2)
      dj = bh - j;
    float d2v = float(dj*dj)*(svr*svr) / ((bh / 2)*(bh / 2)); // v1.7
    for (i = 0; i < outwidth; i++)
    {
      float d2 = d2v + float(i*i) / ((bw / 2)*(bw / 2)); // distance_2 - v1.7
      wsharpen[i] = 1 - exp(-d2 / (2 * scutoff*scutoff));
    }
    wsharpen += outpitch;
  }
  wsharpen -= outpitch*bh; // restore pointer

  // window for dehalo - added in v1.9
  float wmax = 0;
  for (j = 0; j < bh; j++)
  {
    int dj = j;
    if (j >= bh / 2)
      dj = bh - j;
    float d2v = float(dj*dj)*(svr*svr) / ((bh / 2)*(bh / 2));
    for (i = 0; i < outwidth; i++)
    {
      float d2 = d2v + float(i*i) / ((bw / 2)*(bw / 2)); // squared distance in frequency domain
      float d1 = sqrt(d2);
      wdehalo[i] = exp(-0.7f*d2*hr*hr) - exp(-d2*hr*hr); // some window with max around 1/hr, small at low and high frequencies
      if (wdehalo[i] > wmax) wmax = wdehalo[i]; // for normalization
    }
    wdehalo += outpitch;
  }
  wdehalo -= outpitch*bh; // restore pointer

  for (j = 0; j < bh; j++)
  {
    for (i = 0; i < outwidth; i++)
    {
      wdehalo[i] /= wmax; // normalize
    }
    wdehalo += outpitch;
  }
  wdehalo -= outpitch*bh; // restore pointer

  // init nlast
  nlast = -999; // init as nonexistant
  btcurlast = -999; // init as nonexistant

  norm = 1.0f / (bw*bh); // do not forget set FFT normalization factor

  sigmaSquaredNoiseNormed2D = sigma*sigma / norm;
  sigmaNoiseNormed2D = sigma / sqrtf(norm);
  sigmaMotionNormed = sigma*kratio / sqrtf(norm);
  sigmaSquaredSharpenMinNormed = smin*smin / norm;
  sigmaSquaredSharpenMaxNormed = smax*smax / norm;
  ht2n = ht*ht / norm; // halo threshold squared and normed - v1.9

  // init Kalman
  if (bt == 0) // Kalman
  {
    fill_complex(outLast, outsize, 0, 0);
    fill_complex(covar, outsize, sigmaSquaredNoiseNormed2D, sigmaSquaredNoiseNormed2D); // fixed bug in v.1.1
    fill_complex(covarProcess, outsize, sigmaSquaredNoiseNormed2D, sigmaSquaredNoiseNormed2D);// fixed bug in v.1.1
  }

  CPUFlags = env->GetCPUFlags(); //re-enabled in v.1.9
  mean = (float*)malloc(nox*noy * sizeof(float));

  pwin = (float*)malloc(bh*outpitch * sizeof(float)); // pattern window array

  float fw2, fh2;
  for (j = 0; j < bh; j++)
  {
    if (j < bh / 2)
      fh2 = (j*2.0f / bh)*(j*2.0f / bh);
    else
      fh2 = ((bh - 1 - j)*2.0f / bh)*((bh - 1 - j)*2.0f / bh);
    for (i = 0; i < outwidth; i++)
    {
      fw2 = (i*2.0f / bw)*(j*2.0f / bw);
      pwin[i] = (fh2 + fw2) / (fh2 + fw2 + pcutoff*pcutoff);
    }
    pwin += outpitch;
  }
  pwin -= outpitch*bh; // restore pointer

  pattern2d = (float*)fftwf_malloc(bh*outpitch * sizeof(float)); // noise pattern window array
  pattern3d = (float*)fftwf_malloc(bh*outpitch * sizeof(float)); // noise pattern window array

  if ((sigma2 != sigma || sigma3 != sigma || sigma4 != sigma) && pfactor == 0)
  {// we have different sigmas, so create pattern from sigmas
    SigmasToPattern(sigma, sigma2, sigma3, sigma4, bh, outwidth, outpitch, norm, pattern2d);
    isPatternSet = true;
    pfactor = 1;
  }
  else
  {
    isPatternSet = false; // pattern must be estimated
  }

  // prepare  window compensation array gridsample
  // allocate large array for simplicity :)
  // but use one block only for speed
  // Attention: other block could be the same, but we do not calculate them!
  plan1 = fftwf_plan_many_dft_r2c(rank, ndim, 1,
    in, inembed, istride, idist, outrez, onembed, ostride, odist, planFlags); // 1 block

  // avs+
  switch (pixelsize) {
  case 1: memset(coverbuf, 255, coverheight*coverpitch); 
    break;
  case 2: std::fill_n((uint16_t *)coverbuf, coverheight*coverpitch, (1 << bits_per_pixel) - 1); 
    break; // 255 
  case 4: std::fill_n((float *)coverbuf, coverheight*coverpitch, 1.0f); 
    break; // 255 
  }
  FFT3DFilter::InitOverlapPlane(in, coverbuf, coverpitch, 0, 0.0f);
  // make FFT 2D
  fftwf_execute_dft_r2c(plan1, in, gridsample);

  messagebuf = (char *)malloc(80); //1.8.5

//	fullwinan = (float *)fftwf_malloc(sizeof(float) * insize);
//	FFT3DFilter::InitFullWin(fullwinan, wanxl, wanxr, wanyl, wanyr);
//	fullwinsyn = (float *)fftwf_malloc(sizeof(float) * insize);
//	FFT3DFilter::InitFullWin(fullwinsyn, wsynxl, wsynxr, wsynyl, wsynyr);

}
//-------------------------------------------------------------------------------------------

// This is where any actual destructor code used goes
FFT3DFilter::~FFT3DFilter() {
  // This is where you can deallocate any memory you might have used.
  fftwf_destroy_plan(plan);
  fftwf_destroy_plan(plan1);
  fftwf_destroy_plan(planinv);
  fftwf_free(in);
  //	fftwf_free(out);
  free(wanxl);
  free(wanxr);
  free(wanyl);
  free(wanyr);
  free(wsynxl);
  free(wsynxr);
  free(wsynyl);
  free(wsynyr);
  fftwf_free(wsharpen);
  fftwf_free(wdehalo);
  free(mean);
  free(pwin);
  fftwf_free(pattern2d);
  fftwf_free(pattern3d);
  //	if (bt >= 2)
  //		fftwf_free(outprev);
  //	if (bt >= 3)
  //		fftwf_free(outnext);
  //	if (bt >= 4)
  //		fftwf_free(outprev2);
  fftwf_free(outrez);
  if (bt == 0) // Kalman
  {
    fftwf_free(outLast);
    fftwf_free(covar);
    fftwf_free(covarProcess);
  }
  free(coverbuf);
  free(cachewhat);
  for (int i = 0; i < cachesize; i++)
  {
    fftwf_free(cachefft[i]);
  }
  fftwf_free(cachefft);
  fftwf_free(gridsample); //fixed memory leakage in v1.8.5
//	fftwf_free(fullwinan);
//	fftwf_free(fullwinsyn);
//	fftwf_free(shiftedprev);
//	fftwf_free(shiftedprev2);
//	fftwf_free(shiftednext);
//	fftwf_free(shiftednext2);
//	fftwf_free(fftcorrel);
//	fftwf_free(correl);
//	free(xshifts);
//	free(yshifts);

  if (hinstLib != NULL)
    FreeLibrary(hinstLib);
  free(messagebuf); //v1.8.5
}
//-----------------------------------------------------------------------
//
template<typename pixel_t>
void PlanarPlaneToCoverbuf(const pixel_t *srcp, int src_width, int src_height, int src_pitch, pixel_t *coverbuf, int coverwidth, int coverheight, int coverpitch, int mirw, int mirh, bool interlaced, IScriptEnvironment* env)
{
  int h, w;
  int width2 = src_width + src_width + mirw + mirw - 2;
  pixel_t * coverbuf1 = coverbuf + coverpitch*mirh;

  int pixelsize = sizeof(pixel_t);

  if (!interlaced) //progressive
  {
    for (h = mirh; h < src_height + mirh; h++)
    {
      env->BitBlt((byte *)(coverbuf1 + mirw), coverpitch*pixelsize, (const byte *)srcp, src_pitch*pixelsize, src_width*pixelsize, 1); // copy line
      for (w = 0; w < mirw; w++)
      {
        coverbuf1[w] = coverbuf1[mirw + mirw - w]; // mirror left border
      }
      for (w = src_width + mirw; w < coverwidth; w++)
      {
        coverbuf1[w] = coverbuf1[width2 - w]; // mirror right border
      }
      coverbuf1 += coverpitch;
      srcp += src_pitch;
    }
  }
  else // interlaced
  {
    for (h = mirh; h < src_height / 2 + mirh; h++) // first field
    {
      env->BitBlt((byte *)(coverbuf1 + mirw), coverpitch*pixelsize, (const byte *)srcp, src_pitch*pixelsize, src_width*pixelsize, 1); // copy line
      for (w = 0; w < mirw; w++)
      {
        coverbuf1[w] = coverbuf1[mirw + mirw - w]; // mirror left border
      }
      for (w = src_width + mirw; w < coverwidth; w++)
      {
        coverbuf1[w] = coverbuf1[width2 - w]; // mirror right border
      }
      coverbuf1 += coverpitch;
      srcp += src_pitch * 2;
    }

    srcp -= src_pitch;
    for (h = src_height / 2 + mirh; h < src_height + mirh; h++) // flip second field
    {
      env->BitBlt((byte *)(coverbuf1 + mirw), coverpitch*pixelsize, (const byte *)srcp, src_pitch*pixelsize, src_width*pixelsize, 1); // copy line
      for (w = 0; w < mirw; w++)
      {
        coverbuf1[w] = coverbuf1[mirw + mirw - w]; // mirror left border
      }
      for (w = src_width + mirw; w < coverwidth; w++)
      {
        coverbuf1[w] = coverbuf1[width2 - w]; // mirror right border
      }
      coverbuf1 += coverpitch;
      srcp -= src_pitch * 2;
    }
  }

  pixel_t * pmirror = coverbuf1 - coverpitch * 2; // pointer to vertical mirror
  for (h = src_height + mirh; h < coverheight; h++)
  {
    env->BitBlt((byte *)coverbuf1, coverpitch*pixelsize, (const byte *)pmirror, coverpitch*pixelsize, coverwidth*pixelsize, 1); // mirror bottom line by line
    coverbuf1 += coverpitch;
    pmirror -= coverpitch;
  }
  coverbuf1 = coverbuf;
  pmirror = coverbuf1 + coverpitch*mirh * 2; // pointer to vertical mirror
  for (h = 0; h < mirh; h++)
  {
    env->BitBlt((byte *)coverbuf1, coverpitch*pixelsize, (const byte *)pmirror, coverpitch*pixelsize, coverwidth*pixelsize, 1); // mirror bottom line by line
    coverbuf1 += coverpitch;
    pmirror -= coverpitch;
  }
}
//-----------------------------------------------------------------------
//
template<typename pixel_t>
void CoverbufToPlanarPlane(const pixel_t *coverbuf, int coverwidth, int coverheight, int coverpitch, pixel_t *dstp, int dst_width, int dst_height, int dst_pitch, int mirw, int mirh, bool interlaced, IScriptEnvironment* env)
{
  int h;

  int pixelsize = sizeof(pixel_t);

  const pixel_t *coverbuf1 = coverbuf + coverpitch*mirh + mirw;
  if (!interlaced) // progressive
  {
    for (h = 0; h < dst_height; h++)
    {
      env->BitBlt((byte *)dstp, dst_pitch*pixelsize, (const byte *)coverbuf1, coverpitch*pixelsize, dst_width*pixelsize, 1); // copy pure frame size only
      dstp += dst_pitch;
      coverbuf1 += coverpitch;
    }
  }
  else // interlaced
  {
    for (h = 0; h < dst_height; h += 2)
    {
      env->BitBlt((byte *)dstp, dst_pitch*pixelsize, (const byte *)coverbuf1, coverpitch*pixelsize, dst_width*pixelsize, 1); // copy pure frame size only
      dstp += dst_pitch * 2;
      coverbuf1 += coverpitch;
    }
    // second field is flipped
    dstp -= dst_pitch;
    for (h = 0; h < dst_height; h += 2)
    {
      env->BitBlt((byte *)dstp, dst_pitch*pixelsize, (const byte *)coverbuf1, coverpitch*pixelsize, dst_width*pixelsize, 1); // copy pure frame size only
      dstp -= dst_pitch * 2;
      coverbuf1 += coverpitch;
    }
  }
}
//-----------------------------------------------------------------------
// not planar
void YUY2PlaneToCoverbuf(int plane, const BYTE *srcp, int src_width, int src_height, int src_pitch, BYTE *coverbuf, int coverwidth, int coverheight, int coverpitch, int mirw, int mirh, bool interlaced)
{
  int h, w;
  int src_width_plane;
  int width2;
  BYTE * coverbuf1 = coverbuf + coverpitch*mirh + mirw; // start of image (not mirrored) v.1.0.1

  if (!interlaced)
  {
    if (plane == 0) // Y
    {
      src_width_plane = src_width / 2;
      for (h = mirh; h < src_height + mirh; h++)
      {
        for (w = 0; w < src_width_plane; w++)
        {
          coverbuf1[w] = srcp[w << 1];// copy image line
        }
        coverbuf1 += coverpitch;
        srcp += src_pitch;
      }

    }
    else if (plane == 1) // U
    {
      src_width_plane = src_width / 4;
      for (h = mirh; h < src_height + mirh; h++)
      {
        for (w = 0; w < src_width_plane; w++)
        {
          coverbuf1[w] = srcp[(w << 2) + 1];// copy line
        }
        coverbuf1 += coverpitch;
        srcp += src_pitch;
      }
    }
    else if (plane == 2) // V
    {
      src_width_plane = src_width / 4;
      for (h = mirh; h < src_height + mirh; h++)
      {
        for (w = 0; w < src_width_plane; w++)
        {
          coverbuf1[w] = srcp[(w << 2) + 3];// copy line
        }
        coverbuf1 += coverpitch;
        srcp += src_pitch;
      }
    }
  }
  else // interlaced
  {
    if (plane == 0) // Y
    {
      src_width_plane = src_width / 2;
      for (h = mirh; h < src_height / 2 + mirh; h++)
      {
        for (w = 0; w < src_width_plane; w++)
        {
          coverbuf1[w] = srcp[w << 1];// copy image line
        }
        coverbuf1 += coverpitch;
        srcp += src_pitch * 2;
      }
      srcp -= src_pitch;
      for (h = mirh; h < src_height / 2 + mirh; h++)
      {
        for (w = 0; w < src_width_plane; w++)
        {
          coverbuf1[w] = srcp[w << 1];// copy image line
        }
        coverbuf1 += coverpitch;
        srcp -= src_pitch * 2;
      }

    }
    else if (plane == 1) // U
    {
      src_width_plane = src_width / 4;
      for (h = mirh; h < src_height / 2 + mirh; h++)
      {
        for (w = 0; w < src_width_plane; w++)
        {
          coverbuf1[w] = srcp[(w << 2) + 1];// copy line
        }
        coverbuf1 += coverpitch;
        srcp += src_pitch * 2;
      }
      srcp -= src_pitch;
      for (h = mirh; h < src_height / 2 + mirh; h++)
      {
        for (w = 0; w < src_width_plane; w++)
        {
          coverbuf1[w] = srcp[(w << 2) + 1];// copy line
        }
        coverbuf1 += coverpitch;
        srcp -= src_pitch * 2;
      }
    }
    else if (plane == 2) // V
    {
      src_width_plane = src_width / 4;
      for (h = mirh; h < src_height / 2 + mirh; h++)
      {
        for (w = 0; w < src_width_plane; w++)
        {
          coverbuf1[w] = srcp[(w << 2) + 3];// copy line
        }
        coverbuf1 += coverpitch;
        srcp += src_pitch * 2;
      }
      srcp -= src_pitch;
      for (h = mirh; h < src_height / 2 + mirh; h++)
      {
        for (w = 0; w < src_width_plane; w++)
        {
          coverbuf1[w] = srcp[(w << 2) + 3];// copy line
        }
        coverbuf1 += coverpitch;
        srcp -= src_pitch * 2;
      }
    }
  }

  // make mirrors
  coverbuf1 = coverbuf + coverpitch*mirh; //  start of first image line
  width2 = src_width_plane * 2 + mirw * 2 - 2; // for right position

  for (h = mirh; h < src_height + mirh; h++)
  {
    for (w = 0; w < mirw; w++)
    {
      coverbuf1[w] = coverbuf1[(mirw + mirw - w)]; // mirror left border
    }
    for (w = src_width_plane + mirw; w < coverwidth; w++)
    {
      coverbuf1[w] = coverbuf1[width2 - w]; // mirror right border
    }
    coverbuf1 += coverpitch;
    //			srcp += src_pitch;
  }
  // make bottom mirror
  BYTE * pmirror = coverbuf1 - coverpitch * 2; // pointer to vertical mirror
  for (h = src_height + mirh; h < coverheight; h++)
  {
    for (w = 0; w < coverwidth; w++)
    {
      coverbuf1[w] = pmirror[w];// copy line
    }
    coverbuf1 += coverpitch;
    pmirror -= coverpitch;
  }
  // make top mirror
  coverbuf1 = coverbuf;
  pmirror = coverbuf1 + coverpitch*mirh * 2; // pointer to vertical mirror
  for (h = 0; h < mirh; h++)
  {
    for (w = 0; w < coverwidth; w++)
    {
      coverbuf1[w] = pmirror[w];// copy line
    }
    coverbuf1 += coverpitch;
    pmirror -= coverpitch;
  }


}
//-----------------------------------------------------------------------
// not planar
void CoverbufToYUY2Plane(int plane, const BYTE *coverbuf, int coverwidth, int coverheight, int coverpitch, BYTE *dstp, int dst_width, int dst_height, int dst_pitch, int mirw, int mirh, bool interlaced)
{
  int h, w;
  int dst_width_plane;
  //	int width2;
  const BYTE *coverbuf1 = coverbuf + coverpitch*mirh + mirw;

  if (!interlaced)
  {
    if (plane == 0) // Y
    {
      dst_width_plane = dst_width / 2;
      for (h = 0; h < dst_height; h++)
      {
        for (w = 0; w < dst_width_plane; w++)
        {
          dstp[w << 1] = coverbuf1[w];// copy line
        }
        coverbuf1 += coverpitch;
        dstp += dst_pitch;
      }
    }
    else if (plane == 1) // U
    {
      dst_width_plane = dst_width / 4;
      for (h = 0; h < dst_height; h++)
      {
        for (w = 0; w < dst_width_plane; w++)
        {
          dstp[(w << 2) + 1] = coverbuf1[w];// copy line
        }
        coverbuf1 += coverpitch;
        dstp += dst_pitch;
      }
    }
    else if (plane == 2) // V
    {
      dst_width_plane = dst_width / 4;
      for (h = 0; h < dst_height; h++)
      {
        for (w = 0; w < dst_width_plane; w++)
        {
          dstp[(w << 2) + 3] = coverbuf1[w];// copy line
        }
        coverbuf1 += coverpitch;
        dstp += dst_pitch;
      }
    }
  }
  else //progressive
  {
    if (plane == 0) // Y
    {
      dst_width_plane = dst_width / 2;
      for (h = 0; h < dst_height; h += 2)
      {
        for (w = 0; w < dst_width_plane; w++)
        {
          dstp[w << 1] = coverbuf1[w];// copy line
        }
        coverbuf1 += coverpitch;
        dstp += dst_pitch * 2;
      }
      dstp -= dst_pitch;
      for (h = 0; h < dst_height; h += 2)
      {
        for (w = 0; w < dst_width_plane; w++)
        {
          dstp[w << 1] = coverbuf1[w];// copy line
        }
        coverbuf1 += coverpitch;
        dstp -= dst_pitch * 2;
      }
    }
    else if (plane == 1) // U
    {
      dst_width_plane = dst_width / 4;
      for (h = 0; h < dst_height; h += 2)
      {
        for (w = 0; w < dst_width_plane; w++)
        {
          dstp[(w << 2) + 1] = coverbuf1[w];// copy line
        }
        coverbuf1 += coverpitch;
        dstp += dst_pitch * 2;
      }
      dstp -= dst_pitch;
      for (h = 0; h < dst_height; h += 2)
      {
        for (w = 0; w < dst_width_plane; w++)
        {
          dstp[(w << 2) + 1] = coverbuf1[w];// copy line
        }
        coverbuf1 += coverpitch;
        dstp -= dst_pitch * 2;
      }
    }
    else if (plane == 2) // V
    {
      dst_width_plane = dst_width / 4;
      for (h = 0; h < dst_height; h += 2)
      {
        for (w = 0; w < dst_width_plane; w++)
        {
          dstp[(w << 2) + 3] = coverbuf1[w];// copy line
        }
        coverbuf1 += coverpitch;
        dstp += dst_pitch * 2;
      }
      dstp -= dst_pitch;
      for (h = 0; h < dst_height; h += 2)
      {
        for (w = 0; w < dst_width_plane; w++)
        {
          dstp[(w << 2) + 3] = coverbuf1[w];// copy line
        }
        coverbuf1 += coverpitch;
        dstp -= dst_pitch * 2;
      }
    }
  }
}
//-----------------------------------------------------------------------
//
void FramePlaneToCoverbuf(int plane, PVideoFrame &src, VideoInfo vi1, BYTE *coverbuf, int coverwidth, int coverheight, int coverpitch, int mirw, int mirh, bool interlaced, int bits_per_pixel, IScriptEnvironment* env)
{
  const BYTE *srcp;
  int src_width, src_height, src_pitch;
  int planarNum;

  int pixelsize = bits_per_pixel == 8 ? 1 : (bits_per_pixel == 32 ? 4 : 2);

  if (vi1.IsPlanar()/* || vi1.IsY8()*/) // was: YV12 || Y8
  {
    int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    int *planes = (vi1.IsYUV() || vi1.IsYUVA()) ? planes_y : planes_r;
    planarNum = planes[plane];

    srcp = src->GetReadPtr(planarNum);
    src_height = src->GetHeight(planarNum);
    src_width = src->GetRowSize(planarNum) / pixelsize; // real width!
    src_pitch = src->GetPitch(planarNum) / pixelsize; // same pixel_t granularity as coverXXX
    switch (bits_per_pixel)
    {
    case 8: PlanarPlaneToCoverbuf<uint8_t>(srcp, src_width, src_height, src_pitch, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, env); break;
    case 10: case 12: case 14: case 16: PlanarPlaneToCoverbuf<uint16_t>((uint16_t *)srcp, src_width, src_height, src_pitch, (uint16_t *)coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, env); break;
    case 32: PlanarPlaneToCoverbuf<float>((float *)srcp, src_width, src_height, src_pitch, (float *)coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, env); break;
    }
  }
  else // YUY2
  {
    srcp = src->GetReadPtr();
    src_height = src->GetHeight();
    src_width = src->GetRowSize();
    src_pitch = src->GetPitch();
    YUY2PlaneToCoverbuf(plane, srcp, src_width, src_height, src_pitch, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced);
  }
}
//-----------------------------------------------------------------------
//
void CoverbufToFramePlane(int plane, const BYTE *coverbuf, int coverwidth, int coverheight, int coverpitch, PVideoFrame &dst, VideoInfo vi1, int mirw, int mirh, bool interlaced, int bits_per_pixel, IScriptEnvironment* env)
{
  BYTE *dstp;
  int dst_width, dst_height, dst_pitch;
  int planarNum;

  int pixelsize = bits_per_pixel == 8 ? 1 : (bits_per_pixel == 32 ? 4 : 2);

  if (vi1.IsPlanar()/* || vi1.IsY8()*/) // was: YV12 || Y8
  {
    int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    int *planes = (vi1.IsYUV() || vi1.IsYUVA()) ? planes_y : planes_r;
    planarNum = planes[plane];

    dstp = dst->GetWritePtr(planarNum);
    dst_height = dst->GetHeight(planarNum);
    dst_width = dst->GetRowSize(planarNum) / pixelsize; // real width!;
    dst_pitch = dst->GetPitch(planarNum) / pixelsize; // same pixel_t granularity as coverXXX;
    switch (bits_per_pixel)
    {
    case 8: CoverbufToPlanarPlane<uint8_t>(coverbuf, coverwidth, coverheight, coverpitch, dstp, dst_width, dst_height, dst_pitch, mirw, mirh, interlaced, env); break;
    case 10: case 12: case 14: case 16: CoverbufToPlanarPlane<uint16_t>((uint16_t *)coverbuf, coverwidth, coverheight, coverpitch, (uint16_t *)dstp, dst_width, dst_height, dst_pitch, mirw, mirh, interlaced, env); break;
    case 32: CoverbufToPlanarPlane<float>((float *)coverbuf, coverwidth, coverheight, coverpitch, (float *)dstp, dst_width, dst_height, dst_pitch, mirw, mirh, interlaced, env); break;
    }

  }
  else // YUY2
  {
    dstp = dst->GetWritePtr();
    dst_height = dst->GetHeight();
    dst_width = dst->GetRowSize();
    dst_pitch = dst->GetPitch();
    CoverbufToYUY2Plane(plane, coverbuf, coverwidth, coverheight, coverpitch, dstp, dst_width, dst_height, dst_pitch, mirw, mirh, interlaced);
  }
}
//-----------------------------------------------------------------------
// put source bytes to float array of overlapped blocks
// use analysis windows
//

void FFT3DFilter::InitOverlapPlane(float * inp0, const BYTE *srcp0, int src_pitch, int planeBase, float planeBase_f)
{
  switch (pixelsize) {
  case 1: do_InitOverlapPlane<uint8_t>(inp0, srcp0, src_pitch, planeBase, planeBase_f); break;
  case 2: do_InitOverlapPlane<uint16_t>(inp0, srcp0, src_pitch, planeBase, planeBase_f); break;
  case 4: do_InitOverlapPlane<float>(inp0, srcp0, src_pitch, planeBase, planeBase_f); break;
  }
}

template<typename pixel_t>
void FFT3DFilter::do_InitOverlapPlane(float * inp0, const BYTE *srcp0, int src_pitch, int planeBase_i, float planeBase_f)
{
  // pitch is pixel_t granularity, can be used directly as scrp+=pitch
  int w, h;
  int ihx, ihy;
  const pixel_t *srcp = reinterpret_cast<const pixel_t *>(srcp0);// + (hrest/2)*src_pitch + wrest/2; // centered
  float ftmp;
  int xoffset = bh*bw - (bw - ow); // skip frames
  int yoffset = bw*nox*bh - bw*(bh - oh); // vertical offset of same block (overlap)

  float *inp = inp0;
  //	char debugbuf[1536];
  //	wsprintf(debugbuf,"FFT3DFilter: InitOverlapPlane");
  //	OutputDebugString(debugbuf);
  typedef std::conditional<sizeof(pixel_t) == 4, float, int>::type cast_t;
  cast_t planeBase = sizeof(pixel_t) == 4 ? cast_t(planeBase_f) : cast_t(planeBase_i); // anti warning

  ihy = 0; // first top (big non-overlapped) part
  {
    for (h = 0; h < oh; h++)
    {
      inp = inp0 + h*bw;
      for (w = 0; w < ow; w++)   // left part  (non-overlapped) row of first block
      {
        inp[w] = float(wanxl[w] * wanyl[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
      }
      for (w = ow; w < bw - ow; w++)   // left part  (non-overlapped) row of first block
      {
        inp[w] = float(wanyl[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
      }
      inp += bw - ow;
      srcp += bw - ow;
      for (ihx = 1; ihx < nox; ihx += 1) // middle horizontal blocks
      {
        for (w = 0; w < ow; w++)   // first part (overlapped) row of block
        {
          ftmp = float(wanyl[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp * wanxr[w]; // cur block
          inp[w + xoffset] = ftmp *wanxl[w];   // overlapped Copy - next block
        }
        inp += ow;
        inp += xoffset;
        srcp += ow;
        for (w = 0; w < bw - ow - ow; w++)   // center part  (non-overlapped) row of first block
        {
          inp[w] = float(wanyl[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
        }
        inp += bw - ow - ow;
        srcp += bw - ow - ow;
      }
      for (w = 0; w < ow; w++)   // last part (non-overlapped) of line of last block
      {
        inp[w] = float(wanxr[w] * wanyl[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
      }
      inp += ow;
      srcp += ow;
      srcp += (src_pitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
    }
    for (h = oh; h < bh - oh; h++)
    {
      inp = inp0 + h*bw;
      for (w = 0; w < ow; w++)   // left part  (non-overlapped) row of first block
      {
        inp[w] = float(wanxl[w] * (srcp[w] - planeBase));   // Copy each byte from source to float array
      }
      for (w = ow; w < bw - ow; w++)   // left part  (non-overlapped) row of first block
      {
        inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
      }
      inp += bw - ow;
      srcp += bw - ow;
      for (ihx = 1; ihx < nox; ihx += 1) // middle horizontal blocks
      {
        for (w = 0; w < ow; w++)   // first part (overlapped) row of block
        {
          ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp * wanxr[w]; // cur block
          inp[w + xoffset] = ftmp *wanxl[w];   // overlapped Copy - next block
        }
        inp += ow;
        inp += xoffset;
        srcp += ow;
        for (w = 0; w < bw - ow - ow; w++)   // center part  (non-overlapped) row of first block
        {
          inp[w] = float((srcp[w] - planeBase));   // Copy each byte from source to float array
        }
        inp += bw - ow - ow;
        srcp += bw - ow - ow;
      }
      for (w = 0; w < ow; w++)   // last part (non-overlapped) line of last block
      {
        inp[w] = float(wanxr[w] * (srcp[w] - planeBase));   // Copy each byte from source to float array
      }
      inp += ow;
      srcp += ow;

      srcp += (src_pitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
    }
  }

  for (ihy = 1; ihy < noy; ihy += 1) // middle vertical
  {
    for (h = 0; h < oh; h++) // top overlapped part
    {
      inp = inp0 + (ihy - 1)*(yoffset + (bh - oh)*bw) + (bh - oh)*bw + h*bw;
      for (w = 0; w < ow; w++)   // first half line of first block
      {
        ftmp = float(wanxl[w] * (srcp[w] - planeBase));
        inp[w] = ftmp*wanyr[h];   // Copy each byte from source to float array
        inp[w + yoffset] = ftmp*wanyl[h];   // y overlapped
      }
      for (w = ow; w < bw - ow; w++)   // first half line of first block
      {
        ftmp = float((srcp[w] - planeBase));
        inp[w] = ftmp*wanyr[h];   // Copy each byte from source to float array
        inp[w + yoffset] = ftmp*wanyl[h];   // y overlapped
      }
      inp += bw - ow;
      srcp += bw - ow;
      for (ihx = 1; ihx < nox; ihx++) // middle blocks
      {
        for (w = 0; w < ow; w++)   // half overlapped line of block
        {
          ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp * wanxr[w] * wanyr[h];
          inp[w + xoffset] = ftmp *wanxl[w] * wanyr[h];   // x overlapped
          inp[w + yoffset] = ftmp * wanxr[w] * wanyl[h];
          inp[w + xoffset + yoffset] = ftmp *wanxl[w] * wanyl[h];   // x overlapped
        }
        inp += ow;
        inp += xoffset;
        srcp += ow;
        for (w = 0; w < bw - ow - ow; w++)   // half non-overlapped line of block
        {
          ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp * wanyr[h];
          inp[w + yoffset] = ftmp * wanyl[h];
        }
        inp += bw - ow - ow;
        srcp += bw - ow - ow;
      }
      for (w = 0; w < ow; w++)   // last half line of last block
      {
        ftmp = float(wanxr[w] * (srcp[w] - planeBase));// Copy each byte from source to float array
        inp[w] = ftmp*wanyr[h];
        inp[w + yoffset] = ftmp*wanyl[h];
      }
      inp += ow;
      srcp += ow;

      srcp += (src_pitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
    }
    // middle  vertical nonovelapped part
    for (h = 0; h < bh - oh - oh; h++)
    {
      inp = inp0 + (ihy - 1)*(yoffset + (bh - oh)*bw) + (bh)*bw + h*bw + yoffset;
      for (w = 0; w < ow; w++)   // first half line of first block
      {
        ftmp = float(wanxl[w] * (srcp[w] - planeBase));
        inp[w] = ftmp;   // Copy each byte from source to float array
      }
      for (w = ow; w < bw - ow; w++)   // first half line of first block
      {
        ftmp = float((srcp[w] - planeBase));
        inp[w] = ftmp;   // Copy each byte from source to float array
      }
      inp += bw - ow;
      srcp += bw - ow;
      for (ihx = 1; ihx < nox; ihx++) // middle blocks
      {
        for (w = 0; w < ow; w++)   // half overlapped line of block
        {
          ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp * wanxr[w];
          inp[w + xoffset] = ftmp *wanxl[w];   // x overlapped
        }
        inp += ow;
        inp += xoffset;
        srcp += ow;
        for (w = 0; w < bw - ow - ow; w++)   // half non-overlapped line of block
        {
          ftmp = float((srcp[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp;
        }
        inp += bw - ow - ow;
        srcp += bw - ow - ow;
      }
      for (w = 0; w < ow; w++)   // last half line of last block
      {
        ftmp = float(wanxr[w] * (srcp[w] - planeBase));// Copy each byte from source to float array
        inp[w] = ftmp;
      }
      inp += ow;
      srcp += ow;

      srcp += (src_pitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
    }

  }

  ihy = noy; // last bottom  part
  {
    for (h = 0; h < oh; h++)
    {
      inp = inp0 + (ihy - 1)*(yoffset + (bh - oh)*bw) + (bh - oh)*bw + h*bw;
      for (w = 0; w < ow; w++)   // first half line of first block
      {
        ftmp = float(wanxl[w] * wanyr[h] * (srcp[w] - planeBase));
        inp[w] = ftmp;   // Copy each byte from source to float array
      }
      for (w = ow; w < bw - ow; w++)   // first half line of first block
      {
        ftmp = float(wanyr[h] * (srcp[w] - planeBase));
        inp[w] = ftmp;   // Copy each byte from source to float array
      }
      inp += bw - ow;
      srcp += bw - ow;
      for (ihx = 1; ihx < nox; ihx++) // middle blocks
      {
        for (w = 0; w < ow; w++)   // half line of block
        {
          float ftmp = float(wanyr[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
          inp[w] = ftmp * wanxr[w];
          inp[w + xoffset] = ftmp *wanxl[w];   // overlapped Copy
        }
        inp += ow;
        inp += xoffset;
        srcp += ow;
        for (w = 0; w < bw - ow - ow; w++)   // center part  (non-overlapped) row of first block
        {
          inp[w] = float(wanyr[h] * (srcp[w] - planeBase));   // Copy each byte from source to float array
        }
        inp += bw - ow - ow;
        srcp += bw - ow - ow;
      }
      for (w = 0; w < ow; w++)   // last half line of last block
      {
        ftmp = float(wanxr[w] * wanyr[h] * (srcp[w] - planeBase));
        inp[w] = ftmp;   // Copy each byte from source to float array
      }
      inp += ow;
      srcp += ow;

      srcp += (src_pitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
    }

  }
}
/*
//-----------------------------------------------------------------------
// create multiple windows for overlapped blocks
//
void FFT3DFilter::InitFullWin(float * inp0, float *wanxl, float *wanxr, float *wanyl, float *wanyr)
{
  int w,h;
  int ihx,ihy;
  float ftmp;
  int xoffset = bh*bw - (bw-ow); // skip frames
  int yoffset = bw*nox*bh - bw*(bh-oh); // vertical offset of same block (overlap)

  float *inp = inp0;
//	char debugbuf[1536];
//	wsprintf(debugbuf,"FFT3DFilter: InitOverlapPlane");
//	OutputDebugString(debugbuf);

  ihy =0; // first top (big non-overlapped) part
  {
    for (h=0; h < oh; h++)
    {
      inp = inp0 + h*bw;
      for (w = 0; w < ow; w++)   // left part  (non-overlapped) row of first block
      {
        inp[w] = float(wanxl[w]*wanyl[h]);   // Copy each byte from source to float array
      }
      for (w = ow; w < bw-ow; w++)   // left part  (non-overlapped) row of first block
      {
        inp[w] = float(wanyl[h]);   // Copy each byte from source to float array
      }
      inp += bw-ow;
      for (ihx =1; ihx < nox; ihx+=1) // middle horizontal blocks
      {
        for (w = 0; w < ow; w++)   // first part (overlapped) row of block
        {
          ftmp = float(wanyl[h]);   // Copy each byte from source to float array
          inp[w] = ftmp * wanxr[w]; // cur block
          inp[w+xoffset] = ftmp *wanxl[w];   // overlapped Copy - next block
        }
        inp += ow;
        inp += xoffset;
        for (w = 0; w < bw-ow-ow; w++)   // center part  (non-overlapped) row of first block
        {
          inp[w] = float(wanyl[h]);   // Copy each byte from source to float array
        }
        inp += bw-ow-ow;
      }
      for (w = 0; w < ow; w++)   // last part (non-overlapped) of line of last block
      {
        inp[w] = float(wanxr[w]*wanyl[h]);   // Copy each byte from source to float array
      }
      inp += ow;
    }
    for (h=oh; h < bh-oh; h++)
    {
      inp = inp0 + h*bw;
      for (w = 0; w < ow; w++)   // left part  (non-overlapped) row of first block
      {
        inp[w] = float(wanxl[w]);   // Copy each byte from source to float array
      }
      for (w = ow; w < bw-ow; w++)   // left part  (non-overlapped) row of first block
      {
        inp[w] = float(1);   // Copy each byte from source to float array
      }
      inp += bw-ow;
      for (ihx =1; ihx < nox; ihx+=1) // middle horizontal blocks
      {
        for (w = 0; w < ow; w++)   // first part (overlapped) row of block
        {
          ftmp = float(1);   // Copy each byte from source to float array
          inp[w] = ftmp * wanxr[w]; // cur block
          inp[w+xoffset] = ftmp *wanxl[w];   // overlapped Copy - next block
        }
        inp += ow;
        inp += xoffset;
        for (w = 0; w < bw-ow-ow; w++)   // center part  (non-overlapped) row of first block
        {
          inp[w] = float(1);   // Copy each byte from source to float array
        }
        inp += bw-ow-ow;
      }
      for (w = 0; w < ow; w++)   // last part (non-overlapped) line of last block
      {
        inp[w] = float(wanxr[w]);   // Copy each byte from source to float array
      }
      inp += ow;

    }
  }

  for (ihy =1; ihy < noy; ihy+=1 ) // middle vertical
  {
    for (h=0; h < oh; h++) // top overlapped part
    {
      inp = inp0 + (ihy-1)*(yoffset + (bh-oh)*bw) + (bh-oh)*bw + h*bw;
      for (w = 0; w < ow; w++)   // first half line of first block
      {
        ftmp = float(wanxl[w]);
        inp[w] = ftmp*wanyr[h];   // Copy each byte from source to float array
        inp[w+yoffset] = ftmp*wanyl[h];   // y overlapped
      }
      for (w = ow; w < bw-ow; w++)   // first half line of first block
      {
        ftmp = float(1);
        inp[w] = ftmp*wanyr[h];   // Copy each byte from source to float array
        inp[w+yoffset] = ftmp*wanyl[h];   // y overlapped
      }
      inp += bw-ow;
      for (ihx =1; ihx < nox; ihx++) // middle blocks
      {
        for (w = 0; w < ow; w++)   // half overlapped line of block
        {
          ftmp = float(1);   // Copy each byte from source to float array
          inp[w] = ftmp * wanxr[w]*wanyr[h];
          inp[w+xoffset] = ftmp *wanxl[w]*wanyr[h];   // x overlapped
          inp[w+yoffset] = ftmp * wanxr[w]*wanyl[h];
          inp[w+xoffset+yoffset] = ftmp *wanxl[w]*wanyl[h];   // x overlapped
        }
        inp += ow;
        inp += xoffset;
        for (w = 0; w < bw-ow-ow; w++)   // half non-overlapped line of block
        {
          ftmp = float(1);   // Copy each byte from source to float array
          inp[w] = ftmp * wanyr[h];
          inp[w+yoffset] = ftmp * wanyl[h];
        }
        inp +=  bw-ow-ow;
      }
      for (w = 0; w < ow; w++)   // last half line of last block
      {
        ftmp = float(wanxr[w]);// Copy each byte from source to float array
        inp[w] = ftmp*wanyr[h];
        inp[w+yoffset] = ftmp*wanyl[h];
      }
      inp += ow;
    }
    // middle  vertical nonovelapped part
    for (h=0; h < bh-oh-oh; h++)
    {
      inp = inp0 + (ihy-1)*(yoffset + (bh-oh)*bw) + (bh)*bw + h*bw +yoffset;
      for (w = 0; w < ow; w++)   // first half line of first block
      {
        ftmp = float(wanxl[w]);
        inp[w] = ftmp;   // Copy each byte from source to float array
      }
      for (w = ow; w < bw-ow; w++)   // first half line of first block
      {
        ftmp = float(1);
        inp[w] = ftmp;   // Copy each byte from source to float array
      }
      inp += bw-ow;
      for (ihx =1; ihx < nox; ihx++) // middle blocks
      {
        for (w = 0; w < ow; w++)   // half overlapped line of block
        {
          ftmp = float(1);   // Copy each byte from source to float array
          inp[w] = ftmp * wanxr[w];
          inp[w+xoffset] = ftmp *wanxl[w];   // x overlapped
        }
        inp += ow;
        inp += xoffset;
        for (w = 0; w < bw-ow-ow; w++)   // half non-overlapped line of block
        {
          ftmp = float(1);   // Copy each byte from source to float array
          inp[w] = ftmp ;
        }
        inp +=  bw-ow-ow;
      }
      for (w = 0; w < ow; w++)   // last half line of last block
      {
        ftmp = float(wanxr[w]);// Copy each byte from source to float array
        inp[w] = ftmp;
      }
      inp += ow;
    }

  }

  ihy = noy ; // last bottom  part
  {
    for (h=0; h < oh; h++)
    {
      inp = inp0 + (ihy-1)*(yoffset + (bh-oh)*bw) + (bh-oh)*bw + h*bw ;
      for (w = 0; w < ow; w++)   // first half line of first block
      {
        ftmp = float(wanxl[w]*wanyr[h]);
        inp[w] = ftmp;   // Copy each byte from source to float array
      }
      for (w = ow; w < bw-ow; w++)   // first half line of first block
      {
        ftmp = float(wanyr[h]);
        inp[w] = ftmp;   // Copy each byte from source to float array
      }
      inp += bw-ow;
      for (ihx =1; ihx < nox; ihx++) // middle blocks
      {
        for (w = 0; w < ow; w++)   // half line of block
        {
          float ftmp = float(wanyr[h]);   // Copy each byte from source to float array
          inp[w] = ftmp * wanxr[w];
          inp[w+xoffset] = ftmp *wanxl[w];   // overlapped Copy
        }
        inp += ow;
        inp += xoffset;
        for (w = 0; w < bw-ow-ow; w++)   // center part  (non-overlapped) row of first block
        {
          inp[w] = float(wanyr[h]);   // Copy each byte from source to float array
        }
        inp += bw-ow-ow;
      }
      for (w = 0; w < ow; w++)   // last half line of last block
      {
        ftmp = float(wanxr[w]*wanyr[h]);
        inp[w] = ftmp;   // Copy each byte from source to float array
      }
      inp += ow;
    }

  }
}

void FillBlock(float * inp, float * fullwin, const BYTE *srcp, int bw, int bh, int src_pitch, int planeBase)
{
  for (int h=0; h < bh; h++)
  {
    for (int w = 0; w < bw; w++)
    {
      // Copy each byte from source to float array and multiply by window
      inp[w] =  fullwin[w]*float((srcp[w]-planeBase));
    }
    inp += bw;
    fullwin += bw;
    srcp += src_pitch;
  }
}

void FillBlock_sse(float * inp, float * fullwin, const BYTE *srcp, int bw, int blockh, int src_pitch, int planeBase)
{
//	int w, h;
//	for (h=0; h < bh; h++)
//	{
//		for (w = 0; w < bw; w++)
//		{
  _asm
  {
    pxor mm7, mm7; // 0
    movd mm6, planeBase;
    cvtpi2ps xmm7, mm6; // convert 2 int to 2 float to low qword
    shufps xmm7, xmm7, 0 // form 4 doubleword
    mov esi, srcp;
    mov edx, fullwin;
    mov edi, inp;
    mov ecx, bw;
    mov ebx, blockh;
    mov eax, 0;


align 16
next4bytes:
      movd mm0, [esi]; // read 4 source bytes
      punpcklbw mm0, mm7; // to 4 words
      movq mm1, mm0; // copy
      punpcklbw mm0, mm7; // first 2 bytes to 2 integer
      punpckhbw mm1, mm7; // second 2 bytes to 2 integer
      cvtpi2ps xmm0, mm0; // convert 2 int to 2 float to low qword
      cvtpi2ps xmm1, mm1; // convert 2 int to 2 float to low qword
      shufps xmm0, xmm1, 4+64; // form 4 doubleword
      subps xmm0, xmm7; // subtract planebase
      movaps xmm1, [edx]; // 4 fullwin
      mulps xmm0, xmm1;
      movaps [edi], xmm0;
      add eax, 4;
      cmp eax, ecx;
      jg nextline;
      add edx, 16;
      add edi, 16;
      add esi, 4;
      jmp next4bytes
nextline:
      dec ebx;
      jz finish;
      mov eax, src_pitch;
      add esi, eax;
      sub esi, ecx;
      mov eax, 0;
      jmp next4bytes

//			inp[w] =  fullwin[w]*float((srcp[w]-planeBase));
      // Copy each byte from source to float array and multiply by window
//		}
//		inp += bw;
//		fullwin += bw;
//		srcp += src_pitch;
finish:
  }
}

//-----------------------------------------------------------------------
// put source bytes to float array of overlapped blocks
// use analysis windows
//
void FFT3DFilter::InitOverlapPlaneWin(float * inp, const BYTE *srcp, int src_pitch, int planeBase, float * fullwin)
{
  int ihx,ihy;

  int bwbh = bw*bh;

    if ((CPUFlags & CPUF_SSE) && (bw%4 == 0) && (bwbh%16 == 0))
    {
        for (ihy =0; ihy < noy; ihy+=1 ) //  vertical
        {
            for (ihx =0; ihx < nox; ihx++) //  blocks
            {
                FillBlock_sse(inp, fullwin, srcp, bw, bh, src_pitch, planeBase);
                inp += bwbh;
                fullwin += bwbh;
                srcp += bw-ow;
            }
            srcp += (bh-oh)*src_pitch - nox*(bw-ow);
        }
        _asm emms;
  }
  else
    {
        for (ihy =0; ihy < noy; ihy+=1 ) //  vertical
        {
            for (ihx =0; ihx < nox; ihx++) //  blocks
            {
                FillBlock(inp, fullwin, srcp, bw, bh, src_pitch, planeBase);
                inp += bwbh;
                fullwin += bwbh;
                srcp += bw-ow;
            }
            srcp += (bh-oh)*src_pitch - nox*(bw-ow);
        }
    }
}
*/
//
//-----------------------------------------------------------------------------------------
// make destination frame plane from overlaped blocks
// use synthesis windows wsynxl, wsynxr, wsynyl, wsynyr
void FFT3DFilter::DecodeOverlapPlane(float *inp0, float norm, BYTE *dstp0, int dst_pitch, int planeBase, float planeBase_f)
{
  switch (bits_per_pixel) {
  case 8: do_DecodeOverlapPlane<uint8_t,8>(inp0, norm, dstp0, dst_pitch, planeBase, planeBase_f); break;
  case 10: do_DecodeOverlapPlane<uint16_t,10>(inp0, norm, dstp0, dst_pitch, planeBase, planeBase_f); break;
  case 12: do_DecodeOverlapPlane<uint16_t, 12>(inp0, norm, dstp0, dst_pitch, planeBase, planeBase_f); break;
  case 14: do_DecodeOverlapPlane<uint16_t, 14>(inp0, norm, dstp0, dst_pitch, planeBase, planeBase_f); break;
  case 16: do_DecodeOverlapPlane<uint16_t, 16>(inp0, norm, dstp0, dst_pitch, planeBase, planeBase_f); break;
  case 32: do_DecodeOverlapPlane<float,0>(inp0, norm, dstp0, dst_pitch, planeBase, planeBase_f); break;
  }
}

template<typename pixel_t, int bits_per_pixel>
void FFT3DFilter::do_DecodeOverlapPlane(float *inp0, float norm, BYTE *dstp0, int dst_pitch, int planeBase_i, float planeBase_f)
{
  int w, h;
  int ihx, ihy;
  pixel_t *dstp = reinterpret_cast<pixel_t *>(dstp0);// + (hrest/2)*dst_pitch + wrest/2; // centered
  float *inp = inp0;
  int xoffset = bh*bw - (bw - ow);
  int yoffset = bw*nox*bh - bw*(bh - oh); // vertical offset of same block (overlap)
#if 0
  // PF
  // debug to check what happens when planeBase is not param, but set to fix 0
  // optimizer can rule one one addition -> faster
  // consider using template for this
  planeBase_i = 0; // debug;!+hj4l
#endif
  typedef std::conditional<sizeof(pixel_t) == 4, float, int>::type cast_t;

  cast_t planeBase = sizeof(pixel_t) == 4 ? cast_t(planeBase_f) : cast_t(planeBase_i); // anti warning

  cast_t max_pixel_value = sizeof(pixel_t) == 4 ? (pixel_t)1.0f : (pixel_t)((1 << bits_per_pixel) - 1);

  ihy = 0; // first top big non-overlapped) part
  {
    for (h = 0; h < bh - oh; h++)
    {
      inp = inp0 + h*bw;
      for (w = 0; w < bw - ow; w++)   // first half line of first block
      {
        dstp[w] = min(cast_t(max_pixel_value), max((cast_t)0, (cast_t)(inp[w] * norm) + planeBase));   // Copy each byte from float array to dest with windows
      }
      inp += bw - ow;
      dstp += bw - ow;
      for (ihx = 1; ihx < nox; ihx++) // middle horizontal half-blocks
      {
        for (w = 0; w < ow; w++)   // half line of block
        {
          dstp[w] = min(cast_t(max_pixel_value), max(0, (cast_t)((inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w])*norm) + planeBase));   // overlapped Copy
        }
        inp += xoffset + ow;
        dstp += ow;
        for (w = 0; w < bw - ow - ow; w++)   // first half line of first block
        {
          dstp[w] = min(cast_t(max_pixel_value), max(0, (cast_t)(inp[w] * norm) + planeBase));   // Copy each byte from float array to dest with windows
        }
        inp += bw - ow - ow;
        dstp += bw - ow - ow;
      }
      for (w = 0; w < ow; w++)   // last half line of last block
      {
        dstp[w] = min(cast_t(max_pixel_value), max(0, (cast_t)(inp[w] * norm) + planeBase));
      }
      inp += ow;
      dstp += ow;

      dstp += (dst_pitch - coverwidth);  // Add the pitch of one line (in bytes) to the dest image.
    }
  }

  for (ihy = 1; ihy < noy; ihy += 1) // middle vertical
  {
    for (h = 0; h < oh; h++) // top overlapped part
    {
      inp = inp0 + (ihy - 1)*(yoffset + (bh - oh)*bw) + (bh - oh)*bw + h*bw;

      float wsynyrh = wsynyr[h] * norm; // remove from cycle for speed
      float wsynylh = wsynyl[h] * norm;

      for (w = 0; w < bw - ow; w++)   // first half line of first block
      {
        dstp[w] = min(cast_t(max_pixel_value), max(0, (cast_t)((inp[w] * wsynyrh + inp[w + yoffset] * wsynylh)) + planeBase));   // y overlapped
      }
      inp += bw - ow;
      dstp += bw - ow;
      for (ihx = 1; ihx < nox; ihx++) // middle blocks
      {
        for (w = 0; w < ow; w++)   // half overlapped line of block
        {
          dstp[w] = min(cast_t(max_pixel_value), max(0, (cast_t)(((inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w])*wsynyrh
            + (inp[w + yoffset] * wsynxr[w] + inp[w + xoffset + yoffset] * wsynxl[w])*wsynylh)) + planeBase));   // x overlapped
        }
        inp += xoffset + ow;
        dstp += ow;
        for (w = 0; w < bw - ow - ow; w++)   // double minus - half non-overlapped line of block
        {
          dstp[w] = min(cast_t(max_pixel_value), max(0, (cast_t)((inp[w] * wsynyrh + inp[w + yoffset] * wsynylh)) + planeBase));
        }
        inp += bw - ow - ow;
        dstp += bw - ow - ow;
      }
      for (w = 0; w < ow; w++)   // last half line of last block
      {
        dstp[w] = min(cast_t(max_pixel_value), max(0, (cast_t)((inp[w] * wsynyrh + inp[w + yoffset] * wsynylh)) + planeBase));
      }
      inp += ow;
      dstp += ow;

      dstp += (dst_pitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
    }
    // middle  vertical non-ovelapped part
    for (h = 0; h < (bh - oh - oh); h++)
    {
      inp = inp0 + (ihy - 1)*(yoffset + (bh - oh)*bw) + (bh)*bw + h*bw + yoffset;
      for (w = 0; w < bw - ow; w++)   // first half line of first block
      {
        dstp[w] = min(cast_t(max_pixel_value), max(0, (cast_t)((inp[w])*norm) + planeBase));
      }
      inp += bw - ow;
      dstp += bw - ow;
      for (ihx = 1; ihx < nox; ihx++) // middle blocks
      {
        for (w = 0; w < ow; w++)   // half overlapped line of block
        {
          dstp[w] = min(cast_t(max_pixel_value), max(0, (cast_t)((inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w])*norm) + planeBase));   // x overlapped
        }
        inp += xoffset + ow;
        dstp += ow;
        for (w = 0; w < bw - ow - ow; w++)   // half non-overlapped line of block
        {
          dstp[w] = min(cast_t(max_pixel_value), max(0, (cast_t)((inp[w])*norm) + planeBase));
        }
        inp += bw - ow - ow;
        dstp += bw - ow - ow;
      }
      for (w = 0; w < ow; w++)   // last half line of last block
      {
        dstp[w] = min(cast_t(max_pixel_value), max(0, (cast_t)((inp[w])*norm) + planeBase));
      }
      inp += ow;
      dstp += ow;

      dstp += (dst_pitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
    }

  }

  ihy = noy; // last bottom part
  {
    for (h = 0; h < oh; h++)
    {
      inp = inp0 + (ihy - 1)*(yoffset + (bh - oh)*bw) + (bh - oh)*bw + h*bw;
      for (w = 0; w < bw - ow; w++)   // first half line of first block
      {
        dstp[w] = min(cast_t(max_pixel_value), max(0, (cast_t)(inp[w] * norm) + planeBase));
      }
      inp += bw - ow;
      dstp += bw - ow;
      for (ihx = 1; ihx < nox; ihx++) // middle blocks
      {
        for (w = 0; w < ow; w++)   // half line of block
        {
          dstp[w] = min(cast_t(max_pixel_value), max(0, (cast_t)((inp[w] * wsynxr[w] + inp[w + xoffset] * wsynxl[w])*norm) + planeBase));   // overlapped Copy
        }
        inp += xoffset + ow;
        dstp += ow;
        for (w = 0; w < bw - ow - ow; w++)   // half line of block
        {
          dstp[w] = min(cast_t(max_pixel_value), max(0, (cast_t)((inp[w])*norm) + planeBase));
        }
        inp += bw - ow - ow;
        dstp += bw - ow - ow;
      }
      for (w = 0; w < ow; w++)   // last half line of last block
      {
        dstp[w] = min(cast_t(max_pixel_value), max(0, (cast_t)(inp[w] * norm) + planeBase));
      }
      inp += ow;
      dstp += ow;

      dstp += (dst_pitch - coverwidth);  // Add the pitch of one line (in bytes) to the source image.
    }
  }
}

//-------------------------------------------------------------------------------------------
void GetAndSubtactMean(float *in, int howmanyblocks, int bw, int bh, int ow, int oh, float *wxl, float *wxr, float *wyl, float *wyr, float *mean)
{
  int h, w, block;
  float meanblock;
  float norma;

  // calculate norma
  norma = 0;
  for (h = 0; h < oh; h++)
  {
    for (w = 0; w < ow; w++)
    {
      norma += wxl[w] * wyl[h];
    }
    for (w = ow; w < bw - ow; w++)
    {
      norma += wyl[h];
    }
    for (w = bw - ow; w < bw; w++)
    {
      norma += wxr[w - bw + ow] * wyl[h];
    }
  }
  for (h = oh; h < bh - oh; h++)
  {
    for (w = 0; w < ow; w++)
    {
      norma += wxl[w];
    }
    for (w = ow; w < bw - ow; w++)
    {
      norma += 1;
    }
    for (w = bw - ow; w < bw; w++)
    {
      norma += wxr[w - bw + ow];
    }
  }
  for (h = bh - oh; h < bh; h++)
  {
    for (w = 0; w < ow; w++)
    {
      norma += wxl[w] * wyr[h - bh + oh];
    }
    for (w = ow; w < bw - ow; w++)
    {
      norma += wyr[h - bh + oh];
    }
    for (w = bw - ow; w < bw; w++)
    {
      norma += wxr[w - bw + ow] * wyr[h - bh + oh];
    }
  }


  for (block = 0; block < howmanyblocks; block++)
  {
    meanblock = 0;
    for (h = 0; h < bh; h++)
    {
      for (w = 0; w < bw; w++)
      {
        meanblock += in[w];
      }
      in += bw;
    }
    meanblock /= (bw*bh);
    mean[block] = meanblock;

    in -= bw*bh; // restore pointer
    for (h = 0; h < bh; h++)
    {
      for (w = 0; w < bw; w++)
      {
        in[w] -= meanblock;
      }
      in += bw;
    }

  }
}
//-------------------------------------------------------------------------------------------
void RestoreMean(float *in, int howmanyblocks, int bw, int bh, float *mean)
{
  int h, w, block;
  float meanblock;

  for (block = 0; block < howmanyblocks; block++)
  {
    meanblock = mean[block] * (bw*bh);

    for (h = 0; h < bh; h++)
    {
      for (w = 0; w < bw; w++)
      {
        in[w] += meanblock;
      }
      in += bw;
    }
  }
}
//-------------------------------------------------------------------------------------------
void ShowIn(float *in0, int nox, int noy, int bw, int bh, BYTE* srcp0, int src_width, int src_height, int src_pitch)
{
  float *in = in0;
  BYTE *srcp = srcp0;
  int in_pitch = nox*bw*bh;
  int noxmax = src_width / bw;
  int noymax = src_height / bh;
  for (int by = 0; by < noymax; by++)
  {
    srcp = srcp0 + src_pitch*bh*by;
    in = in0 + in_pitch*by;
    for (int bx = 0; bx < noxmax; bx++)
    {
      for (int h = 0; h < bh; h++)
      {
        for (int w = 0; w < bw; w++)
        {
          srcp[w] = (int)in[w];
        }
        srcp += src_pitch;
        in += bw;
      }
      srcp -= src_pitch*bh;
      srcp += bw;
    }
  }
}
//-------------------------------------------------------------------------------------------
void FindPatternBlock(fftwf_complex *outcur0, int outwidth, int outpitch, int bh, int nox, int noy, int &px, int &py, float *pwin, float degrid, fftwf_complex *gridsample)
{
  // since v1.7 outwidth must be really an outpitch
  int h;
  int w;
  fftwf_complex *outcur;
  float psd;
  float sigmaSquaredcur;
  float sigmaSquared;
  sigmaSquared = 1e15f;

  for (int by = 2; by < noy - 2; by++)
  {
    for (int bx = 2; bx < nox - 2; bx++)
    {
      outcur = outcur0 + nox*by*bh*outpitch + bx*bh*outpitch;
      sigmaSquaredcur = 0;
      float gcur = degrid*outcur[0][0] / gridsample[0][0]; // grid (windowing) correction factor
      for (h = 0; h < bh; h++)
      {
        for (w = 0; w < outwidth; w++)
        {
          //					psd = outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1];
          float grid0 = gcur*gridsample[w][0];
          float grid1 = gcur*gridsample[w][1];
          float corrected0 = outcur[w][0] - grid0;
          float corrected1 = outcur[w][1] - grid1;
          psd = corrected0*corrected0 + corrected1*corrected1;
          sigmaSquaredcur += psd*pwin[w]; // windowing
        }
        outcur += outpitch;
        pwin += outpitch;
        gridsample += outpitch;
      }
      pwin -= outpitch*bh; // restore
      if (sigmaSquaredcur < sigmaSquared)
      {
        px = bx;
        py = by;
        sigmaSquared = sigmaSquaredcur;
      }
    }
  }
}
//-------------------------------------------------------------------------------------------
void SetPattern(fftwf_complex *outcur, int outwidth, int outpitch, int bh, int nox, int noy, int px, int py, float *pwin, float *pattern2d, float &psigma, float degrid, fftwf_complex *gridsample)
{
  int h;
  int w;
  outcur += nox*py*bh*outpitch + px*bh*outpitch;
  float psd;
  float sigmaSquared = 0;
  float weight = 0;

  for (h = 0; h < bh; h++)
  {
    for (w = 0; w < outwidth; w++)
    {
      weight += pwin[w];
    }
    pwin += outpitch;
  }
  pwin -= outpitch*bh; // restore

  float gcur = degrid*outcur[0][0] / gridsample[0][0]; // grid (windowing) correction factor

  for (h = 0; h < bh; h++)
  {
    for (w = 0; w < outwidth; w++)
    {
      float grid0 = gcur*gridsample[w][0];
      float grid1 = gcur*gridsample[w][1];
      float corrected0 = outcur[w][0] - grid0;
      float corrected1 = outcur[w][1] - grid1;
      psd = corrected0*corrected0 + corrected1*corrected1;
      //			psd = outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1];
      pattern2d[w] = psd*pwin[w]; // windowing
      sigmaSquared += pattern2d[w]; // sum
    }
    outcur += outpitch;
    pattern2d += outpitch;
    pwin += outpitch;
    gridsample += outpitch;
  }
  psigma = sqrt(sigmaSquared / (weight*bh*outwidth)); // mean std deviation (sigma)
}
//-------------------------------------------------------------------------------------------
void PutPatternOnly(fftwf_complex *outcur, int outwidth, int outpitch, int bh, int nox, int noy, int px, int py)
{
  int h, w;
  int block;
  int pblock = py*nox + px;
  int blocks = nox*noy;

  for (block = 0; block < pblock; block++)
  {
    for (h = 0; h < bh; h++)
    {
      for (w = 0; w < outwidth; w++)
      {
        outcur[w][0] = 0;
        outcur[w][1] = 0;
      }
      outcur += outpitch;
    }
  }

  outcur += bh*outpitch;

  for (block = pblock + 1; block < blocks; block++)
  {
    for (h = 0; h < bh; h++)
    {
      for (w = 0; w < outwidth; w++)
      {
        outcur[w][0] = 0;
        outcur[w][1] = 0;
      }
      outcur += outpitch;
    }
  }

}
//-------------------------------------------------------------------------------------------
void Pattern2Dto3D(float *pattern2d, int bh, int outwidth, int outpitch, float mult, float *pattern3d)
{
  // slow, but executed once only per clip
  int size = bh*outpitch;
  for (int i = 0; i < size; i++)
  { // get 3D pattern
    pattern3d[i] = pattern2d[i] * mult;
  }
}
//-------------------------------------------------------------------------------------------
void Copyfft(fftwf_complex *outrez, fftwf_complex *outprev, int outsize, IScriptEnvironment* env)
{ // save outprev to outrez to prevent cache change (inverse fft2d will destroy the array)
/*	for (int i=0; i<outsize; i++)
  {
    outrez[i][0] = outprev[i][0];
    outrez[i][1] = outprev[i][1];
  }
*/
  env->BitBlt((BYTE*)&outrez[0][0], outsize * 8, (BYTE*)&outprev[0][0], outsize * 8, outsize * 8, 1); // more fast
}
//-------------------------------------------------------------------------------------------
void SortCache(int *cachewhat, fftwf_complex **cachefft, int cachesize, int cachestart, int cachestartold)
{
  // sort ordered series, put existant ffts to proper places
  int i;
  int ctemp;
  fftwf_complex *ffttemp;

  int offset = cachestart - cachestartold;
  if (offset > 0) // right
  {
    for (i = 0; i < cachesize; i++)
    {
      if ((i + offset) < cachesize)
      {
        //swap
        ctemp = cachewhat[i + offset];
        cachewhat[i + offset] = cachewhat[i];
        cachewhat[i] = ctemp;
        ffttemp = cachefft[i + offset];
        cachefft[i + offset] = cachefft[i];
        cachefft[i] = ffttemp;
      }
    }
  }
  else if (offset < 0)
  {
    for (i = cachesize - 1; i >= 0; i--)
    {
      if ((i + offset) >= 0)
      {
        ctemp = cachewhat[i + offset];
        cachewhat[i + offset] = cachewhat[i];
        cachewhat[i] = ctemp;
        ffttemp = cachefft[i + offset];
        cachefft[i + offset] = cachefft[i];
        cachefft[i] = ffttemp;
      }
    }
  }
}
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
void CopyFrame(PVideoFrame &src, PVideoFrame &dst, VideoInfo vi, int planeskip, IScriptEnvironment* env)
{
  const BYTE * srcp;
  BYTE * dstp;
  int src_height, src_width, src_pitch;
  int dst_height, dst_width, dst_pitch;
  int planarNum, plane;

  // greyscale is planar as well
  if (vi.IsPlanar()) // copy all planes besides given
  {
    for (plane = 0; plane < (vi.IsY() ? 1 : 3); plane++)
    {
      if (plane != planeskip)
      {
        int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
        int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
        int *planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;
        planarNum = planes[plane];

        srcp = src->GetReadPtr(planarNum);
        src_height = src->GetHeight(planarNum);
        src_width = src->GetRowSize(planarNum);
        src_pitch = src->GetPitch(planarNum);
        dstp = dst->GetWritePtr(planarNum);
        dst_height = dst->GetHeight(planarNum);
        dst_width = dst->GetRowSize(planarNum);
        dst_pitch = dst->GetPitch(planarNum);
        env->BitBlt(dstp, dst_pitch, srcp, src_pitch, dst_width, dst_height); // copy one plane
      }
    }

  }
  /*
  else if (vi.IsY8()) // copy Y planes if not given
  {
    if (0 != planeskip)
    {

      srcp = src->GetReadPtr(0);
      src_height = src->GetHeight(0);
      src_width = src->GetRowSize(0);
      src_pitch = src->GetPitch(0);
      dstp = dst->GetWritePtr(0);
      dst_height = dst->GetHeight(0);
      dst_width = dst->GetRowSize(0);
      dst_pitch = dst->GetPitch(0);
      env->BitBlt(dstp, dst_pitch, srcp, src_pitch, dst_width, dst_height); // copy one plane
    }

  }
  */
  else if (vi.IsYUY2()) // copy all
  {
    srcp = src->GetReadPtr();
    src_height = src->GetHeight();
    src_width = src->GetRowSize();
    src_pitch = src->GetPitch();
    dstp = dst->GetWritePtr();
    dst_height = dst->GetHeight();
    dst_width = dst->GetRowSize();
    dst_pitch = dst->GetPitch();
    env->BitBlt(dstp, dst_pitch, srcp, src_pitch, dst_width, dst_height); // copy full frame
  }
}
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------

PVideoFrame __stdcall FFT3DFilter::GetFrame(int n, IScriptEnvironment* env) {
  // This is the implementation of the GetFrame function.
  // See the header definition for further info.

  PVideoFrame prev2, prev, src, next, psrc, dst, next2;
  int pxf, pyf;
  int i;
  int cachecur, cachestart, cachestartold;
  //	char debugbuf[1536];
  //	wsprintf(debugbuf,"FFT3DFilter: n=%d \n", n);
  //	OutputDebugString(debugbuf);
  //	fftwf_complex * tmpoutrez, *tmpoutnext, *tmpoutprev, *tmpoutnext2; // store pointers
  _RPT2(0, "FFT3DFilter GetFrame, frame=%d instance_id=%d\n", n, _instance_id);
  if (reentrancy_check) {
    _RPT2(0, "FFT3DFilter GetFrame, Reentrant call detected! Frame=%d instance_id=%d\n", n, _instance_id);
    env->ThrowError("FFT3DFilter: cannot work in reentrant multithread mode!");
  }
  reentrancy_check = true;

#ifndef X86_64
  _mm_empty(); // _asm emms;
#endif

  bool isRGB = vi.IsRGB();

  if (plane == 0 || isRGB) {
    planeBase = 0;
    planeBase_f = 0.0f;
  }
  else {
    planeBase = (1 << (bits_per_pixel - 1)); // neutral chroma value
    planeBase_f = 0.5f;
  }

  if (pfactor != 0 && isPatternSet == false && pshow == false) // get noise pattern
  {
    psrc = child->GetFrame(pframe, env); // get noise pattern frame

    // put source bytes to float array of overlapped blocks
    FramePlaneToCoverbuf(plane, psrc, vi, coverbuf, coverwidth, coverheight, coverpitch * pixelsize, mirw, mirh, interlaced, bits_per_pixel, env);
    FFT3DFilter::InitOverlapPlane(in, coverbuf, coverpitch, planeBase, planeBase_f);
    // make FFT 2D
    fftwf_execute_dft_r2c(plan, in, outrez);
    if (px == 0 && py == 0) // try find pattern block with minimal noise sigma
      FindPatternBlock(outrez, outwidth, outpitch, bh, nox, noy, px, py, pwin, degrid, gridsample);
    SetPattern(outrez, outwidth, outpitch, bh, nox, noy, px, py, pwin, pattern2d, psigma, degrid, gridsample);
    isPatternSet = true;
  }
  else if (pfactor != 0 && pshow == true)
  {
    // show noise pattern window
    src = child->GetFrame(n, env); // get noise pattern frame
//		env->MakeWritable(&src); // it produced bug for separated fields
    dst = env->NewVideoFrame(vi);
    CopyFrame(src, dst, vi, plane, env);

    // put source bytes to float array of overlapped blocks
    FramePlaneToCoverbuf(plane, src, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, bits_per_pixel, env);
    FFT3DFilter::InitOverlapPlane(in, coverbuf, coverpitch, planeBase, planeBase_f);
    // make FFT 2D
    fftwf_execute_dft_r2c(plan, in, outrez);
    if (px == 0 && py == 0) // try find pattern block with minimal noise sigma
      FindPatternBlock(outrez, outwidth, outpitch, bh, nox, noy, pxf, pyf, pwin, degrid, gridsample);
    else
    {
      pxf = px; // fixed bug in v1.6
      pyf = py;
    }
    SetPattern(outrez, outwidth, outpitch, bh, nox, noy, pxf, pyf, pwin, pattern2d, psigma, degrid, gridsample);

    // change analysis and synthesis window to constant to show
    for (i = 0; i < ow; i++)
    {
      wanxl[i] = 1;	wanxr[i] = 1;	wsynxl[i] = 1;	wsynxr[i] = 1;
    }
    for (i = 0; i < oh; i++)
    {
      wanyl[i] = 1;	wanyr[i] = 1;	wsynyl[i] = 1;	wsynyr[i] = 1;
    }

    if (isRGB) {
      planeBase = 0;
      planeBase_f = 0.0f;
    }
    else {
      planeBase = (1 << (bits_per_pixel - 1)); // 128
      planeBase_f = 0.5f;
    }

    // put source bytes to float array of overlapped blocks
    // cur frame
    FramePlaneToCoverbuf(plane, src, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, bits_per_pixel, env);
    FFT3DFilter::InitOverlapPlane(in, coverbuf, coverpitch, planeBase, planeBase_f);
    // make FFT 2D
    fftwf_execute_dft_r2c(plan, in, outrez);

    PutPatternOnly(outrez, outwidth, outpitch, bh, nox, noy, pxf, pyf);
    // do inverse 2D FFT, get filtered 'in' array
    fftwf_execute_dft_c2r(planinv, outrez, in);

    // make destination frame plane from current overlaped blocks
    FFT3DFilter::DecodeOverlapPlane(in, norm, coverbuf, coverpitch, planeBase, planeBase_f);
    CoverbufToFramePlane(plane, coverbuf, coverwidth, coverheight, coverpitch, dst, vi, mirw, mirh, interlaced, bits_per_pixel, env);
    int psigmaint = ((int)(10 * psigma)) / 10;
    int psigmadec = (int)((psigma - psigmaint) * 10);
    wsprintf(messagebuf, " frame=%d, px=%d, py=%d, sigma=%d.%d", n, pxf, pyf, psigmaint, psigmadec);
    DrawString(dst, vi, 0, 0, messagebuf);

    _RPT2(0, "FFT3DFilter GetFrame END, frame=%d instance_id=%d\n", n, _instance_id);
    reentrancy_check = false;
    return dst; // return pattern frame to show
  }

  _RPT2(0, "FFT3DFilter child GetFrame, frame=%d instance_id=%d\n", n, _instance_id);
  // Request frame 'n' from the child (source) clip.
  src = child->GetFrame(n, env);
  _RPT2(0, "FFT3DFilter child GetFrame END, frame=%d instance_id=%d\n", n, _instance_id);
  dst = env->NewVideoFrame(vi);

  /*
  _multiplane == 0 : process Y, copy U, copy V
  _multiplane == 1 : copy Y, process U, copy V
  _multiplane == 2 : copy Y, copy U, process V
  _multiplane == 3 : copy Y, process U, process V
  _multiplane == 4 : process Y, process U, process V
  */
  if (multiplane < 3 || (multiplane == 3 && plane == 1)) // v1.8.4
  {
    CopyFrame(src, dst, vi, plane, env);
  }

  int btcur = bt; // bt used for current frame
//	if ( (bt/2 > n) || bt==3 && n==vi.num_frames-1 )
  if ((bt / 2 > n) || (bt - 1) / 2 > (vi.num_frames - 1 - n))
  {
    btcur = 1; //	do 2D filter for first and last frames
  }
  // return src //first  frame was not processed prior v.0.7

  if (btcur > 0) // Wiener
  {
    sigmaSquaredNoiseNormed = btcur*sigma*sigma / norm; // normalized variation=sigma^2

    if (btcur != btcurlast) // !! global state, multithreading warning
      Pattern2Dto3D(pattern2d, bh, outwidth, outpitch, (float)btcur, pattern3d);

    // get power spectral density (abs quadrat) for every block and apply filter

//		env->MakeWritable(&src);

    // put source bytes to float array of overlapped blocks

    if (btcur == 1) // 2D
    {
      // cur frame
      FramePlaneToCoverbuf(plane, src, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, bits_per_pixel, env);
      FFT3DFilter::InitOverlapPlane(in, coverbuf, coverpitch, planeBase, planeBase_f);
      //			FFT3DFilter::InitOverlapPlaneWin(in, coverbuf,  coverpitch, planeBase, fullwinan); // slower
            // make FFT 2D
      fftwf_execute_dft_r2c(plan, in, outrez);
      if (degrid != 0)
      {
        if (pfactor != 0)
        {
          ApplyPattern2D_degrid_C(outrez, outwidth, outpitch, bh, howmanyblocks, pfactor, pattern2d, beta, degrid, gridsample);
          Sharpen_degrid(outrez, outwidth, outpitch, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen, degrid, gridsample, dehalo, wdehalo, ht2n, CPUFlags);
        }
        else
          ApplyWiener2D_degrid_C(outrez, outwidth, outpitch, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen, degrid, gridsample, dehalo, wdehalo, ht2n);
      }
      else
      {
        if (pfactor != 0)
        {
          ApplyPattern2D(outrez, outwidth, outpitch, bh, howmanyblocks, pfactor, pattern2d, beta, CPUFlags);
          Sharpen(outrez, outwidth, outpitch, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen, dehalo, wdehalo, ht2n, CPUFlags);
        }
        else
          ApplyWiener2D(outrez, outwidth, outpitch, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen, dehalo, wdehalo, ht2n, CPUFlags);
      }

      // do inverse FFT 2D, get filtered 'in' array
      fftwf_execute_dft_c2r(planinv, outrez, in);
    }
    else if (btcur == 2)  // 3D2
    {
      cachecur = 2;
      cachestart = n - cachecur;
      cachestartold = nlast - cachecur;
      SortCache(cachewhat, cachefft, cachesize, cachestart, cachestartold);
      // cur frame
      out = cachefft[cachecur];
      if (cachewhat[cachecur] != n)
      {
        FramePlaneToCoverbuf(plane, src, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, bits_per_pixel, env);
        FFT3DFilter::InitOverlapPlane(in, coverbuf, coverpitch, planeBase, planeBase_f);
        // make FFT 2D
        fftwf_execute_dft_r2c(plan, in, out);
        cachewhat[cachecur] = n;
      }
      // prev frame
      outprev = cachefft[cachecur - 1];
      if (cachewhat[cachecur - 1] != n - 1)
      {
        _RPT2(0, "FFT3DFilter child-1 GetFrame, frame=%d instance_id=%d\n", n-1, _instance_id);
        prev = child->GetFrame(n - 1, env);
        _RPT2(0, "FFT3DFilter child-1 GetFrame END, frame=%d instance_id=%d\n", n - 1, _instance_id);
        FramePlaneToCoverbuf(plane, prev, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, bits_per_pixel, env);
      }
      if (cachewhat[cachecur - 1] != n - 1)
      {
        // calculate prev
        FFT3DFilter::InitOverlapPlane(in, coverbuf, coverpitch, planeBase, planeBase_f);
        // make FFT 2D
        fftwf_execute_dft_r2c(plan, in, outprev);
        cachewhat[cachecur - 1] = n - 1;
      }
      if (n != nlast + 1)//(not direct sequential access)
      {
        Copyfft(outrez, outprev, outsize, env); // save outprev to outrez to prevent its change in cache
      }
      else
      {
        // swap
        outtemp = outrez;
        outrez = outprev;
        outprev = outtemp;
        cachefft[cachecur - 1] = outtemp;
        cachewhat[cachecur - 1] = -1; // will be destroyed
      }
      if (degrid != 0)
      {
        if (pfactor != 0)
          ApplyPattern3D2_degrid_C(out, outrez, outwidth, outpitch, bh, howmanyblocks, pattern3d, beta, degrid, gridsample);
        else
          ApplyWiener3D2_degrid_C(out, outrez, outwidth, outpitch, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, degrid, gridsample);
        Sharpen_degrid(outrez, outwidth, outpitch, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen, degrid, gridsample, dehalo, wdehalo, ht2n, CPUFlags);
      }
      else
      {
        if (pfactor != 0)
          ApplyPattern3D2(out, outrez, outwidth, outpitch, bh, howmanyblocks, pattern3d, beta, CPUFlags);
        else
          ApplyWiener3D2(out, outrez, outwidth, outpitch, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, CPUFlags); // get result in outpret
        Sharpen(outrez, outwidth, outpitch, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen, dehalo, wdehalo, ht2n, CPUFlags);
      }
      // do inverse FFT 3D, get filtered 'in' array
      // note: input "outrez" array is destroyed by execute algo.
      fftwf_execute_dft_c2r(planinv, outrez, in);
    }
    else if (btcur == 3) // 3D3
    {
      cachecur = 2;
      cachestart = n - cachecur;
      cachestartold = nlast - cachecur;
      SortCache(cachewhat, cachefft, cachesize, cachestart, cachestartold);
      // cur frame
      out = cachefft[cachecur];
      if (cachewhat[cachecur] != n)
      {
        FramePlaneToCoverbuf(plane, src, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, bits_per_pixel, env);
        FFT3DFilter::InitOverlapPlane(in, coverbuf, coverpitch, planeBase, planeBase_f);
        // make FFT 2D
        fftwf_execute_dft_r2c(plan, in, out);
        cachewhat[cachecur] = n;
      }
      // prev frame
      outprev = cachefft[cachecur - 1];
      if (cachewhat[cachecur - 1] != n - 1)
      {
        // calculate prev
        _RPT2(0, "FFT3DFilter child-1 GetFrame, frame=%d instance_id=%d\n", n - 1, _instance_id);
        prev = child->GetFrame(n - 1, env);
        _RPT2(0, "FFT3DFilter child-1 GetFrame END, frame=%d instance_id=%d\n", n - 1, _instance_id);
        FramePlaneToCoverbuf(plane, prev, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, bits_per_pixel, env);
      }
      if (cachewhat[cachecur - 1] != n - 1)
      {
        FFT3DFilter::InitOverlapPlane(in, coverbuf, coverpitch, planeBase, planeBase_f);
        // make FFT 2D
        fftwf_execute_dft_r2c(plan, in, outprev);
        cachewhat[cachecur - 1] = n - 1;
      }
      if (n != nlast + 1)
      {
        Copyfft(outrez, outprev, outsize, env); // save outprev to outrez to preventits change in cache
      }
      else
      {
        // swap
        outtemp = outrez;
        outrez = outprev;
        outprev = outtemp;
        cachefft[cachecur - 1] = outtemp;
        cachewhat[cachecur - 1] = -1; // will be destroyed
      }
      // calculate next
      outnext = cachefft[cachecur + 1];
      if (cachewhat[cachecur + 1] != n + 1)
      {
        _RPT2(0, "FFT3DFilter child+1 GetFrame, frame=%d instance_id=%d\n", n + 1, _instance_id);
        next = child->GetFrame(n + 1, env);
        _RPT2(0, "FFT3DFilter child+1 GetFrame END, frame=%d instance_id=%d\n", n + 1, _instance_id);
        FramePlaneToCoverbuf(plane, next, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, bits_per_pixel, env);
      }
      if (cachewhat[cachecur + 1] != n + 1)
      {
        FFT3DFilter::InitOverlapPlane(in, coverbuf, coverpitch, planeBase, planeBase_f);
        // make FFT 2D
        fftwf_execute_dft_r2c(plan, in, outnext);
        cachewhat[cachecur + 1] = n + 1;
      }
      if (degrid != 0)
      {
        if (pfactor != 0)
          ApplyPattern3D3_degrid(out, outrez, outnext, outwidth, outpitch, bh, howmanyblocks, pattern3d, beta, degrid, gridsample, CPUFlags);
        else
          ApplyWiener3D3_degrid(out, outrez, outnext, outwidth, outpitch, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, degrid, gridsample, CPUFlags);
        Sharpen_degrid(outrez, outwidth, outpitch, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen, degrid, gridsample, dehalo, wdehalo, ht2n, CPUFlags);
      }
      else
      {
        if (pfactor != 0)
          ApplyPattern3D3(out, outrez, outnext, outwidth, outpitch, bh, howmanyblocks, pattern3d, beta, CPUFlags);
        else
          ApplyWiener3D3(out, outrez, outnext, outwidth, outpitch, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, CPUFlags);
        Sharpen(outrez, outwidth, outpitch, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen, dehalo, wdehalo, ht2n, CPUFlags);
      }
      // do inverse FFT 2D, get filtered 'in' array
      // note: input "outrez" array is destroyed by execute algo.
      fftwf_execute_dft_c2r(planinv, outrez, in);
    }
    else if (btcur == 4) // 3D4
    {
      // cycle prev2, prev, cur and next
      cachecur = 3;
      cachestart = n - cachecur;
      cachestartold = nlast - cachecur;
      SortCache(cachewhat, cachefft, cachesize, cachestart, cachestartold);
      // cur frame
      out = cachefft[cachecur];
      if (cachewhat[cachecur] != n)
      {
        FramePlaneToCoverbuf(plane, src, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, bits_per_pixel, env);
        FFT3DFilter::InitOverlapPlane(in, coverbuf, coverpitch, planeBase, planeBase_f);
        // make FFT 2D
        fftwf_execute_dft_r2c(plan, in, out);
        cachewhat[cachecur] = n;
      }
      // prev2 frame
      outprev2 = cachefft[cachecur - 2];
      if (cachewhat[cachecur - 2] != n - 2)
      {
        // calculate prev2
        _RPT2(0, "FFT3DFilter child-2 GetFrame, frame=%d instance_id=%d\n", n - 2, _instance_id);
        prev2 = child->GetFrame(n - 2, env);
        _RPT2(0, "FFT3DFilter child-2 GetFrame END, frame=%d instance_id=%d\n", n - 2, _instance_id);
        FramePlaneToCoverbuf(plane, prev2, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, bits_per_pixel, env);
      }
      if (cachewhat[cachecur - 2] != n - 2)
      {
        FFT3DFilter::InitOverlapPlane(in, coverbuf, coverpitch, planeBase, planeBase_f);
        // make FFT 2D
        fftwf_execute_dft_r2c(plan, in, outprev2);
        cachewhat[cachecur - 2] = n - 2;
      }
      if (n != nlast + 1)
      {
        Copyfft(outrez, outprev2, outsize, env); // save outprev2 to outrez to prevent its change in cache
      }
      else
      {
        // swap
        outtemp = outrez;
        outrez = outprev2;
        outprev2 = outtemp;
        cachefft[cachecur - 2] = outtemp;
        cachewhat[cachecur - 2] = -1; // will be destroyed
      }
      // prev frame
      outprev = cachefft[cachecur - 1];
      if (cachewhat[cachecur - 1] != n - 1)
      {
        _RPT2(0, "FFT3DFilter child-1 GetFrame, frame=%d instance_id=%d\n", n - 1, _instance_id);
        prev = child->GetFrame(n - 1, env);
        _RPT2(0, "FFT3DFilter child-1 GetFrame END, frame=%d instance_id=%d\n", n - 1, _instance_id);
        FramePlaneToCoverbuf(plane, prev, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, bits_per_pixel, env);
      }
      if (cachewhat[cachecur - 1] != n - 1)
      {
        FFT3DFilter::InitOverlapPlane(in, coverbuf, coverpitch, planeBase, planeBase_f);
        // make FFT 2D
        fftwf_execute_dft_r2c(plan, in, outprev);
        cachewhat[cachecur - 1] = n - 1;
      }
      // next frame
      outnext = cachefft[cachecur + 1];
      if (cachewhat[cachecur + 1] != n + 1)
      {
        _RPT2(0, "FFT3DFilter child+1 GetFrame, frame=%d instance_id=%d\n", n + 1, _instance_id);
        next = child->GetFrame(n + 1, env);
        _RPT2(0, "FFT3DFilter child+1 GetFrame END, frame=%d instance_id=%d\n", n + 1, _instance_id);
        FramePlaneToCoverbuf(plane, next, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, bits_per_pixel, env);
      }
      if (cachewhat[cachecur + 1] != n + 1)
      {
        FFT3DFilter::InitOverlapPlane(in, coverbuf, coverpitch, planeBase, planeBase_f);
        // make FFT 2D
        fftwf_execute_dft_r2c(plan, in, outnext);
        cachewhat[cachecur + 1] = n + 1;
      }
      if (degrid != 0)
      {
        if (pfactor != 0)
          ApplyPattern3D4_degrid(out, outrez, outprev, outnext, outwidth, outpitch, bh, howmanyblocks, pattern3d, beta, degrid, gridsample, CPUFlags);
        else
          ApplyWiener3D4_degrid(out, outrez, outprev, outnext, outwidth, outpitch, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, degrid, gridsample, CPUFlags);
        Sharpen_degrid(outrez, outwidth, outpitch, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen, degrid, gridsample, dehalo, wdehalo, ht2n, CPUFlags);
      }
      else
      {
        if (pfactor != 0)
          ApplyPattern3D4(out, outrez, outprev, outnext, outwidth, outpitch, bh, howmanyblocks, pattern3d, beta, CPUFlags);
        else
          ApplyWiener3D4(out, outrez, outprev, outnext, outwidth, outpitch, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, CPUFlags);
        Sharpen(outrez, outwidth, outpitch, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen, dehalo, wdehalo, ht2n, CPUFlags);
      }
      // do inverse FFT 2D, get filtered 'in' array
      // note: input "outrez" array is destroyed by execute algo.
      fftwf_execute_dft_c2r(planinv, outrez, in);
    }
    else if (btcur == 5) // 3D5
    {
      // cycle prev2, prev, cur, next and next2
      cachecur = 3;
      cachestart = n - cachecur;
      cachestartold = nlast - cachecur;
      SortCache(cachewhat, cachefft, cachesize, cachestart, cachestartold);
      // cur frame
      out = cachefft[cachecur];
      if (cachewhat[cachecur] != n)
      {
        FramePlaneToCoverbuf(plane, src, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, bits_per_pixel, env);
        FFT3DFilter::InitOverlapPlane(in, coverbuf, coverpitch, planeBase, planeBase_f);
        // make FFT 2D
        fftwf_execute_dft_r2c(plan, in, out);
        cachewhat[cachecur] = n;
      }
      // prev2 frame
      outprev2 = cachefft[cachecur - 2];
      if (cachewhat[cachecur - 2] != n - 2)
      {
        // calculate prev2
        prev2 = child->GetFrame(n - 2, env);
        FramePlaneToCoverbuf(plane, prev2, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, bits_per_pixel, env);
      }
      if (cachewhat[cachecur - 2] != n - 2)
      {
        FFT3DFilter::InitOverlapPlane(in, coverbuf, coverpitch, planeBase, planeBase_f);
        // make FFT 2D
        fftwf_execute_dft_r2c(plan, in, outprev2);
        cachewhat[cachecur - 2] = n - 2;
      }
      if (n != nlast + 1)
      {
        Copyfft(outrez, outprev2, outsize, env); // save outprev2 to outrez to prevent its change in cache
      }
      else
      {
        // swap
        outtemp = outrez;
        outrez = outprev2;
        outprev2 = outtemp;
        cachefft[cachecur - 2] = outtemp;
        cachewhat[cachecur - 2] = -1; // will be destroyed
      }
      // prev frame
      outprev = cachefft[cachecur - 1];
      if (cachewhat[cachecur - 1] != n - 1)
      {
        prev = child->GetFrame(n - 1, env);
        FramePlaneToCoverbuf(plane, prev, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, bits_per_pixel, env);
      }
      if (cachewhat[cachecur - 1] != n - 1)
      {
        FFT3DFilter::InitOverlapPlane(in, coverbuf, coverpitch, planeBase, planeBase_f);
        // make FFT 2D
        fftwf_execute_dft_r2c(plan, in, outprev);
        cachewhat[cachecur - 1] = n - 1;
      }
      // next frame
      outnext = cachefft[cachecur + 1];
      if (cachewhat[cachecur + 1] != n + 1)
      {
        next = child->GetFrame(n + 1, env);
        FramePlaneToCoverbuf(plane, next, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, bits_per_pixel, env);
      }
      if (cachewhat[cachecur + 1] != n + 1)
      {
        FFT3DFilter::InitOverlapPlane(in, coverbuf, coverpitch, planeBase, planeBase_f);
        // make FFT 2D
        fftwf_execute_dft_r2c(plan, in, outnext);
        cachewhat[cachecur + 1] = n + 1;
      }
      // next2 frame
      outnext2 = cachefft[cachecur + 2];
      if (cachewhat[cachecur + 2] != n + 2)
      {
        next2 = child->GetFrame(n + 2, env);
        FramePlaneToCoverbuf(plane, next2, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, bits_per_pixel, env);
      }
      if (cachewhat[cachecur + 2] != n + 2)
      {
        FFT3DFilter::InitOverlapPlane(in, coverbuf, coverpitch, planeBase, planeBase_f);
        // make FFT 2D
        fftwf_execute_dft_r2c(plan, in, outnext2);
        cachewhat[cachecur + 2] = n + 2;
      }
      if (degrid != 0)
      {
        if (pfactor != 0)
          ApplyPattern3D5_degrid_C(out, outrez, outprev, outnext, outnext2, outwidth, outpitch, bh, howmanyblocks, pattern3d, beta, degrid, gridsample);
        else
          ApplyWiener3D5_degrid_C(out, outrez, outprev, outnext, outnext2, outwidth, outpitch, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta, degrid, gridsample);
        Sharpen_degrid(outrez, outwidth, outpitch, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen, degrid, gridsample, dehalo, wdehalo, ht2n, CPUFlags);
      }
      else
      {
        if (pfactor != 0)
          ApplyPattern3D5_C(out, outrez, outprev, outnext, outnext2, outwidth, outpitch, bh, howmanyblocks, pattern3d, beta);
        else
          ApplyWiener3D5_C(out, outrez, outprev, outnext, outnext2, outwidth, outpitch, bh, howmanyblocks, sigmaSquaredNoiseNormed, beta);
        Sharpen(outrez, outwidth, outpitch, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen, dehalo, wdehalo, ht2n, CPUFlags);
      }
      // do inverse FFT 2D, get filtered 'in' array
      // note: input "outrez" array is destroyed by execute algo.
      fftwf_execute_dft_c2r(planinv, outrez, in);
    }
    // make destination frame plane from current overlaped blocks
    FFT3DFilter::DecodeOverlapPlane(in, norm, coverbuf, coverpitch, planeBase, planeBase_f);
    CoverbufToFramePlane(plane, coverbuf, coverwidth, coverheight, coverpitch, dst, vi, mirw, mirh, interlaced, bits_per_pixel, env);

  }
  else if (bt == 0) //Kalman filter
  {
    // get power spectral density (abs quadrat) for every block and apply filter

    if (n == 0)
    {
      _RPT2(0, "FFT3DFilter GetFrame END, frame=%d instance_id=%d\n", n, _instance_id);
      reentrancy_check = false;
      return src; // first frame  not processed
    }
    /* PF 170302 comment: accumulated error?
      orig = BlankClip(...)
      new = orig.FFT3DFilter(sigma = 3, plane = 4, bt = 0, degrid = 0)
      Subtract(orig,new).Levels(120, 1, 255 - 120, 0, 255, coring = false)
    */

    // put source bytes to float array of overlapped blocks
    // cur frame
    FramePlaneToCoverbuf(plane, src, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, bits_per_pixel, env);
    FFT3DFilter::InitOverlapPlane(in, coverbuf, coverpitch, planeBase, planeBase_f);
    // make FFT 2D
    fftwf_execute_dft_r2c(plan, in, outrez);
    if (pfactor != 0)
      ApplyKalmanPattern(outrez, outLast, covar, covarProcess, outwidth, outpitch, bh, howmanyblocks, pattern2d, kratio*kratio, CPUFlags);
    else
      ApplyKalman(outrez, outLast, covar, covarProcess, outwidth, outpitch, bh, howmanyblocks, sigmaSquaredNoiseNormed2D, kratio*kratio, CPUFlags);

    // copy outLast to outrez
    env->BitBlt((BYTE*)&outrez[0][0], outsize * sizeof(fftwf_complex), (BYTE*)&outLast[0][0], outsize * sizeof(fftwf_complex), outsize * sizeof(fftwf_complex), 1);  //v.0.9.2
    if (degrid != 0)
      Sharpen_degrid(outrez, outwidth, outpitch, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen, degrid, gridsample, dehalo, wdehalo, ht2n, CPUFlags);
    else
      Sharpen(outrez, outwidth, outpitch, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen, dehalo, wdehalo, ht2n, CPUFlags);
    // do inverse FFT 2D, get filtered 'in' array
    // note: input "out" array is destroyed by execute algo.
    // that is why we must have its copy in "outLast" array
    fftwf_execute_dft_c2r(planinv, outrez, in);
    // make destination frame plane from current overlaped blocks
    FFT3DFilter::DecodeOverlapPlane(in, norm, coverbuf, coverpitch, planeBase, planeBase_f);
    CoverbufToFramePlane(plane, coverbuf, coverwidth, coverheight, coverpitch, dst, vi, mirw, mirh, interlaced, bits_per_pixel, env);

  }
  else if (bt == -1) /// sharpen only
  {
    //		env->MakeWritable(&src);
        // put source bytes to float array of overlapped blocks
    FramePlaneToCoverbuf(plane, src, vi, coverbuf, coverwidth, coverheight, coverpitch, mirw, mirh, interlaced, bits_per_pixel, env);
    FFT3DFilter::InitOverlapPlane(in, coverbuf, coverpitch, planeBase, planeBase_f);
    // make FFT 2D
    fftwf_execute_dft_r2c(plan, in, outrez);
    if (degrid != 0)
      Sharpen_degrid(outrez, outwidth, outpitch, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen, degrid, gridsample, dehalo, wdehalo, ht2n, CPUFlags);
    else
      Sharpen(outrez, outwidth, outpitch, bh, howmanyblocks, sharpen, sigmaSquaredSharpenMinNormed, sigmaSquaredSharpenMaxNormed, wsharpen, dehalo, wdehalo, ht2n, CPUFlags);
    // do inverse FFT 2D, get filtered 'in' array
    fftwf_execute_dft_c2r(planinv, outrez, in);
    // make destination frame plane from current overlaped blocks
    FFT3DFilter::DecodeOverlapPlane(in, norm, coverbuf, coverpitch, planeBase, planeBase_f);
    CoverbufToFramePlane(plane, coverbuf, coverwidth, coverheight, coverpitch, dst, vi, mirw, mirh, interlaced, bits_per_pixel, env);

  }

  if (btcur == bt)
  {// for normal step
    nlast = n; // set last frame to current
  }
  btcurlast = btcur;

  // As we now are finished processing the image, we return the destination image.
  _RPT2(0, "FFT3DFilter GetFrame END, frame=%d instance_id=%d\n", n, _instance_id);
  reentrancy_check = false;
  return dst;
}

//-------------------------------------------------------------------------------------------

// This is the function that created the filter, when the filter has been called.
// This can be used for simple parameter checking, so it is possible to create different filters,
// based on the arguments recieved.

// PF: never called. Create_FFT3DFilterMulti is the real filter
AVSValue __cdecl Create_FFT3DFilter(AVSValue args, void* user_data, IScriptEnvironment* env)
{
  // Calls the constructor with the arguments provided.
  float sigma1 = (float)args[1].AsFloat(2.0);

  return new FFT3DFilter(args[0].AsClip(), // the 0th parameter is the source clip
    sigma1, // sigma
    (float)args[2].AsFloat(1.0), // beta
    args[3].AsInt(0), // plane
    args[4].AsInt(48), // bw -new default in v.1.2
    args[5].AsInt(48), // bh -new default in v.1.2
    args[6].AsInt(3), //  bt (=0 for Kalman mode) // new default=3 in v.0.9.3
    args[7].AsInt(-1), //  ow
    args[8].AsInt(-1), //  oh
    (float)args[9].AsFloat(2.0f), // kratio for Kalman mode
    (float)args[10].AsFloat(0.0f), // sharpen strength
    (float)args[11].AsFloat(0.3f), // sharpen cufoff frequency (relative to max) - v1.7
    (float)args[12].AsFloat(1.0f), // svr - sharpen vertical ratio
    (float)args[13].AsFloat(4.0f), // smin -  minimum limit for sharpen (prevent noise amplifying)
    (float)args[14].AsFloat(20.0f), // smax - maximum limit for sharpen (prevent oversharping)
    args[15].AsBool(true), // measure - switched to true in v.0.9.2
    args[16].AsBool(false), // interlaced - v.1.3
    args[17].AsInt(0), // wintype - v1.4, v1.8
    args[18].AsInt(0), //  pframe
    args[19].AsInt(0), //  px
    args[20].AsInt(0), //  py
    args[21].AsBool(false), //  pshow
    (float)args[22].AsFloat(0.1f), //  pcutoff
    (float)args[23].AsFloat(0.0f), //  pfactor
    (float)args[24].AsFloat(sigma1), // sigma2
    (float)args[25].AsFloat(sigma1), // sigma3
    (float)args[26].AsFloat(sigma1), // sigma4
    (float)args[27].AsFloat(1.0f), // degrid
    (float)args[28].AsFloat(0.0f), // dehalo
    (float)args[29].AsFloat(2.0f), // halo radius
    (float)args[30].AsFloat(50.0f), // halo threshold - v 1.9
    args[31].AsInt(1), //  ncpu
    args[32].AsInt(0), //  multiplane
    env);
}
//-------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------
class FFT3DFilterMulti : public GenericVideoFilter {
  // FFT3DFilter defines the name of your filter class.
  // This name is only used internally, and does not affect the name of your filter or similar.
  // This filter extends GenericVideoFilter, which incorporates basic functionality.
  // All functions present in the filter must also be present here.
  int _instance_id; // debug unique id
  std::atomic<bool> reentrancy_check;

  PClip filtered;
  PClip YClip, UClip, VClip;
  int multiplane;

  // avs+
  int pixelsize;
  int bits_per_pixel;

public:
  // This defines that these functions are present in your class.
  // These functions must be that same as those actually implemented.
  // Since the functions are "public" they are accessible to other classes.
  // Otherwise they can only be called from functions within the class itself.

  FFT3DFilterMulti(PClip _child, float _sigma, float _beta, int _plane, int _bw, int _bh, int _bt, int _ow, int _oh,
    float _kratio, float _sharpen, float _scutoff, float _svr, float _smin, float _smax,
    bool _measure, bool _interlaced, int _wintype,
    int _pframe, int _px, int _py, bool _pshow, float _pcutoff, float _pfactor,
    float _sigma2, float _sigma3, float _sigma4, float _degrid,
    float _dehalo, float _hr, float _ht, int _ncpu, IScriptEnvironment* env);
  // This is the constructor. It does not return any value, and is always used,
  //  when an instance of the class is created.
  // Since there is no code in this, this is the definition.

  ~FFT3DFilterMulti();
  // The is the destructor definition. This is called when the filter is destroyed.


  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
  // This is the function that AviSynth calls to get a given frame.
  // So when this functions gets called, the filter is supposed to return frame n.

  // Auto register AVS+ mode: serialized
  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    return cachehints == CACHE_GET_MTMODE ? MT_SERIALIZED : 0;
  }

};


//-------------------------------------------------------------------

// The following is the implementation
// of the defined functions.

//Here is the acutal constructor code used
FFT3DFilterMulti::FFT3DFilterMulti(PClip _child, float _sigma, float _beta, int _multiplane, int _bw, int _bh, int _bt, int _ow, int _oh,
  float _kratio, float _sharpen, float _scutoff, float _svr, float _smin, float _smax,
  bool _measure, bool _interlaced, int _wintype,
  int _pframe, int _px, int _py, bool _pshow, float _pcutoff, float _pfactor,
  float _sigma2, float _sigma3, float _sigma4, float _degrid,
  float _dehalo, float _hr, float _ht, int _ncpu, IScriptEnvironment* env) :

  GenericVideoFilter(_child) {

  static int id = 0; _instance_id = id++;
  reentrancy_check = false;
  _RPT1(0, "FFT3DFilterMulti.Create instance_id=%d\n", _instance_id);

  pixelsize = vi.ComponentSize();
  bits_per_pixel = vi.BitsPerComponent();

  // adaptive default: all planes for RGB
  if (_multiplane == -1) {
    if (vi.IsRGB())
      _multiplane = 4;
    else
      _multiplane = 0;
  }
  /*
  if (bits_per_pixel > 8)
    env->ThrowError("FFT3DFilter: only 8 bit clips are allowed!");
    */

  // This is the implementation of the constructor.
  // The child clip (source clip) is inherited by the GenericVideoFilter,
  //  where the following variables gets defined:
  //   PClip child;   // Contains the source clip.
  //   VideoInfo vi;  // Contains videoinfo on the source clip.

  // convert parameters for RGB clip
  // planar RGB internal frame order G B R, plane=0->R, plane=1->G, plane2->B
  if (vi.IsRGB() && vi.IsPlanar())
  {
    switch (_multiplane) {
    case 0: _multiplane = 2; break; // R
    case 1: break; // B
    case 2: _multiplane = 0; break;
    case 3: break; // n/a
    case 4: break; // all planes
    }
  }

  if (vi.IsY() && _multiplane == 4)
    _multiplane = 0; // no accidental U and V!

  multiplane = _multiplane;

  /*
  _multiplane == 0 : process Y, copy U, copy V
  _multiplane == 1 : copy Y, process U, copy V
  _multiplane == 2 : copy Y, copy U, process V
  _multiplane == 3 : copy Y, process U, process V
  _multiplane == 4 : process Y, process U, process V
  */


  if (_multiplane == 0 || _multiplane == 1 || _multiplane == 2)
  {
    // fallback to single plane mode
    filtered = new FFT3DFilter(_child, _sigma, _beta, _multiplane, _bw, _bh, _bt, _ow, _oh,
      _kratio, _sharpen, _scutoff, _svr, _smin, _smax,
      _measure, _interlaced, _wintype,
      _pframe, _px, _py, _pshow, _pcutoff, _pfactor,
      _sigma2, _sigma3, _sigma4, _degrid, _dehalo, _hr, _ht, _ncpu, _multiplane, env);
  }
  else if (_multiplane == 3 || _multiplane == 4)
  {
    UClip = new FFT3DFilter(_child, _sigma, _beta, 1, _bw, _bh, _bt, _ow, _oh,
      _kratio, _sharpen, _scutoff, _svr, _smin, _smax,
      _measure, _interlaced, _wintype,
      _pframe, _px, _py, _pshow, _pcutoff, _pfactor,
      _sigma2, _sigma3, _sigma4, _degrid, _dehalo, _hr, _ht, _ncpu, _multiplane, env);

    VClip = new FFT3DFilter(_child, _sigma, _beta, 2, _bw, _bh, _bt, _ow, _oh,
      _kratio, _sharpen, _scutoff, _svr, _smin, _smax,
      _measure, _interlaced, _wintype,
      _pframe, _px, _py, _pshow, _pcutoff, _pfactor,
      _sigma2, _sigma3, _sigma4, _degrid, _dehalo, _hr, _ht, _ncpu, _multiplane, env);

    if (_multiplane == 3)
    {
      YClip = _child;
    }
    else
    {
      YClip = new FFT3DFilter(_child, _sigma, _beta, 0, _bw, _bh, _bt, _ow, _oh,
        _kratio, _sharpen, _scutoff, _svr, _smin, _smax,
        _measure, _interlaced, _wintype,
        _pframe, _px, _py, _pshow, _pcutoff, _pfactor,
        _sigma2, _sigma3, _sigma4, _degrid, _dehalo, _hr, _ht, _ncpu, _multiplane, env);
    }

    // replaced by internal processing in v1.9.2
    //			AVSValue argsUToY[1] = { UClip };
    //			UClip = env->Invoke("UToY", AVSValue(argsUToY,1)).AsClip();
    //			AVSValue argsVToY[1] = { VClip };
    //			VClip = env->Invoke("VToY", AVSValue(argsVToY,1)).AsClip();
    //			AVSValue argsYToUV[3] = { UClip, VClip, YClip };
    //			filtered = env->Invoke("YToUV", AVSValue(argsYToUV,3)).AsClip();
  }
  else
    env->ThrowError("FFT3DFilter: plane must be from 0 to 4!");

}

// This is where any actual destructor code used goes
FFT3DFilterMulti::~FFT3DFilterMulti() {
  // This is where you can deallocate any memory you might have used.
}

PVideoFrame __stdcall FFT3DFilterMulti::GetFrame(int n, IScriptEnvironment* env) {
  // This is the implementation of the GetFrame function.
  // See the header definition for further info.
  _RPT2(0, "FFT3DFilterMulti GetFrame, frame=%d instance_id=%d\n", n, _instance_id);
  if (reentrancy_check) {
    _RPT2(0, "FFT3DFilterMulti GetFrame, Reentrant call detected! Frame=%d instance_id=%d\n", n, _instance_id);
    env->ThrowError("FFT3DFilterMulti: cannot work in reentrant multithread mode!");
  }
  reentrancy_check = true;

  PVideoFrame dst;
  if (multiplane < 3)
    dst = filtered->GetFrame(n, env);
  else
  {
    _RPT2(0, "FFT3DFilterMulti GetFrame, before fY Frame=%d instance_id=%d\n", n, _instance_id);
    PVideoFrame fY = YClip->GetFrame(n, env);
    PVideoFrame fU = UClip->GetFrame(n, env);
    PVideoFrame fV = VClip->GetFrame(n, env);
    _RPT2(0, "FFT3DFilterMulti GetFrame, have all fY, fU, fV Frame=%d instance_id=%d\n", n, _instance_id);
    dst = env->NewVideoFrame(vi);
    if (vi.IsPlanar())
    {
      env->BitBlt(dst->GetWritePtr(PLANAR_Y), dst->GetPitch(PLANAR_Y), fY->GetReadPtr(PLANAR_Y),
        fY->GetPitch(PLANAR_Y), fY->GetRowSize(PLANAR_Y), fY->GetHeight(PLANAR_Y));
      env->BitBlt(dst->GetWritePtr(PLANAR_U), dst->GetPitch(PLANAR_U), fU->GetReadPtr(PLANAR_U),
        fU->GetPitch(PLANAR_U), fU->GetRowSize(PLANAR_U), fU->GetHeight(PLANAR_U));
      env->BitBlt(dst->GetWritePtr(PLANAR_V), dst->GetPitch(PLANAR_V), fV->GetReadPtr(PLANAR_V),
        fV->GetPitch(PLANAR_V), fV->GetRowSize(PLANAR_V), fV->GetHeight(PLANAR_V));
    }
    else if (vi.IsY8())
    {
      env->BitBlt(dst->GetWritePtr(PLANAR_Y), dst->GetPitch(PLANAR_Y), fY->GetReadPtr(PLANAR_Y),
        fY->GetPitch(PLANAR_Y), fY->GetRowSize(PLANAR_Y), fY->GetHeight(PLANAR_Y));
    }
    else // YUY2 - not optimal
    {
      int height = dst->GetHeight();
      int width = dst->GetRowSize();
      BYTE * pdst = dst->GetWritePtr();
      const BYTE * pY = fY->GetReadPtr();
      const BYTE * pU = fU->GetReadPtr();
      const BYTE * pV = fV->GetReadPtr();
      for (int h = 0; h < height; h++)
      {
        for (int w = 0; w < width; w += 4)
        {
          pdst[w] = pY[w];
          pdst[w + 1] = pU[w + 1];
          pdst[w + 2] = pY[w + 2];
          pdst[w + 3] = pV[w + 3];
        }
        pdst += dst->GetPitch();
        pY += fY->GetPitch();
        pU += fU->GetPitch();
        pV += fV->GetPitch();
      }
    }

  }
  reentrancy_check = false;
  return dst;
}

AVSValue __cdecl Create_FFT3DFilterMulti(AVSValue args, void* user_data, IScriptEnvironment* env)
{
  // Calls the constructor with the arguments provided.
  float sigma1 = (float)args[1].AsFloat(2.0);

  // v2.3 pre-check: if plane to process for greyscale is U and/or V return original clip
  int plane = args[3].AsInt(-1);
  if (args[0].AsClip()->GetVideoInfo().IsY()) {
    if (plane == 1 || plane == 2 || plane == 3) // U, V, U and V
      return args[0].AsClip();
  }

  return new FFT3DFilterMulti(args[0].AsClip(), // the 0th parameter is the source clip
    sigma1, // sigma
    (float)args[2].AsFloat(1.0), // beta
    args[3].AsInt(-1), // plane. default to -1 to allow adaptive default 4 for RGB, 0 for YUV
    args[4].AsInt(32), // bw - changed default from 48 to 32 in v.1.9.2
    args[5].AsInt(32), // bh - changed default from 48 to 32 in v.1.9.2
    args[6].AsInt(3), //  bt (=0 for Kalman mode) // new default=3 in v.0.9.3
    args[7].AsInt(-1), //  ow
    args[8].AsInt(-1), //  oh
    (float)args[9].AsFloat(2.0f), // kratio for Kalman mode
    (float)args[10].AsFloat(0.0f), // sharpen strength
    (float)args[11].AsFloat(0.3f), // sharpen cutoff frequency (relative to max) - v1.7
    (float)args[12].AsFloat(1.0f), // svr - sharpen vertical ratio
    (float)args[13].AsFloat(4.0f), // smin -  minimum limit for sharpen (prevent noise amplifying)
    (float)args[14].AsFloat(20.0f), // smax - maximum limit for sharpen (prevent oversharping)
    args[15].AsBool(true), // measure - switched to true in v.0.9.2
    args[16].AsBool(false), // interlaced - v.1.3
    args[17].AsInt(0), // wintype - v1.4, v1.8
    args[18].AsInt(0), //  pframe
    args[19].AsInt(0), //  px
    args[20].AsInt(0), //  py
    args[21].AsBool(false), //  pshow
    (float)args[22].AsFloat(0.1f), //  pcutoff
    (float)args[23].AsFloat(0.0f), //  pfactor
    (float)args[24].AsFloat(sigma1), // sigma2
    (float)args[25].AsFloat(sigma1), // sigma3
    (float)args[26].AsFloat(sigma1), // sigma4
    (float)args[27].AsFloat(1.0f), // degrid
    (float)args[28].AsFloat(0.0f), // dehalo - v 1.9
    (float)args[29].AsFloat(2.0f), // halo radius - v 1.9
    (float)args[30].AsFloat(50.0f), // halo threshold - v 1.9
    args[31].AsInt(1), //  ncpu
    env);
}

//-------------------------------------------------------------------------------------------

AVSValue __cdecl FFT3DFilter_VersionNumber(AVSValue args, void* user_data, IScriptEnvironment* env) {
  double v = VERSION_NUMBER;
  return  v;
}

//-------------------------------------------------------------------------------------------

// The following function is the function that actually registers the filter in AviSynth
// It is called automatically, when the plugin is loaded to see which functions this filter contains.

// new V2.6 requirement
const AVS_Linkage *AVS_linkage = NULL;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {

  AVS_linkage = vectors;

  env->AddFunction("FFT3DFilter_VersionNumber", "", FFT3DFilter_VersionNumber, 0);

  env->AddFunction("FFT3DFilter", "c[sigma]f[beta]f[plane]i[bw]i[bh]i[bt]i[ow]i[oh]i[kratio]f[sharpen]f[scutoff]f[svr]f[smin]f[smax]f[measure]b[interlaced]b[wintype]i[pframe]i[px]i[py]i[pshow]b[pcutoff]f[pfactor]f[sigma2]f[sigma3]f[sigma4]f[degrid]f[dehalo]f[hr]f[ht]f[ncpu]i", Create_FFT3DFilterMulti, 0);

  // The AddFunction has the following parameters:
    // AddFunction(Filtername , Arguments, Function to call,0);

    // Arguments is a string that defines the types and optional names of the arguments for you filter.
    // c - Video Clip
    // i - Integer number
    // f - Float number
    // s - String
    // b - boolean


  return "`FFT3DFilter' FFT3DFilter plugin";

}

