//
//	FFT3DFilter plugin for Avisynth 2.5 - 3D Frequency Domain filter
//  pure C++ filtering functions
//  v1.9.2
//	Copyright(C)2004-2006 A.G.Balakhnin aka Fizick, bag@hotmail.ru, http://avisynth.org.ru
//
//	This program is free software; you can redistribute it and/or modify
//	it under the terms of the GNU General Public License version 2 as published by
//	the Free Software Foundation.
//
//	This program is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//	GNU General Public License for more details.
//
//	You should have received a copy of the GNU General Public License
//	along with this program; if not, write to the Free Software
//	Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
//
//-----------------------------------------------------------------------------------------

#ifdef _WIN32
#define NOMINMAX
#include "Windows.h"
#endif

#include "fftwlite.h"
#include "math.h" // for sqrtf
#include <algorithm>

// since v1.7 we use outpitch instead of outwidth

//-----------------------------------------------------------------------------------------
//
void ApplyWiener2D_C(fftwf_complex *outcur, int outwidth, int outpitch, int bh, 
				   int howmanyblocks, float sigmaSquaredNoiseNormed, float beta, 
				   float sharpen, float sigmaSquaredSharpenMin, 
				   float sigmaSquaredSharpenMax, float *wsharpen, float dehalo, float *wdehalo, float ht2n)
{
	// this function take 25% CPU time and may be easy optimized for AMD Athlon 3DNOW assembler
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0
	int h,w, block;
	float psd;
	float WienerFactor;


	if (sharpen == 0 && dehalo == 0)// no sharpen, no dehalo
	{
		for (block =0; block <howmanyblocks; block++)
		{
			for (h=0; h<bh; h++) // middle
			{
				for (w=0; w<outwidth; w++) // not skip first v.1.2
				{
					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]) + 1e-15f;// power spectrum density
					WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
					outcur[w][0] *= WienerFactor; // apply filter on real  part	
					outcur[w][1] *= WienerFactor; // apply filter on imaginary part
				}
				outcur += outpitch;
			}
		}
	}
	else if (sharpen != 0 && dehalo==0) // sharpen
	{
		for (block =0; block <howmanyblocks; block++)
		{
			for (h=0; h<bh; h++) // middle
			{
				for (w=0; w<outwidth; w++) // not skip first
				{
					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]) + 1e-15f;// power spectrum density
					WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
					WienerFactor *= 1 + sharpen*wsharpen[w]*sqrt( psd*sigmaSquaredSharpenMax/((psd + sigmaSquaredSharpenMin)*(psd + sigmaSquaredSharpenMax)) ); // sharpen factor - changed in v.1.1
					outcur[w][0] *= WienerFactor; // apply filter on real  part	
					outcur[w][1] *= WienerFactor; // apply filter on imaginary part
				}
				outcur += outpitch;
				wsharpen += outpitch;
			}
			wsharpen -= outpitch*bh;
		}
	}
	else if (sharpen == 0 && dehalo != 0)
	{
		for (block =0; block <howmanyblocks; block++)
		{
			for (h=0; h<bh; h++) // middle
			{
				for (w=0; w<outwidth; w++) // not skip first
				{
					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]) + 1e-15f;// power spectrum density
					WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
					WienerFactor *= (psd + ht2n)/((psd + ht2n) + dehalo*wdehalo[w] * psd ); 
					outcur[w][0] *= WienerFactor; // apply filter on real  part	
					outcur[w][1] *= WienerFactor; // apply filter on imaginary part
				}
				outcur += outpitch;
				wdehalo += outpitch;
			}
			wdehalo -= outpitch*bh;
		}
	}
	else if (sharpen != 0 && dehalo != 0)
	{
		for (block =0; block <howmanyblocks; block++)
		{
			for (h=0; h<bh; h++) // middle
			{
				for (w=0; w<outwidth; w++) // not skip first
				{
					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]) + 1e-15f;// power spectrum density
					WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
          WienerFactor *= 1 + sharpen * wsharpen[w] * sqrt(psd * sigmaSquaredSharpenMax / ((psd + sigmaSquaredSharpenMin) * (psd + sigmaSquaredSharpenMax)));
          WienerFactor *= (psd + ht2n)/((psd + ht2n) + dehalo*wdehalo[w] * psd );
					outcur[w][0] *= WienerFactor; // apply filter on real  part	
					outcur[w][1] *= WienerFactor; // apply filter on imaginary part
				}
				outcur += outpitch;
				wsharpen += outpitch;
				wdehalo += outpitch;
			}
			wsharpen -= outpitch*bh;
			wdehalo -= outpitch*bh;
		}
	}
		
}
//-------------------------------------------------------------------------------------------
//
void ApplyPattern2D_C(fftwf_complex *outcur, int outwidth, int outpitch, int bh, int howmanyblocks, float pfactor, float *pattern2d0, float beta)
{
	int h,w, block;
	float psd;
	float patternfactor;
	float *pattern2d;
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0

	if (pfactor != 0)
	{

		for (block =0; block <howmanyblocks; block++)
		{
			pattern2d = pattern2d0;
			for (h=0; h<bh; h++) // middle
			{
				for (w=0; w<outwidth; w++) 
				{
					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]) + 1e-15f;
					patternfactor = std::max((psd - pfactor*pattern2d[w])/psd, lowlimit);
					outcur[w][0] *= patternfactor;
					outcur[w][1] *= patternfactor;
				}
				outcur += outpitch;
				pattern2d += outpitch;
			}
		}
	}
}
//
//-----------------------------------------------------------------------------------------
//
void ApplyWiener3D2_C(fftwf_complex *outcur, fftwf_complex *outprev, 
					int outwidth, int outpitch, int bh, int howmanyblocks, 
					float sigmaSquaredNoiseNormed, 	float beta)
{
	// this function take 25% CPU time and may be easy optimized for AMD Athlon 3DNOW assembler
	// return result in outprev
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0
	float psd;
	float WienerFactor;
	float f3d0r, f3d1r, f3d0i, f3d1i;
	int block;
	int h,w;

	for (block=0; block <howmanyblocks; block++)
	{
		for (h=0; h<bh; h++)  
		{
			for (w=0; w<outwidth; w++) 
			{
				// dft 3d (very short - 2 points)
				f3d0r =  outcur[w][0] + outprev[w][0]; // real 0 (sum)
				f3d0i =  outcur[w][1] + outprev[w][1]; // im 0 (sum)
				f3d1r =  outcur[w][0] - outprev[w][0]; // real 1 (dif)
				f3d1i =  outcur[w][1] - outprev[w][1]; // im 1 (dif)
				psd = f3d0r*f3d0r + f3d0i*f3d0i + 1e-15f; // power spectrum density 0
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				f3d0r *= WienerFactor; // apply filter on real  part	
				f3d0i *= WienerFactor; // apply filter on imaginary part
				psd = f3d1r*f3d1r + f3d1i*f3d1i + 1e-15f; // power spectrum density 1
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				f3d1r *= WienerFactor; // apply filter on real  part	
				f3d1i *= WienerFactor; // apply filter on imaginary part
				// reverse dft for 2 points
				outprev[w][0] = (f3d0r + f3d1r)*0.5f; // get  real  part	
				outprev[w][1] = (f3d0i + f3d1i)*0.5f; // get imaginary part
				// Attention! return filtered "out" in "outprev" to preserve "out" for next step
			}
			outcur += outpitch;
			outprev += outpitch;
		}
	}
}
//
//-----------------------------------------------------------------------------------------
//
void ApplyPattern3D2_C(fftwf_complex *outcur, fftwf_complex *outprev, 
					 int outwidth, int outpitch, int bh, 
					 int howmanyblocks, float *pattern3d, float beta)
{
	// this function take 25% CPU time and may be easy optimized for AMD Athlon 3DNOW assembler
	// return result in outprev
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0
	float psd;
	float WienerFactor;
	float f3d0r, f3d1r, f3d0i, f3d1i;
	int block;
	int h,w;

	for (block=0; block <howmanyblocks; block++)
	{
		for (h=0; h<bh; h++)  
		{
			for (w=0; w<outwidth; w++) 
			{
				// dft 3d (very short - 2 points)
				f3d0r =  outcur[w][0] + outprev[w][0]; // real 0 (sum)
				f3d0i =  outcur[w][1] + outprev[w][1]; // im 0 (sum)
				f3d1r =  outcur[w][0] - outprev[w][0]; // real 1 (dif)
				f3d1i =  outcur[w][1] - outprev[w][1]; // im 1 (dif)
				psd = f3d0r*f3d0r + f3d0i*f3d0i + 1e-15f; // power spectrum density 0
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				f3d0r *= WienerFactor; // apply filter on real  part	
				f3d0i *= WienerFactor; // apply filter on imaginary part
				psd = f3d1r*f3d1r + f3d1i*f3d1i + 1e-15f; // power spectrum density 1
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				f3d1r *= WienerFactor; // apply filter on real  part	
				f3d1i *= WienerFactor; // apply filter on imaginary part
				// reverse dft for 2 points
				outprev[w][0] = (f3d0r + f3d1r)*0.5f; // get  real  part	
				outprev[w][1] = (f3d0i + f3d1i)*0.5f; // get imaginary part
				// Attention! return filtered "out" in "outprev" to preserve "out" for next step
			}
			outcur += outpitch;
			outprev += outpitch;
			pattern3d += outpitch;
		}
		pattern3d -= outpitch*bh; // restore pointer for new block
	}
}
//
//-----------------------------------------------------------------------------------------
//
void ApplyWiener3D3_C(fftwf_complex *outcur, fftwf_complex *outprev, fftwf_complex *outnext, 
					int outwidth, int outpitch, int bh, int howmanyblocks, 
					float sigmaSquaredNoiseNormed, float beta)
{
	// this function take 25% CPU time and may be easy optimized for AMD Athlon 3DNOW assembler
	// return result in outprev
	float fcr, fci, fpr, fpi, fnr, fni;
	float WienerFactor;
	float psd;
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0
	float sin120 = 0.86602540378443864676372317075294f;//sqrtf(3.0f)*0.5f;

	int block;
	int h,w;

	for (block=0; block <howmanyblocks; block++)
	{
		for (h=0; h<bh; h++) // first half
		{
			for (w=0; w<outwidth; w++) // 
			{
				// dft 3d (very short - 3 points)
				float pnr = outprev[w][0] + outnext[w][0];
				float pni = outprev[w][1] + outnext[w][1];
				fcr = outcur[w][0] + pnr; // real cur
				fci = outcur[w][1] + pni; // im cur
				float di = sin120*(outprev[w][1]-outnext[w][1]);
				float dr = sin120*(outnext[w][0]-outprev[w][0]);
				fpr = outcur[w][0] - 0.5f*pnr + di; // real prev
//				fnr = outcur[w][0] - 0.5f*pnr - di; // real next
				fnr = fpr - di - di; //v1.8.1
				fpi = outcur[w][1] - 0.5f*pni + dr; // im prev
//				fni = outcur[w][1] - 0.5f*pni - dr ; // im next
				fni = fpi - dr - dr; //v1.8.1
				psd = fcr*fcr + fci*fci + 1e-15f; // power spectrum density cur
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				fcr *= WienerFactor; // apply filter on real  part	
				fci *= WienerFactor; // apply filter on imaginary part
				psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density prev
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				fpr *= WienerFactor; // apply filter on real  part	
				fpi *= WienerFactor; // apply filter on imaginary part
				psd = fnr*fnr + fni*fni + 1e-15f; // power spectrum density next
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				fnr *= WienerFactor; // apply filter on real  part	
				fni *= WienerFactor; // apply filter on imaginary part
				// reverse dft for 3 points
				outprev[w][0] = (fcr + fpr + fnr)*0.33333333333f; // get  real  part	
				outprev[w][1] = (fci + fpi + fni)*0.33333333333f; // get imaginary part
				// Attention! return filtered "out" in "outprev" to preserve "out" for next step
			}
			outcur += outpitch;
			outprev += outpitch;
			outnext += outpitch;
		}
	}
}
//-----------------------------------------------------------------------------------------
//
void ApplyPattern3D3_C(fftwf_complex *outcur, fftwf_complex *outprev, fftwf_complex *outnext, 
					 int outwidth, int outpitch, int bh, int howmanyblocks, 
					 float *pattern3d, float beta)
{
	// this function take 25% CPU time and may be easy optimized for AMD Athlon 3DNOW assembler
	// return result in outprev
	float fcr, fci, fpr, fpi, fnr, fni;
	float WienerFactor;
	float psd;
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0
	float sin120 = 0.86602540378443864676372317075294f;//sqrtf(3.0f)*0.5f;

	int block;
	int h,w;

	for (block=0; block <howmanyblocks; block++)
	{
		for (h=0; h<bh; h++) // first half
		{
			for (w=0; w<outwidth; w++) // 
			{
				// dft 3d (very short - 3 points)
				float pnr = outprev[w][0] + outnext[w][0];
				float pni = outprev[w][1] + outnext[w][1];
				fcr = outcur[w][0] + pnr; // real cur
				fci = outcur[w][1] + pni; // im cur
				float di = sin120*(outprev[w][1]-outnext[w][1]);
				float dr = sin120*(outnext[w][0]-outprev[w][0]);
				fpr = outcur[w][0] - 0.5f*pnr + di; // real prev
				fnr = outcur[w][0] - 0.5f*pnr - di; // real next
				fpi = outcur[w][1] - 0.5f*pni + dr; // im prev
				fni = outcur[w][1] - 0.5f*pni - dr ; // im next
				psd = fcr*fcr + fci*fci + 1e-15f; // power spectrum density cur
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fcr *= WienerFactor; // apply filter on real  part	
				fci *= WienerFactor; // apply filter on imaginary part
				psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density prev
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fpr *= WienerFactor; // apply filter on real  part	
				fpi *= WienerFactor; // apply filter on imaginary part
				psd = fnr*fnr + fni*fni + 1e-15f; // power spectrum density next
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fnr *= WienerFactor; // apply filter on real  part	
				fni *= WienerFactor; // apply filter on imaginary part
				// reverse dft for 3 points
				outprev[w][0] = (fcr + fpr + fnr)*0.33333333333f; // get  real  part	
				outprev[w][1] = (fci + fpi + fni)*0.33333333333f; // get imaginary part
				// Attention! return filtered "out" in "outprev" to preserve "out" for next step
			}
			outcur += outpitch;
			outprev += outpitch;
			outnext += outpitch;
			pattern3d += outpitch;
		}
		pattern3d -= outpitch*bh; // restore pointer for new block
	}
}


//-----------------------------------------------------------------------------------------
//
void ApplyWiener3D4_C(fftwf_complex *outcur, fftwf_complex *outprev2, fftwf_complex *outprev, 
					fftwf_complex *outnext, int outwidth, int outpitch, int bh, 
					int howmanyblocks, float sigmaSquaredNoiseNormed, float beta)
{
	// dft with 4 points
	// this function take 25% CPU time and may be easy optimized for AMD Athlon 3DNOW assembler
	// return result in outprev
	float fcr, fci, fpr, fpi, fnr, fni, fp2r, fp2i;
	float WienerFactor;
	float psd;
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0

	int block;
	int h,w;

	for (block=0; block <howmanyblocks; block++)
	{
		for (h=0; h<bh; h++) // first half
		{
			for (w=0; w<outwidth; w++) // 
			{
				// dft 3d (very short - 4 points)
				fpr = -outprev2[w][0] + outprev[w][1] + outcur[w][0] - outnext[w][1]; // real prev
				fpi = -outprev2[w][1] - outprev[w][0] + outcur[w][1] + outnext[w][0]; // im cur
				fcr = outprev2[w][0] + outprev[w][0] + outcur[w][0] + outnext[w][0]; // real cur
				fci = outprev2[w][1] + outprev[w][1] + outcur[w][1] + outnext[w][1]; // im cur
				fnr = -outprev2[w][0] - outprev[w][1] + outcur[w][0] + outnext[w][1]; // real next
				fni = -outprev2[w][1] + outprev[w][0] + outcur[w][1] - outnext[w][0]; // im next
				fp2r = outprev2[w][0] - outprev[w][0] + outcur[w][0] - outnext[w][0]; // real prev2
				fp2i = outprev2[w][1] - outprev[w][1] + outcur[w][1] - outnext[w][1]; // im cur

				psd = fp2r*fp2r + fp2i*fp2i + 1e-15f; // power spectrum density prev2
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				fp2r *= WienerFactor; // apply filter on real  part	
				fp2i *= WienerFactor; // apply filter on imaginary part

				psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density prev
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				fpr *= WienerFactor; // apply filter on real  part	
				fpi *= WienerFactor; // apply filter on imaginary part

				psd = fcr*fcr + fci*fci + 1e-15f; // power spectrum density cur
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				fcr *= WienerFactor; // apply filter on real  part	
				fci *= WienerFactor; // apply filter on imaginary part

				psd = fnr*fnr + fni*fni + 1e-15f; // power spectrum density next
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				fnr *= WienerFactor; // apply filter on real  part	
				fni *= WienerFactor; // apply filter on imaginary part

				// reverse dft for 4 points
				outprev2[w][0] = (fp2r + fpr + fcr + fnr)*0.25f; // get  real  part	
				outprev2[w][1] = (fp2i + fpi + fci + fni)*0.25f; // get imaginary part
				// Attention! return filtered "out" in "outprev2" to preserve "out" for next step
			}
			outcur += outpitch;
			outprev2 += outpitch;
			outprev += outpitch;
			outnext += outpitch;
		}
	}
}
//-----------------------------------------------------------------------------------------
//
void ApplyPattern3D4_C(fftwf_complex *outcur, fftwf_complex *outprev2, fftwf_complex *outprev, 
					 fftwf_complex *outnext, int outwidth, int outpitch, int bh, 
					 int howmanyblocks, float* pattern3d, float beta)
{
	// dft with 4 points
	// this function take 25% CPU time and may be easy optimized for AMD Athlon 3DNOW assembler
	// return result in outprev
	float fcr, fci, fpr, fpi, fnr, fni, fp2r, fp2i;
	float WienerFactor;
	float psd;
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0

	int block;
	int h,w;

	for (block=0; block <howmanyblocks; block++)
	{
		for (h=0; h<bh; h++) // first half
		{
			for (w=0; w<outwidth; w++) // 
			{
				// dft 3d (very short - 4 points)
				fp2r = outprev2[w][0] - outprev[w][0] + outcur[w][0] - outnext[w][0]; // real prev2
				fp2i = outprev2[w][1] - outprev[w][1] + outcur[w][1] - outnext[w][1]; // im cur
				fpr = -outprev2[w][0] + outprev[w][1] + outcur[w][0] - outnext[w][1]; // real prev
				fpi = -outprev2[w][1] - outprev[w][0] + outcur[w][1] + outnext[w][0]; // im cur
				fcr = outprev2[w][0] + outprev[w][0] + outcur[w][0] + outnext[w][0]; // real cur
				fci = outprev2[w][1] + outprev[w][1] + outcur[w][1] + outnext[w][1]; // im cur
				fnr = -outprev2[w][0] - outprev[w][1] + outcur[w][0] + outnext[w][1]; // real next
				fni = -outprev2[w][1] + outprev[w][0] + outcur[w][1] - outnext[w][0]; // im next

				psd = fp2r*fp2r + fp2i*fp2i + 1e-15f; // power spectrum density prev2
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fp2r *= WienerFactor; // apply filter on real  part	
				fp2i *= WienerFactor; // apply filter on imaginary part

				psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density prev
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fpr *= WienerFactor; // apply filter on real  part	
				fpi *= WienerFactor; // apply filter on imaginary part

				psd = fcr*fcr + fci*fci + 1e-15f; // power spectrum density cur
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fcr *= WienerFactor; // apply filter on real  part	
				fci *= WienerFactor; // apply filter on imaginary part

				psd = fnr*fnr + fni*fni + 1e-15f; // power spectrum density next
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fnr *= WienerFactor; // apply filter on real  part	
				fni *= WienerFactor; // apply filter on imaginary part

				// reverse dft for 4 points
				outprev2[w][0] = (fp2r + fpr + fcr + fnr)*0.25f; // get  real  part	
				outprev2[w][1] = (fp2i + fpi + fci + fni)*0.25f; // get imaginary part
				// Attention! return filtered "out" in "outprev2" to preserve "out" for next step
			}
			outcur += outpitch;
			outprev2 += outpitch;
			outprev += outpitch;
			outnext += outpitch;
			pattern3d += outpitch;
		}
		pattern3d -= outpitch*bh; // restore pointer
	}
}
//
//-----------------------------------------------------------------------------------------
//
void ApplyKalmanPattern_C( fftwf_complex *outcur, fftwf_complex *outLast, 
						fftwf_complex *covar, fftwf_complex *covarProcess,
						int outwidth, int outpitch, int bh, int howmanyblocks,  
						float *covarNoiseNormed, float kratio2)
{
// return result in outLast
	float GainRe, GainIm;  // Kalman Gain 
//	float filteredRe, filteredIm;
	float sumre, sumim;
	int block;
	int h,w;

	for (block=0; block <howmanyblocks; block++)
	{
		for (h=0; h<bh; h++) // 
		{
			for (w=0; w<outwidth; w++) 
			{
				// use one of possible method for motion detection:
				if ( (outcur[w][0]-outLast[w][0])*(outcur[w][0]-outLast[w][0]) > covarNoiseNormed[w]*kratio2 || 
				 	 (outcur[w][1]-outLast[w][1])*(outcur[w][1]-outLast[w][1]) > covarNoiseNormed[w]*kratio2 )
				{
					// big pixel variation due to motion etc
					// reset filter
					covar[w][0] = covarNoiseNormed[w]; 
					covar[w][1] = covarNoiseNormed[w]; 
					covarProcess[w][0] = covarNoiseNormed[w]; 
					covarProcess[w][1] = covarNoiseNormed[w]; 
					outLast[w][0] = outcur[w][0];
					outLast[w][1] = outcur[w][1];
					//return result in outLast
				}
				else
				{ // small variation
					// useful sum
					sumre = (covar[w][0] + covarProcess[w][0]);
					sumim = (covar[w][1] + covarProcess[w][1]);
					// real gain, imagine gain
					GainRe = sumre/(sumre + covarNoiseNormed[w]);
					GainIm = sumim/(sumim + covarNoiseNormed[w]);
					// update process
					covarProcess[w][0] = (GainRe*GainRe*covarNoiseNormed[w]);
					covarProcess[w][1] = (GainIm*GainIm*covarNoiseNormed[w]);
					// update variation
					covar[w][0] =  (1-GainRe)*sumre ;
					covar[w][1] =  (1-GainIm)*sumim ;
					outLast[w][0] = ( GainRe*outcur[w][0] + (1 - GainRe)*outLast[w][0] );
					outLast[w][1] = ( GainIm*outcur[w][1] + (1 - GainIm)*outLast[w][1] );
					//return filtered result in outLast
				}
			}
			outcur += outpitch;
			outLast += outpitch; 
			covar += outpitch; 
			covarProcess += outpitch; 
			covarNoiseNormed += outpitch;
		}
		covarNoiseNormed -= outpitch*bh;
	}
	
}
//-----------------------------------------------------------------------------------------
//
void ApplyKalman_C( fftwf_complex *outcur, fftwf_complex *outLast, fftwf_complex *covar, 
				 fftwf_complex *covarProcess, int outwidth, int outpitch, int bh, 
				 int howmanyblocks,  float covarNoiseNormed, float kratio2)
{
// return result in outLast
	float GainRe, GainIm;  // Kalman Gain 
//	float filteredRe, filteredIm;
	float sumre, sumim;
	int block;
	int h,w;

	float sigmaSquaredMotionNormed = covarNoiseNormed*kratio2;

	for (block=0; block <howmanyblocks; block++)
	{
		for (h=0; h<bh; h++) // 
		{
			for (w=0; w<outwidth; w++) 
			{
				// use one of possible method for motion detection:
				if ( (outcur[w][0]-outLast[w][0])*(outcur[w][0]-outLast[w][0]) > sigmaSquaredMotionNormed || 
				 	 (outcur[w][1]-outLast[w][1])*(outcur[w][1]-outLast[w][1]) > sigmaSquaredMotionNormed )
				{
					// big pixel variation due to motion etc
					// reset filter
					covar[w][0] = covarNoiseNormed; 
					covar[w][1] = covarNoiseNormed; 
					covarProcess[w][0] = covarNoiseNormed; 
					covarProcess[w][1] = covarNoiseNormed; 
					outLast[w][0] = outcur[w][0];
					outLast[w][1] = outcur[w][1];
					//return result in outLast
				}
				else
				{ // small variation
					// useful sum
					sumre = (covar[w][0] + covarProcess[w][0]);
					sumim = (covar[w][1] + covarProcess[w][1]);
					// real gain, imagine gain
					GainRe = sumre/(sumre + covarNoiseNormed);
					GainIm = sumim/(sumim + covarNoiseNormed);
					// update process
					covarProcess[w][0] = (GainRe*GainRe*covarNoiseNormed);
					covarProcess[w][1] = (GainIm*GainIm*covarNoiseNormed);
					// update variation
					covar[w][0] =  (1-GainRe)*sumre ;
					covar[w][1] =  (1-GainIm)*sumim ;
					outLast[w][0] = ( GainRe*outcur[w][0] + (1 - GainRe)*outLast[w][0] );
					outLast[w][1] = ( GainIm*outcur[w][1] + (1 - GainIm)*outLast[w][1] );
					//return filtered result in outLast
				}
			}
			outcur += outpitch;
			outLast += outpitch; 
			covar += outpitch; 
			covarProcess += outpitch; 
		}
	}
	
}

//-------------------------------------------------------------------------------------------
//
void Sharpen_C(fftwf_complex *outcur, int outwidth, int outpitch, int bh, 
			 int howmanyblocks, float sharpen, float sigmaSquaredSharpenMin, 
			 float sigmaSquaredSharpenMax, float *wsharpen, float dehalo, float *wdehalo, float ht2n)
{
	int h,w, block;
	float psd;
	float sfact;

	if (sharpen != 0 && dehalo==0)
	{

		for (block =0; block <howmanyblocks; block++)
		{
			for (h=0; h<bh; h++) // middle
			{
				for (w=0; w<outwidth; w++) 
				{
					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]);
					//improved sharpen mode to prevent grid artifactes and to limit sharpening both fo low and high amplitudes
					sfact = (1 + sharpen*wsharpen[w]*sqrt( psd*sigmaSquaredSharpenMax/((psd + sigmaSquaredSharpenMin)*(psd + sigmaSquaredSharpenMax)) ) ) ;
					// sharpen factor - changed in v1.1c
					outcur[w][0] *= sfact;
					outcur[w][1] *= sfact;
				}
				outcur += outpitch;
				wsharpen += outpitch;
			}
			wsharpen -= outpitch*bh;
		}
	}
	else if (sharpen == 0 && dehalo != 0)
	{

		for (block =0; block <howmanyblocks; block++)
		{
			for (h=0; h<bh; h++) // middle
			{
				for (w=0; w<outwidth; w++) 
				{
					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]);
					//improved sharpen mode to prevent grid artifactes and to limit sharpening both fo low and high amplitudes
					sfact = (psd + ht2n)/((psd + ht2n) + dehalo*wdehalo[w] * psd ); 
					outcur[w][0] *= sfact;
					outcur[w][1] *= sfact;
				}
				outcur += outpitch;
				wdehalo += outpitch;
			}
			wdehalo -= outpitch*bh;
		}
	}
	else if (sharpen != 0 && dehalo != 0)
	{

		for (block =0; block <howmanyblocks; block++)
		{
			for (h=0; h<bh; h++) // middle
			{
				for (w=0; w<outwidth; w++) 
				{
					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]);
					//improved sharpen mode to prevent grid artifactes and to limit sharpening both fo low and high amplitudes
					sfact = (1 + sharpen*wsharpen[w]*sqrt( psd*sigmaSquaredSharpenMax/((psd + sigmaSquaredSharpenMin)*(psd + sigmaSquaredSharpenMax)) ) ) *
						(psd + ht2n) / ((psd + ht2n) + dehalo*wdehalo[w] * psd ); 
					outcur[w][0] *= sfact;
					outcur[w][1] *= sfact;
				}
				outcur += outpitch;
				wsharpen += outpitch;
				wdehalo += outpitch;
			}
			wsharpen -= outpitch*bh;
			wdehalo -= outpitch*bh;
		}
	}
}
//-----------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------
// DEGRID
//-----------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//
void Sharpen_degrid_C(fftwf_complex *outcur, int outwidth, int outpitch, int bh, 
			 int howmanyblocks, float sharpen, float sigmaSquaredSharpenMin, 
			 float sigmaSquaredSharpenMax, float *wsharpen, 
			 float degrid, fftwf_complex *gridsample, float dehalo, float *wdehalo, float ht2n)
{
	int h,w, block;
	float psd;
	float sfact;

	if (sharpen != 0 && dehalo==0)
	{

		for (block =0; block <howmanyblocks; block++)
		{
			float gridfraction = degrid*outcur[0][0]/gridsample[0][0];
			for (h=0; h<bh; h++) // middle
			{
				for (w=0; w<outwidth; w++) 
				{
					float gridcorrection0 = gridfraction*gridsample[w][0];
					float re = outcur[w][0] - gridcorrection0;
					float gridcorrection1 = gridfraction*gridsample[w][1];
					float im = outcur[w][1] - gridcorrection1;
					psd = (re*re + im*im) + 1e-15f;// power spectrum density
//					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]);
					//improved sharpen mode to prevent grid artifactes and to limit sharpening both fo low and high amplitudes
					sfact = (1 + sharpen*wsharpen[w]*sqrt( psd*sigmaSquaredSharpenMax/((psd + sigmaSquaredSharpenMin)*(psd + sigmaSquaredSharpenMax)) )) ; 
					// sharpen factor - changed in v1.1c
					re *= sfact; // apply filter on real  part	
					im *= sfact; // apply filter on imaginary part
					outcur[w][0] = re + gridcorrection0;
					outcur[w][1] = im + gridcorrection1;
				}
				outcur += outpitch;
				wsharpen += outpitch;
				gridsample += outpitch;
			}
			wsharpen -= outpitch*bh;
			gridsample -= outpitch*bh; // restore pointer to only valid first block - bug fixed in v1.8.1
		}
	}
	if (sharpen == 0 && dehalo != 0)
	{

		for (block =0; block <howmanyblocks; block++)
		{
			float gridfraction = degrid*outcur[0][0]/gridsample[0][0];
			for (h=0; h<bh; h++) // middle
			{
				for (w=0; w<outwidth; w++) 
				{
					float gridcorrection0 = gridfraction*gridsample[w][0];
					float re = outcur[w][0] - gridcorrection0;
					float gridcorrection1 = gridfraction*gridsample[w][1];
					float im = outcur[w][1] - gridcorrection1;
					psd = (re*re + im*im) + 1e-15f;// power spectrum density
//					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]);
					//improved sharpen mode to prevent grid artifactes and to limit sharpening both fo low and high amplitudes
					sfact = (psd + ht2n) / ((psd + ht2n) + dehalo*wdehalo[w] * psd ); 
					re *= sfact; // apply filter on real  part	
					im *= sfact; // apply filter on imaginary part
					outcur[w][0] = re + gridcorrection0;
					outcur[w][1] = im + gridcorrection1;
				}
				outcur += outpitch;
				wsharpen += outpitch;
				wdehalo += outpitch;
				gridsample += outpitch;
			}
			wsharpen -= outpitch*bh;
			wdehalo -= outpitch*bh;
			gridsample -= outpitch*bh; // restore pointer to only valid first block - bug fixed in v1.8.1
		}
	}
	if (sharpen != 0 && dehalo != 0)
	{

		for (block =0; block <howmanyblocks; block++)
		{
			float gridfraction = degrid*outcur[0][0]/gridsample[0][0];
			for (h=0; h<bh; h++) // middle
			{
				for (w=0; w<outwidth; w++) 
				{
					float gridcorrection0 = gridfraction*gridsample[w][0];
					float re = outcur[w][0] - gridcorrection0;
					float gridcorrection1 = gridfraction*gridsample[w][1];
					float im = outcur[w][1] - gridcorrection1;
					psd = (re*re + im*im) + 1e-15f;// power spectrum density
//					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]);
					//improved sharpen mode to prevent grid artifactes and to limit sharpening both fo low and high amplitudes
					sfact = (1 + sharpen*wsharpen[w]*sqrt( psd*sigmaSquaredSharpenMax/((psd + sigmaSquaredSharpenMin)*(psd + sigmaSquaredSharpenMax)) )) * 
						(psd + ht2n)/((psd + ht2n) + dehalo*wdehalo[w] * psd ); 
					re *= sfact; // apply filter on real  part	
					im *= sfact; // apply filter on imaginary part
					outcur[w][0] = re + gridcorrection0;
					outcur[w][1] = im + gridcorrection1;
				}
				outcur += outpitch;
				wsharpen += outpitch;
				wdehalo += outpitch;
				gridsample += outpitch;
			}
			wsharpen -= outpitch*bh;
			wdehalo -= outpitch*bh;
			gridsample -= outpitch*bh; // restore pointer to only valid first block - bug fixed in v1.8.1
		}
	}
}
//-----------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------
void ApplyWiener2D_degrid_C(fftwf_complex *outcur, int outwidth, int outpitch, int bh, 
				   int howmanyblocks, float sigmaSquaredNoiseNormed, float beta, 
				   float sharpen, float sigmaSquaredSharpenMin, 
				   float sigmaSquaredSharpenMax, float *wsharpen, 
				   float degrid, fftwf_complex *gridsample, float dehalo, float *wdehalo, float ht2n)
{
	// this function take 25% CPU time and may be easy optimized for AMD Athlon 3DNOW assembler
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0
	int h,w, block;
	float psd;
	float WienerFactor;
	float gridfraction; 

	if (sharpen == 0  && dehalo == 0)// no sharpen, no dehalo
	{
		for (block =0; block <howmanyblocks; block++)
		{
			gridfraction = degrid*outcur[0][0]/gridsample[0][0];
			for (h=0; h<bh; h++) // middle
			{
				for (w=0; w<outwidth; w++) // not skip first v.1.2
				{
					float gridcorrection0 = gridfraction*gridsample[w][0];
					float corrected0 = outcur[w][0] - gridcorrection0;
					float gridcorrection1 = gridfraction*gridsample[w][1];
					float corrected1 = outcur[w][1] - gridcorrection1;
					psd = (corrected0*corrected0 + corrected1*corrected1 ) + 1e-15f;// power spectrum density
//					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]) + 1e-15f;
					WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
					corrected0 *= WienerFactor; // apply filter on real  part	
					corrected1 *= WienerFactor; // apply filter on imaginary part
					outcur[w][0] = corrected0 + gridcorrection0;
					outcur[w][1] = corrected1 + gridcorrection1;
				}
				outcur += outpitch;
				gridsample += outpitch;
			}
			gridsample -= outpitch*bh; // restore pointer to only valid first block
		}
	}
	else if (sharpen != 0 && dehalo==0) // sharpen
	{
		for (block =0; block <howmanyblocks; block++)
		{
			gridfraction = degrid*outcur[0][0]/gridsample[0][0];
			for (h=0; h<bh; h++) // middle
			{
				for (w=0; w<outwidth; w++) // not skip first
				{
//					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]) + 1e-15f;// power spectrum density
					float gridcorrection0 = gridfraction*gridsample[w][0];
					float corrected0 = outcur[w][0] - gridcorrection0;
					float gridcorrection1 = gridfraction*gridsample[w][1];
					float corrected1 = outcur[w][1] - gridcorrection1;
					psd = (corrected0*corrected0 + corrected1*corrected1 ) + 1e-15f;// power spectrum density
					WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
					WienerFactor *= 1 + sharpen*wsharpen[w]*sqrt( psd*sigmaSquaredSharpenMax/((psd + sigmaSquaredSharpenMin)*(psd + sigmaSquaredSharpenMax)) ); // sharpen factor - changed in v.1.1
//					outcur[w][0] *= WienerFactor; // apply filter on real  part	
//					outcur[w][1] *= WienerFactor; // apply filter on imaginary part
					corrected0 *= WienerFactor; // apply filter on real  part	
					corrected1 *= WienerFactor; // apply filter on imaginary part
					outcur[w][0] = corrected0 + gridcorrection0;
					outcur[w][1] = corrected1 + gridcorrection1;
				}
				outcur += outpitch;
				wsharpen += outpitch;
				gridsample += outpitch;
			}
			wsharpen -= outpitch*bh;
			gridsample -= outpitch*bh; // restore pointer to only valid first block
		}
	}
	else if (sharpen == 0 && dehalo != 0)
	{
		for (block =0; block <howmanyblocks; block++)
		{
			gridfraction = degrid*outcur[0][0]/gridsample[0][0];
			for (h=0; h<bh; h++) // middle
			{
				for (w=0; w<outwidth; w++) // not skip first
				{
//					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]) + 1e-15f;// power spectrum density
					float gridcorrection0 = gridfraction*gridsample[w][0];
					float corrected0 = outcur[w][0] - gridcorrection0;
					float gridcorrection1 = gridfraction*gridsample[w][1];
					float corrected1 = outcur[w][1] - gridcorrection1;
					psd = (corrected0*corrected0 + corrected1*corrected1 ) + 1e-15f;// power spectrum density
					WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
					WienerFactor *= (psd + ht2n)/((psd + ht2n) + dehalo*wdehalo[w] * psd ); 
//					outcur[w][0] *= WienerFactor; // apply filter on real  part	
//					outcur[w][1] *= WienerFactor; // apply filter on imaginary part
					corrected0 *= WienerFactor; // apply filter on real  part	
					corrected1 *= WienerFactor; // apply filter on imaginary part
					outcur[w][0] = corrected0 + gridcorrection0;
					outcur[w][1] = corrected1 + gridcorrection1;
				}
				outcur += outpitch;
				wdehalo += outpitch;
				gridsample += outpitch;
			}
			wdehalo -= outpitch*bh;
			gridsample -= outpitch*bh; // restore pointer to only valid first block
		}
	}
	else if (sharpen != 0 && dehalo != 0)
	{
		for (block =0; block <howmanyblocks; block++)
		{
			gridfraction = degrid*outcur[0][0]/gridsample[0][0];
			for (h=0; h<bh; h++) // middle
			{
				for (w=0; w<outwidth; w++) // not skip first
				{
//					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]) + 1e-15f;// power spectrum density
					float gridcorrection0 = gridfraction*gridsample[w][0];
					float corrected0 = outcur[w][0] - gridcorrection0;
					float gridcorrection1 = gridfraction*gridsample[w][1];
					float corrected1 = outcur[w][1] - gridcorrection1;
					psd = (corrected0*corrected0 + corrected1*corrected1 ) + 1e-15f;// power spectrum density
					WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
          // fixed in .10
          WienerFactor *= 1 + sharpen * wsharpen[w] * sqrt(psd * sigmaSquaredSharpenMax / ((psd + sigmaSquaredSharpenMin) * (psd + sigmaSquaredSharpenMax)));
          WienerFactor *= (psd + ht2n)/((psd + ht2n)+ dehalo*wdehalo[w] * psd );
//					outcur[w][0] *= WienerFactor; // apply filter on real  part	
//					outcur[w][1] *= WienerFactor; // apply filter on imaginary part
					corrected0 *= WienerFactor; // apply filter on real  part	
					corrected1 *= WienerFactor; // apply filter on imaginary part
					outcur[w][0] = corrected0 + gridcorrection0;
					outcur[w][1] = corrected1 + gridcorrection1;
				}
				outcur += outpitch;
				wsharpen += outpitch;
				gridsample += outpitch;
				wdehalo += outpitch;
			}
			wsharpen -= outpitch*bh;
			wdehalo -= outpitch*bh;
			gridsample -= outpitch*bh; // restore pointer to only valid first block
		}
	}
		
}
//-----------------------------------------------------------------------------------------
//
void ApplyWiener3D2_degrid_C(fftwf_complex *outcur, fftwf_complex *outprev, 
					int outwidth, int outpitch, int bh, int howmanyblocks, 
					float sigmaSquaredNoiseNormed, 	float beta, 
					float degrid, fftwf_complex *gridsample)
{
	// this function take 25% CPU time and may be easy optimized for AMD Athlon 3DNOW assembler
	// return result in outprev
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0
	float psd;
	float WienerFactor;
	float f3d0r, f3d1r, f3d0i, f3d1i;
	int block;
	int h,w;

	for (block=0; block <howmanyblocks; block++)
	{
		float gridfraction = degrid*outcur[0][0]/gridsample[0][0];
		for (h=0; h<bh; h++)  
		{
			for (w=0; w<outwidth; w++) 
			{
				// dft 3d (very short - 2 points)
				float gridcorrection0_2 = gridfraction*gridsample[w][0]*2; // grid correction
				float gridcorrection1_2 = gridfraction*gridsample[w][1]*2;
				f3d0r =  outcur[w][0] + outprev[w][0] - gridcorrection0_2; // real 0 (sum)
				f3d0i =  outcur[w][1] + outprev[w][1] - gridcorrection1_2; // im 0 (sum)
				psd = f3d0r*f3d0r + f3d0i*f3d0i + 1e-15f; // power spectrum density 0
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				f3d0r *= WienerFactor; // apply filter on real  part	
				f3d0i *= WienerFactor; // apply filter on imaginary part

				f3d1r =  outcur[w][0] - outprev[w][0]; // real 1 (dif)
				f3d1i =  outcur[w][1] - outprev[w][1]; // im 1 (dif)
				psd = f3d1r*f3d1r + f3d1i*f3d1i + 1e-15f; // power spectrum density 1
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				f3d1r *= WienerFactor; // apply filter on real  part	
				f3d1i *= WienerFactor; // apply filter on imaginary part
				// reverse dft for 2 points
				outprev[w][0] = (f3d0r + f3d1r + gridcorrection0_2)*0.5f ; // get  real  part	
				outprev[w][1] = (f3d0i + f3d1i + gridcorrection1_2)*0.5f ; // get imaginary part
				// Attention! return filtered "out" in "outprev" to preserve "out" for next step
			}
			outcur += outpitch;
			outprev += outpitch;
			gridsample += outpitch;
		}
		gridsample -= outpitch*bh; // restore pointer to only valid first block
	}
}
//-----------------------------------------------------------------------------------------
//
void ApplyWiener3D3_degrid_C(fftwf_complex *outcur, fftwf_complex *outprev, fftwf_complex *outnext, 
					int outwidth, int outpitch, int bh, int howmanyblocks, 
					float sigmaSquaredNoiseNormed, float beta, 
					float degrid, fftwf_complex *gridsample)
{
	// this function take 25% CPU time and may be easy optimized for AMD Athlon 3DNOW assembler
	// return result in outprev
	float fcr, fci, fpr, fpi, fnr, fni;
	float WienerFactor;
	float psd;
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0
	float sin120 = 0.86602540378443864676372317075294f;//sqrtf(3.0f)*0.5f;

	int block;
	int h,w;

	for (block=0; block <howmanyblocks; block++)
	{
		float gridfraction = degrid*outcur[0][0]/gridsample[0][0];
		for (h=0; h<bh; h++) // first half
		{
			for (w=0; w<outwidth; w++) // 
			{
				float gridcorrection0_3 = gridfraction*gridsample[w][0]*3;
				float gridcorrection1_3 = gridfraction*gridsample[w][1]*3;
				// dft 3d (very short - 3 points)
				float pnr = outprev[w][0] + outnext[w][0];
				float pni = outprev[w][1] + outnext[w][1];
				fcr = outcur[w][0] + pnr; // real cur
				fcr -= gridcorrection0_3;
				fci = outcur[w][1] + pni; // im cur
				fci -= gridcorrection1_3;
				float di = sin120*(outprev[w][1]-outnext[w][1]);
				float dr = sin120*(outnext[w][0]-outprev[w][0]);
				fpr = outcur[w][0] - 0.5f*pnr + di; // real prev
				fnr = outcur[w][0] - 0.5f*pnr - di; // real next
				fpi = outcur[w][1] - 0.5f*pni + dr; // im prev
				fni = outcur[w][1] - 0.5f*pni - dr ; // im next
				psd = fcr*fcr + fci*fci + 1e-15f; // power spectrum density cur
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				fcr *= WienerFactor; // apply filter on real  part	
				fci *= WienerFactor; // apply filter on imaginary part
				psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density prev
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				fpr *= WienerFactor; // apply filter on real  part	
				fpi *= WienerFactor; // apply filter on imaginary part
				psd = fnr*fnr + fni*fni + 1e-15f; // power spectrum density next
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				fnr *= WienerFactor; // apply filter on real  part	
				fni *= WienerFactor; // apply filter on imaginary part
				// reverse dft for 3 points
				outprev[w][0] = (fcr + fpr + fnr + gridcorrection0_3)*0.33333333333f; // get  real  part	
				outprev[w][1] = (fci + fpi + fni + gridcorrection1_3)*0.33333333333f; // get imaginary part
				// Attention! return filtered "out" in "outprev" to preserve "out" for next step
			}
			outcur += outpitch;
			outprev += outpitch;
			outnext += outpitch;
			gridsample += outpitch;
		}
		gridsample -= outpitch*bh; // restore pointer to only valid first block
	}
}
//-----------------------------------------------------------------------------------------
//
void ApplyWiener3D4_degrid_C(fftwf_complex *outcur, fftwf_complex *outprev2, fftwf_complex *outprev, 
					fftwf_complex *outnext, int outwidth, int outpitch, int bh, 
					int howmanyblocks, float sigmaSquaredNoiseNormed, float beta, 
					float degrid, fftwf_complex *gridsample)
{
	// dft with 4 points
	// this function take 25% CPU time and may be easy optimized for AMD Athlon 3DNOW assembler
	// return result in outprev
	float fcr, fci, fpr, fpi, fnr, fni, fp2r, fp2i;
	float WienerFactor;
	float psd;
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0

	int block;
	int h,w;

	for (block=0; block <howmanyblocks; block++)
	{
		float gridfraction = degrid*outcur[0][0]/gridsample[0][0];
		for (h=0; h<bh; h++) // first half
		{
			for (w=0; w<outwidth; w++) // 
			{
				float gridcorrection0_4 = gridfraction*gridsample[w][0]*4;
				float gridcorrection1_4 = gridfraction*gridsample[w][1]*4;
				// dft 3d (very short - 4 points)
				fpr = -outprev2[w][0] + outprev[w][1] + outcur[w][0] - outnext[w][1]; // real prev
				fpi = -outprev2[w][1] - outprev[w][0] + outcur[w][1] + outnext[w][0]; // im cur
				fcr = outprev2[w][0] + outprev[w][0] + outcur[w][0] + outnext[w][0]; // real cur
				fcr -= gridcorrection0_4;
				fci = outprev2[w][1] + outprev[w][1] + outcur[w][1] + outnext[w][1]; // im cur
				fci -= gridcorrection1_4;
				fnr = -outprev2[w][0] - outprev[w][1] + outcur[w][0] + outnext[w][1]; // real next
				fni = -outprev2[w][1] + outprev[w][0] + outcur[w][1] - outnext[w][0]; // im next
				fp2r = outprev2[w][0] - outprev[w][0] + outcur[w][0] - outnext[w][0]; // real prev2
				fp2i = outprev2[w][1] - outprev[w][1] + outcur[w][1] - outnext[w][1]; // im cur

				psd = fp2r*fp2r + fp2i*fp2i + 1e-15f; // power spectrum density prev2
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				fp2r *= WienerFactor; // apply filter on real  part	
				fp2i *= WienerFactor; // apply filter on imaginary part

				psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density prev
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				fpr *= WienerFactor; // apply filter on real  part	
				fpi *= WienerFactor; // apply filter on imaginary part

				psd = fcr*fcr + fci*fci + 1e-15f; // power spectrum density cur
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				fcr *= WienerFactor; // apply filter on real  part	
				fci *= WienerFactor; // apply filter on imaginary part

				psd = fnr*fnr + fni*fni + 1e-15f; // power spectrum density next
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				fnr *= WienerFactor; // apply filter on real  part	
				fni *= WienerFactor; // apply filter on imaginary part

				// reverse dft for 4 points
				outprev2[w][0] = (fp2r + fpr + fcr + fnr + gridcorrection0_4)*0.25f ; // get  real  part	
				outprev2[w][1] = (fp2i + fpi + fci + fni + gridcorrection1_4)*0.25f; // get imaginary part
				// Attention! return filtered "out" in "outprev2" to preserve "out" for next step
			}
			outcur += outpitch;
			outprev2 += outpitch;
			outprev += outpitch;
			outnext += outpitch;
			gridsample += outpitch;
		}
		gridsample -= outpitch*bh; // restore pointer to only valid first block
	}
}
//-------------------------------------------------------------------------------------------
//
void ApplyPattern2D_degrid_C(fftwf_complex *outcur, int outwidth, int outpitch, int bh, 
							 int howmanyblocks, float pfactor, float *pattern2d0, float beta, 
							 float degrid, fftwf_complex *gridsample)
{
	int h,w, block;
	float psd;
	float WienerFactor;
	float *pattern2d;
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0

	if (pfactor != 0)
	{

		for (block =0; block <howmanyblocks; block++)
		{
			float gridfraction = degrid*outcur[0][0]/gridsample[0][0];
			pattern2d = pattern2d0;
			for (h=0; h<bh; h++) // middle
			{
				for (w=0; w<outwidth; w++) 
				{
					float gridcorrection0 = gridfraction*gridsample[w][0];
					float corrected0 = outcur[w][0] - gridcorrection0;
					float gridcorrection1 = gridfraction*gridsample[w][1];
					float corrected1 = outcur[w][1] - gridcorrection1;
					psd = (corrected0*corrected0 + corrected1*corrected1 ) + 1e-15f;// power spectrum density
//					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]) + 1e-15f;
					WienerFactor = std::max((psd - pfactor*pattern2d[w])/psd, lowlimit); // limited Wiener filter
					corrected0 *= WienerFactor; // apply filter on real  part	
					corrected1 *= WienerFactor; // apply filter on imaginary part
					outcur[w][0] = corrected0 + gridcorrection0;
					outcur[w][1] = corrected1 + gridcorrection1;
				}
				outcur += outpitch;
				pattern2d += outpitch;
				gridsample += outpitch;
			}
			gridsample -= outpitch*bh; // restore pointer to only valid first block
		}
	}
}
//-----------------------------------------------------------------------------------------
//
void ApplyPattern3D2_degrid_C(fftwf_complex *outcur, fftwf_complex *outprev, 
					 int outwidth, int outpitch, int bh, 
					 int howmanyblocks, float *pattern3d, float beta,
					 float degrid, fftwf_complex *gridsample)
{
	// this function take 25% CPU time and may be easy optimized for AMD Athlon 3DNOW assembler
	// return result in outprev
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0
	float psd;
	float WienerFactor;
	float f3d0r, f3d1r, f3d0i, f3d1i;
	int block;
	int h,w;

	for (block=0; block <howmanyblocks; block++)
	{
		float gridfraction = degrid*outcur[0][0]/gridsample[0][0];
		for (h=0; h<bh; h++)  
		{
			for (w=0; w<outwidth; w++) 
			{
				float gridcorrection0_2 = gridfraction*gridsample[w][0]*2; // grid correction
				float gridcorrection1_2 = gridfraction*gridsample[w][1]*2;
				// dft 3d (very short - 2 points)
				f3d0r =  outcur[w][0] + outprev[w][0] - gridcorrection0_2; // real 0 (sum)
				f3d0i =  outcur[w][1] + outprev[w][1] - gridcorrection1_2; // im 0 (sum)
				f3d1r =  outcur[w][0] - outprev[w][0]; // real 1 (dif)
				f3d1i =  outcur[w][1] - outprev[w][1]; // im 1 (dif)
				psd = f3d0r*f3d0r + f3d0i*f3d0i + 1e-15f; // power spectrum density 0
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				f3d0r *= WienerFactor; // apply filter on real  part	
				f3d0i *= WienerFactor; // apply filter on imaginary part
				psd = f3d1r*f3d1r + f3d1i*f3d1i + 1e-15f; // power spectrum density 1
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				f3d1r *= WienerFactor; // apply filter on real  part	
				f3d1i *= WienerFactor; // apply filter on imaginary part
				// reverse dft for 2 points
				outprev[w][0] = (f3d0r + f3d1r + gridcorrection0_2)*0.5f; // get  real  part	
				outprev[w][1] = (f3d0i + f3d1i + gridcorrection1_2)*0.5f; // get imaginary part
				// Attention! return filtered "out" in "outprev" to preserve "out" for next step
			}
			outcur += outpitch;
			outprev += outpitch;
			pattern3d += outpitch;
			gridsample += outpitch;
		}
		pattern3d -= outpitch*bh; // restore pointer for new block
		gridsample -= outpitch*bh; // restore pointer to only valid first block
	}
}
//
//-----------------------------------------------------------------------------------------
//
void ApplyPattern3D3_degrid_C(fftwf_complex *outcur, fftwf_complex *outprev, fftwf_complex *outnext, 
					 int outwidth, int outpitch, int bh, int howmanyblocks, 
					 float *pattern3d, float beta,
					 float degrid, fftwf_complex *gridsample)
{
	// this function take 25% CPU time and may be easy optimized for AMD Athlon 3DNOW assembler
	// return result in outprev
	float fcr, fci, fpr, fpi, fnr, fni;
	float WienerFactor;
	float psd;
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0
	float sin120 = 0.86602540378443864676372317075294f;//sqrtf(3.0f)*0.5f;

	int block;
	int h,w;

	for (block=0; block <howmanyblocks; block++)
	{
		float gridfraction = degrid*outcur[0][0]/gridsample[0][0];
		for (h=0; h<bh; h++) // first half
		{
			for (w=0; w<outwidth; w++) // 
			{
				float gridcorrection0_3 = gridfraction*gridsample[w][0]*3;
				float gridcorrection1_3 = gridfraction*gridsample[w][1]*3;
				// dft 3d (very short - 3 points)
				float pnr = outprev[w][0] + outnext[w][0];
				float pni = outprev[w][1] + outnext[w][1];
				fcr = outcur[w][0] + pnr; // real cur
				fcr -= gridcorrection0_3;
				fci = outcur[w][1] + pni; // im cur
				fci -= gridcorrection1_3;
				float di = sin120*(outprev[w][1]-outnext[w][1]);
				float dr = sin120*(outnext[w][0]-outprev[w][0]);
				fpr = outcur[w][0] - 0.5f*pnr + di; // real prev
				fnr = outcur[w][0] - 0.5f*pnr - di; // real next
				fpi = outcur[w][1] - 0.5f*pni + dr; // im prev
				fni = outcur[w][1] - 0.5f*pni - dr ; // im next
				psd = fcr*fcr + fci*fci + 1e-15f; // power spectrum density cur
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fcr *= WienerFactor; // apply filter on real  part	
				fci *= WienerFactor; // apply filter on imaginary part
				psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density prev
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fpr *= WienerFactor; // apply filter on real  part	
				fpi *= WienerFactor; // apply filter on imaginary part
				psd = fnr*fnr + fni*fni + 1e-15f; // power spectrum density next
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fnr *= WienerFactor; // apply filter on real  part	
				fni *= WienerFactor; // apply filter on imaginary part
				// reverse dft for 3 points
				outprev[w][0] = (fcr + fpr + fnr + gridcorrection0_3)*0.33333333333f; // get  real  part	
				outprev[w][1] = (fci + fpi + fni + gridcorrection1_3)*0.33333333333f; // get imaginary part
				// Attention! return filtered "out" in "outprev" to preserve "out" for next step
			}
			outcur += outpitch;
			outprev += outpitch;
			outnext += outpitch;
			pattern3d += outpitch;
			gridsample += outpitch;
		}
		pattern3d -= outpitch*bh; // restore pointer for new block
		gridsample -= outpitch*bh; // restore pointer to only valid first block
	}
}

//-----------------------------------------------------------------------------------------
//
void ApplyPattern3D4_degrid_C(fftwf_complex *outcur, fftwf_complex *outprev2, fftwf_complex *outprev, 
					 fftwf_complex *outnext, int outwidth, int outpitch, int bh, 
					 int howmanyblocks, float* pattern3d, float beta,
					 float degrid, fftwf_complex *gridsample)
{
	// dft with 4 points
	// this function take 25% CPU time and may be easy optimized for AMD Athlon 3DNOW assembler
	// return result in outprev
	float fcr, fci, fpr, fpi, fnr, fni, fp2r, fp2i;
	float WienerFactor;
	float psd;
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0

	int block;
	int h,w;

	for (block=0; block <howmanyblocks; block++)
	{
		float gridfraction = degrid*outcur[0][0]/gridsample[0][0];
		for (h=0; h<bh; h++) // first half
		{
			for (w=0; w<outwidth; w++) // 
			{
				float gridcorrection0_4 = gridfraction*gridsample[w][0]*4;
				float gridcorrection1_4 = gridfraction*gridsample[w][1]*4;
				// dft 3d (very short - 4 points)
				fp2r = outprev2[w][0] - outprev[w][0] + outcur[w][0] - outnext[w][0]; // real prev2
				fp2i = outprev2[w][1] - outprev[w][1] + outcur[w][1] - outnext[w][1]; // im cur
				fpr = -outprev2[w][0] + outprev[w][1] + outcur[w][0] - outnext[w][1]; // real prev
				fpi = -outprev2[w][1] - outprev[w][0] + outcur[w][1] + outnext[w][0]; // im cur
				fcr = outprev2[w][0] + outprev[w][0] + outcur[w][0] + outnext[w][0]; // real cur
				fcr -= gridcorrection0_4;
				fci = outprev2[w][1] + outprev[w][1] + outcur[w][1] + outnext[w][1]; // im cur
				fci -= gridcorrection1_4;
				fnr = -outprev2[w][0] - outprev[w][1] + outcur[w][0] + outnext[w][1]; // real next
				fni = -outprev2[w][1] + outprev[w][0] + outcur[w][1] - outnext[w][0]; // im next

				psd = fp2r*fp2r + fp2i*fp2i + 1e-15f; // power spectrum density prev2
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fp2r *= WienerFactor; // apply filter on real  part	
				fp2i *= WienerFactor; // apply filter on imaginary part

				psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density prev
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fpr *= WienerFactor; // apply filter on real  part	
				fpi *= WienerFactor; // apply filter on imaginary part

				psd = fcr*fcr + fci*fci + 1e-15f; // power spectrum density cur
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fcr *= WienerFactor; // apply filter on real  part	
				fci *= WienerFactor; // apply filter on imaginary part

				psd = fnr*fnr + fni*fni + 1e-15f; // power spectrum density next
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fnr *= WienerFactor; // apply filter on real  part	
				fni *= WienerFactor; // apply filter on imaginary part

				// reverse dft for 4 points
				outprev2[w][0] = (fp2r + fpr + fcr + fnr + gridcorrection0_4)*0.25f; // get  real  part	
				outprev2[w][1] = (fp2i + fpi + fci + fni + gridcorrection1_4)*0.25f; // get imaginary part
				// Attention! return filtered "out" in "outprev2" to preserve "out" for next step
			}
			outcur += outpitch;
			outprev2 += outpitch;
			outprev += outpitch;
			outnext += outpitch;
			pattern3d += outpitch;
			gridsample += outpitch;
		}
		pattern3d -= outpitch*bh; // restore pointer
		gridsample -= outpitch*bh; // restore pointer to only valid first block
	}
}
//-----------------------------------------------------------------------------------------
//
void ApplyPattern3D5_degrid_C(fftwf_complex *outcur, fftwf_complex *outprev2, fftwf_complex *outprev, 
					fftwf_complex *outnext, fftwf_complex *outnext2, int outwidth, int outpitch, int bh, 
					int howmanyblocks, float *pattern3d, float beta, 
					float degrid, fftwf_complex *gridsample)
{
	// dft with 5 points
	// return result in outprev2
	float fcr, fci, fpr, fpi, fnr, fni, fp2r, fp2i, fn2r, fn2i;
	float WienerFactor;
	float psd;
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0
	float sin72 = 0.95105651629515357211643933337938f;// 2*pi/5
	float cos72 = 0.30901699437494742410229341718282f;
	float sin144 = 0.58778525229247312916870595463907f; 
	float cos144 = -0.80901699437494742410229341718282f;

	int block;
	int h,w;

	for (block=0; block <howmanyblocks; block++)
	{
		float gridfraction = degrid*outcur[0][0]/gridsample[0][0];
		for (h=0; h<bh; h++) // first half
		{
			for (w=0; w<outwidth; w++) // 
			{
				float gridcorrection0_5 = gridfraction*gridsample[w][0]*5;
				float gridcorrection1_5 = gridfraction*gridsample[w][1]*5;
				// dft 3d (very short - 5 points)
				float sum = (outprev2[w][0] + outnext2[w][0])*cos72	+ (outprev[w][0] + outnext[w][0])*cos144 + + outcur[w][0];
				float dif = (- outprev2[w][1] + outnext2[w][1])*sin72 + (outprev[w][1]  - outnext[w][1])*sin144;
				fp2r = sum + dif; // real prev2
				fn2r = sum - dif; // real next2
				sum = (outprev2[w][1] + outnext2[w][1])*cos72 + (outprev[w][1] + outnext[w][1])*cos144 + outcur[w][1];
				dif = (outprev2[w][0] - outnext2[w][0])*sin72 + (- outprev[w][0] + outnext[w][0])*sin144;
				fp2i = sum + dif; // im prev2
				fn2i = sum - dif; // im next2
				sum = (outprev2[w][0] + outnext2[w][0])*cos144 + (outprev[w][0] + outnext[w][0])*cos72 + outcur[w][0];
				dif = (outprev2[w][1] - outnext2[w][1])*sin144 + (outprev[w][1] - outnext[w][1])*sin72;
				fpr = sum + dif; // real prev
				fnr = sum - dif; // real next
				sum = (outprev2[w][1] + outnext2[w][1])*cos144 + (outprev[w][1] + outnext[w][1])*cos72 + outcur[w][1];
				dif =  (- outprev2[w][0] + outnext2[w][0])*sin144 + (- outprev[w][0] + outnext[w][0])*sin72;
				fpi = sum + dif; // im prev
				fni = sum - dif; // im next
				fcr = outprev2[w][0] + outprev[w][0] + outcur[w][0] + outnext[w][0] + outnext2[w][0]; // real cur
				fcr -= gridcorrection0_5;
				fci = outprev2[w][1] + outprev[w][1] + outcur[w][1] + outnext[w][1] + outnext2[w][1]; // im cur
				fci -= gridcorrection1_5;

				psd = fp2r*fp2r + fp2i*fp2i + 1e-15f; // power spectrum density prev2
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fp2r *= WienerFactor; // apply filter on real  part	
				fp2i *= WienerFactor; // apply filter on imaginary part

				psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density prev
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fpr *= WienerFactor; // apply filter on real  part	
				fpi *= WienerFactor; // apply filter on imaginary part

				psd = fcr*fcr + fci*fci + 1e-15f; // power spectrum density cur
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fcr *= WienerFactor; // apply filter on real  part	
				fci *= WienerFactor; // apply filter on imaginary part

				psd = fnr*fnr + fni*fni + 1e-15f; // power spectrum density next
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fnr *= WienerFactor; // apply filter on real  part	
				fni *= WienerFactor; // apply filter on imaginary part

				psd = fn2r*fn2r + fn2i*fn2i + 1e-15f; // power spectrum density next
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fn2r *= WienerFactor; // apply filter on real  part	
				fn2i *= WienerFactor; // apply filter on imaginary part

				// reverse dft for 5 points
				outprev2[w][0] = (fp2r + fpr + fcr + fnr + fn2r + gridcorrection0_5)*0.2f ; // get  real  part	
				outprev2[w][1] = (fp2i + fpi + fci + fni + fn2i + gridcorrection1_5)*0.2f; // get imaginary part
				// Attention! return filtered "out" in "outprev2" to preserve "out" for next step
			}
			outcur += outpitch;
			outprev2 += outpitch;
			outprev += outpitch;
			outnext += outpitch;
			outnext2 += outpitch;
			gridsample += outpitch;
			pattern3d += outpitch;
		}
		gridsample -= outpitch*bh; // restore pointer to only valid first block
		pattern3d -= outpitch*bh; // restore pointer
	}
}

//-----------------------------------------------------------------------------------------
//
void ApplyWiener3D5_degrid_C(fftwf_complex *outcur, fftwf_complex *outprev2, fftwf_complex *outprev, 
					fftwf_complex *outnext, fftwf_complex *outnext2, int outwidth, int outpitch, int bh, 
					int howmanyblocks, float sigmaSquaredNoiseNormed, float beta, 
					float degrid, fftwf_complex *gridsample)
{
	// dft with 5 points
	// return result in outprev2
	float fcr, fci, fpr, fpi, fnr, fni, fp2r, fp2i, fn2r, fn2i;
	float WienerFactor;
	float psd;
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0
	float sin72 = 0.95105651629515357211643933337938f;// 2*pi/5
	float cos72 = 0.30901699437494742410229341718282f;
	float sin144 = 0.58778525229247312916870595463907f; 
	float cos144 = -0.80901699437494742410229341718282f;

	int block;
	int h,w;

	for (block=0; block <howmanyblocks; block++)
	{
		float gridfraction = degrid*outcur[0][0]/gridsample[0][0];
		for (h=0; h<bh; h++) // first half
		{
			for (w=0; w<outwidth; w++) // 
			{
				float gridcorrection0_5 = gridfraction*gridsample[w][0]*5;
				float gridcorrection1_5 = gridfraction*gridsample[w][1]*5;
				// dft 3d (very short - 5 points)
//				fp2r = outprev2[w][0]*cos72 - outprev2[w][1]*sin72 + outprev[w][0]*cos144 + outprev[w][1]*sin144 + outcur[w][0] 
//					+ outnext[w][0]*cos144 - outnext[w][1]*sin144 + outnext2[w][0]*cos72 + outnext2[w][1]*sin72; // real prev2
//				fn2r = outprev2[w][0]*cos72 + outprev2[w][1]*sin72 + outprev[w][0]*cos144 - outprev[w][1]*sin144 + outcur[w][0] 
//					+ outnext[w][0]*cos144 + outnext[w][1]*sin144 + outnext2[w][0]*cos72 - outnext2[w][1]*sin72; // real prev2
//				fp2i = outprev2[w][1]*cos72 + outprev2[w][0]*sin72 + outprev[w][1]*cos144 - outprev[w][0]*sin144 + outcur[w][1] 
//					+ outnext[w][1]*cos144 + outnext[w][0]*sin144 + outnext2[w][1]*cos72 - outnext2[w][0]*sin72; // im prev2
//				fn2i = outprev2[w][1]*cos72 - outprev2[w][0]*sin72 + outprev[w][1]*cos144 + outprev[w][0]*sin144 + outcur[w][1] 
//					+ outnext[w][1]*cos144 - outnext[w][0]*sin144 + outnext2[w][1]*cos72 + outnext2[w][0]*sin72; // im prev2
//				fpr = outprev2[w][0]*cos144 + outprev2[w][1]*sin144 + outprev[w][0]*cos72 + outprev[w][1]*sin72 + outcur[w][0] 
//					+ outnext[w][0]*cos72 - outnext[w][1]*sin72 + outnext2[w][0]*cos144 - outnext2[w][1]*sin144; // real prev
//				fnr = outprev2[w][0]*cos144 - outprev2[w][1]*sin144 + outprev[w][0]*cos72 - outprev[w][1]*sin72 + outcur[w][0] 
//					+ outnext[w][0]*cos72 + outnext[w][1]*sin72 + outnext2[w][0]*cos144 + outnext2[w][1]*sin144; // real prev
//				fpi = outprev2[w][1]*cos144 - outprev2[w][0]*sin144 + outprev[w][1]*cos72 - outprev[w][0]*sin72 + outcur[w][1] 
//					+ outnext[w][1]*cos72 + outnext[w][0]*sin72 + outnext2[w][1]*cos144 + outnext2[w][0]*sin144; // im prev
//				fni = outprev2[w][1]*cos144 + outprev2[w][0]*sin144 + outprev[w][1]*cos72 + outprev[w][0]*sin72 + outcur[w][1] 
//					+ outnext[w][1]*cos72 - outnext[w][0]*sin72 + outnext2[w][1]*cos144 - outnext2[w][0]*sin144; // im prev
				float sum = (outprev2[w][0] + outnext2[w][0])*cos72	+ (outprev[w][0] + outnext[w][0])*cos144 + + outcur[w][0];
				float dif = (- outprev2[w][1] + outnext2[w][1])*sin72 + (outprev[w][1]  - outnext[w][1])*sin144;
				fp2r = sum + dif; // real prev2
				fn2r = sum - dif; // real next2
				sum = (outprev2[w][1] + outnext2[w][1])*cos72 + (outprev[w][1] + outnext[w][1])*cos144 + outcur[w][1];
				dif = (outprev2[w][0] - outnext2[w][0])*sin72 + (- outprev[w][0] + outnext[w][0])*sin144;
				fp2i = sum + dif; // im prev2
				fn2i = sum - dif; // im next2
				sum = (outprev2[w][0] + outnext2[w][0])*cos144 + (outprev[w][0] + outnext[w][0])*cos72 + outcur[w][0];
				dif = (outprev2[w][1] - outnext2[w][1])*sin144 + (outprev[w][1] - outnext[w][1])*sin72;
				fpr = sum + dif; // real prev
				fnr = sum - dif; // real next
				sum = (outprev2[w][1] + outnext2[w][1])*cos144 + (outprev[w][1] + outnext[w][1])*cos72 + outcur[w][1];
				dif =  (- outprev2[w][0] + outnext2[w][0])*sin144 + (- outprev[w][0] + outnext[w][0])*sin72;
				fpi = sum + dif; // im prev
				fni = sum - dif; // im next
				fcr = outprev2[w][0] + outprev[w][0] + outcur[w][0] + outnext[w][0] + outnext2[w][0]; // real cur
				fcr -= gridcorrection0_5;
				fci = outprev2[w][1] + outprev[w][1] + outcur[w][1] + outnext[w][1] + outnext2[w][1]; // im cur
				fci -= gridcorrection1_5;

				psd = fp2r*fp2r + fp2i*fp2i + 1e-15f; // power spectrum density prev2
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				fp2r *= WienerFactor; // apply filter on real  part	
				fp2i *= WienerFactor; // apply filter on imaginary part

				psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density prev
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				fpr *= WienerFactor; // apply filter on real  part	
				fpi *= WienerFactor; // apply filter on imaginary part

				psd = fcr*fcr + fci*fci + 1e-15f; // power spectrum density cur
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				fcr *= WienerFactor; // apply filter on real  part	
				fci *= WienerFactor; // apply filter on imaginary part

				psd = fnr*fnr + fni*fni + 1e-15f; // power spectrum density next
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				fnr *= WienerFactor; // apply filter on real  part	
				fni *= WienerFactor; // apply filter on imaginary part

				psd = fn2r*fn2r + fn2i*fn2i + 1e-15f; // power spectrum density next
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				fn2r *= WienerFactor; // apply filter on real  part	
				fn2i *= WienerFactor; // apply filter on imaginary part

				// reverse dft for 5 points
				outprev2[w][0] = (fp2r + fpr + fcr + fnr + fn2r + gridcorrection0_5)*0.2f ; // get  real  part	
				outprev2[w][1] = (fp2i + fpi + fci + fni + fn2i + gridcorrection1_5)*0.2f; // get imaginary part
				// Attention! return filtered "out" in "outprev2" to preserve "out" for next step
			}
			outcur += outpitch;
			outprev2 += outpitch;
			outprev += outpitch;
			outnext += outpitch;
			outnext2 += outpitch;
			gridsample += outpitch;
		}
		gridsample -= outpitch*bh; // restore pointer to only valid first block
	}
}
//-----------------------------------------------------------------------------------------
//
void ApplyPattern3D5_C(fftwf_complex *outcur, fftwf_complex *outprev2, fftwf_complex *outprev, 
					fftwf_complex *outnext, fftwf_complex *outnext2, int outwidth, int outpitch, int bh, 
					int howmanyblocks, float *pattern3d, float beta)
{
	// dft with 5 points
	// return result in outprev2
	float fcr, fci, fpr, fpi, fnr, fni, fp2r, fp2i, fn2r, fn2i;
	float WienerFactor;
	float psd;
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0
	float sin72 = 0.95105651629515357211643933337938f;// 2*pi/5
	float cos72 = 0.30901699437494742410229341718282f;
	float sin144 = 0.58778525229247312916870595463907f; 
	float cos144 = -0.80901699437494742410229341718282f;

	int block;
	int h,w;

	for (block=0; block <howmanyblocks; block++)
	{
		for (h=0; h<bh; h++) // first half
		{
			for (w=0; w<outwidth; w++) // 
			{
				// dft 3d (very short - 5 points)
				float sum = (outprev2[w][0] + outnext2[w][0])*cos72	+ (outprev[w][0] + outnext[w][0])*cos144 + + outcur[w][0];
				float dif = (- outprev2[w][1] + outnext2[w][1])*sin72 + (outprev[w][1]  - outnext[w][1])*sin144;
				fp2r = sum + dif; // real prev2
				fn2r = sum - dif; // real next2
				sum = (outprev2[w][1] + outnext2[w][1])*cos72 + (outprev[w][1] + outnext[w][1])*cos144 + outcur[w][1];
				dif = (outprev2[w][0] - outnext2[w][0])*sin72 + (- outprev[w][0] + outnext[w][0])*sin144;
				fp2i = sum + dif; // im prev2
				fn2i = sum - dif; // im next2
				sum = (outprev2[w][0] + outnext2[w][0])*cos144 + (outprev[w][0] + outnext[w][0])*cos72 + outcur[w][0];
				dif = (outprev2[w][1] - outnext2[w][1])*sin144 + (outprev[w][1] - outnext[w][1])*sin72;
				fpr = sum + dif; // real prev
				fnr = sum - dif; // real next
				sum = (outprev2[w][1] + outnext2[w][1])*cos144 + (outprev[w][1] + outnext[w][1])*cos72 + outcur[w][1];
				dif =  (- outprev2[w][0] + outnext2[w][0])*sin144 + (- outprev[w][0] + outnext[w][0])*sin72;
				fpi = sum + dif; // im prev
				fni = sum - dif; // im next
				fcr = outprev2[w][0] + outprev[w][0] + outcur[w][0] + outnext[w][0] + outnext2[w][0]; // real cur
				fci = outprev2[w][1] + outprev[w][1] + outcur[w][1] + outnext[w][1] + outnext2[w][1]; // im cur

				psd = fp2r*fp2r + fp2i*fp2i + 1e-15f; // power spectrum density prev2
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fp2r *= WienerFactor; // apply filter on real  part	
				fp2i *= WienerFactor; // apply filter on imaginary part

				psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density prev
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fpr *= WienerFactor; // apply filter on real  part	
				fpi *= WienerFactor; // apply filter on imaginary part

				psd = fcr*fcr + fci*fci + 1e-15f; // power spectrum density cur
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fcr *= WienerFactor; // apply filter on real  part	
				fci *= WienerFactor; // apply filter on imaginary part

				psd = fnr*fnr + fni*fni + 1e-15f; // power spectrum density next
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fnr *= WienerFactor; // apply filter on real  part	
				fni *= WienerFactor; // apply filter on imaginary part

				psd = fn2r*fn2r + fn2i*fn2i + 1e-15f; // power spectrum density next
				WienerFactor = std::max((psd - pattern3d[w])/psd, lowlimit); // limited Wiener filter
				fn2r *= WienerFactor; // apply filter on real  part	
				fn2i *= WienerFactor; // apply filter on imaginary part

				// reverse dft for 5 points
				outprev2[w][0] = (fp2r + fpr + fcr + fnr + fn2r)*0.2f ; // get  real  part	
				outprev2[w][1] = (fp2i + fpi + fci + fni + fn2i)*0.2f; // get imaginary part
				// Attention! return filtered "out" in "outprev2" to preserve "out" for next step
			}
			outcur += outpitch;
			outprev2 += outpitch;
			outprev += outpitch;
			outnext += outpitch;
			outnext2 += outpitch;
			pattern3d += outpitch;
		}
		pattern3d -= outpitch*bh; // restore pointer
	}
}

//-----------------------------------------------------------------------------------------
//
void ApplyWiener3D5_C(fftwf_complex *outcur, fftwf_complex *outprev2, fftwf_complex *outprev, 
					fftwf_complex *outnext, fftwf_complex *outnext2, int outwidth, int outpitch, int bh, 
					int howmanyblocks, float sigmaSquaredNoiseNormed, float beta)
{
	// dft with 5 points
	// return result in outprev2
	float fcr, fci, fpr, fpi, fnr, fni, fp2r, fp2i, fn2r, fn2i;
	float WienerFactor;
	float psd;
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0
	float sin72 = 0.95105651629515357211643933337938f;// 2*pi/5
	float cos72 = 0.30901699437494742410229341718282f;
	float sin144 = 0.58778525229247312916870595463907f; 
	float cos144 = -0.80901699437494742410229341718282f;

	int block;
	int h,w;

	for (block=0; block <howmanyblocks; block++)
	{
		for (h=0; h<bh; h++) // first half
		{
			for (w=0; w<outwidth; w++) // 
			{
				// dft 3d (very short - 5 points)
				float sum = (outprev2[w][0] + outnext2[w][0])*cos72	+ (outprev[w][0] + outnext[w][0])*cos144 + + outcur[w][0];
				float dif = (- outprev2[w][1] + outnext2[w][1])*sin72 + (outprev[w][1]  - outnext[w][1])*sin144;
				fp2r = sum + dif; // real prev2
				fn2r = sum - dif; // real next2
				sum = (outprev2[w][1] + outnext2[w][1])*cos72 + (outprev[w][1] + outnext[w][1])*cos144 + outcur[w][1];
				dif = (outprev2[w][0] - outnext2[w][0])*sin72 + (- outprev[w][0] + outnext[w][0])*sin144;
				fp2i = sum + dif; // im prev2
				fn2i = sum - dif; // im next2
				sum = (outprev2[w][0] + outnext2[w][0])*cos144 + (outprev[w][0] + outnext[w][0])*cos72 + outcur[w][0];
				dif = (outprev2[w][1] - outnext2[w][1])*sin144 + (outprev[w][1] - outnext[w][1])*sin72;
				fpr = sum + dif; // real prev
				fnr = sum - dif; // real next
				sum = (outprev2[w][1] + outnext2[w][1])*cos144 + (outprev[w][1] + outnext[w][1])*cos72 + outcur[w][1];
				dif =  (- outprev2[w][0] + outnext2[w][0])*sin144 + (- outprev[w][0] + outnext[w][0])*sin72;
				fpi = sum + dif; // im prev
				fni = sum - dif; // im next
				fcr = outprev2[w][0] + outprev[w][0] + outcur[w][0] + outnext[w][0] + outnext2[w][0]; // real cur
				fci = outprev2[w][1] + outprev[w][1] + outcur[w][1] + outnext[w][1] + outnext2[w][1]; // im cur

				psd = fp2r*fp2r + fp2i*fp2i + 1e-15f; // power spectrum density prev2
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				fp2r *= WienerFactor; // apply filter on real  part	
				fp2i *= WienerFactor; // apply filter on imaginary part

				psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density prev
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				fpr *= WienerFactor; // apply filter on real  part	
				fpi *= WienerFactor; // apply filter on imaginary part

				psd = fcr*fcr + fci*fci + 1e-15f; // power spectrum density cur
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				fcr *= WienerFactor; // apply filter on real  part	
				fci *= WienerFactor; // apply filter on imaginary part

				psd = fnr*fnr + fni*fni + 1e-15f; // power spectrum density next
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				fnr *= WienerFactor; // apply filter on real  part	
				fni *= WienerFactor; // apply filter on imaginary part

				psd = fn2r*fn2r + fn2i*fn2i + 1e-15f; // power spectrum density next2
				WienerFactor = std::max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				fn2r *= WienerFactor; // apply filter on real  part	
				fn2i *= WienerFactor; // apply filter on imaginary part

				// reverse dft for 5 points
				outprev2[w][0] = (fp2r + fpr + fcr + fnr + fn2r)*0.2f ; // get  real  part	
				outprev2[w][1] = (fp2i + fpi + fci + fni + fn2i)*0.2f; // get imaginary part
				// Attention! return filtered "out" in "outprev2" to preserve "out" for next step
			}
			outcur += outpitch;
			outprev2 += outpitch;
			outprev += outpitch;
			outnext += outpitch;
			outnext2 += outpitch;
		}
	}
}
