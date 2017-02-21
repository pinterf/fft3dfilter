//
//	FFT3DFilter plugin for Avisynth 2.5 - 3D Frequency Domain filter
//  Intel Pentium3 SSE filtering functions
//
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
//
#include "windows.h"
#include <avs/config.h> // x64
#include "fftwlite.h"

typedef struct xmmregstruct
{
	float f1;
	float f2;
	float f3;
	float f4;
} xmmreg; // v.1.9.1

// since v1.7 we use outpitch instead of outwidth

void ApplyWiener3D2_SSE(fftwf_complex *outcur, fftwf_complex *outprev, 
						int outwidth, int outpitch, int bh, int howmanyblocks, 
						float sigmaSquaredNoiseNormed, float beta)
{
	//  optimized for SSE assembler
	// return result in outprev
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0
//	float psd;
//	float WienerFactor;
//	float f3d0r, f3d1r, f3d0i, f3d1i;
//	int block;
//	int h,w;
	float smallf = 1e-15f;
	float onehalf = 0.5f;
	int totalbytes = howmanyblocks*bh*outpitch*8;

#ifndef X86_64

//	for (block=0; block <howmanyblocks; block++)
//	{
//		for (h=0; h<bh; h++)  
//		{
//			for (w=0; w<outwidth; w++) 
//			{
		__asm
		{
			emms;
			mov edi, outprev;
			mov esi, outcur; // current
			mov ecx, totalbytes; // counter
			movss xmm7, smallf;
			shufps xmm7, xmm7, 0 ;// xmm7 = smallf
			movss xmm6, sigmaSquaredNoiseNormed;
			shufps xmm6, xmm6, 0 ; // xmm6 = sigmaSquaredNoiseNormed
			movss xmm5, lowlimit;
			shufps xmm5, xmm5, 0; // xmm5 =lowlimit
			movss xmm4, onehalf;
			shufps xmm4, xmm4, 0; // xmm4 =onehalf
			mov eax, 0;
align 16
nextnumber:
				// take two complex numbers
				movaps xmm0, [edi+eax]; // xmm0=prev real | img
				movaps xmm2, xmm0;
				movaps xmm1, [esi+eax]; // xmm1=cur real | img 
//				f3d0r =  outcur[w][0] + outprev[w][0]; // real 0 (sum)
//				f3d0i =  outcur[w][1] + outprev[w][1]; // im 0 (sum)
				addps xmm2, xmm1; // xmm2 =sum

//				f3d1r =  outcur[w][0] - outprev[w][0]; // real 1 (dif)
//				f3d1i =  outcur[w][1] - outprev[w][1]; // im 1 (dif)
				movaps xmm3, xmm1;
				subps xmm3, xmm0;
				movaps xmm0, xmm3;
				// xmm0= dif  xmm1-xmm0 = cur-prev

				movaps xmm1, xmm2; // copy sum
				mulps xmm1, xmm1; // xmm1 =sumre*sumre | sumin*sumim
				movaps xmm3, xmm1; //copy
//				pswapd mm3, mm3; // swap re & im
				shufps xmm3, xmm3, (1 + 0 + 48 + 128);// swap re & im
				addps xmm1, xmm3; // xmm1 = sumre*sumre + sumim*sumim
//				psd = f3d0r*f3d0r + f3d0i*f3d0i + 1e-15f; // power spectrum density 0
				addps xmm1, xmm7; // xmm1 = psd of sum = sumre*sumre + sumim*sumim + smallf
				
				movaps xmm3, xmm1; // xmm3 =copy psd
				subps xmm3, xmm6; // xmm3= psd - sigma
				rcpps xmm1, xmm1; // xmm1= 1/psd // bug fixed in v.0.9.3
				mulps xmm3, xmm1; //  (psd-sigma)/psd
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				maxps xmm3, xmm5; // xmm3 =wienerfactor
//				f3d0r *= WienerFactor; // apply filter on real  part	
//				f3d0i *= WienerFactor; // apply filter on imaginary part
				mulps xmm2, xmm3; // xmm2 = final wiener sum f3d0



				movaps xmm1, xmm0; // copy dif
				mulps xmm1, xmm1; // xmm1 = difre*difre | difim*difim
				movaps xmm3, xmm1; // copy
//				pswapd mm3, mm3;
				shufps xmm3, xmm3, (1 + 0 + 48 + 128);// swap re & im
				addps xmm1, xmm3;
//				psd = f3d1r*f3d1r + f3d1i*f3d1i + 1e-15f; // power spectrum density 1
				addps xmm1, xmm7; // mm3 = psd of dif

				movaps xmm3, xmm1; //copy of psd
				subps xmm3, xmm6; // xmm3= psd - sigma
				rcpps xmm1, xmm1; // xmm1= 1/psd // bug fixed in v.0.9.3
				mulps xmm3, xmm1; //  (psd-sigma)/psd
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				maxps xmm3, xmm5; // xmm3 =wienerfactor
//				f3d1r *= WienerFactor; // apply filter on real  part	
//				f3d1i *= WienerFactor; // apply filter on imaginary part
				mulps xmm0, xmm3; // xmm0 = fimal wiener dif f3d1

				// reverse dft for 2 points
				addps xmm2, xmm0; // filterd sum + dif
//				outprev[w][0] = (f3d0r + f3d1r)*0.5f; // get  real  part	
//				outprev[w][1] = (f3d0i + f3d1i)*0.5f; // get imaginary part
				mulps xmm2, xmm4; // filterd (sum+dif)*0.5
				movaps [edi+eax], xmm2;
				// Attention! return filtered "outcur" in "outprev" to preserve "outcur" for next step
			
//			outcur += outpitch;
//			outprev += outpitch;
//		}
				add eax, 16; // the lastest number may be skipped
				cmp eax, ecx;
				jge finish;
				jmp nextnumber;
finish:			emms;
		}
#endif // !x64
}
//-----------------------------------------------------------------------------------------
//
void ApplyPattern3D2_SSE(fftwf_complex *outcur, fftwf_complex *outprev, 
						 int outwidth, int outpitch, int bh, int howmanyblocks, 
						 float * pattern3d, float beta)
{
	//  optimized SSE assembler
	// return result in outprev
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0
//	float psd;
//	float WienerFactor;
//	float f3d0r, f3d1r, f3d0i, f3d1i;
//	int block;
//	int h,w;
	float smallf = 1e-15f;
	float onehalf = 0.5f;
//	int totalbytes = howmanyblocks*bh*outwidth*8;
	int bytesperblock = bh*outpitch*8;
	int blockscounter = howmanyblocks;

#ifndef X86_64

//	for (block=0; block <howmanyblocks; block++)
//	{
//		for (h=0; h<bh; h++)  
//		{
//			for (w=0; w<outwidth; w++) 
//			{
		__asm
		{
			emms;
			mov edi, outprev;
			mov esi, outcur; // current
			mov ebx, pattern3d;
			mov ecx, bytesperblock; // counter
			movss xmm7, smallf;
			shufps xmm7, xmm7, 0 ;// xmm7 = smallf
			movss xmm5, lowlimit;
			shufps xmm5, xmm5, 0; // xmm5 =lowlimit
			movss xmm4, onehalf;
			shufps xmm4, xmm4, 0; // xmm4 =onehalf
			mov eax, 0;
			mov edx, 0;
align 16
nextnumber:
//				shr eax, 1;
				movlps xmm6, [ebx+edx];// pattern3d - two values
				shufps xmm6, xmm6, (64 + 16 + 0 + 0) ;// 01 01 00 00 low
//				shl eax, 1;

				// take two complex numbers
				movaps xmm0, [edi+eax]; // xmm0=prev real | img
				movaps xmm2, xmm0;
				movaps xmm1, [esi+eax]; // xmm1=cur real | img 
//				f3d0r =  outcur[w][0] + outprev[w][0]; // real 0 (sum)
//				f3d0i =  outcur[w][1] + outprev[w][1]; // im 0 (sum)
				addps xmm2, xmm1; // xmm2 =sum

//				f3d1r =  outcur[w][0] - outprev[w][0]; // real 1 (dif)
//				f3d1i =  outcur[w][1] - outprev[w][1]; // im 1 (dif)
				movaps xmm3, xmm1;
				subps xmm3, xmm0;
				movaps xmm0, xmm3;
				// xmm0= dif  xmm1-xmm0 = cur-prev

				movaps xmm1, xmm2; // copy sum
				mulps xmm1, xmm1; // xmm1 =sumre*sumre | sumim*sumim
				movaps xmm3, xmm1; //copy
//				pswapd mm3, mm3; // swap re & im
				shufps xmm3, xmm3, (128 + 48 + 0 + 1);// swap re & im
				addps xmm1, xmm3; // xmm1 = sumre*sumre + sumim*sumim
//				psd = f3d0r*f3d0r + f3d0i*f3d0i + 1e-15f; // power spectrum density 0
				addps xmm1, xmm7; // xmm1 = psd of sum = sumre*sumre + sumim*sumim + smallf
				
				movaps xmm3, xmm1; // xmm3 =copy psd

				subps xmm3, xmm6; // xmm3= psd - pattern sigma
				rcpps xmm1, xmm1; // xmm1= 1/psd // bug fixed in v.0.9.3
				mulps xmm3, xmm1; //  (psd-sigma)/psd
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				maxps xmm3, xmm5; // xmm3 =wienerfactor
//				f3d0r *= WienerFactor; // apply filter on real  part	
//				f3d0i *= WienerFactor; // apply filter on imaginary part
				mulps xmm2, xmm3; // xmm2 = final wiener sum f3d0



				movaps xmm1, xmm0; // copy dif
				mulps xmm1, xmm1; // xmm1 = difre*difre | difim*difim
				movaps xmm3, xmm1; // copy
//				pswapd mm3, mm3;
				shufps xmm3, xmm3, (128 + 48 + 0 + 1);// swap re & im
				addps xmm1, xmm3;
//				psd = f3d1r*f3d1r + f3d1i*f3d1i + 1e-15f; // power spectrum density 1
				addps xmm1, xmm7; // mm3 = psd of dif

				movaps xmm3, xmm1; //copy of psd
				subps xmm3, xmm6; // xmm3= psd - pattern sigma
				rcpps xmm1, xmm1; // xmm1= 1/psd // bug fixed in v.0.9.3
				mulps xmm3, xmm1; //  (psd-sigma)/psd
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				maxps xmm3, xmm5; // xmm3 =wienerfactor
//				f3d1r *= WienerFactor; // apply filter on real  part	
//				f3d1i *= WienerFactor; // apply filter on imaginary part
				mulps xmm0, xmm3; // xmm0 = fimal wiener dif f3d1

				// reverse dft for 2 points
				addps xmm2, xmm0; // filterd sum + dif
//				outprev[w][0] = (f3d0r + f3d1r)*0.5f; // get  real  part	
//				outprev[w][1] = (f3d0i + f3d1i)*0.5f; // get imaginary part
				mulps xmm2, xmm4; // filterd (sum+dif)*0.5
				movaps [edi+eax], xmm2;
				// Attention! return filtered "outcur" in "outprev" to preserve "outcur" for next step
			
//			outcur += outpitch;
//			outprev += outpitch;
//		}
				add edx, 8; // pattern
				add eax, 16; // two numbers, that is why we use pitch
				cmp eax, ecx;
				jge blockend;
				jmp nextnumber;
blockend:		mov eax, blockscounter;
				dec eax
				je	finish;
				mov blockscounter, eax;
				mov eax, 0;
				mov edx, 0; 
				add edi, ecx; // new block
				add esi, ecx;
				jmp nextnumber;
finish:			emms;
		}
#endif
}
//-----------------------------------------------------------------------------------------
//
void ApplyWiener3D3_SSE(fftwf_complex *outcur, fftwf_complex *outprev, 
						fftwf_complex *outnext, int outwidth, int outpitch, int bh, 
						int howmanyblocks, float sigmaSquaredNoiseNormed, float beta)
{
	// dft 3d (very short - 3 points)
	// optimized for SSE assembler
	// return result in outprev
//	float fcr, fci, fpr, fpi, fnr, fni;
//	float pnr, pni, di, dr;
//	float WienerFactor =1;
//	float psd;
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0
	float sin120 = 0.86602540378443864676372317075294f; //sqrtf(3.0f)*0.5f;
	float smallf = 1e-15f;
	float onethird = 0.33333333333f;
	float onehalf = 0.5f;


//	int block;
//	int h,w;

	int totalbytes = howmanyblocks*bh*outpitch*8;

#ifndef X86_64

//	for (line=0; line <totalnumber; line++)
//	{
		__asm
		{
			emms;
			mov edi, outprev;
			mov edx, outnext;
			mov esi, outcur; // current
			mov ecx, totalbytes; // counter
			mov eax, 0;
align 16
nextnumber:
				movaps xmm0, [edi+eax]; // xmm0=prev real | img
				movaps xmm1, [edx+eax]; // xmm1=next real | img
				movaps xmm3, [esi+eax]; // mm3=cur real | img 

				movaps xmm2, xmm0; // copy prev
				//pnr = outprev[w][0] + outnext[w][0];
				//pni = outprev[w][1] + outnext[w][1];
				addps xmm2, xmm1; // xmm2= summa prev and next  , pnr | pni

				//fcr = outcur[w][0] + pnr; // real cur
				//fci = outcur[w][1] + pni; // im cur
				movaps xmm4, xmm3 ;// copy of cur
				addps xmm4, xmm2; // mm4= fcr | fci

				movss xmm7, onehalf; // 0.5
				shufps xmm7, xmm7, 0 ; // xmm7 = onehalf
				mulps xmm2, xmm7; // mm2 = 0.5*pnr | 0.5*pni


				movaps xmm5, xmm4; // xmm5= copy   fcr | fci
				mulps xmm5, xmm4; // xmm5 = fcr*fcr | fci*fci

				//pfacc mm5, mm5; // mm5 = fcr*fcr+fci*fci

				movaps xmm6, xmm5; //copy
				shufps xmm6, xmm6, (128 + 48 + 0 + 1);//  10 11 00 01 low - swap re & im
				addps xmm5, xmm6; // xmm1 = sumre*sumre + sumim*sumim
				
				movss xmm7, smallf; // 1e-15f
				shufps xmm7, xmm7, 0 ;
//				psd = fcr*fcr + fci*fci + 1e-15f; // power spectrum density cur
				addps xmm5, xmm7;// xmm5 =psd cur
				movss xmm7, sigmaSquaredNoiseNormed;
				shufps xmm7, xmm7, 0 ;

//				pfsubr mm7, mm5; // psd - sigma
				movaps xmm6, xmm5;
				subps xmm6, xmm7; // psd - sigma
				rcpps xmm5, xmm5; // xmm5= 1/psd
				mulps xmm5, xmm6; // // (psd-sigma)/psd
				movss xmm7, lowlimit;
				shufps xmm7, xmm7, 0 ;
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
//				fcr *= WienerFactor; // apply filter on real  part	
//				fci *= WienerFactor; // apply filter on imaginary part
				maxps xmm5, xmm7; // xmm5 =wienerfactor
				mulps xmm4, xmm5; // xmm4 = final wiener fcr | fci
				// mm5, mm6, mm7 are free
				
				
				movaps xmm5, xmm1;  // copy of next
				subps xmm5, xmm0; // next-prev   real | img
				movaps xmm6, xmm0; // copy of prev
				subps xmm6, xmm1; // prev-next  real | img

//				pswapd mm6, mm6; // swap real and img,    img | real
//				punpckldq mm5, mm6;  // use low dwords:   real next-prev | img prev-next
//				pswapd mm5, mm5 ; // mm5 = im(prev-next) | re(next-prev)

				shufps xmm5, xmm5, (128 + 0 + 8 + 0); // 10 00 10 00 low // 2 0 2 0 low //low Re(next-prev) | Re2(next-prev) || same second
				shufps xmm6, xmm6, (192 +16 + 12 + 1);// 11 01 11 01 low // 3 1 3 1 low //low Im(prev-next) | Im2(prev-next) || same second
				unpcklps xmm6, xmm5;// low Im(prev-next) | Re(next-prev) || Im2(prev-next) | Re2(next-prev)  

				movss xmm7, sin120;
				shufps xmm7, xmm7, 0 ; // xmm7 = sin120 | sin120
//				di = sin120*(outprev[w][1]-outnext[w][1]);
//				dr = sin120*(outnext[w][0]-outprev[w][0]);
				mulps xmm6, xmm7; // xmm6= di | dr

				//  xmm1, xmm5 are free
				subps xmm3, xmm2; // xmm3=cur-0.5*pn
				movaps xmm1, xmm3; // copy 
//				fpr = out[w][0] - 0.5f*pnr + di; // real prev
//				fpi = out[w][1] - 0.5f*pni + dr; // im prev
				addps xmm3, xmm6;// xmm3 = fpr | fpi
				
//				fnr = out[w][0] - 0.5f*pnr - di; // real next
//				fni = out[w][1] - 0.5f*pni - dr ; // im next
				subps xmm1, xmm6;// mm1 = fnr | fni
				// mm2, mm5, mm6, mm7 are free


				movaps xmm5, xmm3; // xmm5= copy   fpr | fpi
				mulps xmm5, xmm3; // xmm5 = fpr*fpr | fpi*fpi
//				pfacc mm5, mm5; // mm5 = fpr*fpr+fpi*fpi

				movss xmm7, smallf; // 1e-15f
				shufps xmm7, xmm7, 0;

				movaps xmm6, xmm5; //copy
				shufps xmm6, xmm6, (128 + 48 + 0 + 1);//  10 11 00 01 low// swap re & im
				addps xmm5, xmm6; // xmm5 = sumre*sumre + sumim*sumim

//				psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density cur
				addps xmm5, xmm7;// xmm5 =psd cur
				movss xmm7, sigmaSquaredNoiseNormed;
				shufps xmm7, xmm7, 0;

//				pfsubr mm7, mm5; // psd - sigma
				movaps xmm6, xmm5; //psd
				subps xmm6, xmm7; // psd - sigma

				rcpps xmm5, xmm5; // mm5= 1/psd
				mulps xmm5, xmm6; // // (psd-sigma)/psd
				movss xmm7, lowlimit;
				shufps xmm7, xmm7, 0;
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
//				fpr *= WienerFactor; // apply filter on real  part	
//				fpi *= WienerFactor; // apply filter on imaginary part
				maxps xmm5, xmm7; // mm5 =wienerfactor
				mulps xmm3, xmm5; // mm3 = final wiener fpr | fpi
				// mm2, mm5, mm6, mm7 are free
				


				movaps xmm5, xmm1; // mm5= copy   fnr | fni
				mulps xmm5, xmm1; // mm5 = fnr*fnr | fni*fni

				movss xmm7, smallf; // 1e-15f
				shufps xmm7, xmm7, 0;

//				pfacc mm5, mm5; // mm5 = fnr*fnr+fni*fni
				movaps xmm6, xmm5;
				shufps xmm6, xmm6, (128 + 48 + 0 + 1);//  10 11 00 01 low// swap re & im
				addps xmm5, xmm6; // xmm5 = sumre*sumre + sumim*sumim

//				psd = fnr*fnr + fni*fni + 1e-15f; // power spectrum density cur
				addps xmm5, xmm7;// xmm5 =psd cur
				movss xmm7, sigmaSquaredNoiseNormed;
				shufps xmm7, xmm7, 0;

//				pfsubr mm7, mm5; // psd - sigma
				movaps xmm6, xmm5; //psd
				subps xmm6, xmm7; // psd - sigma

				rcpps xmm5, xmm5; // mm5= 1/psd
				mulps xmm5, xmm6; // // (psd-sigma)/psd
				movss xmm7, lowlimit;
//				shufps xmm7, xmm7,0;
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
//				fnr *= WienerFactor; // apply filter on real  part	
//				fni *= WienerFactor; // apply filter on imaginary part
				maxps xmm5, xmm7; // mm5 =wienerfactor
				mulps xmm1, xmm5; // mm1 = final wiener fmr | fmi
				// mm2, mm5, mm6, mm7 are free
				
				// reverse dft for 3 points
				addps xmm4, xmm3; // fc + fp
				addps xmm4, xmm1; // fc + fp + fn
				movss xmm7, onethird;
				shufps xmm7, xmm7, 0;
				mulps xmm4, xmm7;
//				outprev[w][0] = (fcr + fpr + fnr)*0.33333333333f; // get  real  part	
//				outprev[w][1] = (fci + fpi + fni)*0.33333333333f; // get imaginary part
				movaps [edi+eax], xmm4; // write output to prev array
				// Attention! return filtered "out" in "outprev" to preserve "out" for next step

				add eax, 16;
				cmp eax, ecx;
				jge finish;
				jmp nextnumber;
finish:			emms;
		}
#endif // !x64
}
//-----------------------------------------------------------------------------------------
//
void ApplyPattern3D3_SSE(fftwf_complex *outcur, fftwf_complex *outprev, 
						 fftwf_complex *outnext, int outwidth, int outpitch, int bh, 
						 int howmanyblocks, float *pattern3d, float beta)
{
	// dft 3d (very short - 3 points)
	// SSE assembler
	// return result in outprev
//	float fcr, fci, fpr, fpi, fnr, fni;
//	float pnr, pni, di, dr;
//	float WienerFactor =1;
//	float psd;
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0
	float sin120 = 0.86602540378443864676372317075294f;// sqrtf(3.0f)*0.5f;
	float smallf = 1e-15f;
	float onethird = 0.33333333333f;
	float onehalf = 0.5f;

//	int block;
//	int h,w;

//	int totalbytes = howmanyblocks*bh*outwidth*8;
	int bytesperblock = bh*outpitch*8;
	int blockscounter = howmanyblocks;

#ifndef X86_64

//	for (line=0; line <totalnumber; line++)
//	{
		__asm
		{
			emms;
			mov edi, outprev;
			mov edx, outnext;
			mov esi, outcur; // current
			mov ebx, pattern3d;
			mov ecx, bytesperblock; // counter
//			movd mm0, lowlimit;
//			movd mm1, sin120;
//			movd mm2, smallf;
//			movd mm3, onethird;
//			movd mm4, onehalf;
			mov eax, 0;
align 16
nextnumber:
				movaps xmm0, [edi+eax]; // xmm0=prev real | img
				movaps xmm1, [edx+eax]; // xmm1=next real | img
				movaps xmm3, [esi+eax]; // mm3=cur real | img 

				movaps xmm2, xmm0; // copy prev
				//pnr = outprev[w][0] + outnext[w][0];
				//pni = outprev[w][1] + outnext[w][1];
				addps xmm2, xmm1; // xmm2= summa prev and next  , pnr | pni

				//fcr = outcur[w][0] + pnr; // real cur
				//fci = outcur[w][1] + pni; // im cur
				movaps xmm4, xmm3 ;// copy of cur
				addps xmm4, xmm2; // mm4= fcr | fci

				movss xmm7, onehalf; // 0.5
				//punpckldq mm7, mm7 ; // mm7 = 0.5 | 0.5
				shufps xmm7, xmm7, 0 ; // xmm7 = onehalf
				mulps xmm2, xmm7; // mm2 = 0.5*pnr | 0.5*pni


				movaps xmm5, xmm4; // xmm5= copy   fcr | fci
				mulps xmm5, xmm4; // xmm5 = fcr*fcr | fci*fci

				//pfacc mm5, mm5; // mm5 = fcr*fcr+fci*fci

				movaps xmm6, xmm5; //copy
				shufps xmm6, xmm6, (128 + 48 + 0 + 1);//  10 11 00 01  low- swap re & im
				addps xmm5, xmm6; // xmm1 = sumre*sumre + sumim*sumim
				
				movss xmm7, smallf; // 1e-15f
				shufps xmm7, xmm7, 0 ;
//				psd = fcr*fcr + fci*fci + 1e-15f; // power spectrum density cur
				addps xmm5, xmm7;// xmm5 =psd cur
				shr eax, 1;
				movlps xmm7, [ebx+eax];// pattern3d - two values
				shufps xmm7, xmm7, (64 + 16 + 0 + 0) ;// 01 01 00 00 low
				shl eax, 1;

//				pfsubr mm7, mm5; // psd - sigma
				movaps xmm6, xmm5;
				subps xmm6, xmm7; // psd - sigma
				rcpps xmm5, xmm5; // xmm5= 1/psd
				mulps xmm5, xmm6; // // (psd-sigma)/psd
				movss xmm7, lowlimit;
				shufps xmm7, xmm7, 0 ;
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
//				fcr *= WienerFactor; // apply filter on real  part	
//				fci *= WienerFactor; // apply filter on imaginary part
				maxps xmm5, xmm7; // xmm5 =wienerfactor
				mulps xmm4, xmm5; // xmm4 = final wiener fcr | fci
				// mm5, mm6, mm7 are free
				
				
				movaps xmm5, xmm1;  // copy of next
				subps xmm5, xmm0; // next-prev   real | img
				movaps xmm6, xmm0; // copy of prev
				subps xmm6, xmm1; // prev-next  real | img

//				pswapd mm6, mm6; // swap real and img,    img | real
//				punpckldq mm5, mm6;  // use low dwords:   real next-prev | img prev-next
//				pswapd mm5, mm5 ; // mm5 = im(prev-next) | re(next-prev)

				shufps xmm5, xmm5, (128 + 0 + 8 + 0);// 10 00 10 00 low // 2 0 2 0 low//low Re(next-prev) | Re2(next-prev) high || same second
				shufps xmm6, xmm6, (192 +16 +12 + 1);// 11 01 11 01 low // 3 1 3 1 low//low Im(prev-next) | Im2(prev-next) high || same second
				unpcklps xmm6, xmm5;//low Im(prev-next) | Re(next-prev) || Im2(prev-next) | Re2(next-prev)  

				movss xmm7, sin120;
				shufps xmm7, xmm7, 0 ; // xmm7 = sin120 | sin120
//				di = sin120*(outprev[w][1]-outnext[w][1]);
//				dr = sin120*(outnext[w][0]-outprev[w][0]);
				mulps xmm6, xmm7; // xmm6= di | dr

				//  xmm1, xmm5 are free
				subps xmm3, xmm2; // xmm3=cur-0.5*pn
				movaps xmm1, xmm3; // copy 
//				fpr = out[w][0] - 0.5f*pnr + di; // real prev
//				fpi = out[w][1] - 0.5f*pni + dr; // im prev
				addps xmm3, xmm6;// xmm3 = fpr | fpi
				
//				fnr = out[w][0] - 0.5f*pnr - di; // real next
//				fni = out[w][1] - 0.5f*pni - dr ; // im next
				subps xmm1, xmm6;// mm1 = fnr | fni
				// mm2, mm5, mm6, mm7 are free


				movaps xmm5, xmm3; // xmm5= copy   fpr | fpi
				mulps xmm5, xmm3; // xmm5 = fpr*fpr | fpi*fpi
//				pfacc mm5, mm5; // mm5 = fpr*fpr+fpi*fpi

				movss xmm7, smallf; // 1e-15f
				shufps xmm7, xmm7, 0;

				movaps xmm6, xmm5; //copy
				shufps xmm6, xmm6, (128 + 48 + 0 + 1);//  10 11 00 01 low// swap re & im
				addps xmm5, xmm6; // xmm5 = sumre*sumre + sumim*sumim

//				psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density cur
				addps xmm5, xmm7;// xmm5 =psd cur
				shr eax, 1;
				movlps xmm7, [ebx+eax];// pattern3d - two values
				shufps xmm7, xmm7, (64 + 16 + 0 + 0) ;// 01 01 00 00 low
				shl eax, 1;

//				pfsubr mm7, mm5; // psd - sigma
				movaps xmm6, xmm5; //psd
				subps xmm6, xmm7; // psd - sigma

				rcpps xmm5, xmm5; // mm5= 1/psd
				mulps xmm5, xmm6; // // (psd-sigma)/psd
				movss xmm7, lowlimit;
				shufps xmm7, xmm7, 0;
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
//				fpr *= WienerFactor; // apply filter on real  part	
//				fpi *= WienerFactor; // apply filter on imaginary part
				maxps xmm5, xmm7; // mm5 =wienerfactor
				mulps xmm3, xmm5; // mm3 = final wiener fpr | fpi
				// mm2, mm5, mm6, mm7 are free
				


				movaps xmm5, xmm1; // mm5= copy   fnr | fni
				mulps xmm5, xmm1; // mm5 = fnr*fnr | fni*fni

				movss xmm7, smallf; // 1e-15f
				shufps xmm7, xmm7, 0;

//				pfacc mm5, mm5; // mm5 = fnr*fnr+fni*fni
				movaps xmm6, xmm5;
				shufps xmm6, xmm6, (128 + 48 + 0 + 1);//  10 11 00 01 low// swap re & im
				addps xmm5, xmm6; // xmm5 = sumre*sumre + sumim*sumim

//				psd = fnr*fnr + fni*fni + 1e-15f; // power spectrum density cur
				addps xmm5, xmm7;// xmm5 =psd cur
				shr eax, 1;
				movlps xmm7, [ebx+eax];// pattern3d - two values
				shufps xmm7, xmm7, (64 + 16 + 0 + 0) ;// 01 01 00 00 low
				shl eax, 1;

//				pfsubr mm7, mm5; // psd - sigma
				movaps xmm6, xmm5; //psd
				subps xmm6, xmm7; // psd - sigma

				rcpps xmm5, xmm5; // mm5= 1/psd
				mulps xmm5, xmm6; // // (psd-sigma)/psd
				movss xmm7, lowlimit;
				shufps xmm7, xmm7,0;
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
//				fnr *= WienerFactor; // apply filter on real  part	
//				fni *= WienerFactor; // apply filter on imaginary part
				maxps xmm5, xmm7; // mm5 =wienerfactor
				mulps xmm1, xmm5; // mm1 = final wiener fmr | fmi
				// mm2, mm5, mm6, mm7 are free
				
				// reverse dft for 3 points
				addps xmm4, xmm3; // fc + fp
				addps xmm4, xmm1; // fc + fp + fn
				movss xmm7, onethird;
				shufps xmm7, xmm7, 0;
				mulps xmm4, xmm7;
//				outprev[w][0] = (fcr + fpr + fnr)*0.33333333333f; // get  real  part	
//				outprev[w][1] = (fci + fpi + fni)*0.33333333333f; // get imaginary part
				movaps [edi+eax], xmm4; // write output to prev array
				// Attention! return filtered "out" in "outprev" to preserve "out" for next step

				add eax, 16; // two numbers, that is why we use pitch
				cmp eax, ecx;
				jge blockend;
				jmp nextnumber;
blockend:		mov eax, blockscounter;
				dec eax
				je	finish;
				mov blockscounter, eax;
				mov eax, 0;
				add edx, ecx; // new block
				add edi, ecx;
				add esi, ecx;
				jmp nextnumber;
finish:			emms;
		}
#endif // !x64
}
//-------------------------------------------------------------------------------------------
//
void Sharpen_SSE(fftwf_complex *outcur, int outwidth, int outpitch, int bh, 
				 int howmanyblocks, float sharpen, float sigmaSquaredSharpenMin, 
				 float sigmaSquaredSharpenMax, float *wsharpen)
{
//	int h,w, block;
//	float psd;
//	float sfact;
//	float one = 1.0f;
	int bytesperblock = bh*outpitch*8;
	int blockscounter = howmanyblocks;

#ifndef X86_64

	if (sharpen != 0 )
	{
		__asm {


//		for (block =0; block <howmanyblocks; block++)
//		{
//			for (h=0; h<bh; h++) // middle
//			{
//				for (w=0; w<outwidth; w++) // skip leftmost column w=0
//				{
			emms;
			mov esi, outcur; // current
			mov ebx, wsharpen;
			mov ecx, bytesperblock; // counter
			movss xmm7, sharpen;
			shufps xmm7, xmm7, 0 ;// xmm7 = sharpen
			movss xmm5, sigmaSquaredSharpenMin;
			shufps xmm5, xmm5, 0; // xmm5 =sigmaSquaredSharpenMin
			movss xmm4, sigmaSquaredSharpenMax;
			shufps xmm4, xmm4, 0; // xmm4 =sigmaSquaredSharpenMax
			mov eax, 0;
			mov edx, 0;
align 16
nextnumber:
//				shr eax, 1;
				movlps xmm6, [ebx+edx];// wsharpen - two values
				shufps xmm6, xmm6, (64 + 16 + 0 + 0) ;// 01 01 00 00 low
//				shl eax, 1;

				// take two complex numbers
				movaps xmm1, [esi+eax]; // xmm1=cur real | img 
				movaps xmm0, xmm1; // copy sum
				mulps xmm0, xmm0; // xmm0 =sumre*sumre | sumim*sumim
				movaps xmm3, xmm0; //copy
				shufps xmm3, xmm3, (128 + 48 + 0 + 1);// swap re & im
//					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]);
				addps xmm0, xmm3; // xmm0 = psd = sumre*sumre + sumim*sumim
				movaps xmm2, xmm0; //copy psd
				addps xmm2, xmm5; // psd + smin
				movaps xmm3, xmm0; //copy psd
				addps xmm3, xmm4; // psd + smax
				mulps xmm3, xmm2; // (psd + smin)*(psd + smax)
				mulps xmm0, xmm4; // psd*smax
				rcpps xmm3, xmm3; // 1/(psd + smin)*(psd + smax)
				mulps xmm0, xmm3; // psd*smax/((psd + smin)*(psd + smax))
				sqrtps xmm0, xmm0; // sqrt()
//improved sharpen mode to prevent grid artifactes and to limit sharpening both for low and high amplitudes
//				sfact = (1 + sharpen*wsharpen[w]*sqrt( psd*sigmaSquaredSharpenMax/((psd + sigmaSquaredSharpenMin)*(psd + sigmaSquaredSharpenMax)) ) ); // sharpen factor - changed in v1.1c
				mulps xmm0, xmm6; // wsharpen*sqrt()
				mulps xmm0, xmm7; // sharpen*wsharpen*sqrt()
				mulps xmm0, xmm1; // outcur*sharpen*wsharpen*sqrt()
				addps xmm0, xmm1; // outcur + outcur*sharpen*wsharpen*sqrt()
//				outcur[w][0] *= sfact;
//				outcur[w][1] *= sfact;
				movaps [esi+eax], xmm0;
//				}
//				outcur += outpitch;
//				wsharpen += outpitch;
//			}
//			wsharpen -= outpitch*bh;
//		}
				add edx, 8; // sharpen
				add eax, 16; // two numbers, that is why we use pitch  
				cmp eax, ecx;
				jge blockend;
				jmp nextnumber;
blockend:		mov eax, blockscounter;
				dec eax
				je	finish;
				mov blockscounter, eax;
				mov eax, 0;
				mov edx, 0; 
				add esi, ecx;// new block
				jmp nextnumber;
finish:			emms;
		}
	}
#endif // !x64
}
//-------------------------------------------------------------------------------------------
//
void Sharpen_degrid_SSE(fftwf_complex *outcur, int outwidth, int outpitch, int bh, 
				 int howmanyblocks, float sharpen, float sigmaSquaredSharpenMin, 
				 float sigmaSquaredSharpenMax, float *wsharpen,
				 float degrid, fftwf_complex *gridsample, float dehalo, float *wdehalo, float ht2n)
{
//	int h,w, block;
//	float psd;
//	float sfact;
//	float one = 1.0f;
	int bytesperblock = bh*outpitch*8;
	int blockscounter = howmanyblocks;

	xmmreg gridcorrection;
	float gridfraction;

#ifndef X86_64

	if (sharpen != 0 && dehalo == 0)
	{
		__asm {


//		for (block =0; block <howmanyblocks; block++)
//		{
//			for (h=0; h<bh; h++) // middle
//			{
//				for (w=0; w<outwidth; w++) // skip leftmost column w=0
//				{
			emms;
			mov esi, outcur; // current
			mov edx, wsharpen;
			mov edi, wdehalo;
			mov ecx, bytesperblock; // counter
			movss xmm7, sharpen;
			shufps xmm7, xmm7, 0 ;// xmm7 = sharpen
			movss xmm4, sigmaSquaredSharpenMax;
			shufps xmm4, xmm4, 0; // xmm4 =sigmaSquaredSharpenMax
blockend:		mov eax, blockscounter;
				test eax, eax;
				je	finish;
				dec eax
				mov blockscounter, eax;
				
				movss xmm3, [esi]; // mm3=cur real | img 
				movss xmm7, degrid;
				mulps xmm7, xmm3;
				mov ebx, gridsample;
				movss xmm3, [ebx];
				rcpps xmm3, xmm3;
				mulps xmm7, xmm3;
				movss gridfraction, xmm7;
				mov eax, 0;
align 16
nextnumber:
				movaps xmm3, [ebx+eax]; // mm3=grid real | img 
				movss xmm7, gridfraction;
				shufps xmm7, xmm7, 0;
				mulps xmm3, xmm7; // fraction*sample
				movaps xmm7, xmm3; // copy
				movups gridcorrection, xmm7;

				// take two complex numbers
				movaps xmm1, [esi+eax]; // xmm1=cur real | img 

				subps xmm1, xmm7; // - gridcorrection

				movaps xmm0, xmm1; // copy sum
				mulps xmm0, xmm0; // xmm0 =sumre*sumre | sumim*sumim
				movaps xmm3, xmm0; //copy
				shufps xmm3, xmm3, (128 + 48 + 0 + 1);// swap re & im
//					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]);
				addps xmm0, xmm3; // xmm0 = psd = sumre*sumre + sumim*sumim
				movss xmm5, sigmaSquaredSharpenMin;
				movaps xmm2, xmm0; //copy psd
				shufps xmm5, xmm5, 0; // xmm5 =sigmaSquaredSharpenMin
				addps xmm2, xmm5; // psd + smin
				movaps xmm5, xmm0; //copy psd for dehalo
				movaps xmm3, xmm0; //copy psd
				addps xmm3, xmm4; // psd + smax
				mulps xmm3, xmm2; // (psd + smin)*(psd + smax)
				mulps xmm0, xmm4; // psd*smax
				rcpps xmm3, xmm3; // 1/(psd + smin)*(psd + smax)
				mulps xmm0, xmm3; // psd*smax/((psd + smin)*(psd + smax))
				sqrtps xmm0, xmm0; // sqrt()
//improved sharpen mode to prevent grid artifactes and to limit sharpening both for low and high amplitudes
//				sfact = (1 + sharpen*wsharpen[w]*sqrt( psd*sigmaSquaredSharpenMax/((psd + sigmaSquaredSharpenMin)*(psd + sigmaSquaredSharpenMax)) ) ); // sharpen factor - changed in v1.1c

				shr eax, 1;
				movlps xmm6, [edx+eax];// wsharpen - two values
				shufps xmm6, xmm6, (64 + 16 + 0 + 0) ;// 01 01 00 00 low
				shl eax, 1;

				mulps xmm0, xmm6; // wsharpen*sqrt()
				movss xmm7, sharpen;
				shufps xmm7, xmm7, 0 ;// xmm7 = sharpen

				mulps xmm0, xmm7; // sharpen*wsharpen*sqrt()
				mulps xmm0, xmm1; // outcur*sharpen*wsharpen*sqrt()
				addps xmm0, xmm1; // outcur + outcur*sharpen*wsharpen*sqrt()

				movups xmm7, gridcorrection;
				addps xmm0, xmm7;

//				outcur[w][0] *= sfact;
//				outcur[w][1] *= sfact;
				movaps [esi+eax], xmm0;
//				}
//				outcur += outpitch;
//				wsharpen += outpitch;
//			}
//			wsharpen -= outpitch*bh;
//		}
				add eax, 16;
				cmp eax, ecx;
				jl nextnumber;
				add esi, ecx;// new block
				jmp blockend;
finish:			emms;
		}
	}
	if (sharpen == 0 && dehalo != 0)
	{
		__asm {


//		for (block =0; block <howmanyblocks; block++)
//		{
//			for (h=0; h<bh; h++) // middle
//			{
//				for (w=0; w<outwidth; w++) // skip leftmost column w=0
//				{
			emms;
			mov esi, outcur; // current
			mov edx, wsharpen;
			mov edi, wdehalo;
			mov ecx, bytesperblock; // counter
			movss xmm7, sharpen;
			shufps xmm7, xmm7, 0 ;// xmm7 = sharpen
			movss xmm4, sigmaSquaredSharpenMax;
			shufps xmm4, xmm4, 0; // xmm4 =sigmaSquaredSharpenMax
blockend2:		mov eax, blockscounter;
				test eax, eax;
				je	finish2;
				dec eax
				mov blockscounter, eax;
				
				movss xmm3, [esi]; // mm3=cur real | img 
				movss xmm7, degrid;
				mulps xmm7, xmm3;
				mov ebx, gridsample;
				movss xmm3, [ebx];
				rcpps xmm3, xmm3;
				mulps xmm7, xmm3;
				movss gridfraction, xmm7;
				mov eax, 0;
align 16
nextnumber2:
				movaps xmm3, [ebx+eax]; // mm3=grid real | img 
				movss xmm7, gridfraction;
				shufps xmm7, xmm7, 0;
				mulps xmm3, xmm7; // fraction*sample
				movaps xmm7, xmm3; // copy
				movups gridcorrection, xmm7;

				// take two complex numbers
				movaps xmm1, [esi+eax]; // xmm1=cur real | img 

				subps xmm1, xmm7; // - gridcorrection

				movaps xmm0, xmm1; // copy sum
				mulps xmm0, xmm0; // xmm0 =sumre*sumre | sumim*sumim
				movaps xmm3, xmm0; //copy
				shufps xmm3, xmm3, (128 + 48 + 0 + 1);// swap re & im
//					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]);
				addps xmm0, xmm3; // xmm0 = psd = sumre*sumre + sumim*sumim
				movaps xmm5, xmm0; //copy psd for dehalo
				movaps xmm0, xmm1; //copy current

//			(psd + ht2n)/((psd + ht2n) + dehalo*wdehalo[w] * psd ); // dehalo factor
				shr eax, 1;
				movlps xmm6, [edi+eax];// wdehalo - two values
				shufps xmm6, xmm6, (64 + 16 + 0 + 0) ;// 01 01 00 00 low
				shl eax, 1;

				movss xmm7, dehalo;
				shufps xmm7, xmm7, 0 ;// xmm7 = dehalo
				mulps xmm6, xmm7; // dehalo*wdehalo
				mulps xmm6, xmm5; // dehalo*wdehalo*psd
				addps xmm6, xmm5; // dehalo*wdehalo*psd + psd
				movss xmm7, ht2n;
				shufps xmm7, xmm7, 0; // xmm7=ht2n
				addps xmm6, xmm7; // dehalo*wdehalo*psd + psd + ht2n
				rcpps xmm6, xmm6; // inverse
				addps xmm5, xmm7; // psd + ht2n
				mulps xmm6, xmm5; // dehalo factor
				mulps xmm0, xmm6; // halo-corrected currect

				movups xmm7, gridcorrection;
				addps xmm0, xmm7;

//				outcur[w][0] *= sfact;
//				outcur[w][1] *= sfact;
				movaps [esi+eax], xmm0;
//				}
//				outcur += outpitch;
//				wsharpen += outpitch;
//			}
//			wsharpen -= outpitch*bh;
//		}
				add eax, 16;
				cmp eax, ecx;
				jl nextnumber2;
				add esi, ecx;// new block
				jmp blockend2;
finish2:			emms;
		}
	}
	if (sharpen != 0 || dehalo != 0)
	{
		__asm {


//		for (block =0; block <howmanyblocks; block++)
//		{
//			for (h=0; h<bh; h++) // middle
//			{
//				for (w=0; w<outwidth; w++) // skip leftmost column w=0
//				{
			emms;
			mov esi, outcur; // current
			mov edx, wsharpen;
			mov edi, wdehalo;
			mov ecx, bytesperblock; // counter
			movss xmm7, sharpen;
			shufps xmm7, xmm7, 0 ;// xmm7 = sharpen
			movss xmm4, sigmaSquaredSharpenMax;
			shufps xmm4, xmm4, 0; // xmm4 =sigmaSquaredSharpenMax
blockend3:		mov eax, blockscounter;
				test eax, eax;
				je	finish3;
				dec eax
				mov blockscounter, eax;
				
				movss xmm3, [esi]; // mm3=cur real | img 
				movss xmm7, degrid;
				mulps xmm7, xmm3;
				mov ebx, gridsample;
				movss xmm3, [ebx];
				rcpps xmm3, xmm3;
				mulps xmm7, xmm3;
				movss gridfraction, xmm7;
				mov eax, 0;
align 16
nextnumber3:
				movaps xmm3, [ebx+eax]; // mm3=grid real | img 
				movss xmm7, gridfraction;
				shufps xmm7, xmm7, 0;
				mulps xmm3, xmm7; // fraction*sample
				movaps xmm7, xmm3; // copy
				movups gridcorrection, xmm7;

				// take two complex numbers
				movaps xmm1, [esi+eax]; // xmm1=cur real | img 

				subps xmm1, xmm7; // - gridcorrection

				movaps xmm0, xmm1; // copy sum
				mulps xmm0, xmm0; // xmm0 =sumre*sumre | sumim*sumim
				movaps xmm3, xmm0; //copy
				shufps xmm3, xmm3, (128 + 48 + 0 + 1);// swap re & im
//					psd = (outcur[w][0]*outcur[w][0] + outcur[w][1]*outcur[w][1]);
				addps xmm0, xmm3; // xmm0 = psd = sumre*sumre + sumim*sumim
				movss xmm5, sigmaSquaredSharpenMin;
				movaps xmm2, xmm0; //copy psd
				shufps xmm5, xmm5, 0; // xmm5 =sigmaSquaredSharpenMin
				addps xmm2, xmm5; // psd + smin
				movaps xmm5, xmm0; //copy psd for dehalo
				movaps xmm3, xmm0; //copy psd
				addps xmm3, xmm4; // psd + smax
				mulps xmm3, xmm2; // (psd + smin)*(psd + smax)
				mulps xmm0, xmm4; // psd*smax
				rcpps xmm3, xmm3; // 1/(psd + smin)*(psd + smax)
				mulps xmm0, xmm3; // psd*smax/((psd + smin)*(psd + smax))
				sqrtps xmm0, xmm0; // sqrt()
//improved sharpen mode to prevent grid artifactes and to limit sharpening both for low and high amplitudes
//				sfact = (1 + sharpen*wsharpen[w]*sqrt( psd*sigmaSquaredSharpenMax/((psd + sigmaSquaredSharpenMin)*(psd + sigmaSquaredSharpenMax)) ) ); // sharpen factor - changed in v1.1c

				shr eax, 1;
				movlps xmm6, [edx+eax];// wsharpen - two values
				shufps xmm6, xmm6, (64 + 16 + 0 + 0) ;// 01 01 00 00 low
				shl eax, 1;

				mulps xmm0, xmm6; // wsharpen*sqrt()
				movss xmm7, sharpen;
				shufps xmm7, xmm7, 0 ;// xmm7 = sharpen

				mulps xmm0, xmm7; // sharpen*wsharpen*sqrt()
				mulps xmm0, xmm1; // outcur*sharpen*wsharpen*sqrt()
				addps xmm0, xmm1; // outcur + outcur*sharpen*wsharpen*sqrt()

//			(psd + ht2n)/((psd + ht2n) + dehalo*wdehalo[w] * psd ); // dehalo factor
				shr eax, 1;
				movlps xmm6, [edi+eax];// wdehalo - two values
				shufps xmm6, xmm6, (64 + 16 + 0 + 0) ;// 01 01 00 00 low
				shl eax, 1;

				movss xmm7, dehalo;
				shufps xmm7, xmm7, 0 ;// xmm7 = dehalo
				mulps xmm6, xmm7; // dehalo*wdehalo
				mulps xmm6, xmm5; // dehalo*wdehalo*psd
				addps xmm6, xmm5; // dehalo*wdehalo*psd + psd
				movss xmm7, ht2n;
				shufps xmm7, xmm7, 0; // xmm7=ht2n
				addps xmm6, xmm7; // dehalo*wdehalo*psd + psd + ht2n
				rcpps xmm6, xmm6; // inverse
				addps xmm5, xmm7; // psd + ht2n
				mulps xmm6, xmm5; // dehalo factor
				mulps xmm0, xmm6; // halo-corrected currect

				movups xmm7, gridcorrection;
				addps xmm0, xmm7;

//				outcur[w][0] *= sfact;
//				outcur[w][1] *= sfact;
				movaps [esi+eax], xmm0;
//				}
//				outcur += outpitch;
//				wsharpen += outpitch;
//			}
//			wsharpen -= outpitch*bh;
//		}
				add eax, 16;
				cmp eax, ecx;
				jl nextnumber3;
				add esi, ecx;// new block
				jmp blockend3;
finish3:			emms;
		}
	}
#endif // !x64
}

//-----------------------------------------------------------------------------------------
//
void ApplyWiener3D3_degrid_SSE(fftwf_complex *outcur, fftwf_complex *outprev, 
						fftwf_complex *outnext, int outwidth, int outpitch, int bh, 
						int howmanyblocks, float sigmaSquaredNoiseNormed, float beta,
					float degrid, fftwf_complex *gridsample)
{
	// dft 3d (very short - 3 points)
	// optimized for SSE assembler
	// return result in outprev
//	float fcr, fci, fpr, fpi, fnr, fni;
//	float pnr, pni, di, dr;
//	float WienerFactor =1;
//	float psd;
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0
	float sin120 = 0.86602540378443864676372317075294f; //sqrtf(3.0f)*0.5f;
	float smallf = 1e-15f;
	float onethird = 0.33333333333f;
	float onehalf = 0.5f;

	xmmreg gridcorrection;
	float gridfraction;

//	int block;
//	int h,w;

//	int totalbytes = howmanyblocks*bh*outpitch*8;
	int bytesperblock = bh*outpitch*8;
	int blockscounter = howmanyblocks;

#ifndef X86_64

//	for (line=0; line <totalnumber; line++)
//	{
		__asm
		{
			emms;
			mov esi, outcur; // current
			mov ecx, bytesperblock; // counter
			mov edi, outprev;
			mov edx, outnext;

blockend:		mov eax, blockscounter;
				test eax, eax;
				je	finish;
				dec eax
				mov blockscounter, eax;
				
				movaps xmm3, [esi]; // mm3=cur real | img 
				movss xmm7, degrid;
				shufps xmm7, xmm7, 0 ;
				mulps xmm7, xmm3;
				mov ebx, gridsample;
				movss xmm3, [ebx];
				shufps xmm3, xmm3, 0 ;
				rcpps xmm3, xmm3;
				mulps xmm7, xmm3;
				movss gridfraction, xmm7;
				mov eax, 0;
align 16
nextnumber:
				movaps xmm3, [ebx+eax]; // mm3=grid real | img 
				movss xmm7, gridfraction;
				shufps xmm7, xmm7, 0;
				mulps xmm3, xmm7; // fraction*sample
				movaps xmm7, xmm3; // copy
				addps xmm7, xmm7;
				addps xmm7, xmm3; // gridcorrection 0 | 1
				movups gridcorrection, xmm7;

				movaps xmm0, [edi+eax]; // xmm0=prev real | img
				movaps xmm1, [edx+eax]; // xmm1=next real | img
				movaps xmm3, [esi+eax]; // mm3=cur real | img 

				movaps xmm2, xmm0; // copy prev
				//pnr = outprev[w][0] + outnext[w][0];
				//pni = outprev[w][1] + outnext[w][1];
				addps xmm2, xmm1; // xmm2= summa prev and next  , pnr | pni

				//fcr = outcur[w][0] + pnr; // real cur
				//fci = outcur[w][1] + pni; // im cur
				movaps xmm4, xmm3 ;// copy of cur
				addps xmm4, xmm2; // mm4= fcr | fci

				subps xmm4, xmm7; // - gridcorrection

				movss xmm7, onehalf; // 0.5
				shufps xmm7, xmm7, 0 ; // xmm7 = onehalf
				mulps xmm2, xmm7; // mm2 = 0.5*pnr | 0.5*pni

				movaps xmm5, xmm4; // xmm5= copy   fcr | fci
				mulps xmm5, xmm4; // xmm5 = fcr*fcr | fci*fci

				movaps xmm6, xmm5; //copy
				shufps xmm6, xmm6, (128 + 48 + 0 + 1);//  10 11 00 01 low - swap re & im
				addps xmm5, xmm6; // xmm1 = sumre*sumre + sumim*sumim
				
				movss xmm7, smallf; // 1e-15f
				shufps xmm7, xmm7, 0 ;
//				psd = fcr*fcr + fci*fci + 1e-15f; // power spectrum density cur
				addps xmm5, xmm7;// xmm5 =psd cur
				movss xmm7, sigmaSquaredNoiseNormed;
				shufps xmm7, xmm7, 0 ;

				movaps xmm6, xmm5;
				subps xmm6, xmm7; // psd - sigma
				rcpps xmm5, xmm5; // xmm5= 1/psd
				mulps xmm5, xmm6; // // (psd-sigma)/psd
				movss xmm7, lowlimit;
				shufps xmm7, xmm7, 0 ;
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
//				fcr *= WienerFactor; // apply filter on real  part	
//				fci *= WienerFactor; // apply filter on imaginary part
				maxps xmm5, xmm7; // xmm5 =wienerfactor
				mulps xmm4, xmm5; // xmm4 = final wiener fcr | fci
				// mm5, mm6, mm7 are free
				
				
				movaps xmm5, xmm1;  // copy of next
				subps xmm5, xmm0; // next-prev   real | img
				movaps xmm6, xmm0; // copy of prev
				subps xmm6, xmm1; // prev-next  real | img

				shufps xmm5, xmm5, (128 + 0 + 8 + 0); // 10 00 10 00 low // 2 0 2 0 low //low Re(next-prev) | Re2(next-prev) || same second
				shufps xmm6, xmm6, (192 +16 + 12 + 1);// 11 01 11 01 low // 3 1 3 1 low //low Im(prev-next) | Im2(prev-next) || same second
				unpcklps xmm6, xmm5;// low Im(prev-next) | Re(next-prev) || Im2(prev-next) | Re2(next-prev)  

				movss xmm7, sin120;
				shufps xmm7, xmm7, 0 ; // xmm7 = sin120 | sin120
//				di = sin120*(outprev[w][1]-outnext[w][1]);
//				dr = sin120*(outnext[w][0]-outprev[w][0]);
				mulps xmm6, xmm7; // xmm6= di | dr

				//  xmm1, xmm5 are free
				subps xmm3, xmm2; // xmm3=cur-0.5*pn
				movaps xmm1, xmm3; // copy 
//				fpr = out[w][0] - 0.5f*pnr + di; // real prev
//				fpi = out[w][1] - 0.5f*pni + dr; // im prev
				addps xmm3, xmm6;// xmm3 = fpr | fpi
				
//				fnr = out[w][0] - 0.5f*pnr - di; // real next
//				fni = out[w][1] - 0.5f*pni - dr ; // im next
				subps xmm1, xmm6;// mm1 = fnr | fni
				// mm2, mm5, mm6, mm7 are free


				movaps xmm5, xmm3; // xmm5= copy   fpr | fpi
				mulps xmm5, xmm3; // xmm5 = fpr*fpr | fpi*fpi

				movss xmm7, smallf; // 1e-15f
				shufps xmm7, xmm7, 0;

				movaps xmm6, xmm5; //copy
				shufps xmm6, xmm6, (128 + 48 + 0 + 1);//  10 11 00 01 low// swap re & im
				addps xmm5, xmm6; // xmm5 = sumre*sumre + sumim*sumim

//				psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density cur
				addps xmm5, xmm7;// xmm5 =psd cur
				movss xmm7, sigmaSquaredNoiseNormed;
				shufps xmm7, xmm7, 0;

				movaps xmm6, xmm5; //psd
				subps xmm6, xmm7; // psd - sigma

				rcpps xmm5, xmm5; // mm5= 1/psd
				mulps xmm5, xmm6; // // (psd-sigma)/psd
				movss xmm7, lowlimit;
				shufps xmm7, xmm7, 0;
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
//				fpr *= WienerFactor; // apply filter on real  part	
//				fpi *= WienerFactor; // apply filter on imaginary part
				maxps xmm5, xmm7; // mm5 =wienerfactor
				mulps xmm3, xmm5; // mm3 = final wiener fpr | fpi
				// mm2, mm5, mm6, mm7 are free
				


				movaps xmm5, xmm1; // mm5= copy   fnr | fni
				mulps xmm5, xmm1; // mm5 = fnr*fnr | fni*fni

				movss xmm7, smallf; // 1e-15f
				shufps xmm7, xmm7, 0;

				movaps xmm6, xmm5;
				shufps xmm6, xmm6, (128 + 48 + 0 + 1);//  10 11 00 01 low// swap re & im
				addps xmm5, xmm6; // xmm5 = sumre*sumre + sumim*sumim

//				psd = fnr*fnr + fni*fni + 1e-15f; // power spectrum density cur
				addps xmm5, xmm7;// xmm5 =psd cur
				movss xmm7, sigmaSquaredNoiseNormed;
				shufps xmm7, xmm7, 0;

				movaps xmm6, xmm5; //psd
				subps xmm6, xmm7; // psd - sigma

				rcpps xmm5, xmm5; // mm5= 1/psd
				mulps xmm5, xmm6; // // (psd-sigma)/psd
				movss xmm7, lowlimit;
				shufps xmm7, xmm7,0;
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
//				fnr *= WienerFactor; // apply filter on real  part	
//				fni *= WienerFactor; // apply filter on imaginary part
				maxps xmm5, xmm7; // mm5 =wienerfactor
				mulps xmm1, xmm5; // mm1 = final wiener fmr | fmi
				// mm2, mm5, mm6, mm7 are free
				
				// reverse dft for 3 points
				addps xmm4, xmm3; // fc + fp
				addps xmm4, xmm1; // fc + fp + fn

				movups xmm7, gridcorrection;
				addps xmm4, xmm7;

				movss xmm7, onethird;
				shufps xmm7, xmm7, 0;
				mulps xmm4, xmm7;
//				outprev[w][0] = (fcr + fpr + fnr)*0.33333333333f; // get  real  part	
//				outprev[w][1] = (fci + fpi + fni)*0.33333333333f; // get imaginary part
				movaps [edi+eax], xmm4; // write output to prev array
				// Attention! return filtered "out" in "outprev" to preserve "out" for next step

				add eax, 16;
				cmp eax, ecx;
				jl nextnumber;
				add edx, ecx; // new block
				add edi, ecx;
				add esi, ecx;
				jmp blockend;
finish:			emms;
		}
#endif // !x64
}
//-----------------------------------------------------------------------------------------
//
void ApplyPattern3D3_degrid_SSE(fftwf_complex *outcur, fftwf_complex *outprev, 
						 fftwf_complex *outnext, int outwidth, int outpitch, int bh, 
						 int howmanyblocks, float *pattern3d, float beta,
						 float degrid, fftwf_complex *gridsample)
{
	// dft 3d (very short - 3 points)
	// SSE assembler
	// return result in outprev
//	float fcr, fci, fpr, fpi, fnr, fni;
//	float pnr, pni, di, dr;
//	float WienerFactor =1;
//	float psd;
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0
	float sin120 = 0.86602540378443864676372317075294f;// sqrtf(3.0f)*0.5f;
	float smallf = 1e-15f;
	float onethird = 0.33333333333f;
	float onehalf = 0.5f;

	xmmreg gridcorrection;
//	float gridcorrection[4];
	float gridfraction;

//	int block;
//	int h,w;

//	int totalbytes = howmanyblocks*bh*outwidth*8;
	int bytesperblock = bh*outpitch*8;
	int blockscounter = howmanyblocks;
#ifndef X86_64

//	for (line=0; line <totalnumber; line++)
//	{
		__asm
		{
			emms;
			mov edi, outprev;
			mov edx, outnext;
			mov esi, outcur; // current
			mov ebx, pattern3d;

blockend:		mov eax, blockscounter;
				test eax, eax;
				je	finish;
				dec eax
				mov blockscounter, eax;
				
				movaps xmm3, [esi]; // mm3=cur real | img 
				movss xmm7, degrid;
				shufps xmm7, xmm7, 0 ;
				mulps xmm7, xmm3;
				mov ecx, gridsample;
				movss xmm3, [ecx];
				shufps xmm3, xmm3, 0 ;
				rcpps xmm3, xmm3;
				mulps xmm7, xmm3;
				movss gridfraction, xmm7;
				mov eax, 0;
			
align 16
nextnumber:
				mov ecx, gridsample;
				movaps xmm3, [ecx+eax]; // mm3=grid real | img 
				movss xmm7, gridfraction;
				shufps xmm7, xmm7, 0;
				mulps xmm3, xmm7; // fraction*sample
				movaps xmm7, xmm3; // copy
				addps xmm7, xmm7;
				addps xmm7, xmm3; // gridcorrection 0 | 1
				movups gridcorrection, xmm7;

				movaps xmm0, [edi+eax]; // xmm0=prev real | img
				movaps xmm1, [edx+eax]; // xmm1=next real | img
				movaps xmm3, [esi+eax]; // mm3=cur real | img 

				movaps xmm2, xmm0; // copy prev
				//pnr = outprev[w][0] + outnext[w][0];
				//pni = outprev[w][1] + outnext[w][1];
				addps xmm2, xmm1; // xmm2= summa prev and next  , pnr | pni

				//fcr = outcur[w][0] + pnr; // real cur
				//fci = outcur[w][1] + pni; // im cur
				movaps xmm4, xmm3 ;// copy of cur
				addps xmm4, xmm2; // mm4= fcr | fci

				subps xmm4, xmm7; // - gridcorrection

				movss xmm7, onehalf; // 0.5
				//punpckldq mm7, mm7 ; // mm7 = 0.5 | 0.5
				shufps xmm7, xmm7, 0 ; // xmm7 = onehalf
				mulps xmm2, xmm7; // mm2 = 0.5*pnr | 0.5*pni


				movaps xmm5, xmm4; // xmm5= copy   fcr | fci
				mulps xmm5, xmm4; // xmm5 = fcr*fcr | fci*fci

				//pfacc mm5, mm5; // mm5 = fcr*fcr+fci*fci

				movaps xmm6, xmm5; //copy
				shufps xmm6, xmm6, (128 + 48 + 0 + 1);//  10 11 00 01  low- swap re & im
				addps xmm5, xmm6; // xmm1 = sumre*sumre + sumim*sumim
				
				movss xmm7, smallf; // 1e-15f
				shufps xmm7, xmm7, 0 ;
//				psd = fcr*fcr + fci*fci + 1e-15f; // power spectrum density cur
				addps xmm5, xmm7;// xmm5 =psd cur
				shr eax, 1;
				movlps xmm7, [ebx+eax];// pattern3d - two values
				shufps xmm7, xmm7, (64 + 16 + 0 + 0) ;// 01 01 00 00 low
				shl eax, 1;

//				pfsubr mm7, mm5; // psd - sigma
				movaps xmm6, xmm5;
				subps xmm6, xmm7; // psd - sigma
				rcpps xmm5, xmm5; // xmm5= 1/psd
				mulps xmm5, xmm6; // // (psd-sigma)/psd
				movss xmm7, lowlimit;
				shufps xmm7, xmm7, 0 ;
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
//				fcr *= WienerFactor; // apply filter on real  part	
//				fci *= WienerFactor; // apply filter on imaginary part
				maxps xmm5, xmm7; // xmm5 =wienerfactor
				mulps xmm4, xmm5; // xmm4 = final wiener fcr | fci
				// mm5, mm6, mm7 are free
				
				
				movaps xmm5, xmm1;  // copy of next
				subps xmm5, xmm0; // next-prev   real | img
				movaps xmm6, xmm0; // copy of prev
				subps xmm6, xmm1; // prev-next  real | img

//				pswapd mm6, mm6; // swap real and img,    img | real
//				punpckldq mm5, mm6;  // use low dwords:   real next-prev | img prev-next
//				pswapd mm5, mm5 ; // mm5 = im(prev-next) | re(next-prev)

				shufps xmm5, xmm5, (128 + 0 + 8 + 0);// 10 00 10 00 low // 2 0 2 0 low//low Re(next-prev) | Re2(next-prev) high || same second
				shufps xmm6, xmm6, (192 +16 +12 + 1);// 11 01 11 01 low // 3 1 3 1 low//low Im(prev-next) | Im2(prev-next) high || same second
				unpcklps xmm6, xmm5;//low Im(prev-next) | Re(next-prev) || Im2(prev-next) | Re2(next-prev)  

				movss xmm7, sin120;
				shufps xmm7, xmm7, 0 ; // xmm7 = sin120 | sin120
//				di = sin120*(outprev[w][1]-outnext[w][1]);
//				dr = sin120*(outnext[w][0]-outprev[w][0]);
				mulps xmm6, xmm7; // xmm6= di | dr

				//  xmm1, xmm5 are free
				subps xmm3, xmm2; // xmm3=cur-0.5*pn
				movaps xmm1, xmm3; // copy 
//				fpr = out[w][0] - 0.5f*pnr + di; // real prev
//				fpi = out[w][1] - 0.5f*pni + dr; // im prev
				addps xmm3, xmm6;// xmm3 = fpr | fpi
				
//				fnr = out[w][0] - 0.5f*pnr - di; // real next
//				fni = out[w][1] - 0.5f*pni - dr ; // im next
				subps xmm1, xmm6;// mm1 = fnr | fni
				// mm2, mm5, mm6, mm7 are free


				movaps xmm5, xmm3; // xmm5= copy   fpr | fpi
				mulps xmm5, xmm3; // xmm5 = fpr*fpr | fpi*fpi
//				pfacc mm5, mm5; // mm5 = fpr*fpr+fpi*fpi

				movss xmm7, smallf; // 1e-15f
				shufps xmm7, xmm7, 0;

				movaps xmm6, xmm5; //copy
				shufps xmm6, xmm6, (128 + 48 + 0 + 1);//  10 11 00 01 low// swap re & im
				addps xmm5, xmm6; // xmm5 = sumre*sumre + sumim*sumim

//				psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density cur
				addps xmm5, xmm7;// xmm5 =psd cur
				shr eax, 1;
				movlps xmm7, [ebx+eax];// pattern3d - two values
				shufps xmm7, xmm7, (64 + 16 + 0 + 0) ;// 01 01 00 00 low
				shl eax, 1;

//				pfsubr mm7, mm5; // psd - sigma
				movaps xmm6, xmm5; //psd
				subps xmm6, xmm7; // psd - sigma

				rcpps xmm5, xmm5; // mm5= 1/psd
				mulps xmm5, xmm6; // // (psd-sigma)/psd
				movss xmm7, lowlimit;
				shufps xmm7, xmm7, 0;
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
//				fpr *= WienerFactor; // apply filter on real  part	
//				fpi *= WienerFactor; // apply filter on imaginary part
				maxps xmm5, xmm7; // mm5 =wienerfactor
				mulps xmm3, xmm5; // mm3 = final wiener fpr | fpi
				// mm2, mm5, mm6, mm7 are free
				


				movaps xmm5, xmm1; // mm5= copy   fnr | fni
				mulps xmm5, xmm1; // mm5 = fnr*fnr | fni*fni

				movss xmm7, smallf; // 1e-15f
				shufps xmm7, xmm7, 0;

//				pfacc mm5, mm5; // mm5 = fnr*fnr+fni*fni
				movaps xmm6, xmm5;
				shufps xmm6, xmm6, (128 + 48 + 0 + 1);//  10 11 00 01 low// swap re & im
				addps xmm5, xmm6; // xmm5 = sumre*sumre + sumim*sumim

//				psd = fnr*fnr + fni*fni + 1e-15f; // power spectrum density cur
				addps xmm5, xmm7;// xmm5 =psd cur
				shr eax, 1;
				movlps xmm7, [ebx+eax];// pattern3d - two values
				shufps xmm7, xmm7, (64 + 16 + 0 + 0) ;// 01 01 00 00 low
				shl eax, 1;

//				pfsubr mm7, mm5; // psd - sigma
				movaps xmm6, xmm5; //psd
				subps xmm6, xmm7; // psd - sigma

				rcpps xmm5, xmm5; // mm5= 1/psd
				mulps xmm5, xmm6; // // (psd-sigma)/psd
				movss xmm7, lowlimit;
				shufps xmm7, xmm7,0;
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
//				fnr *= WienerFactor; // apply filter on real  part	
//				fni *= WienerFactor; // apply filter on imaginary part
				maxps xmm5, xmm7; // mm5 =wienerfactor
				mulps xmm1, xmm5; // mm1 = final wiener fmr | fmi
				// mm2, mm5, mm6, mm7 are free
				
				// reverse dft for 3 points
				addps xmm4, xmm3; // fc + fp
				addps xmm4, xmm1; // fc + fp + fn

				movups xmm7, gridcorrection;
				addps xmm4, xmm7;

				movss xmm7, onethird;
				shufps xmm7, xmm7, 0;
				mulps xmm4, xmm7;
//				outprev[w][0] = (fcr + fpr + fnr)*0.33333333333f; // get  real  part	
//				outprev[w][1] = (fci + fpi + fni)*0.33333333333f; // get imaginary part
				movaps [edi+eax], xmm4; // write output to prev array
				// Attention! return filtered "out" in "outprev" to preserve "out" for next step

				add eax, 16; // two numbers, that is why we use pitch
				mov ecx, bytesperblock; // counter
				cmp eax, ecx;
				jl nextnumber;
				add edx, ecx; // new block
				add edi, ecx;
				add esi, ecx;
				jmp blockend;
finish:			emms;
		}
#endif // !x64
}
//-----------------------------------------------------------------------------------------
//
void ApplyWiener3D4_degrid_SSE(fftwf_complex *outcur, fftwf_complex *outprev2, fftwf_complex *outprev, 
						fftwf_complex *outnext, int outwidth, int outpitch, int bh, 
						int howmanyblocks, float sigmaSquaredNoiseNormed, float beta,
					float degrid, fftwf_complex *gridsample)
{
	// dft 3d (very short - 4 points)
	// optimized for SSE assembler
	// return result in outprev
//	float fcr, fci, fpr, fpi, fnr, fni, fp2r, fp2i;
//	float pnr, pni, di, dr;
//	float WienerFactor =1;
//	float psd;
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0
	float smallf = 1e-15f;
	float onefourth = 0.25f;

	xmmreg fp2; //temporary array for prev2, prev, cur
	xmmreg fp;
	xmmreg fc;
	xmmreg gridcorrection;
	float gridfraction;

//	int block;
//	int h,w;

//	int totalbytes = howmanyblocks*bh*outpitch*8;
	int bytesperblock = bh*outpitch*8;
	int blockscounter = howmanyblocks;

#ifndef X86_64

//	for (line=0; line <totalnumber; line++)
//	{
		__asm
		{
			emms;
			mov esi, outcur; // current
			mov ebx, outprev2; 
			mov edi, outprev;
			mov edx, outnext;

blockend:		mov eax, blockscounter;
				test eax, eax;
				je	finish;
				dec eax
				mov blockscounter, eax;
				
				movaps xmm3, [esi]; // mm3=cur real | img 
				movss xmm7, degrid;
				shufps xmm7, xmm7, 0 ;
				mulps xmm7, xmm3;
				mov ecx, gridsample;
				movss xmm3, [ecx];
				shufps xmm3, xmm3, 0 ;
				rcpps xmm3, xmm3;
				mulps xmm7, xmm3;
				movss gridfraction, xmm7;
				mov eax, 0;
align 16
nextnumber:
				mov ecx, gridsample;
				movaps xmm3, [ecx+eax]; // mm3=grid real | img 
				movss xmm7, gridfraction;
				shufps xmm7, xmm7, 0;
				mulps xmm3, xmm7; // fraction*sample
				movaps xmm7, xmm3; // copy
				addps xmm7, xmm7; // *2
				addps xmm7, xmm7; // *2, gridcorrection 0 | 1
				movups gridcorrection, xmm7;

				movaps xmm0, [ebx+eax]; // xmm0=prev2 real | img
				movaps xmm1, [edi+eax]; // xmm1=prev real | img
				movaps xmm2, [esi+eax]; // xmm2=cur real | img 
				movaps xmm3, [edx+eax]; // xmm3=next real | img 
// cur
//				fcr = outprev2[w][0] + outprev[w][0] + outcur[w][0] + outnext[w][0]; // real cur
//				fci = outprev2[w][1] + outprev[w][1] + outcur[w][1] + outnext[w][1]; // im cur
				movaps xmm4, xmm0; // copy prev2
				addps xmm4, xmm1;
				addps xmm4, xmm2;
				addps xmm4, xmm3;
//				fcr -= gridcorrection0_4;
//				fci -= gridcorrection1_4;
				subps xmm4, xmm7; // fcr

//				psd = fcr*fcr + fci*fci + 1e-15f; // power spectrum density cur
				movaps xmm5, xmm4; // xmm5= copy   fp2r | fp2i
				mulps xmm5, xmm4; // xmm5 = r*r | i*i
				movaps xmm6, xmm5; //copy
				shufps xmm6, xmm6, (128 + 48 + 0 + 1);//  10 11 00 01 low - swap re & im
				addps xmm5, xmm6; // xmm1 = sumre*sumre + sumim*sumim
				movss xmm7, smallf; // 1e-15f
				shufps xmm7, xmm7, 0 ;
				addps xmm5, xmm7;// xmm5 =psd cur
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				movss xmm7, sigmaSquaredNoiseNormed;
				shufps xmm7, xmm7, 0 ;

				movaps xmm6, xmm5;
				subps xmm6, xmm7; // psd - sigma
				rcpps xmm5, xmm5; // xmm5= 1/psd
				mulps xmm5, xmm6; // // (psd-sigma)/psd
				movss xmm7, lowlimit;
				shufps xmm7, xmm7, 0 ;
				maxps xmm5, xmm7; // xmm5 =wienerfactor
//				fcr *= WienerFactor; // apply filter on real  part	
//				fci *= WienerFactor; // apply filter on imaginary part
				mulps xmm4, xmm5; // xmm4 = final wiener fcr | fci
				movups fc, xmm4;// store fc

// prev2
//				fp2r = outprev2[w][0] - outprev[w][0] + outcur[w][0] - outnext[w][0]; // real prev2
//				fp2i = outprev2[w][1] - outprev[w][1] + outcur[w][1] - outnext[w][1]; // im cur
				movaps xmm4, xmm0; // copy prev2
				subps xmm4, xmm1;
				addps xmm4, xmm2;
				subps xmm4, xmm3;

//				psd = fp2r*fp2r + fp2i*fp2i + 1e-15f; // power spectrum density prev2
				movaps xmm5, xmm4; // xmm5= copy   fp2r | fp2i
				mulps xmm5, xmm4; // xmm5 = r*r | i*i
				movaps xmm6, xmm5; //copy
				shufps xmm6, xmm6, (128 + 48 + 0 + 1);//  10 11 00 01 low - swap re & im
				addps xmm5, xmm6; // xmm1 = sumre*sumre + sumim*sumim
				movss xmm7, smallf; // 1e-15f
				shufps xmm7, xmm7, 0 ;
				addps xmm5, xmm7;// xmm5 =psd prev2
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				movss xmm7, sigmaSquaredNoiseNormed;
				shufps xmm7, xmm7, 0 ;

				movaps xmm6, xmm5;
				subps xmm6, xmm7; // psd - sigma
				rcpps xmm5, xmm5; // xmm5= 1/psd
				mulps xmm5, xmm6; // // (psd-sigma)/psd
				movss xmm7, lowlimit;
				shufps xmm7, xmm7, 0 ;
				maxps xmm5, xmm7; // xmm5 =wienerfactor
//				fp2r *= WienerFactor; // apply filter on real  part	
//				fp2i *= WienerFactor; // apply filter on imaginary part
				mulps xmm4, xmm5; // xmm4 = final wiener fp2r | fp2i
				movups fp2, xmm4;// store fp2


// prev
//				fpr = -outprev2[w][0] + outprev[w][1] + outcur[w][0] - outnext[w][1]; // real prev
//				fpi = -outprev2[w][1] - outprev[w][0] + outcur[w][1] + outnext[w][0]; // im cur
				movaps xmm5, xmm3;  // copy of next
				subps xmm5, xmm1; // next-prev   real | img
				movaps xmm4, xmm1; // copy of prev
				subps xmm4, xmm3; // prev-next  real | img

				shufps xmm5, xmm5, (128 + 0 + 8 + 0); // 10 00 10 00 low // 2 0 2 0 low //low Re(next-prev) | Re2(next-prev) || same second
				shufps xmm4, xmm4, (192 +16 + 12 + 1);// 11 01 11 01 low // 3 1 3 1 low //low Im(prev-next) | Im2(prev-next) || same second
				unpcklps xmm4, xmm5;// low Im(prev-next) | Re(next-prev) || Im2(prev-next) | Re2(next-prev)  

				subps xmm4, xmm0;
				addps xmm4, xmm2; //fp


//				psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density prev2
				movaps xmm5, xmm4; // xmm5= copy   fpr | fpi
				mulps xmm5, xmm4; // xmm5 = r*r | i*i
				movaps xmm6, xmm5; //copy
				shufps xmm6, xmm6, (128 + 48 + 0 + 1);//  10 11 00 01 low - swap re & im
				addps xmm5, xmm6; // xmm1 = sumre*sumre + sumim*sumim
				movss xmm7, smallf; // 1e-15f
				shufps xmm7, xmm7, 0 ;
				addps xmm5, xmm7;// xmm5 =psd prev
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				movss xmm7, sigmaSquaredNoiseNormed;
				shufps xmm7, xmm7, 0 ;

				movaps xmm6, xmm5;
				subps xmm6, xmm7; // psd - sigma
				rcpps xmm5, xmm5; // xmm5= 1/psd
				mulps xmm5, xmm6; // // (psd-sigma)/psd
				movss xmm7, lowlimit;
				shufps xmm7, xmm7, 0 ;
				maxps xmm5, xmm7; // xmm5 =wienerfactor
//				fpr *= WienerFactor; // apply filter on real  part	
//				fpi *= WienerFactor; // apply filter on imaginary part
				mulps xmm4, xmm5; // xmm4 = final wiener fpr | fpi
				movups fp, xmm4;// store fp

// next
//				fnr = -outprev2[w][0] - outprev[w][1] + outcur[w][0] + outnext[w][1]; // real next
//				fni = -outprev2[w][1] + outprev[w][0] + outcur[w][1] - outnext[w][0]; // im next
//				fpr = -outprev2[w][0] + outprev[w][1] + outcur[w][0] - outnext[w][1]; // real prev
//				fpi = -outprev2[w][1] - outprev[w][0] + outcur[w][1] + outnext[w][0]; // im cur
				movaps xmm5, xmm3;  // copy of next
				subps xmm5, xmm1; // next-prev   real | img
				movaps xmm4, xmm1; // copy of prev
				subps xmm4, xmm3; // prev-next  real | img

				shufps xmm5, xmm5, (128 + 0 + 8 + 0); // 10 00 10 00 low // 2 0 2 0 low //low Re(next-prev) | Re2(next-prev) || same second
				shufps xmm4, xmm4, (192 +16 + 12 + 1);// 11 01 11 01 low // 3 1 3 1 low //low Im(prev-next) | Im2(prev-next) || same second
				unpcklps xmm4, xmm5;// low Im(prev-next) | Re(next-prev) || Im2(prev-next) | Re2(next-prev)  

				movaps xmm5, xmm4;
				movaps xmm4, xmm2; //cur
				subps xmm4, xmm0; //cur-prev2
				subps xmm4, xmm5; //fn


//				psd = fnr*fnr + fni*fni + 1e-15f; // power spectrum density next
				movaps xmm5, xmm4; // xmm5= copy   fnr | fni
				mulps xmm5, xmm4; // xmm5 = r*r | i*i
				movaps xmm6, xmm5; //copy
				shufps xmm6, xmm6, (128 + 48 + 0 + 1);//  10 11 00 01 low - swap re & im
				addps xmm5, xmm6; // xmm1 = sumre*sumre + sumim*sumim
				movss xmm7, smallf; // 1e-15f
				shufps xmm7, xmm7, 0 ;
				addps xmm5, xmm7;// xmm5 =psd prev
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				movss xmm7, sigmaSquaredNoiseNormed;
				shufps xmm7, xmm7, 0 ;

				movaps xmm6, xmm5;
				subps xmm6, xmm7; // psd - sigma
				rcpps xmm5, xmm5; // xmm5= 1/psd
				mulps xmm5, xmm6; // // (psd-sigma)/psd
				movss xmm7, lowlimit;
				shufps xmm7, xmm7, 0 ;
				maxps xmm5, xmm7; // xmm5 =wienerfactor
//				fpr *= WienerFactor; // apply filter on real  part	
//				fpi *= WienerFactor; // apply filter on imaginary part
				mulps xmm4, xmm5; // xmm4 = final wiener fpr | fpi
//				movups fn, xmm4;// store fn

				
				// reverse dft for 4 points
//				outprev2[w][0] = (fp2r + fpr + fcr + fnr + gridcorrection0_4)*0.25f ; // get  real  part	
//				outprev2[w][1] = (fp2i + fpi + fci + fni + gridcorrection1_4)*0.25f; // get imaginary part
				movups xmm0, fp2[0];
				addps xmm4, xmm0;
				movups xmm0, fp;
				addps xmm4, xmm0;
				movups xmm0, fc[0];
				addps xmm4, xmm0;

				movups xmm7, gridcorrection;
				addps xmm4, xmm7;

				movss xmm7, onefourth;
				shufps xmm7, xmm7, 0;
				mulps xmm4, xmm7;
				movaps [ebx+eax], xmm4; // write output to prev2 array
				// Attention! return filtered "out" in "outprev2" to preserve "out" for next step

				add eax, 16;
				mov ecx, bytesperblock
				cmp eax, ecx;
				jl nextnumber;
				add edx, ecx; // new block
				add edi, ecx;
				add esi, ecx;
				add ebx, ecx;
				jmp blockend;
finish:			emms;
		}
#endif // !x64
}
/////
//-----------------------------------------------------------------------------------------
//
void ApplyPattern3D4_degrid_SSE(fftwf_complex *outcur, fftwf_complex *outprev2, fftwf_complex *outprev, 
						fftwf_complex *outnext, int outwidth, int outpitch, int bh, 
						int howmanyblocks, float *pattern3d, float beta,
					float degrid, fftwf_complex *gridsample)
{
	// dft 3d (very short - 4 points)
	// optimized for SSE assembler
	// return result in outprev
//	float fcr, fci, fpr, fpi, fnr, fni, fp2r, fp2i;
//	float pnr, pni, di, dr;
//	float WienerFactor =1;
//	float psd;
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0
	float smallf = 1e-15f;
	float onefourth = 0.25f;

	xmmreg fp2; //temporary array for prev2, prev, cur
	xmmreg fp;
	xmmreg fc;
	xmmreg gridcorrection;
	float gridfraction;

//	int block;
//	int h,w;

//	int totalbytes = howmanyblocks*bh*outpitch*8;
	int bytesperblock = bh*outpitch*8;
	int blockscounter = howmanyblocks;

#ifndef X86_64

//	for (line=0; line <totalnumber; line++)
//	{
		__asm
		{
			emms;
			mov esi, outcur; // current
			mov ebx, outprev2; 
			mov edi, outprev;
			mov edx, outnext;

blockend:		mov eax, blockscounter;
				test eax, eax;
				je	finish;
				dec eax
				mov blockscounter, eax;
				
				movaps xmm3, [esi]; // mm3=cur real | img 
				movss xmm7, degrid;
				shufps xmm7, xmm7, 0 ;
				mulps xmm7, xmm3;
				mov ecx, gridsample;
				movss xmm3, [ecx];
				shufps xmm3, xmm3, 0 ;
				rcpps xmm3, xmm3;
				mulps xmm7, xmm3;
				movss gridfraction, xmm7;
				mov eax, 0;
align 16
nextnumber:
				mov ecx, gridsample;
				movaps xmm3, [ecx+eax]; // mm3=grid real | img 
				movss xmm7, gridfraction;
				shufps xmm7, xmm7, 0;
				mulps xmm3, xmm7; // fraction*sample
				movaps xmm7, xmm3; // copy
				addps xmm7, xmm7; // *2
				addps xmm7, xmm7; // *2, gridcorrection 0 | 1
				movups gridcorrection, xmm7;

				movaps xmm0, [ebx+eax]; // xmm0=prev2 real | img
				movaps xmm1, [edi+eax]; // xmm1=prev real | img
				movaps xmm2, [esi+eax]; // xmm2=cur real | img 
				movaps xmm3, [edx+eax]; // xmm3=next real | img 
// cur
//				fcr = outprev2[w][0] + outprev[w][0] + outcur[w][0] + outnext[w][0]; // real cur
//				fci = outprev2[w][1] + outprev[w][1] + outcur[w][1] + outnext[w][1]; // im cur
				movaps xmm4, xmm0; // copy prev2
				addps xmm4, xmm1;
				addps xmm4, xmm2;
				addps xmm4, xmm3;
//				fcr -= gridcorrection0_4;
//				fci -= gridcorrection1_4;
				subps xmm4, xmm7; // fcr

//				psd = fcr*fcr + fci*fci + 1e-15f; // power spectrum density cur
				movaps xmm5, xmm4; // xmm5= copy   fp2r | fp2i
				mulps xmm5, xmm4; // xmm5 = r*r | i*i
				movaps xmm6, xmm5; //copy
				shufps xmm6, xmm6, (128 + 48 + 0 + 1);//  10 11 00 01 low - swap re & im
				addps xmm5, xmm6; // xmm1 = sumre*sumre + sumim*sumim
				movss xmm7, smallf; // 1e-15f
				shufps xmm7, xmm7, 0 ;
				addps xmm5, xmm7;// xmm5 =psd cur
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
//				movss xmm7, sigmaSquaredNoiseNormed;
//				shufps xmm7, xmm7, 0 ;
				mov ecx, pattern3d;
				shr eax, 1;
				movlps xmm7, [ecx+eax];// pattern3d - two values
				shufps xmm7, xmm7, (64 + 16 + 0 + 0) ;// 01 01 00 00 low
				shl eax, 1;

				movaps xmm6, xmm5;
				subps xmm6, xmm7; // psd - sigma
				rcpps xmm5, xmm5; // xmm5= 1/psd
				mulps xmm5, xmm6; // // (psd-sigma)/psd
				movss xmm7, lowlimit;
				shufps xmm7, xmm7, 0 ;
				maxps xmm5, xmm7; // xmm5 =wienerfactor
//				fcr *= WienerFactor; // apply filter on real  part	
//				fci *= WienerFactor; // apply filter on imaginary part
				mulps xmm4, xmm5; // xmm4 = final wiener fcr | fci
				movups fc, xmm4;// store fc

// prev2
//				fp2r = outprev2[w][0] - outprev[w][0] + outcur[w][0] - outnext[w][0]; // real prev2
//				fp2i = outprev2[w][1] - outprev[w][1] + outcur[w][1] - outnext[w][1]; // im cur
				movaps xmm4, xmm0; // copy prev2
				subps xmm4, xmm1;
				addps xmm4, xmm2;
				subps xmm4, xmm3;

//				psd = fp2r*fp2r + fp2i*fp2i + 1e-15f; // power spectrum density prev2
				movaps xmm5, xmm4; // xmm5= copy   fp2r | fp2i
				mulps xmm5, xmm4; // xmm5 = r*r | i*i
				movaps xmm6, xmm5; //copy
				shufps xmm6, xmm6, (128 + 48 + 0 + 1);//  10 11 00 01 low - swap re & im
				addps xmm5, xmm6; // xmm1 = sumre*sumre + sumim*sumim
				movss xmm7, smallf; // 1e-15f
				shufps xmm7, xmm7, 0 ;
				addps xmm5, xmm7;// xmm5 =psd prev2
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
//				movss xmm7, sigmaSquaredNoiseNormed;
//				shufps xmm7, xmm7, 0 ;
//				mov ecx, pattern3d;
				shr eax, 1;
				movlps xmm7, [ecx+eax];// pattern3d - two values
				shufps xmm7, xmm7, (64 + 16 + 0 + 0) ;// 01 01 00 00 low
				shl eax, 1;

				movaps xmm6, xmm5;
				subps xmm6, xmm7; // psd - sigma
				rcpps xmm5, xmm5; // xmm5= 1/psd
				mulps xmm5, xmm6; // // (psd-sigma)/psd
				movss xmm7, lowlimit;
				shufps xmm7, xmm7, 0 ;
				maxps xmm5, xmm7; // xmm5 =wienerfactor
//				fp2r *= WienerFactor; // apply filter on real  part	
//				fp2i *= WienerFactor; // apply filter on imaginary part
				mulps xmm4, xmm5; // xmm4 = final wiener fp2r | fp2i
				movups fp2, xmm4;// store fp2


// prev
//				fpr = -outprev2[w][0] + outprev[w][1] + outcur[w][0] - outnext[w][1]; // real prev
//				fpi = -outprev2[w][1] - outprev[w][0] + outcur[w][1] + outnext[w][0]; // im cur
				movaps xmm5, xmm3;  // copy of next
				subps xmm5, xmm1; // next-prev   real | img
				movaps xmm4, xmm1; // copy of prev
				subps xmm4, xmm3; // prev-next  real | img

				shufps xmm5, xmm5, (128 + 0 + 8 + 0); // 10 00 10 00 low // 2 0 2 0 low //low Re(next-prev) | Re2(next-prev) || same second
				shufps xmm4, xmm4, (192 +16 + 12 + 1);// 11 01 11 01 low // 3 1 3 1 low //low Im(prev-next) | Im2(prev-next) || same second
				unpcklps xmm4, xmm5;// low Im(prev-next) | Re(next-prev) || Im2(prev-next) | Re2(next-prev)  

				subps xmm4, xmm0;
				addps xmm4, xmm2; //fp


//				psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density prev2
				movaps xmm5, xmm4; // xmm5= copy   fpr | fpi
				mulps xmm5, xmm4; // xmm5 = r*r | i*i
				movaps xmm6, xmm5; //copy
				shufps xmm6, xmm6, (128 + 48 + 0 + 1);//  10 11 00 01 low - swap re & im
				addps xmm5, xmm6; // xmm1 = sumre*sumre + sumim*sumim
				movss xmm7, smallf; // 1e-15f
				shufps xmm7, xmm7, 0 ;
				addps xmm5, xmm7;// xmm5 =psd prev
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
//				movss xmm7, sigmaSquaredNoiseNormed;
//				shufps xmm7, xmm7, 0 ;
//				mov ecx, pattern3d;
				shr eax, 1;
				movlps xmm7, [ecx+eax];// pattern3d - two values
				shufps xmm7, xmm7, (64 + 16 + 0 + 0) ;// 01 01 00 00 low
				shl eax, 1;

				movaps xmm6, xmm5;
				subps xmm6, xmm7; // psd - sigma
				rcpps xmm5, xmm5; // xmm5= 1/psd
				mulps xmm5, xmm6; // // (psd-sigma)/psd
				movss xmm7, lowlimit;
				shufps xmm7, xmm7, 0 ;
				maxps xmm5, xmm7; // xmm5 =wienerfactor
//				fpr *= WienerFactor; // apply filter on real  part	
//				fpi *= WienerFactor; // apply filter on imaginary part
				mulps xmm4, xmm5; // xmm4 = final wiener fpr | fpi
				movups fp, xmm4;// store fp

// next
//				fnr = -outprev2[w][0] - outprev[w][1] + outcur[w][0] + outnext[w][1]; // real next
//				fni = -outprev2[w][1] + outprev[w][0] + outcur[w][1] - outnext[w][0]; // im next
//				fpr = -outprev2[w][0] + outprev[w][1] + outcur[w][0] - outnext[w][1]; // real prev
//				fpi = -outprev2[w][1] - outprev[w][0] + outcur[w][1] + outnext[w][0]; // im cur
				movaps xmm5, xmm3;  // copy of next
				subps xmm5, xmm1; // next-prev   real | img
				movaps xmm4, xmm1; // copy of prev
				subps xmm4, xmm3; // prev-next  real | img

				shufps xmm5, xmm5, (128 + 0 + 8 + 0); // 10 00 10 00 low // 2 0 2 0 low //low Re(next-prev) | Re2(next-prev) || same second
				shufps xmm4, xmm4, (192 +16 + 12 + 1);// 11 01 11 01 low // 3 1 3 1 low //low Im(prev-next) | Im2(prev-next) || same second
				unpcklps xmm4, xmm5;// low Im(prev-next) | Re(next-prev) || Im2(prev-next) | Re2(next-prev)  

				movaps xmm5, xmm4;
				movaps xmm4, xmm2; //cur
				subps xmm4, xmm0; //cur-prev2
				subps xmm4, xmm5; //fn


//				psd = fnr*fnr + fni*fni + 1e-15f; // power spectrum density next
				movaps xmm5, xmm4; // xmm5= copy   fnr | fni
				mulps xmm5, xmm4; // xmm5 = r*r | i*i
				movaps xmm6, xmm5; //copy
				shufps xmm6, xmm6, (128 + 48 + 0 + 1);//  10 11 00 01 low - swap re & im
				addps xmm5, xmm6; // xmm1 = sumre*sumre + sumim*sumim
				movss xmm7, smallf; // 1e-15f
				shufps xmm7, xmm7, 0 ;
				addps xmm5, xmm7;// xmm5 =psd prev
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
//				movss xmm7, sigmaSquaredNoiseNormed;
//				shufps xmm7, xmm7, 0 ;
//				mov ecx, pattern3d;
				shr eax, 1;
				movlps xmm7, [ecx+eax];// pattern3d - two values
				shufps xmm7, xmm7, (64 + 16 + 0 + 0) ;// 01 01 00 00 low
				shl eax, 1;

				movaps xmm6, xmm5;
				subps xmm6, xmm7; // psd - sigma
				rcpps xmm5, xmm5; // xmm5= 1/psd
				mulps xmm5, xmm6; // // (psd-sigma)/psd
				movss xmm7, lowlimit;
				shufps xmm7, xmm7, 0 ;
				maxps xmm5, xmm7; // xmm5 =wienerfactor
//				fpr *= WienerFactor; // apply filter on real  part	
//				fpi *= WienerFactor; // apply filter on imaginary part
				mulps xmm4, xmm5; // xmm4 = final wiener fpr | fpi
//				movups fn, xmm4;// store fn

				
				// reverse dft for 4 points
//				outprev2[w][0] = (fp2r + fpr + fcr + fnr + gridcorrection0_4)*0.25f ; // get  real  part	
//				outprev2[w][1] = (fp2i + fpi + fci + fni + gridcorrection1_4)*0.25f; // get imaginary part
				movups xmm0, fp2[0];
				addps xmm4, xmm0;
				movups xmm0, fp;
				addps xmm4, xmm0;
				movups xmm0, fc[0];
				addps xmm4, xmm0;

				movups xmm7, gridcorrection;
				addps xmm4, xmm7;

				movss xmm7, onefourth;
				shufps xmm7, xmm7, 0;
				mulps xmm4, xmm7;
				movaps [ebx+eax], xmm4; // write output to prev2 array
				// Attention! return filtered "out" in "outprev2" to preserve "out" for next step

				add eax, 16;
				mov ecx, bytesperblock
				cmp eax, ecx;
				jl nextnumber;
				add edx, ecx; // new block
				add edi, ecx;
				add esi, ecx;
				add ebx, ecx;
				jmp blockend;
finish:			emms;
		}
#endif
}
