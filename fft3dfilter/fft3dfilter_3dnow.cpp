//
//	FFT3DFilter plugin for Avisynth 2.5 - 3D Frequency Domain filter
//  AMD Athlon 3DNow! filtering functions
//
//	Copyright(C)2004-2005 A.G.Balakhnin aka Fizick, bag@hotmail.ru, http://bag.hotmail.ru
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

#include "windows.h"
#include "fftwlite.h"

#ifdef WITH3DNOW
//
void ApplyWiener3D2_3DNow(fftwf_complex *outcur, fftwf_complex *outprev, int outwidth, int outpitch, int bh, int howmanyblocks, float sigmaSquaredNoiseNormed, float beta)
{
	//  optimized for AMD Athlon 3DNOW assembler
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
			movd mm7, smallf;
			punpckldq mm7, mm7 ; // mm7 = smallf
			movd mm6, sigmaSquaredNoiseNormed;
			punpckldq mm6, mm6 ; // mm6 = sigmaSquaredNoiseNormed
			movd mm5, lowlimit;
			punpckldq mm5, mm5; // mm5 =lowlimit
			movd mm4, onehalf;
			punpckldq mm4, mm4; // mm4 =onehalf
			mov eax, 0;
align 16
nextnumber:
				movq mm0, [edi+eax]; // mm0=prev real | img
				movq mm1, [esi+eax]; // mm1=cur real | img 
				movq mm2, mm0;
//				f3d0r =  outcur[w][0] + outprev[w][0]; // real 0 (sum)
//				f3d0i =  outcur[w][1] + outprev[w][1]; // im 0 (sum)
				pfadd mm2, mm1; // mm2 =sum

//				f3d1r =  outcur[w][0] - outprev[w][0]; // real 1 (dif)
//				f3d1i =  outcur[w][1] - outprev[w][1]; // im 1 (dif)
				pfsubr mm0, mm1; // mm0= dif  mm1-mm0 = cur-prev

				movq mm1, mm2; // copy sum
				pfmul mm1, mm1; // mm1 =sumre*sumre | sumin*sumim
				movq mm3, mm1; //copy
				pswapd mm3, mm3; // swap re & im
				pfadd mm1, mm3; // mm1 = sumre*sumre + sumim*sumim
//				psd = f3d0r*f3d0r + f3d0i*f3d0i + 1e-15f; // power spectrum density 0
				pfadd mm1, mm7; // mm1 = psd of sum = sumre*sumre + sumim*sumim + smallf
				
				movq mm3, mm1; // mm3 =copy psd
				pfsub mm3, mm6; // mm3= psd - sigma
				pfrcp mm1, mm1; // mm1= 1/psd // bug fixed in v.0.9.3
				pfmul mm3, mm1; //  (psd-sigma)/psd
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				pfmax mm3, mm5; // mm3 =wienerfactor
//				f3d0r *= WienerFactor; // apply filter on real  part	
//				f3d0i *= WienerFactor; // apply filter on imaginary part
				pfmul mm2, mm3; // mm2 = final wiener sum f3d0



				movq mm1, mm0; // copy dif
				pfmul mm1, mm1; // mm1 = difre*difre | difim*difim
				movq mm3, mm1; // copy
				pswapd mm3, mm3;
				pfadd mm1, mm3;
//				psd = f3d1r*f3d1r + f3d1i*f3d1i + 1e-15f; // power spectrum density 1
				pfadd mm1, mm7; // mm3 = psd of dif

				movq mm3, mm1; //copy of psd
				pfsub mm3, mm6; // mm3= psd - sigma
				pfrcp mm1, mm1; // mm1= 1/psd // bug fixed in v.0.9.3
				pfmul mm3, mm1; //  (psd-sigma)/psd
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
				pfmax mm3, mm5; // mm3 =wienerfactor
//				f3d1r *= WienerFactor; // apply filter on real  part	
//				f3d1i *= WienerFactor; // apply filter on imaginary part
				pfmul mm0, mm3; //mm0 = fimal wiener dif f3d1

				// reverse dft for 2 points
				pfadd mm2, mm0; // filterd sum + dif
//				outprev[w][0] = (f3d0r + f3d1r)*0.5f; // get  real  part	
//				outprev[w][1] = (f3d0i + f3d1i)*0.5f; // get imaginary part
				pfmul mm2, mm4; // filterd (sum+dif)*0.5
				movq [edi+eax], mm2;
				// Attention! return filtered "outcur" in "outprev" to preserve "outcur" for next step
			
//			outcur += outpitch;
//			outprev += outpitch;
//		}
				add eax, 8;
				cmp eax, ecx;
				jge finish;
				jmp nextnumber;
finish:			emms;
		}
}
//-----------------------------------------------------------------------------------------
//
void ApplyWiener3D3_3DNow(fftwf_complex *outcur, fftwf_complex *outprev, 
						  fftwf_complex *outnext, int outwidth, int outpitch, int bh, 
						  int howmanyblocks, float sigmaSquaredNoiseNormed, float beta)
{
	// dft 3d (very short - 3 points)
	// optimized for AMD Athlon 3DNOW assembler
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

	int totalbytes = howmanyblocks*bh*outpitch*8;

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
				movq mm0, [edi+eax]; // mm0=prev real | img
				movq mm1, [edx+eax]; // mm1=next real | img
				movq mm3, [esi+eax]; // mm3=cur real | img 

				movq mm2, mm0; // copy prev
				//pnr = outprev[w][0] + outnext[w][0];
				//pni = outprev[w][1] + outnext[w][1];
				pfadd mm2, mm1; // mm2= summa prev and next  , pnr | pni

				//fcr = outcur[w][0] + pnr; // real cur
				//fci = outcur[w][1] + pni; // im cur
				movq mm4, mm3 ;// copy of cur
				pfadd mm4, mm2; // mm4= fcr | fci

				movd mm7, onehalf; // 0.5
				punpckldq mm7, mm7 ; // mm7 = 0.5 | 0.5
				pfmul mm2, mm7; // mm2 = 0.5*pnr | 0.5*pni


				movq mm5, mm4; // mm5= copy   fcr | fci
				pfmul mm5, mm4; // mm5 = fcr*fcr | fci*fci
				pfacc mm5, mm5; // mm5 = fcr*fcr+fci*fci
				movd mm7, smallf; // 1e-15f
				punpckldq mm7, mm7;
//				psd = fcr*fcr + fci*fci + 1e-15f; // power spectrum density cur
				pfadd mm5, mm7;// mm5 =psd cur
				movd mm7, sigmaSquaredNoiseNormed;
				punpckldq mm7, mm7;
				pfsubr mm7, mm5; // psd - sigma
				pfrcp mm5, mm5; // mm5= 1/psd
				pfmul mm5, mm7; // // (psd-sigma)/psd
				movd mm7, lowlimit;
				punpckldq mm7, mm7;
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
//				fcr *= WienerFactor; // apply filter on real  part	
//				fci *= WienerFactor; // apply filter on imaginary part
				pfmax mm5, mm7; // mm5 =wienerfactor
				pfmul mm4, mm5; // mm4 = final wiener fcr | fci
				// mm5, mm6, mm7 are free
				
				
				movq mm5, mm1;  // copy of next
				pfsub mm5, mm0; // next-prev   real | img
				movq mm6, mm0; // copy of prev
				pfsub mm6, mm1; // prev-next  real | img
				pswapd mm6, mm6; // swap real and img,    img | real
				punpckldq mm5, mm6;  // use low dwords:   real next-prev | img prev-next
				movd mm7, sin120;
				punpckldq mm7, mm7 ; // mm7 = sin120 | sin120
//				di = sin120*(outprev[w][1]-outnext[w][1]);
//				dr = sin120*(outnext[w][0]-outprev[w][0]);
				pfmul mm5, mm7; // mm5= dr | di
				pswapd mm5, mm5 ; // mm5 = di | dr

				//  mm1, mm6 are free
				pfsub mm3, mm2; // mm3=cur-0.5*pn
				movq mm1, mm3; // copy 
//				fpr = out[w][0] - 0.5f*pnr + di; // real prev
//				fpi = out[w][1] - 0.5f*pni + dr; // im prev
				pfadd mm3, mm5;// mm3 = fpr | fpi
				
//				fnr = out[w][0] - 0.5f*pnr - di; // real next
//				fni = out[w][1] - 0.5f*pni - dr ; // im next
				pfsub mm1, mm5;// mm1 = fnr | fni
				// mm2, mm5, mm6, mm7 are free


				movq mm5, mm3; // mm5= copy   fpr | fpi
				pfmul mm5, mm3; // mm5 = fpr*fpr | fpi*fpi
				pfacc mm5, mm5; // mm5 = fpr*fpr+fpi*fpi
				movd mm7, smallf; // 1e-15f
				punpckldq mm7, mm7;
//				psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density cur
				pfadd mm5, mm7;// mm5 =psd cur
				movd mm7, sigmaSquaredNoiseNormed;
				punpckldq mm7, mm7;
				pfsubr mm7, mm5; // psd - sigma
				pfrcp mm5, mm5; // mm5= 1/psd
				pfmul mm5, mm7; // // (psd-sigma)/psd
				movd mm7, lowlimit;
				punpckldq mm7, mm7;
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
//				fpr *= WienerFactor; // apply filter on real  part	
//				fpi *= WienerFactor; // apply filter on imaginary part
				pfmax mm5, mm7; // mm5 =wienerfactor
				pfmul mm3, mm5; // mm3 = final wiener fpr | fpi
				// mm2, mm5, mm6, mm7 are free
				


				movq mm5, mm1; // mm5= copy   fnr | fni
				pfmul mm5, mm1; // mm5 = fnr*fnr | fni*fni
				pfacc mm5, mm5; // mm5 = fnr*fnr+fni*fni
				movd mm7, smallf; // 1e-15f
				punpckldq mm7, mm7;
//				psd = fnr*fnr + fni*fni + 1e-15f; // power spectrum density cur
				pfadd mm5, mm7;// mm5 =psd cur
				movd mm7, sigmaSquaredNoiseNormed;
				punpckldq mm7, mm7;
				pfsubr mm7, mm5; // psd - sigma
				pfrcp mm5, mm5; // mm5= 1/psd
				pfmul mm5, mm7; // // (psd-sigma)/psd
				movd mm7, lowlimit;
				punpckldq mm7, mm7;
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
//				fnr *= WienerFactor; // apply filter on real  part	
//				fni *= WienerFactor; // apply filter on imaginary part
				pfmax mm5, mm7; // mm5 =wienerfactor
				pfmul mm1, mm5; // mm1 = final wiener fmr | fmi
				// mm2, mm5, mm6, mm7 are free
				
				// reverse dft for 3 points
				pfadd mm4, mm3; // fc + fp
				pfadd mm4, mm1; // fc + fp + fn
				movd mm7, onethird;
				punpckldq mm7, mm7;
				pfmul mm4, mm7;
//				outprev[w][0] = (fcr + fpr + fnr)*0.33333333333f; // get  real  part	
//				outprev[w][1] = (fci + fpi + fni)*0.33333333333f; // get imaginary part
				movq [edi+eax], mm4; // write output to prev array
				// Attention! return filtered "out" in "outprev" to preserve "out" for next step

				add eax, 8;
				cmp eax, ecx;
				jge finish;
				jmp nextnumber;
finish:			emms;
		}
}

//-----------------------------------------------------------------------------------------
//
void ApplyWiener3D4_3DNow(fftwf_complex *outcur, fftwf_complex *outprev2, 
						  fftwf_complex *outprev, fftwf_complex *outnext, 
						  int outwidth, int outpitch, int bh, int howmanyblocks, 
						  float sigmaSquaredNoiseNormed, float beta)
{
	// dft 3d (very short - 3 points)
	// optimized for AMD Athlon 3DNOW assembler
	// return result in outprev
//	float fcr, fci, fpr, fpi, fnr, fni;
//	float pnr, pni, di, dr;
//	float WienerFactor =1;
//	float psd;
	float lowlimit = (beta-1)/beta; //     (beta-1)/beta>=0
	float smallf = 1e-15f;
	float onefourth = 0.25f;

//	int block;
//	int h,w;

	int totalbytes = howmanyblocks*bh*outpitch*8;

//	for (line=0; line <totalnumber; line++)
//	{
		__asm
		{
			emms;
			mov edi, outprev2;
			mov ebx, outprev;
			mov edx, outnext;
			mov esi, outcur; // current
			mov ecx, totalbytes; // counter
			mov eax, 0;
align 16
nextnumber:
				movq mm0, [edi+eax]; // mm0=prev2 real | img
				movq mm1, [ebx+eax]; // mm1=prev real | img
				movq mm2, [esi+eax]; // mm2=cur real | img
				movq mm3, [edx+eax]; // mm3=next real | img 

				// dft 3d (very short - 4 points)

				// get useful combination
				movq mm4, mm1; //  prev
				pfsub mm4, mm3; // Re (prev-next) | Im(prev-next)
				movq mm5, mm3; // next
				pfsub mm5, mm1; // Re(next-prev) | Im(next-prev)
				pswapd mm5, mm5; // Im(next-prev)  | Re(next-prev)
				punpckldq mm5, mm4;  // use low dwords:   Im(next-prev) |  Re(prev-next)

//				fpr = -outprev2[w][0] + outprev[w][1] + out[w][0] - outnext[w][1]; // real prev
//				fpi = -outprev2[w][1] - outprev[w][0] + out[w][1] + outnext[w][0]; // im cur
				movq mm4, mm2; // cur
				pfsub mm4, mm0; // cur-prev2
				pfsub mm4, mm5; // fp = Re(cur-prev2) - Im(next-prev) | Im(cur-prev) - Re(prev-next)

//				fnr = -outprev2[w][0] - outprev[w][1] + out[w][0] + outnext[w][1]; // real next
//				fni = -outprev2[w][1] + outprev[w][0] + out[w][1] - outnext[w][0]; // im next
				movq mm6, mm2; // cur
				pfsub mm6, mm0; // cur-prev2
				pfadd mm6, mm5; // fn = Re(cur-prev2) + Im(next-prev) | Im(cur-prev) + Re(prev-next)
				// free mm5, mm7

				movq mm5, mm4; // mm5= copy   fpr | fpi
				pfmul mm5, mm5; // mm5 = fpr*fpr | fpi*fpi
				pfacc mm5, mm5; // mm5 = fpr*fpr + fpi*fpi
				movd mm7, smallf; // 1e-15f
				punpckldq mm7, mm7;
//				psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density prev
				pfadd mm5, mm7;// mm5 =psd prev
				movd mm7, sigmaSquaredNoiseNormed;
				punpckldq mm7, mm7;
				pfsubr mm7, mm5; // psd - sigma
				pfrcp mm5, mm5; // mm5= 1/psd
				pfmul mm5, mm7; // // (psd-sigma)/psd
				movd mm7, lowlimit;
				punpckldq mm7, mm7;
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
//				fpr *= WienerFactor; // apply filter on real  part	
//				fpi *= WienerFactor; // apply filter on imaginary part
				pfmax mm5, mm7; // mm5 =wienerfactor
				pfmul mm4, mm5; // mm4 = final wiener fpr | fpi
				// mm5, mm7 are free

				movq mm5, mm6; // mm5= copy   fnr | fni
				pfmul mm5, mm5; // mm5 = fnr*fnr | fni*fni
				pfacc mm5, mm5; // mm5 = fnr*fnr+fni*fni
				movd mm7, smallf; // 1e-15f
				punpckldq mm7, mm7;
//				psd = fnr*fnr + fni*fni + 1e-15f; // power spectrum density cur
				pfadd mm5, mm7;// mm5 =psd cur
				movd mm7, sigmaSquaredNoiseNormed;
				punpckldq mm7, mm7;
				pfsubr mm7, mm5; // psd - sigma
				pfrcp mm5, mm5; // mm5= 1/psd
				pfmul mm5, mm7; // // (psd-sigma)/psd
				movd mm7, lowlimit;
				punpckldq mm7, mm7;
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
//				fnr *= WienerFactor; // apply filter on real  part	
//				fni *= WienerFactor; // apply filter on imaginary part
				pfmax mm5, mm7; // mm5 =wienerfactor
				pfmul mm6, mm5; // mm6 = final wiener fnr | fni

				pfadd mm4, mm6; // mm4 = fp + fn final
				// mm5, mm6, mm7 are free
				
//				fcr = outprev2[w][0] + outprev[w][0] + out[w][0] + outnext[w][0]; // real cur
//				fci = outprev2[w][1] + outprev[w][1] + out[w][1] + outnext[w][1]; // im cur
				movq mm6, mm2; // cur
				pfadd mm6, mm0; // cur+prev2
				pfadd mm6, mm1; // cur+prev2+prev
				pfadd mm6, mm3; // fc = cur+prev2+prev+next
				
				movq mm5, mm6; // mm5= copy   fcr | fci
				pfmul mm5, mm5; // mm5 = fcr*fcr | fci*fci
				pfacc mm5, mm5; // mm5 = fcr*fcr + fci*fci
				movd mm7, smallf; // 1e-15f
				punpckldq mm7, mm7;
//				psd = fpr*fpr + fpi*fpi + 1e-15f; // power spectrum density cur
				pfadd mm5, mm7;// mm5 =psd cur
				movd mm7, sigmaSquaredNoiseNormed;
				punpckldq mm7, mm7;
				pfsubr mm7, mm5; // psd - sigma
				pfrcp mm5, mm5; // mm5= 1/psd
				pfmul mm5, mm7; // // (psd-sigma)/psd
				movd mm7, lowlimit;
				punpckldq mm7, mm7;
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
//				fcr *= WienerFactor; // apply filter on real  part	
//				fci *= WienerFactor; // apply filter on imaginary part
				pfmax mm5, mm7; // mm5 =wienerfactor
				pfmul mm6, mm5; // mm3 = final wiener fcr | fci

				pfadd mm4, mm6; // mm4 = fp + fn + fc
				// mm2, mm5, mm6, mm7 are free
				
//				fp2r = outprev2[w][0] - outprev[w][0] + out[w][0] - outnext[w][0]; // real prev2
//				fp2i = outprev2[w][1] - outprev[w][1] + out[w][1] - outnext[w][1]; // im cur
				movq mm6, mm2; // cur
				pfadd mm6, mm0; // cur+prev2
				pfsub mm6, mm1; // cur+prev2-prev
				pfsub mm6, mm3; // fp2 = cur+prev2-prev-next
				
				movq mm5, mm6; // mm5= copy   fr | fi
				pfmul mm5, mm5; // mm5 = fr*fr | fi*fi
				pfacc mm5, mm5; // mm5 = fr*fr + fi*fi
				movd mm7, smallf; // 1e-15f
				punpckldq mm7, mm7;
//				psd = fp2r*fp2r + fp2i*fp2i + 1e-15f; // power spectrum density cur
				pfadd mm5, mm7;// mm5 =psd cur
				movd mm7, sigmaSquaredNoiseNormed;
				punpckldq mm7, mm7;
				pfsubr mm7, mm5; // psd - sigma
				pfrcp mm5, mm5; // mm5= 1/psd
				pfmul mm5, mm7; // // (psd-sigma)/psd
				movd mm7, lowlimit;
				punpckldq mm7, mm7;
//				WienerFactor = max((psd - sigmaSquaredNoiseNormed)/psd, lowlimit); // limited Wiener filter
//				fp2r *= WienerFactor; // apply filter on real  part	
//				fp2i *= WienerFactor; // apply filter on imaginary part
				pfmax mm5, mm7; // mm5 =wienerfactor
				pfmul mm6, mm5; // mm6 = final wiener fp2r | fp2i

				pfadd mm4, mm6; // mm4 = fp + fn + fc + fp2 final
				// mm5, mm6, mm7 are free
				
				// reverse dft for 4 points
				movd mm7, onefourth;
				punpckldq mm7, mm7;
				pfmul mm4, mm7;
//				outprev2[w][0] = (fcr + fpr + fnr + fp2r)*0.25f; // get  real  part	
//				outprev2[w][1] = (fci + fpi + fni + fp2i)*0.25f; // get imaginary part
				movq [edi+eax], mm4; // write output to prev array
				// Attention! return filtered "out" in "outprev2" to preserve "out" for next step

				add eax, 8;
				cmp eax, ecx;
				jge finish;
				jmp nextnumber;
finish:			emms;
		}
}
//
//-----------------------------------------------------------------------------------------
//void FFT3DFilter::
void ApplyKalman_3DNow( fftwf_complex *outcur, fftwf_complex *outLast, 
					   fftwf_complex *covar, fftwf_complex *covarProcess, 
					   int outwidth, int outpitch, int bh, int howmanyblocks,  
					   float covarNoiseNormed, float kratio2)
{
// return result in outLast
//	float GainRe, GainIm;  // Kalman Gain 
//	float sumre, sumim;

//	int block;
//	int h,w;

	unsigned int totalbytes = howmanyblocks*bh*outpitch*8;


//	for (block=0; block <howmanyblocks; block++)
//	{
//		for (h=0; h<bh; h++) // 
//		{
//			for (w=0; w<outwidth; w++) 
//			{
		__asm
		{
			emms;
			mov esi, outcur; // current
			mov edi, outLast;
			mov edx, covar;
			mov ebx, covarProcess;
			mov ecx, totalbytes;// counter,
			movd mm6, covarNoiseNormed;
			punpckldq mm6, mm6; // covarNoiseNormed
			movd mm7, kratio2;
			punpckldq mm7, mm7; // kratio2 
			pfmul mm7, mm6; // mm7=sigmaSquaredMotionNormed
			mov eax, 0;
			push ebp; // will use register for cmp
align 16
nextnumber:

				movq mm0, [esi+eax]; // mm0=cur real | img
				// use one of possible method for motion detection:
				//if ( (outcur[w][0]-outLast[w][0])*(outcur[w][0]-outLast[w][0]) > sigmaSquaredMotionNormed ||
				//     (outcur[w][1]-outLast[w][1])*(outcur[w][1]-outLast[w][1]) > sigmaSquaredMotionNormed )
				movq mm2, mm0; // copy cur
				movq mm1, [edi+eax]; // mm1=last real | img 
				pfsub mm2, mm1; // mm2 = cur-last

				pfmul mm2, mm2; // mm2 = (cur-last)*(cur-last) // - fixed bug in v1.5.2

				//movq mm3, mm1; // copy last
				//pfsub mm3, mm0; // mm3 = last-cur
				//pfmax mm2, mm3 ; // max = fabs(cur-last)

				pfcmpgt mm2, mm7; //  > motion?
				movq mm4, mm2; // copy
				pswapd mm2, mm4; // swap re - img comparison
				por mm2, mm4; // any of both

				pextrw ebp, mm2, 0;
				cmp ebp, 0; 
				je nomotion; // filter only if no motion


align 16
				//bad
				// big pixel variation due to motion etc
				// reset filter
				//	outLast[w][0] = outcur[w][0];
				//	outLast[w][1] = outcur[w][1];
				movq [edi+eax], mm0; //return result in outLast
				//	covar[w][0] = covarNoiseNormed; 
				//	covar[w][1] = covarNoiseNormed; 
				movq [edx+eax], mm6;
				//	covarProcess[w][0] = covarNoiseNormed; 
				//	covarProcess[w][1] = covarNoiseNormed; 
				movq [ebx+eax], mm6
				jmp nextnumber;

align 16
nomotion:
				// small variation
				movq mm2, [edx+eax]; // covar
				movq mm4, mm2;
				movq mm3, [ebx+eax]; // covarProcess
					// useful sum
				//	sumre = (covar[w][0] + covarProcess[w][0]);
				//	sumim = (covar[w][1] + covarProcess[w][1]);
				pfadd mm4, mm3;  // mm4 = sum = (covar +covarProcess)

					// real gain, imagine gain
				movq mm5, mm4;
				pfadd mm5, mm6; // sum + covarnoise
				pswapd mm3, mm5;// swap re and img
				//	GainRe = sumre/(sumre + covarNoiseNormed);
				pfrcp mm5, mm5; //  1/(sumRe + covarnoise) in both halfs
				//	GainIm = sumim/(sumim + covarNoiseNormed);
				pfrcp mm3, mm3; //  1/(sumIm + covarnoise) in both halfs
				// combine re im
				punpckldq mm5, mm3; 
				pfmul mm5, mm4; // mm5 = gain = 1/(sum + covarnoise) * sum

					// update process
				movq mm3, mm5; // copy gain
				pfmul mm3, mm3; // gain*gain
				pfmul mm3, mm6; // mm3 = gain*gain*covarNoiseNormed
				//	covarProcess[w][0] = (GainRe*GainRe*covarNoiseNormed);
				//	covarProcess[w][1] = (GainIm*GainIm*covarNoiseNormed);
				movq [ebx+eax], mm3;

					// update variation
				movq mm3, mm5; // copy gain
				pfmul mm3, mm4; // gain*sum
				pfsubr mm3, mm4; // sum - gain*sum
				//	covar[w][0] =  (1-GainRe)*sumre ; = sumre - GainRe*sumre
				//	covar[w][1] =  (1-GainIm)*sumim ;
				movq [edx+eax], mm3;

				// make output
//				movq mm0, [esi+eax]; // cur
//				movq mm1, [edi+eax]; // last
				pfsub mm0, mm1; // cur-last
				pfmul mm0, mm5; // (cur-last)*gain
				pfadd mm0, mm1; // (cur-last)*gain + last
				movq [edi+eax], mm0;
				//	outLast[w][0] = ( GainRe*outcur[w][0] + (1 - GainRe)*outLast[w][0] ); = GainRe*(outcur[w][0]-outLast[w][0]) + outLast[w][0]
				//	outLast[w][1] = ( GainIm*outcur[w][1] + (1 - GainIm)*outLast[w][1] );
					//return filtered result in outLast
				
			
//			outcur += outpitch;
//			outLast += outpitch; 
//			covar += outpitch; 
//			covarProcess += outpitch; 
				add eax, 8;
				cmp eax, ecx;
				jge finish;
				jmp nextnumber;

finish:			emms;
				pop ebp; //restore
		}
}
#endif
