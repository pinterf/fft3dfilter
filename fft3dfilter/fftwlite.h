// Lite version of fftw header on base of fftw3.h
// some needed fftwf typedefs added  for delayed loading
// (by Fizick)
// 
#ifndef __FFTWLITE_H__
#define __FFTWLITE_H__
typedef float fftwf_complex[2];
typedef struct fftwf_plan_s  *fftwf_plan;
typedef fftwf_complex* (*fftwf_malloc_proc)(size_t n); 
typedef VOID (*fftwf_free_proc) (void *ppp);
typedef fftwf_plan (*fftwf_plan_dft_r2c_2d_proc) (int winy, int winx, float *realcorrel, fftwf_complex *correl, int flags);
typedef fftwf_plan (*fftwf_plan_dft_c2r_2d_proc) (int winy, int winx, fftwf_complex *correl, float *realcorrel, int flags);
typedef fftwf_plan (*fftwf_plan_many_dft_r2c_proc) (int rank, const int *n,	int howmany,  float *in, const int *inembed, int istride, int idist, fftwf_complex *out, const int *onembed, int ostride, int odist, unsigned flags);
typedef fftwf_plan (*fftwf_plan_many_dft_c2r_proc) (int rank, const int *n,	int howmany,  fftwf_complex *out, const int *inembed, int istride, int idist, float *in, const int *onembed, int ostride, int odist, unsigned flags);
typedef void (*fftwf_destroy_plan_proc) (fftwf_plan);
typedef void (*fftwf_execute_dft_r2c_proc) (fftwf_plan, float *realdata, fftwf_complex *fftsrc);
typedef void (*fftwf_execute_dft_c2r_proc) (fftwf_plan, fftwf_complex *fftsrc, float *realdata);
#define FFTW_MEASURE (0U)
#define FFTW_ESTIMATE (1U << 6)
typedef int (*fftwf_init_threads_proc) ();
typedef void (*fftwf_plan_with_nthreads_proc)(int nthreads);

#endif