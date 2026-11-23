#ifndef NANCM_WORD_FIELD_148_API_H
#define NANCM_WORD_FIELD_148_API_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Opaque native world-line state; loads a strictly paired cloud fingerprint. */
void* wf148_open(const char*model,const char*geom,const char*cloud);
void wf148_close(void*state);
/* Query only bytes strictly before the next word; no future bytes. */
int wf148_begin(void*state,const unsigned char*past,size_t n,float cloud_strength);
/* Probability over a..z, EOS (26); return residual open-vocabulary mass. */
int wf148_next(void*state,const char*observed_prefix,size_t n,float out27[27],float*escape_mass);
/* Parent 256-way distribution + field projection. protect_top5 keeps the
   PARENT top-five byte indices unchanged, not just the aggregate hit rate. */
int wf148_fuse(const float parent256[256],const float word27[27],float escape_mass,
               float strength,int protect_top5,float fused256[256]);
#ifdef __cplusplus
}
#endif
#endif
/* v148 surgical mode: minimum-KL top-1 challenger inside parent's top-5. */
#ifdef __cplusplus
extern "C" {
#endif
int wf148_fuse_surgical(const float parent256[256],const float word27[27],float escape,
                        float max_relative_gap,float fused256[256]);
#ifdef __cplusplus
}
#endif
