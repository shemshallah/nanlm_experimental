#!/usr/bin/env python3
"""NaNCM -- single-file system. See ARCHITECTURE.md (shipped alongside) for the design notes."""
import sys, os, re, struct, math, time, subprocess, argparse, hashlib, glob, ast, bisect, collections, json
import numpy as np
CSRC = r'''// nancm.c -- NaN v9 byte-level context-mixing core (bitwise, PAQ/lpaq family) with SUBWORD contexts.
// Predicts BYTES (so bpc is unambiguous). BPE enters as *causal* context features: the current word prefix is
// segmented with the same merge table as bpe.py (verified by --segdump), giving morphology-aware contexts.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
typedef uint8_t u8; typedef uint16_t u16; typedef uint32_t u32; typedef uint64_t u64;

#define NCTX 32
#define NLM 3
#define NMM 2
#define NPC 4
#define NIN (NCTX*NPC + 2 + NMM*2 + 2 + 1 + NLM*2)
#define NMIX 12
#define MAXH (48u<<20)

typedef struct { u16 p; u16 p2; u8 n; u8 chk; u8 st; u8 pad; } Slot;
static int TBITS[NCTX] = {18,24,24,24,24,24,24, 22,24,24, 22,24, 24,24, 22,22,22,22,22, 22,22, 23,23,22,22,23,22,23, 23,23,22,22};
static const int LIMIT[NCTX] = {250,250,250,160,120,100,80, 250,120,80, 250,120, 200,160, 250,250,250,250,250, 250,250, 80,120,200,200,160,250,120, 60,160,160,160};
static Slot *T[NCTX]; static Slot o0t[256];
static int rt[256]; static float conf[256];
static float stab[4096];
static u8 *hist; static u32 histlen = 0, mm_floor = 0;

static int ns=1; static u8 nxt[256][2]; static short sn0[256],sn1[256]; static u8 sl_[256];
static int allowed(int a,int b){ int m=a<b?a:b,M=a<b?b:a; if(m==0)return M<=40; if(m==1)return M<=20; if(m==2)return M<=10; if(m==3)return M<=6; if(m==4)return M<=5; return 0; }
static int sid[64*64*2];
static void build_states(void){
  memset(sid,-1,sizeof(sid)); ns=0; int q[256]; int qh=0;
  #define ADD(a,b,l) ({ int k=((a)*64+(b))*2+(l); if(sid[k]<0){ sid[k]=ns; sn0[ns]=a; sn1[ns]=b; sl_[ns]=l; q[ns]=ns; ns++; if(ns>255){fprintf(stderr,"too many states\n");exit(1);} } sid[k]; })
  ADD(0,0,0);
  while(qh<ns){ int id=qh++; for(int bit=0;bit<2;bit++){ int n0=sn0[id],n1=sn1[id];
      if(bit){ n1++; if(n0>2) n0=2+(n0-2)/3; } else { n0++; if(n1>2) n1=2+(n1-2)/3; }
      while(!allowed(n0,n1)){ if(bit){ if(n1>=n0) n1--; else n0--; } else { if(n0>=n1) n0--; else n1--; } }
      int l=bit; if(n0==0) l=(n1>0); if(n1==0) l=0; if(n0==0&&n1==0) l=0;
      nxt[id][bit]=(u8)ADD(n0,n1,l); } }
}
typedef struct { float p,n; } SM;
static SM smt[NCTX][8][256]; static int use_sm=3;
static inline float squashf(float x){ if(x>20)x=20; if(x<-20)x=-20; return 1.0f/(1.0f+expf(-x)); }
static inline float clampf(float x,float a,float b){ return x<a?a:(x>b?b:x); }
static inline u32 mix(u32 a,u32 b){ u32 x=a*0x9E3779B1u ^ (b+0x7F4A7C15u)*0x85EBCA6Bu; x^=x>>16; x*=0x7FEB352Du; x^=x>>15; x*=0x846CA68Bu; x^=x>>16; return x; }
static inline int isword(int c){ return (c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c>=0x80; }
static inline int fold(int c){ return (c>='A'&&c<='Z')?c+32:c; }

// ---------- BPE (same table as bpe.py) ----------
static u16 *pr = NULL; static int nmerge = 0; static int use_sub = 1;
static int bpe_load(const char *path){
  pr = (u16*)calloc(4096u*4096u, 2); FILE *f=fopen(path,"r"); if(!f) return -1; int a,b,r=0;
  while(fscanf(f,"%d %d",&a,&b)==2){ if(a<4096&&b<4096) pr[a*4096+b]=(u16)(r+1); r++; } fclose(f); nmerge=r; return r; }
static int seg(const u8 *w,int L,int *t){ int n=L>48?48:L; for(int i=0;i<n;i++) t[i]=w[i];
  while(n>1){ int best=-1; int br=65535; for(int i=0;i<n-1;i++){ int r=pr[t[i]*4096+t[i+1]]; if(r&&r<br){br=r;best=i;} }
    if(best<0)break; t[best]=256+(br-1); for(int i=best+1;i<n-1;i++) t[i]=t[i+1]; n--; } return n; }

// ---------- byte state ----------
typedef struct { u32 c4,c8; u32 wh,pw1,pw2,fwh,fpw1; u32 pw3,pw4,lh,plh,pf; int col,lst,pls,plen,plv,lhd,wic,wpre; int wlen,wsp; u8 w[64]; int mptr[NMM], mlen[NMM]; int mmatched[NMM]; } BS;
static BS bs;
static u32 *mht[NMM]; static const int MINLEN[NMM]={5,12}; static const int MHB=22;
static float msm[NMM][64];
static int lmtop[NLM], lmcf[NLM]; static int gc=1;
static u32 h[NCTX]; static int cls_c1, wl_sel; static int xmask=127; static int ctxmask=31, m6=1, a3on=1; static u16 ind1[65536], ind2[65536]; static int ind_upd=1;
static inline int clsof(int c){ return (c>='a'&&c<='z')?0:(c>='A'&&c<='Z')?1:(c>='0'&&c<='9')?2:(c==' ')?3:(c=='\n')?4:(c>=0x80)?5:6; }

static void mm_byte_update(BS *s,int c){
  for(int j=0;j<NMM;j++){
    if(s->mlen[j]>0 && (int)hist[s->mptr[j]]==c){ s->mlen[j]++; s->mptr[j]++; } else s->mlen[j]=0;
    u32 pos=histlen; // hist already has c appended
    if(pos-mm_floor>=(u32)MINLEN[j]){
      u32 hv=0; for(int k=1;k<=MINLEN[j];k++) hv=hv*0x2F0B4A27u+hist[pos-k]+1; hv=(hv^(hv>>15))*0x9E3779B1u>>(32-MHB);
      if(s->mlen[j]==0){ u32 cand=mht[j][hv]; if(cand>mm_floor && cand<pos){ int l=0; while(l<40 && cand-1-l>=mm_floor && pos-1-l>cand-1-l && hist[cand-1-l]==hist[pos-1-l]) l++;
          if(l>=MINLEN[j]){ s->mlen[j]=l; s->mptr[j]=cand; } } }
      mht[j][hv]=pos;
    }
  }
}
static void bs_push(BS *s,int c){
  hist[histlen++]=(u8)c;
  if(ind_upd){ u32 k1=s->c4&0xFFFF, k2=mix(s->c4&0xFFFFFF,99)>>16; ind1[k1]=(u16)((ind1[k1]<<8)|c); ind2[k2]=(u16)((ind2[k2]<<8)|c); }
  s->c8=(s->c8<<8)|(s->c4>>24); s->c4=(s->c4<<8)|c;
  if(c=='\n'){ s->plh=s->lh; s->pls=s->lst; s->plen=(int)histlen-1-s->lst; s->plv=1; s->lst=(int)histlen; s->col=0; s->lh=0; s->lhd=0; s->wic=0; } else { s->col++; if(!s->lhd){ if(c==' ') s->lhd=1; else s->lh=mix(s->lh+1,c); } }
  if(isword(c)){ if(s->wlen==0){ s->wsp=(((s->c4>>8)&255)==32); s->wpre=(s->c4>>8)&255; s->wic++; } s->wh=mix(s->wh+1,c); if(s->wlen<63) s->w[s->wlen]=(u8)c; s->wlen++; s->fwh=mix(s->fwh+1,fold(c)); }
  else { if(s->wlen>0){ s->pw4=s->pw3; s->pw3=s->pw2; s->pw2=s->pw1; s->pw1=s->wh; s->fpw1=s->fwh; s->pf=(s->pf<<8)|s->w[0]; } s->wh=0; s->wlen=0; s->fwh=0; }
  mm_byte_update(s,c);
}
static void compute_ctx(const BS *s){
  u32 c1=s->c4&255,c2=(s->c4>>8)&255,c3=(s->c4>>16)&255,c4=s->c4>>24;
  h[0]=c1+1;
  h[1]=mix(s->c4&0xFFFF,2);
  h[2]=mix(s->c4&0xFFFFFF,3);
  h[3]=mix(s->c4,4);
  h[4]=mix(s->c4,mix(s->c8&0xFF,5));
  h[5]=mix(s->c4,mix(s->c8&0xFFFF,6));
  h[6]=mix(s->c4,mix(s->c8,8));
  int inw=s->wlen>0;
  h[7]= inw? mix(s->wh,7) : mix(c1,0x77);
  h[8]= mix(inw? s->wh : mix(c1,9), s->pw1+8);
  h[9]= mix(mix(inw? s->wh : mix(c1,10), s->pw1), s->pw2+9);
  u32 sa=0xFFFF,sb=0xFFFF; int ntok=0;
  if(use_sub && inw && pr){ int t[50]; u8 wb[50]; int L=s->wlen>48?48:s->wlen,o=0; if(s->wsp) wb[o++]=32; memcpy(wb+o,s->w,L); ntok=seg(wb,L+o,t); sa=t[ntok-1]; if(ntok>1) sb=t[ntok-2]; }
  h[10]= mix(mix(sa,sb),ntok>3?3:ntok);
  h[11]= mix(mix(mix(sa,sb),s->pw1),11+(inw?0:c1));
  h[12]= mix(mix(c2,c3),12);
  h[13]= mix(mix(fold(c1)|(fold(c2)<<8),fold(c3)|(fold(c4)<<8)),13);
  { int clh=clsof(c1)+7*clsof(c2)+49*clsof(c3)+343*clsof(c4);
    h[14]=mix(c1|(c3<<8),14);
    h[15]=mix(mix(clh,c1),15);
    h[16]=mix(ind1[s->c4&0xFFFF]|(c1<<16),16);
    h[17]=mix(mix(ind2[mix(s->c4&0xFFFFFF,99)>>16],c1|(c2<<8)),17);
    h[18]=mix(mix(s->fwh,s->fpw1),18+(inw?0:c1));
    h[19]=mix(lmtop[0]|(lmcf[0]<<8)|(lmcf[1]<<10),19);
    h[20]=mix(mix(lmtop[0]|(lmtop[1]<<8),lmcf[0]|(lmcf[1]<<2)),20+(inw?0:c1)); }
  { u32 bw = inw? s->wh : mix(c1,21);
    u32 ab=0,ab2=0; if(s->plv && s->col<s->plen){ ab=hist[s->pls+s->col]; if(s->col+1<s->plen) ab2=hist[s->pls+s->col+1]; }
    h[21]=mix(mix(mix(bw,s->pw1),s->pw2),s->pw3+21);
    h[22]=mix(mix(inw?s->wh:mix(c1,22),s->pw2),22);
    h[23]=mix(ab|(ab2<<8)|(c1<<16),23);
    h[24]=mix(mix(s->lh,inw?s->wh:mix(c1,24)),24);
    h[25]=inw? mix(mix(s->wh,(u32)s->wpre),25) : mix(mix(s->pw1,c1|(c2<<8)|(c3<<16)),25);
    h[26]=mix(mix((u32)(s->wic>15?15:s->wic),c1),inw?s->wh:0xA5);
    h[27]=mix(mix(inw?s->wh:mix(c1,27),s->pf&0xFFFFFF),27);
    u32 ab3=0; if(s->plv && s->col+2<s->plen) ab3=hist[s->pls+s->col+2];
    h[28]=mix(mix(mix(mix(bw,s->pw1),s->pw2),mix(s->pw3,s->pw4)),28);
    h[29]=mix(mix(ab|(ab2<<8)|(ab3<<16),c1|(c2<<8)),29);
    h[30]=mix(mix((u32)(s->col>255?255:s->col),ab),30);
    h[31]=mix(mix(s->plh,s->lh),(s->lhd?(inw?s->wh:mix(c1,31)):0x31)); }
  cls_c1 = (c1>='a'&&c1<='z')?0:(c1>='A'&&c1<='Z')?1:(c1>='0'&&c1<='9')?2:(c1==' ')?3:(c1=='\n')?4:(c1>=0x80)?5:6;
  wl_sel = (s->wlen>7?7:s->wlen)*8+cls_c1; // 64
}

// ---------- tables ----------
static inline Slot* find(int i,u32 hv,int c0,int alloc){
  u32 x=(hv^((u32)c0*0x2545F491u))*0x9E3779B1u; x^=x>>15; x*=0x85EBCA6Bu; x^=x>>13;
  u32 s=x>>(32-TBITS[i]); u8 chk=(u8)(x&255)|1; Slot *a=&T[i][s], *b=&T[i][s^1];
  if(a->n && a->chk==chk) return a; if(b->n && b->chk==chk) return b;
  if(!alloc) return NULL;
  Slot *v=(a->n<=b->n)?a:b; v->p=32768; v->p2=32768; v->st=0; v->n=0; v->chk=chk; return v;
}
static inline void supd(Slot *s,int bit,int lim){ int n=s->n; { int t2=bit?65535:0; s->p2=(u16)((int)s->p2+((t2-(int)s->p2)>>2)); s->st=nxt[s->st][bit]; } int tg=bit?65535:0; int d=(int)(((int64_t)(tg-(int)s->p)*rt[n])>>16); int p=(int)s->p+d; if(p<24)p=24; if(p>65511)p=65511; s->p=(u16)p; if(n<lim) s->n++; }
static inline float slogit(Slot *s){ return stab[s->p>>4]; }

// ---------- mixers / APM ----------
static const int SELN[NMIX]={256,64,256,128,64,512,3584,256,512,320,320,320};
static int mx=1,w2b_on=1,fin=1,bm=0; static float lra=0; static u32 *cnt1[NMIX];
static float *W2b,*FW; static float lrf=0.002f;
static float *W1[NMIX]; static float *W2; static float lr1=0.003f, lr2=0.0015f, w0=0.12f, apmr=0.02f;
typedef struct { float *t; int lo; float w; int ctx; } APM;
static APM apm1, apm2, apm3, apm4, apm5;
static void apm_init(APM *a,int n){ a->t=(float*)malloc(sizeof(float)*n*33); for(int i=0;i<n;i++) for(int j=0;j<33;j++) a->t[i*33+j]=squashf((j-16)*0.5f); }
static inline float apm_pp(APM *a,int ctx,float st){ float pos=(clampf(st,-7.99f,7.99f)+8.f)*2.f; int lo=(int)pos; float w=pos-lo; a->lo=lo; a->w=w; a->ctx=ctx; return a->t[ctx*33+lo]*(1-w)+a->t[ctx*33+lo+1]*w; }
static inline void apm_up(APM *a,int bit,float rate){ float *p=&a->t[a->ctx*33+a->lo]; p[0]+=(bit-p[0])*rate*(1-a->w); p[1]+=(bit-p[1])*rate*a->w; }

// ---------- LM inputs: [0]=subword(BPE) bigram, [1]=word bigram (both exact token->byte, in-process), [2]=external sidecar ----------
static float lmnode[NLM][256]; static int lm_on[NLM], lm_cfg[NLM]; static int lm_any=0;
static float h2f(u16 h){ u32 s=(h>>15)&1,e=(h>>10)&31,m=h&1023; float v; if(e==0) v=ldexpf((float)m,-24); else if(e==31) v=0.f; else v=ldexpf((float)(m|1024),(int)e-25); return s?-v:v; }
static void lm_finish(int i,const double *d){ // d[256] normalized byte distribution -> node probs, top guess, confidence bucket
  double lv[512]; for(int v=0;v<256;v++) lv[256+v]=d[v]; for(int k=255;k>=1;k--) lv[k]=lv[2*k]+lv[2*k+1];
  for(int c0=1;c0<256;c0++){ double den=lv[c0]; lmnode[i][c0]=den>1e-12?(float)(lv[2*c0+1]/den):0.5f; }
  int bt=0; for(int v=1;v<256;v++) if(d[v]>d[bt]) bt=v; double pm=d[bt]; lmtop[i]=bt; lmcf[i]=pm>0.9?3:pm>0.6?2:pm>0.3?1:0; lm_on[i]=1; }
// external sidecar (any token LM via tok2byte.py; UNTESTED with real HF weights here)
typedef struct { int mode; u64 n; u16 *d; } LMF;
static LMF lmtr, lmev; static long lmtr_start=0, stream_pos=0, evk=0;
static int lm_load(LMF *f,const char *p){ FILE *fp=fopen(p,"rb"); if(!fp){fprintf(stderr,"cannot open %s\n",p); return -1;} char mg[4]; u32 mode,res; u64 n;
  if(fread(mg,1,4,fp)!=4||fread(&mode,4,1,fp)!=1||fread(&n,8,1,fp)!=1||fread(&res,4,1,fp)!=1) return -1;
  size_t per=mode?256:8; f->d=(u16*)malloc(n*per*2); if(fread(f->d,2,n*per,fp)!=n*per){fprintf(stderr,"short sidecar\n"); return -1;} f->mode=(int)mode; f->n=n; fclose(fp); return 0; }
static void lm_set_ext(LMF *f,long pos,int c){
  if(!f->d||pos<0||(u64)pos>=f->n){ lm_on[2]=0; lmtop[2]=0; lmcf[2]=0; return; }
  if(f->mode==0){ u16 *q=f->d+(size_t)pos*8; for(int bp=0;bp<8;bp++){ int c0=(1<<bp)|(c>>(8-bp)); lmnode[2][c0]=q[bp]*(1.f/65536.f); } lm_on[2]=1; lmtop[2]=0; lmcf[2]=0; }
  else { double d[256],t=0; u16 *q=f->d+(size_t)pos*256; for(int v=0;v<256;v++){ d[v]=h2f(q[v]); t+=d[v]; } for(int v=0;v<256;v++) d[v]/=(t>0?t:1); lm_finish(2,d); } }
// exact token->byte engine (same math as tok2byte.py; bigram state = last token)
#define KMAX 128
typedef struct { int V,nn,nc,K; int *ord,*pos,*nlo,*nhi,*nex,*cs,*cc,*cnode; u8 *cb; double *cum;
  int nh,hlt[2*KMAX+8],hnd[2*KMAX+8]; double hw[2*KMAX+8]; } LME;
static LME lme[2]; static int lmK=8;
static u8 **g_tb; static int *g_tl;
static int cmp_tok(const void*a,const void*b){ int x=*(const int*)a,y=*(const int*)b; int m=g_tl[x]<g_tl[y]?g_tl[x]:g_tl[y]; int c=memcmp(g_tb[x],g_tb[y],m); if(c) return c; return g_tl[x]-g_tl[y]; }
static int lme_build(LME *L,int lo,int hi,int depth){ int id=L->nn++; L->nlo[id]=lo; L->nhi[id]=hi; L->nex[id]=-1; int i=lo;
  if(g_tl[L->ord[i]]==depth){ L->nex[id]=L->ord[i]; i++; }
  int g=0; for(int j=i;j<hi;){ int b=g_tb[L->ord[j]][depth],k=j; while(k<hi&&g_tb[L->ord[k]][depth]==b) k++; g++; j=k; }
  int off=L->nc; L->nc+=g; L->cs[id]=off; L->cc[id]=g; int q=0;
  for(int j=i;j<hi;){ int b=g_tb[L->ord[j]][depth],k=j; while(k<hi&&g_tb[L->ord[k]][depth]==b) k++; L->cb[off+q]=(u8)b; L->cnode[off+q]=lme_build(L,j,k,depth+1); q++; j=k; }
  return id; }
static void lme_reset(LME *L){ L->nh=1; L->hlt[0]=L->V; L->hnd[0]=0; L->hw[0]=1.0; }
static int lme_load(LME *L,const char *path){ FILE *f=fopen(path,"rb"); if(!f){fprintf(stderr,"cannot open %s\n",path);return -1;} char mg[4]; int V;
  if(fread(mg,1,4,f)!=4||fread(&V,4,1,f)!=1) return -1; L->V=V; u8 **tb=(u8**)malloc(sizeof(u8*)*V); int *tl=(int*)malloc(4*V); size_t tot=0;
  for(int i=0;i<V;i++){ u16 n; if(fread(&n,2,1,f)!=1) return -1; tb[i]=(u8*)malloc(n+1); if(fread(tb[i],1,n,f)!=n) return -1; tl[i]=n; tot+=n; }
  size_t R=(size_t)(V+1)*V; float *P=(float*)malloc(R*4); if(fread(P,4,R,f)!=R){fprintf(stderr,"short lm table\n");return -1;} fclose(f);
  g_tb=tb; g_tl=tl; L->ord=(int*)malloc(4*V); for(int i=0;i<V;i++) L->ord[i]=i; qsort(L->ord,V,4,cmp_tok);
  L->pos=(int*)malloc(4*V); for(int i=0;i<V;i++) L->pos[L->ord[i]]=i;
  size_t NN=tot+2; L->nlo=(int*)malloc(4*NN); L->nhi=(int*)malloc(4*NN); L->nex=(int*)malloc(4*NN); L->cs=(int*)malloc(4*NN); L->cc=(int*)malloc(4*NN); L->cnode=(int*)malloc(4*NN); L->cb=(u8*)malloc(NN);
  L->nn=0; L->nc=0; lme_build(L,0,V,0);
  size_t W=V+1; L->cum=(double*)malloc(8*(size_t)(V+1)*W);
  for(int r=0;r<=V;r++){ double *c=L->cum+(size_t)r*W; c[0]=0; double t=0; for(int i=0;i<V;i++){ t+=P[(size_t)r*V+L->ord[i]]; c[i+1]=t; } double inv=t>0?1.0/t:1.0; for(int i=0;i<=V;i++) c[i]*=inv; }
  free(P); L->K=lmK; lme_reset(L); return 0; }
static void lme_dist(LME *L,double *d){ memset(d,0,256*sizeof(double)); size_t W=L->V+1;
  for(int h=0;h<L->nh;h++){ const double *row=L->cum+(size_t)L->hlt[h]*W; int nd=L->hnd[h]; double w=L->hw[h];
    for(int k=0;k<L->cc[nd];k++){ int c=L->cnode[L->cs[nd]+k]; d[L->cb[L->cs[nd]+k]]+=w*(row[L->nhi[c]]-row[L->nlo[c]]); } }
  double t=0; for(int v=0;v<256;v++) t+=d[v]; if(t>0) for(int v=0;v<256;v++) d[v]/=t; else for(int v=0;v<256;v++) d[v]=1.0/256; }
static double lme_score(LME *L,int lt,int nd,double w){ if(nd==0) return w; const double *row=L->cum+(size_t)lt*(L->V+1); double m=row[L->nhi[nd]]-row[L->nlo[nd]]; if(L->nex[nd]>=0){ int p=L->pos[L->nex[nd]]; m-=row[p+1]-row[p]; } return w*(m>1e-300?m:1e-300); }
static void lme_push(LME *L,int b){ int nlt[2*KMAX+8],nnd[2*KMAX+8],n=0; double nw[2*KMAX+8]; size_t W=L->V+1;
  #define ADDH(LT,ND,WW) do{ int f_=-1; for(int q_=0;q_<n;q_++) if(nlt[q_]==(LT)&&nnd[q_]==(ND)){ f_=q_; break; } if(f_>=0) nw[f_]+=(WW); else if(n<2*KMAX+6){ nlt[n]=(LT); nnd[n]=(ND); nw[n]=(WW); n++; } }while(0)
  for(int h=0;h<L->nh;h++){ int nd=L->hnd[h],lt=L->hlt[h]; double w=L->hw[h]; int c=-1; for(int k=0;k<L->cc[nd];k++) if(L->cb[L->cs[nd]+k]==b){ c=L->cnode[L->cs[nd]+k]; break; }
    if(c<0) continue; int hasx=L->nex[c]>=0;
    if(hasx){ int tid=L->nex[c]; const double *row=L->cum+(size_t)lt*W; double pt=row[L->pos[tid]+1]-row[L->pos[tid]]; if(pt>0) ADDH(tid,0,w*pt); }
    if(L->nhi[c]-L->nlo[c]>(hasx?1:0)) ADDH(lt,c,w); }
  if(n==0){ nlt[0]=L->V; nnd[0]=0; nw[0]=1.0; n=1; }
  double sc[2*KMAX+8]; for(int i=0;i<n;i++) sc[i]=lme_score(L,nlt[i],nnd[i],nw[i]);
  int idx[2*KMAX+8]; for(int i=0;i<n;i++) idx[i]=i; // selection of top K by score
  for(int i=0;i<n&&i<L->K;i++){ int bi=i; for(int j=i+1;j<n;j++) if(sc[idx[j]]>sc[idx[bi]]) bi=j; int t=idx[i]; idx[i]=idx[bi]; idx[bi]=t; }
  int m=n<L->K?n:L->K; double tw=0; for(int i=0;i<m;i++) tw+=nw[idx[i]]; if(tw<=0) tw=1;
  for(int i=0;i<m;i++){ L->hlt[i]=nlt[idx[i]]; L->hnd[i]=nnd[idx[i]]; L->hw[i]=nw[idx[i]]/tw; } L->nh=m; }
static void lm_prepare(long ext_pos_tr,LMF *ef,int c_true){ double d[256];
  for(int i=0;i<2;i++){ if(lm_cfg[i]){ lme_dist(&lme[i],d); lm_finish(i,d); } else { lm_on[i]=0; lmtop[i]=0; lmcf[i]=0; } }
  if(lm_cfg[2]) lm_set_ext(ef,ext_pos_tr,c_true); else { lm_on[2]=0; lmtop[2]=0; lmcf[2]=0; } }
static void lm_advance(int c){ for(int i=0;i<2;i++) if(lm_cfg[i]) lme_push(&lme[i],c); }
static void lm_reset_all(void){ for(int i=0;i<2;i++) if(lm_cfg[i]) lme_reset(&lme[i]); }
typedef struct { Slot *sl[NCTX]; SM *sp[NCTX]; Slot *s0; float x[NIN]; int sel[NMIX]; float pm[NMIX], d[NMIX]; float pmixa,pmixb,z[7],q,pavg; int bsel2; float pmix,a1,a2,a3,a4,a5,pf; int mact[NMM],mebit[NMM],midx[NMM]; int c0,bp; } PS;
static PS ps;
static int lenb(int l){ return l<=0?0:(l>=28?7:1+(l>>2)); }

static float predict(int c0,int bp,int alloc){
  int k=0,nhit=0; ps.c0=c0; ps.bp=bp;
  for(int i=0;i<NCTX;i++){ int on=(i<14)||(i<19?((ctxmask>>(i-14))&1):i<21?gc:((xmask>>(i-21))&1)); Slot *s=on?find(i,h[i],c0,alloc):NULL; ps.sl[i]=s;
    ps.sp[i]=NULL;
    if(s&&s->n){ ps.x[k++]=slogit(s); ps.x[k++]=((float)s->p-32768.f)*(1.f/32768.f)*conf[s->n]; if(i>=2&&i<21)nhit++;
      { SM *e=&smt[i][bp][s->st]; ps.sp[i]=e; int ix=(int)(e->p*4096.f); if(ix<0)ix=0; if(ix>4095)ix=4095; ps.x[k++]=(use_sm&1)?stab[ix]:0.f; ps.x[k++]=(use_sm&2)?stab[s->p2>>4]:0.f; } }
    else { ps.x[k++]=0; ps.x[k++]=0; ps.x[k++]=0; ps.x[k++]=0; } }
  { Slot *s=&o0t[c0]; ps.s0=s; if(s->n){ ps.x[k++]=slogit(s); ps.x[k++]=((float)s->p-32768.f)*(1.f/32768.f)*conf[s->n]; } else { ps.x[k++]=0; ps.x[k++]=0; } }
  int mlb=0;
  for(int j=0;j<NMM;j++){ ps.mact[j]=0; int l=bs.mlen[j];
    if(l>0){ int pb=hist[bs.mptr[j]]; if(((pb|256)>>(8-bp))==c0){ int eb=(pb>>(7-bp))&1; int li=l>63?63:l; float st=clampf(logf(msm[j][li]/(1.f-msm[j][li])),-8,8);
        ps.mact[j]=1; ps.mebit[j]=eb; ps.midx[j]=li; ps.x[k++]=eb?st:-st; ps.x[k++]=(eb?1.f:-1.f)*(l>=16?1.f:0.5f); if(j==0) mlb=lenb(l); continue; } }
    ps.x[k++]=0; ps.x[k++]=0; }
  ps.x[k++]=0.4f;
  for(int i=0;i<NLM;i++){ if(lm_on[i]){ float p=clampf(lmnode[i][c0],1e-4f,1.f-1e-4f); float st=clampf(logf(p/(1.f-p)),-8.f,8.f); ps.x[k++]=st; ps.x[k++]=p-0.5f; int lb=(int)((st+8.f)*2.f); if(lb>31)lb=31; ps.sel[9+i]=lb*8+bp; }
    else { ps.x[k++]=0; ps.x[k++]=0; ps.sel[9+i]=32*8+bp; } }
  ps.sel[0]=c0; ps.sel[1]=mlb*8+bp; ps.sel[2]=(int)(bs.c4&255); ps.sel[3]=(nhit>15?15:nhit)*8+bp; ps.sel[4]=(wl_sel&63); ps.sel[5]=clsof(bs.c4&255)+7*clsof((bs.c4>>8)&255)+49*clsof((bs.c4>>16)&255);
  { int ms=(ps.mact[0]?1+ps.mebit[0]:0)+3*(ps.mact[1]?1+ps.mebit[1]:0); ps.sel[6]=bm?((((bs.wlen>3?3:bs.wlen)*7+cls_c1)*4+lmcf[1])*4+lmcf[0])*8+bp:ms*8+bp; ps.sel[7]=(int)((bs.c4>>8)&255);
    int hm=0; static const int hs[6]={3,4,5,6,8,10}; for(int q=0;q<6;q++) if(ps.sl[hs[q]]&&ps.sl[hs[q]]->n) hm|=1<<q; ps.sel[8]=hm*8+bp; }
  for(int m=0;m<NMIX;m++){ if((m==5&&!m6)||((m==6&&!(mx||bm))||(m>=7&&m<9&&!mx))||(m>=9&&!lm_cfg[m-9])){ ps.d[m]=0; ps.pm[m]=0.5f; continue; } float *w=&W1[m][ps.sel[m]*NIN]; float dot=0; for(int i=0;i<NIN;i++) dot+=w[i]*ps.x[i];
    dot=clampf(dot,-10,10); ps.d[m]=dot; ps.pm[m]=squashf(dot); }
  float *w2=&W2[c0*NMIX]; float o2a=0; for(int m=0;m<NMIX;m++) o2a+=w2[m]*ps.d[m]; o2a=clampf(o2a,-10,10); ps.pmixa=squashf(o2a);
  float o2=o2a; ps.bsel2=mlb*8+bp; ps.pmixb=0.5f;
  if(w2b_on){ float *wb=&W2b[ps.bsel2*NMIX]; float o2b=0; for(int m=0;m<NMIX;m++) o2b+=wb[m]*ps.d[m]; o2b=clampf(o2b,-10,10); ps.pmixb=squashf(o2b); o2=0.5f*(o2a+o2b); }
  ps.pmix=squashf(o2);
  ps.a1=apm_pp(&apm1,c0,o2); ps.a2=apm_pp(&apm2,c0*256+(bs.c4&255),o2);
  ps.a3=apm_pp(&apm3,c0|((wl_sel&63)<<8),o2);
  float pf;
  if(fin==0){ pf=a3on?(ps.pmix+ps.a1+2*ps.a2+ps.a3)*0.2f:(ps.pmix+ps.a1+2*ps.a2)*0.25f; }
  else { ps.a4=apm_pp(&apm4,(int)(mix(bs.c4&0xFFFF,c0+77)>>16),o2);
    int pb2=-1; if(ps.mact[0]) pb2=hist[bs.mptr[0]]; ps.a5=apm_pp(&apm5,ps.mact[0]?(c0|(pb2<<8)):c0,o2);
    float pa=(ps.pmix+ps.a1+2*ps.a2+ps.a3+ps.a4+ps.a5)*(1.f/7.f); ps.pavg=pa;
    float av[6]={ps.pmix,ps.a1,ps.a2,ps.a3,ps.a4,ps.a5}; float *fw=&FW[(c0)*7]; float dd=0;
    for(int i=0;i<6;i++){ float pp=clampf(av[i],1e-4f,1-1e-4f); ps.z[i]=logf(pp/(1-pp)); dd+=fw[i]*ps.z[i]; } ps.z[6]=0.3f; dd+=fw[6]*0.3f;
    ps.q=squashf(clampf(dd,-12,12)); pf=(fin==2)?ps.q:0.5f*(ps.q+pa); }
  ps.pf=clampf(pf,1.f/16384,1.f-1.f/16384); return ps.pf;
}
static void update(int bit){
  for(int i=0;i<NCTX;i++) if(ps.sl[i]){ if(ps.sp[i]){ SM *e=ps.sp[i]; float r=1.f/(e->n+1.5f); if(r<0.004f) r=0.004f; e->p+=(bit-e->p)*r; e->n+=1; } supd(ps.sl[i],bit,LIMIT[i]); }
  supd(ps.s0,bit,255);
  for(int j=0;j<NMM;j++) if(ps.mact[j]){ float *p=&msm[j][ps.midx[j]]; *p+=((bit==ps.mebit[j])-*p)*0.02f; *p=clampf(*p,0.02f,0.9995f); }
  for(int m=0;m<NMIX;m++){ if((m==5&&!m6)||((m==6&&!(mx||bm))||(m>=7&&m<9&&!mx))||(m>=9&&!lm_cfg[m-9])) continue; float lr=lr1; if(lra>0){ u32 *c=&cnt1[m][ps.sel[m]]; lr=lr1*(1.f+lra/(1.f+(*c)*(1.f/32.f))); (*c)++; }
    float e=(bit-ps.pm[m])*lr; float *w=&W1[m][ps.sel[m]*NIN]; for(int i=0;i<NIN;i++) w[i]+=e*ps.x[i]; }
  { float e=(bit-ps.pmixa)*lr2; float *w2=&W2[ps.c0*NMIX]; for(int m=0;m<NMIX;m++) w2[m]+=e*ps.d[m]; }
  if(w2b_on){ float e=(bit-ps.pmixb)*lr2; float *wb=&W2b[ps.bsel2*NMIX]; for(int m=0;m<NMIX;m++) wb[m]+=e*ps.d[m]; }
  if(fin){ float e=(bit-ps.q)*lrf; float *fw=&FW[ps.c0*7]; for(int i=0;i<7;i++) fw[i]+=e*ps.z[i]; apm_up(&apm4,bit,apmr); apm_up(&apm5,bit,apmr); }
  apm_up(&apm1,bit,apmr); apm_up(&apm2,bit,apmr); apm_up(&apm3,bit,apmr);
}

// ---------- init ----------
static void init_model(int tb_shift){
  for(int i=0;i<NCTX;i++){ int b=TBITS[i]-tb_shift; if(b<16)b=16; TBITS[i]=b; T[i]=(Slot*)calloc((size_t)1<<b,sizeof(Slot)); if(!T[i]){fprintf(stderr,"oom\n");exit(1);} }
  for(int c=0;c<256;c++){ o0t[c].p=32768; o0t[c].n=0; }
  for(int n=0;n<256;n++){ rt[n]=(int)(65536.0/(n+1.5)); conf[n]=n>=4?1.f:n/4.f; }
  for(int i=0;i<4096;i++){ float p=(i+0.5f)/4096.f; stab[i]=clampf(logf(p/(1-p)),-8.3f,8.3f); }
  for(int m=0;m<NMIX;m++){ W1[m]=(float*)malloc(sizeof(float)*SELN[m]*NIN); for(int i=0;i<SELN[m]*NIN;i++) W1[m][i]=w0; }
  { int na=0; float act[NMIX]; for(int m=0;m<NMIX;m++){ int sk=(m==5&&!m6)||((m==6&&!(mx||bm))||(m>=7&&m<9&&!mx))||(m>=9&&!lm_cfg[m-9]); act[m]=sk?0.f:1.f; na+=!sk; }
    W2=(float*)malloc(sizeof(float)*256*NMIX); for(int i=0;i<256*NMIX;i++) W2[i]=act[i%NMIX]/na;
    W2b=(float*)malloc(sizeof(float)*64*NMIX); for(int i=0;i<64*NMIX;i++) W2b[i]=act[i%NMIX]/na;
    FW=(float*)malloc(sizeof(float)*256*7); for(int i=0;i<256*7;i++) FW[i]=((i%7)<6)?1.f/6.f:0.f;
    for(int m=0;m<NMIX;m++) cnt1[m]=(u32*)calloc(SELN[m],4);
    build_states(); for(int i=0;i<NCTX;i++) for(int b=0;b<8;b++) for(int t=0;t<ns;t++){ smt[i][b][t].p=(sn1[t]+0.4f)/(sn0[t]+sn1[t]+0.8f); smt[i][b][t].n=0; } }
  apm_init(&apm1,256); apm_init(&apm2,65536); apm_init(&apm3,16384); apm_init(&apm4,65536); apm_init(&apm5,65536);
  hist=(u8*)malloc(MAXH);
  for(int j=0;j<NMM;j++){ mht[j]=(u32*)calloc((size_t)1<<MHB,4); for(int l=0;l<64;l++) msm[j][l]=1.f-1.f/(l+2.f); }
  memset(&bs,0,sizeof(bs)); compute_ctx(&bs);
}

// ---------- train / eval ----------
static int train_bytes(const u8 *d,size_t n,double *bits_acc){
  for(size_t i=0;i<n;i++){ int c=d[i]; int c0=1; lm_prepare(stream_pos+(long)i-lmtr_start,&lmtr,c); compute_ctx(&bs);
    for(int bp=0;bp<8;bp++){ int bit=(c>>(7-bp))&1; float p=predict(c0,bp,1); *bits_acc+=-log2f(bit?p:1-p); update(bit); c0=c0*2+bit; }
    bs_push(&bs,c); lm_advance(c); }
  return 0;
}
typedef struct { float nll; u8 t1,t5,pad[2]; } Rec;
static void dist256(float *pr256){ // full byte distribution from node predictions (no allocation / no update)
  float p1[256]; for(int c0=1;c0<256;c0++){ int bp=0; int t=c0; while(t>1){t>>=1;bp++;} p1[c0]=predict(c0,bp,0); }
  for(int v=0;v<256;v++){ float pr=1.f; int c0=1; for(int b=7;b>=0;b--){ int bit=(v>>b)&1; pr*= bit?p1[c0]:(1.f-p1[c0]); c0=c0*2+bit; } pr256[v]=pr; }
}

// spelled-unit accuracy: a unit is "guessed" at top-k if every one of its bytes is within the model's top-k given the true prefix
typedef struct { long n[3],h1[3],h5[3],fm[3][8]; } UM; static UM um; static u8 *evfl=NULL; static size_t evfl_n=0;
static void unit_metrics(const u8 *f,const u8 *rk,u32 len,u32 skip){
  for(int kd=0;kd<3;kd++) for(u32 i=skip;i<len;i++){
    int st = kd==0 ? (f[i]&1) : kd==1 ? ((f[i]&1)&&!(f[i]&2)) : (f[i]&4); if(!st) continue;
    u32 j=i+1; if(kd<2) while(j<len&&!(f[j]&1)) j++; else while(j<len&&!(f[j]&2)) j++;
    if(j>=len) continue; int a1=1,a5=1; for(u32 k=i;k<j;k++){ if(rk[k]!=0) a1=0; if(rk[k]>=5) a5=0; } um.n[kd]++; um.h1[kd]+=a1; um.h5[kd]+=a5; if(!a1){ u32 k=i; while(rk[k]==0) k++; um.fm[kd][(k-i)>7?7:(k-i)]++; } } }
static void eval_file(const char *wpath,int online,const char *opath,double *o_nats,double *o_t1,double *o_t5,long *o_n){
  FILE *f=fopen(wpath,"rb"); if(!f){fprintf(stderr,"no windows\n");exit(1);} u32 nw,skip; fread(&nw,4,1,f); fread(&skip,4,1,f);
  evk=0; FILE *fo=fopen(opath,"wb"); BS save=bs; int iu_save=ind_upd; if(!online) ind_upd=0; u32 hsave=histlen; u32 fsave=mm_floor; double nats=0,t1=0,t5=0; long n=0; float pr256[256];
  memset(&um,0,sizeof(um)); size_t flo=0;
  for(u32 w=0;w<nw;w++){ u32 len; fread(&len,4,1,f); u8 *buf=(u8*)malloc(len); fread(buf,1,len,f); u8 *rk=(u8*)calloc(len,1);
    if(!online){ histlen=hsave; }
    mm_floor=histlen; memset(&bs,0,sizeof(bs)); bs.lst=bs.pls=(int)histlen; lm_reset_all();
    for(u32 i=0;i<len;i++){ int c=buf[i]; lm_prepare(evk++,&lmev,c); compute_ctx(&bs);
      if(i>=skip){ dist256(pr256); float pt=pr256[c]; int rank=0; for(int v=0;v<256;v++) if(pr256[v]>pt) rank++;
        Rec r; r.nll=0; r.t1=(rank==0); r.t5=(rank<5); r.pad[0]=r.pad[1]=0; rk[i]=(u8)(rank>254?254:rank);
        if(online){ int c0=1; double nl=0; for(int bp=0;bp<8;bp++){ int bit=(c>>(7-bp))&1; float p=predict(c0,bp,1); nl+=-logf(bit?p:1-p); update(bit); c0=c0*2+bit; } r.nll=(float)nl; }
        else r.nll=-logf(fmaxf(pt,1e-9f));
        nats+=r.nll; t1+=r.t1; t5+=r.t5; n++; fwrite(&r,sizeof(r),1,fo);
      } else if(online){ int c0=1; for(int bp=0;bp<8;bp++){ int bit=(c>>(7-bp))&1; predict(c0,bp,1); update(bit); c0=c0*2+bit; } }
      bs_push(&bs,c); lm_advance(c); }
    if(evfl && flo+len<=evfl_n) unit_metrics(evfl+flo,rk,len,skip); flo+=len;
    free(buf); free(rk); }
  fclose(f); fclose(fo);
  if(!online){ bs=save; histlen=hsave; } mm_floor=fsave; ind_upd=iu_save;
  *o_nats=nats/n; *o_t1=t1/n; *o_t5=t5/n; *o_n=n;
}
static void print_units(const char *tag){ if(!um.n[0]) return; const char *nm[3]={"token","subword","word"};
  printf("UNITS %s:",tag); for(int k=0;k<3;k++) printf(" %s n=%ld top1=%.4f top5=%.4f |",nm[k],um.n[k],um.n[k]?(double)um.h1[k]/um.n[k]:0.0,um.n[k]?(double)um.h5[k]/um.n[k]:0.0); printf("\n");
  for(int k=0;k<3;k++){ printf("FIRSTMISS %s %s:",tag,nm[k]); for(int j=0;j<8;j++) printf(" %d=%ld",j,um.fm[k][j]); printf("\n"); } fflush(stdout); }

int main(int argc,char **argv){
  const char *evalp=NULL,*bpep=NULL,*outp="res"; int tbs=0; long evalat[32]; int nea=0; int online_final=1; char *files[16]; int nf=0; long maxb=1L<<60; const char *lmpath[2]={NULL,NULL}; const char *chk_in=NULL,*chk_out=NULL; int chk_i=0,chk_n=0;
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"-eval")) evalp=argv[++i];
    else if(!strcmp(argv[i],"-bpe")) bpep=argv[++i];
    else if(!strcmp(argv[i],"-o")) outp=argv[++i];
    else if(!strcmp(argv[i],"-tshift")) tbs=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-nosub")) use_sub=0;
    else if(!strcmp(argv[i],"-cm")) ctxmask=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-xc")) xmask=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-m6")) m6=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-a3")) a3on=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-lmtr")){ if(lm_load(&lmtr,argv[++i])) return 1; }
    else if(!strcmp(argv[i],"-lmtrstart")) lmtr_start=atol(argv[++i]);
    else if(!strcmp(argv[i],"-lmev")){ if(lm_load(&lmev,argv[++i])) return 1; }
    else if(!strcmp(argv[i],"-lmsub")) lmpath[0]=argv[++i];
    else if(!strcmp(argv[i],"-lmword")) lmpath[1]=argv[++i];
    else if(!strcmp(argv[i],"-lmk")) lmK=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-gc")) gc=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-lmcheck")){ chk_i=atoi(argv[i+1]); chk_in=argv[i+2]; chk_n=atoi(argv[i+3]); chk_out=argv[i+4]; i+=4; }
    else if(!strcmp(argv[i],"-evalu")){ FILE *fu=fopen(argv[++i],"rb"); if(!fu){fprintf(stderr,"no unit flags\n");return 1;} fseek(fu,0,SEEK_END); evfl_n=ftell(fu); fseek(fu,0,SEEK_SET); evfl=(u8*)malloc(evfl_n); fread(evfl,1,evfl_n,fu); fclose(fu); }
    else if(!strcmp(argv[i],"-w0")) w0=atof(argv[++i]);
    else if(!strcmp(argv[i],"-apmr")) apmr=atof(argv[++i]);
    else if(!strcmp(argv[i],"-sm")) use_sm=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-mx")) mx=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-bm")) bm=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-w2b")) w2b_on=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-fin")) fin=atoi(argv[++i]);
    else if(!strcmp(argv[i],"-lra")) lra=atof(argv[++i]);
    else if(!strcmp(argv[i],"-lrf")) lrf=atof(argv[++i]);
    else if(!strcmp(argv[i],"-lr1")) lr1=atof(argv[++i]);
    else if(!strcmp(argv[i],"-lr2")) lr2=atof(argv[++i]);
    else if(!strcmp(argv[i],"-maxb")) maxb=atol(argv[++i]);
    else if(!strcmp(argv[i],"-noonline")) online_final=0;
    else if(!strcmp(argv[i],"-evalat")){ char *s=strtok(argv[++i],","); while(s){ evalat[nea++]=atol(s); s=strtok(NULL,","); } }
    else if(!strcmp(argv[i],"-segdump")){ // segdump <merges> <file> <out>
      bpe_load(argv[i+1]); FILE *f=fopen(argv[i+2],"rb"); fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); u8 *d=(u8*)malloc(n); fread(d,1,n,f); fclose(f);
      FILE *fo=fopen(argv[i+3],"wb"); long i0=0; while(i0<n){ if(isword(d[i0])){ long j=i0; while(j<n&&isword(d[j])) j++; int t[48]; long L=j-i0; long off=0;
          // long words: segment in 48-byte chunks (C context cap); python test restricts to words <=48
          while(off<L){ int cl=(L-off>48)?48:(int)(L-off); int nt=seg(d+i0+off,cl,t); for(int k=0;k<nt;k++){ u32 v=t[k]; fwrite(&v,4,1,fo);} off+=cl; } i0=j; }
        else { u32 v=d[i0]; fwrite(&v,4,1,fo); i0++; } }
      fclose(fo); return 0; }
    else files[nf++]=argv[i];
  }
  if(bpep && bpe_load(bpep)<0){ fprintf(stderr,"bpe load failed\n"); return 1; }
  if(bpep) fprintf(stderr,"bpe merges loaded: %d subword=%d\n",nmerge,use_sub);
  lm_cfg[2]=(lmtr.d!=NULL)||(lmev.d!=NULL); if(lmK>KMAX) lmK=KMAX;
  for(int i=0;i<2;i++) if(lmpath[i]){ if(lme_load(&lme[i],lmpath[i])) return 1; lm_cfg[i]=1; fprintf(stderr,"LM[%d] loaded: V=%d nodes=%d beam=%d\n",i,lme[i].V,lme[i].nn,lmK); }
  lm_any=lm_cfg[0]||lm_cfg[1]||lm_cfg[2];
  if(lmev.d&&lmev.mode!=1){ fprintf(stderr,"eval sidecar must be full (mode 1)\n"); return 1; }
  if(chk_out){ FILE *fi=fopen(chk_in,"rb"); u8 *b=(u8*)malloc(chk_n); fread(b,1,chk_n,fi); fclose(fi); FILE *fo=fopen(chk_out,"wb"); lm_reset_all(); double d[256];
    for(int i=0;i<chk_n;i++){ lme_dist(&lme[chk_i],d); fwrite(d,8,256,fo); lme_push(&lme[chk_i],b[i]); } fclose(fo); return 0; }
  init_model(tbs);
  long total=0; double bits=0, bits_win=0; long win=0; int eai=0; double nats,t1,t5; long ne;
  for(int fi=0;fi<nf;fi++){ FILE *f=fopen(files[fi],"rb"); if(!f){fprintf(stderr,"cannot open %s\n",files[fi]);return 1;} fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    if(n>maxb-total) n=maxb-total; u8 *d=(u8*)malloc(n); fread(d,1,n,f); fclose(f);
    long done=0; const long CH=1<<18; lm_reset_all();
    while(done<n){ long m=n-done<CH?n-done:CH; stream_pos=total; train_bytes(d+done,m,&bits_win); done+=m; total+=m; win+=m;
      if(win>=(1L<<20)){ fprintf(stderr,"[train %s] %ld bytes  running bpc(last 1M)=%.3f\n",files[fi],total,bits_win/win); win=0; bits_win=0; }
      while(eai<nea && total>=evalat[eai]){ if(evalp){ char o[256]; snprintf(o,256,"%s_frozen_%ld.bin",outp,evalat[eai]); eval_file(evalp,0,o,&nats,&t1,&t5,&ne); lm_reset_all();
          printf("EVAL frozen @%ld bytes: nats=%.4f bpc=%.4f top1=%.4f top5=%.4f n=%ld\n",total,nats,nats/log(2),t1,t5,ne); fflush(stdout);} eai++; } }
    free(d); }
  if(evalp){ char o[256]; snprintf(o,256,"%s_frozen.bin",outp); eval_file(evalp,0,o,&nats,&t1,&t5,&ne);
    printf("FINAL frozen @%ld bytes: nats=%.4f bpc=%.4f top1=%.4f top5=%.4f n=%ld\n",total,nats,nats/log(2),t1,t5,ne); print_units("frozen");
    if(online_final){ snprintf(o,256,"%s_online.bin",outp); eval_file(evalp,1,o,&nats,&t1,&t5,&ne);
      printf("FINAL online @%ld bytes: nats=%.4f bpc=%.4f top1=%.4f top5=%.4f n=%ld\n",total,nats,nats/log(2),t1,t5,ne); print_units("online"); } }
  return 0;
}
'''
# ======================= Python side: corpus prep, BPE, LM tables, token->byte reference, CLI =======================
PRE = re.compile(rb" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")

def pretok(b): return PRE.findall(b)

def train_bpe(counts, nmerge):
    words = [list(w) for w in counts]; freq = [counts[w] for w in counts]
    pc = collections.Counter(); where = collections.defaultdict(set)
    for i, w in enumerate(words):
        for p in zip(w, w[1:]): pc[p] += freq[i]; where[p].add(i)
    merges = []
    for r in range(nmerge):
        if not pc: break
        pair = max(pc, key=pc.get)
        if pc[pair] < 2: break
        nid = 256 + r; merges.append(pair)
        for i in list(where[pair]):
            w = words[i]; f = freq[i]
            for p in zip(w, w[1:]):
                pc[p] -= f
                if pc[p] <= 0: del pc[p]
            nw = []; j = 0
            while j < len(w):
                if j < len(w) - 1 and (w[j], w[j + 1]) == pair: nw.append(nid); j += 2
                else: nw.append(w[j]); j += 1
            words[i] = nw
            for p in zip(nw, nw[1:]): pc[p] += f; where[p].add(i)
        where.pop(pair, None)
    return merges

class BPE:
    def __init__(self, merges):
        self.merges = merges; self.rank = {m: i for i, m in enumerate(merges)}; self.cache = {}
        self.vocab = [bytes([i]) for i in range(256)]
        for a, b in merges: self.vocab.append(self.vocab[a] + self.vocab[b])
    def enc_pre(self, pt):
        r = self.cache.get(pt)
        if r is None:
            t = list(pt)
            while len(t) > 1:
                best = None; br = 1 << 30
                for i in range(len(t) - 1):
                    k = self.rank.get((t[i], t[i + 1]))
                    if k is not None and k < br: br = k; best = i
                if best is None: break
                t[best:best + 2] = [256 + br]
            r = self.cache[pt] = t
        return r
    def save(self, p): open(p, "w").write("\n".join("%d %d" % m for m in self.merges) + "\n")
    @staticmethod
    def load(p): return BPE([tuple(map(int, l.split())) for l in open(p) if l.strip()])

def write_lm_table(path, tokbytes, P):
    """P: (V+1, V) float32 next-token probabilities, row V = BOS/unknown-context."""
    with open(path, "wb") as f:
        f.write(b"LMT1" + struct.pack("<i", len(tokbytes)))
        for t in tokbytes: f.write(struct.pack("<H", len(t)) + t)
        f.write(np.ascontiguousarray(P, np.float32).tobytes())

def bigram_matrix(ids, V, D=0.75):
    ids = np.asarray(ids, np.int64); prev, nxt = ids[:-1], ids[1:]
    C = np.bincount(prev * V + nxt, minlength=V * V).reshape(V, V).astype(np.float64)
    uni = np.bincount(ids, minlength=V) + 0.5; uni /= uni.sum(); tot = C.sum(1, keepdims=True)
    lam = (D * (C > 0).sum(1, keepdims=True) + 1e-9) / np.maximum(tot, 1e-9); lam = np.where(tot > 0, lam, 1.0)
    M = np.maximum(C - D, 0) / np.maximum(tot, 1e-9) + lam * uni[None, :]; M /= M.sum(1, keepdims=True)
    return np.vstack([M, uni[None, :] / uni.sum()]).astype(np.float32)

def build_word_vocab(pcount, nwords):
    singles = [bytes([i]) for i in range(256)]; sset = set(singles)
    top = [w for w, _ in pcount.most_common() if len(w) > 1 and w not in sset][:nwords]
    return singles + top

def encode_words(pts, vocab):
    idx = {t: i for i, t in enumerate(vocab)}; out = []
    for pt in pts:
        i = idx.get(pt)
        if i is not None: out.append(i)
        else: out.extend(pt)
    return out

def unit_flags(b, bpe):
    """per-byte flags: 1=BPE token starts, 2=pretoken starts, 4=pretoken is an alphabetic word (with optional leading space)."""
    f = bytearray(len(b)); o = 0
    for pt in pretok(b):
        f[o] |= 2
        if pt.lstrip(b" ")[:1].isalpha(): f[o] |= 4
        k = o
        for t in bpe.enc_pre(pt): f[k] |= 1; k += len(bpe.vocab[t])
        o += len(pt)
    return bytes(f)

def gather_corpus(args):
    """returns list of (name, bytes). Explicit --corpus paths win; otherwise English-ish prose found on this machine."""
    items = []
    def add(p, t):
        if len(t) > 2000: items.append((p, t))
    if args.corpus:
        for p in args.corpus:
            fs = [p] if os.path.isfile(p) else [x for x in glob.glob(os.path.join(p, "**", "*"), recursive=True) if os.path.isfile(x)]
            for x in sorted(fs): add(x, open(x, "rb").read())
    else:
        for p in glob.glob("/usr/share/perl*/**/*.pod", recursive=True) + glob.glob("/usr/share/texlive/**/*-en.tex", recursive=True) + glob.glob("/usr/share/texlive/**/README*", recursive=True) + glob.glob("/usr/share/texlive/**/*.md", recursive=True):
            try: add(p, open(p, "rb").read())
            except Exception: pass
        for p in glob.glob("/usr/lib/python3*/**/*.py", recursive=True):
            try:
                t = ast.parse(open(p, encoding="utf8").read()); ds = [ast.get_docstring(n) for n in ast.walk(t) if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
                add(p, "\n\n".join(d for d in ds if d).encode("utf8"))
            except Exception: pass
    seen = set(); out = []
    for p, t in items:
        t = t.replace(b"\r", b"")
        h = hashlib.md5(t).digest()
        if h in seen: continue
        seen.add(h); out.append((p, t))
    return out

def cmd_prep(a):
    t0 = time.time(); os.makedirs(a.work, exist_ok=True); W = lambda n: os.path.join(a.work, n)
    items = gather_corpus(a)
    ev, tr = [], []
    for p, t in items: (ev if int(hashlib.md5(p.encode()).hexdigest(), 16) % a.holdout == 0 else tr).append((p, t))
    tr.sort(key=lambda x: hashlib.md5(x[0].encode()).hexdigest())
    pre = b""
    if a.wordlists:  # priming: wordlists go first in the training stream, one word per line, so the CM sees them before prose
        ws = []
        for p in a.wordlists:
            fs = [p] if os.path.isfile(p) else [x for x in glob.glob(os.path.join(p, "**", "*"), recursive=True) if os.path.isfile(x)]
            for x in sorted(fs): ws += [l.strip() for l in open(x, "rb").read().split(b"\n") if 1 < len(l.strip()) < 40]
        pre = b"\n".join(ws[:a.wl_max]) + b"\n\n"
    stream = pre + b"\n\n".join(t for _, t in tr)
    stream = stream[:a.maxb]
    open(W("train.bin"), "wb").write(stream)
    # eval windows from held-out files
    rs = np.random.RandomState(7); wins = []
    evt = [t for _, t in ev if len(t) > a.wlen * 2]
    for i in range(a.nwin):
        t = evt[rs.randint(len(evt))]; o = rs.randint(0, len(t) - a.wlen); wins.append(t[o:o + a.wlen])
    with open(W("win.bin"), "wb") as f:
        f.write(struct.pack("<II", len(wins), a.skip)); [f.write(struct.pack("<I", len(w)) + w) for w in wins]
    # BPE
    pcount = collections.Counter(pretok(stream)); bpe = BPE(train_bpe(pcount, a.merges)); bpe.save(W("bpe.txt"))
    open(W("win.u"), "wb").write(b"".join(unit_flags(w, bpe) for w in wins))
    # LMs are built from the first lmfrac of the training stream (so the mixer meets an LM that is honest on unseen text)
    lmpart = stream[:int(len(stream) * a.lmfrac)]; pts = pretok(lmpart)
    ids = [t for pt in pts for t in bpe.enc_pre(pt)]; V = len(bpe.vocab)
    write_lm_table(W("lm_sub.bin"), bpe.vocab, bigram_matrix(ids, V))
    wv = build_word_vocab(pcount, a.words); wids = encode_words(pts, wv)
    write_lm_table(W("lm_word.bin"), wv, bigram_matrix(wids, len(wv)))
    json.dump(dict(train_bytes=len(stream), files_train=len(tr), files_eval=len(ev), windows=len(wins), skip=a.skip, bpe_vocab=V, word_vocab=len(wv), lm_bytes=len(lmpart), prime_bytes=len(pre)), open(W("prep.json"), "w"))
    print("prep: %d train files (%d B), %d held-out files, %d windows x %d B, BPE V=%d, word V=%d, LM built on %d B, %.0fs" % (len(tr), len(stream), len(ev), len(wins), a.wlen, V, len(wv), len(lmpart), time.time() - t0))

def ensure_bin(a):
    src = os.path.abspath(__file__); exe = os.path.join(a.work, "nancm_core")
    c = os.path.join(a.work, "nancm_core.c"); os.makedirs(a.work, exist_ok=True)
    code = CSRC
    if "@@CORE@@" in code: code = open(os.path.join(os.path.dirname(src), "core.c")).read()
    if not os.path.exists(exe) or not os.path.exists(c) or open(c).read() != code:
        open(c, "w").write(code); subprocess.check_call(["gcc", "-O3", "-march=native", "-o", exe, c, "-lm"], stderr=subprocess.DEVNULL)
    return exe

def run_core(a, flags, maxb=None, label="run", out="res", extra_eval=None):
    exe = ensure_bin(a); W = lambda n: os.path.join(a.work, n); flags = list(flags) + os.environ.get("NANCM_FLAGS", "").split()
    cmd = [exe, "-bpe", W("bpe.txt"), "-eval", W("win.bin"), "-evalu", W("win.u"), "-tshift", str(a.tshift), "-o", W(out), "-maxb", str(maxb or a.maxb)] + flags + [W("train.bin")]
    t0 = time.time(); r = subprocess.run(cmd, capture_output=True, text=True); dt = time.time() - t0
    res = {"label": label, "sec": round(dt, 1), "flags": " ".join(flags)}
    for ln in r.stdout.splitlines():
        m = re.match(r"FINAL (frozen|online) @(\d+) bytes: nats=\S+ bpc=(\S+) top1=(\S+) top5=(\S+)", ln)
        if m: res[m.group(1)] = dict(bpc=float(m.group(3)), top1=float(m.group(4)), top5=float(m.group(5)))
        m = re.match(r"FIRSTMISS (frozen|online) (token|subword|word):(.*)", ln)
        if m: res.setdefault("firstmiss_" + m.group(1), {})[m.group(2)] = [int(x) for x in re.findall(r"=(\d+)", m.group(3))]
        m = re.match(r"UNITS (frozen|online):(.*)", ln)
        if m:
            for k in re.finditer(r"(token|subword|word) n=(\d+) top1=(\S+) top5=(\S+)", m.group(2)): res.setdefault(m.group(1), {})[k.group(1)] = (float(k.group(3)), float(k.group(4)))
    if "frozen" not in res: res["err"] = (r.stderr or r.stdout)[-400:]
    return res

def fmt(res):
    if "err" in res: return "%-14s ERROR %s" % (res["label"], res["err"])
    f, o = res.get("frozen", {}), res.get("online", {})
    u = lambda d, k: "%.3f/%.3f" % d[k] if k in d else "-"
    return "%-14s frozen %.4f t1/t5 %.3f/%.3f | online %.4f t1/t5 %.3f/%.3f | spelled(online) tok %s sub %s word %s | %ss" % (res["label"], f["bpc"], f["top1"], f["top5"], o.get("bpc", 0), o.get("top1", 0), o.get("top5", 0), u(o, "token"), u(o, "subword"), u(o, "word"), res["sec"])

BASE = ["-cm", "7", "-m6", "1", "-a3", "0", "-sm", "1", "-w2b", "1", "-fin", "1", "-lr1", "0.0015", "-lr2", "0.0005", "-lrf", "0.0005", "-apmr", "0.01", "-mx", "0"]
def lmflags(a, sub=True, word=True, gc=1):
    W = lambda n: os.path.join(a.work, n); f = ["-gc", str(gc), "-lmk", str(a.beam)]
    if sub: f += ["-lmsub", W("lm_sub.bin")]
    if word: f += ["-lmword", W("lm_word.bin")]
    return f

def cmd_run(a):
    r = run_core(a, BASE + lmflags(a, not a.nosub, not a.noword, a.gc) + ["-bm", str(a.bm)] + a.flags, label="run"); print(fmt(r)); print(json.dumps(r))

def cmd_ablate(a):
    plan = [("base-noLM", BASE + ["-gc", "0"]), ("sub", BASE + lmflags(a, 1, 0, 0)), ("sub+word", BASE + lmflags(a, 1, 1, 0)), ("sub+word+gc", BASE + lmflags(a, 1, 1, 1)), ("gc-only-word", BASE + lmflags(a, 0, 1, 1)),
            ("+mx3", BASE[:-2] + ["-mx", "1"] + lmflags(a, 1, 1, 1)), ("+fin0", [x for x in BASE] + lmflags(a, 1, 1, 1) + ["-fin", "0"]), ("+sm3", BASE + lmflags(a, 1, 1, 1) + ["-sm", "3"])]
    if a.only: plan = [p for p in plan if p[0] in a.only]
    out = []
    for lab, fl in plan:
        r = run_core(a, fl, maxb=a.maxb, label=lab); print(fmt(r), flush=True); out.append(r)
        open(os.path.join(a.work, "ablate_%s.json" % a.tag), "w").write(json.dumps(out, indent=1))

# ---------------- token->byte reference (python), selftest, sidecar for external LMs
class ByteTrie:
    def __init__(self, tokbytes):
        self.tb = tokbytes; self.V = len(tokbytes); self.order = sorted(range(self.V), key=lambda i: tokbytes[i]); self.sb = [tokbytes[i] for i in self.order]
        self.pos = np.empty(self.V, np.int64); self.pos[self.order] = np.arange(self.V); self.order = np.array(self.order, np.int64)
        self.exact = {tokbytes[i]: i for i in range(self.V)}; self._rng = {}; self._kids = {}
    def rng(self, p):
        r = self._rng.get(p)
        if r is None:
            lo = bisect.bisect_left(self.sb, p); hi = lo
            while hi < self.V and self.sb[hi].startswith(p): hi += 1
            r = self._rng[p] = (lo, hi)
        return r
    def kids(self, p):
        k = self._kids.get(p)
        if k is None:
            lo, hi = self.rng(p); n = len(p); bs, los, his = [], [], []; i = lo
            while i < hi:
                if len(self.sb[i]) == n: i += 1; continue
                b = self.sb[i][n]; j = i
                while j < hi and len(self.sb[j]) > n and self.sb[j][n] == b: j += 1
                bs.append(b); los.append(i); his.append(j); i = j
            k = self._kids[p] = (np.array(bs, np.int64), np.array(los, np.int64), np.array(his, np.int64))
        return k

class Byte_LM:
    """exact causal token->byte conversion for ANY token LM exposing probs(T)->ndarray (T = tuple of token ids). Works for BPE subwords and whole-word vocabularies alike."""
    def __init__(self, lm, tokbytes, beam=8): self.lm, self.trie, self.K = lm, ByteTrie(tokbytes), beam; self._cs = {}
    def _cumsum(self, T):
        c = self._cs.get(T)
        if c is None:
            pr = np.asarray(self.lm.probs(T), np.float64)[self.trie.order]; c = self._cs[T] = (np.concatenate([[0.0], np.cumsum(pr)]), pr)
        return c
    def _score(self, T, p, lw):
        if not p: return lw
        c, _ = self._cumsum(T); lo, hi = self.trie.rng(p); m = c[hi] - c[lo]
        if p in self.trie.exact: j = self.trie.pos[self.trie.exact[p]]; m -= c[j + 1] - c[j]
        return lw + math.log(max(m, 1e-300))
    def dists(self, s):
        hyps = [((), b"", 0.0)]; out = np.zeros((len(s), 256))
        if len(self._cs) > 8192: self._cs = {}
        for i in range(len(s)):
            d = np.zeros(256)
            for T, p, lw in hyps:
                c, pr = self._cumsum(T); bs, lo, hi = self.trie.kids(p)
                if len(bs): d[bs] += math.exp(lw) * (c[hi] - c[lo])
            tot = d.sum(); out[i] = d / tot if tot > 0 else 1.0 / 256; b = s[i]; new = []
            for T, p, lw in hyps:
                q = p + bytes([b]); lo, hi = self.trie.rng(q)
                if hi == lo: continue
                if q in self.trie.exact:
                    tid = self.trie.exact[q]; _, pr = self._cumsum(T); pt = pr[self.trie.pos[tid]]
                    if pt > 0: new.append((T + (tid,), b"", lw + math.log(pt)))
                    if hi - lo > 1: new.append((T, q, lw))
                else: new.append((T, q, lw))
            if not new: new = [((), b"", 0.0)]
            if len(new) > self.K: new.sort(key=lambda h: -self._score(*h)); new = new[:self.K]
            hyps = new
        return out

def node_probs(D):
    D = np.atleast_2d(D); n = D.shape[0]; lv = [None] * 9; lv[8] = D
    for k in range(7, -1, -1): lv[k] = lv[k + 1].reshape(n, -1, 2).sum(2)
    P = np.full((n, 256), 0.5)
    for d in range(8):
        den = lv[d]; num = lv[d + 1][:, 1::2]; P[:, (1 << d):(2 << d)] = np.where(den > 0, num / np.maximum(den, 1e-300), 0.5)
    return P

def write_sidecar(path, mode, rows):
    parts = []; n = 0
    for chunk, D in rows:
        if mode == 1: parts.append(D.astype(np.float16).tobytes())
        else:
            P = node_probs(D); a = np.frombuffer(chunk, np.uint8).astype(np.int64); q = np.empty((len(a), 8), np.uint16)
            for bp in range(8): c0 = (1 << bp) | (a >> (8 - bp)); q[:, bp] = np.clip(P[np.arange(len(a)), c0] * 65536, 16, 65520).astype(np.uint16)
            parts.append(q.tobytes())
        n += len(chunk)
    with open(path, "wb") as f:
        f.write(b"LMB1" + struct.pack("<IQI", mode, n, 0)); [f.write(x) for x in parts]
    return n

def read_windows(path):
    f = open(path, "rb").read(); nw, skip = struct.unpack("<II", f[:8]); o = 8; ws = []
    for _ in range(nw): (l,) = struct.unpack("<I", f[o:o + 4]); o += 4; ws.append(f[o:o + l]); o += l
    return ws, skip

def brute_dist(lm, tokbytes, s):
    V = len(tokbytes); memo = {}
    def G(T, r):
        if not r: return 1.0
        key = (T, r)
        if key in memo: return memo[key]
        pr = lm.probs(T); tot = 0.0
        for t in range(V):
            tb = tokbytes[t]
            if r.startswith(tb): tot += pr[t] * G(T + (t,), r[len(tb):]) if len(tb) < len(r) else pr[t]
            elif tb.startswith(r): tot += pr[t]
        memo[key] = tot; return tot
    q = np.array([G((), s + bytes([b])) for b in range(256)]); return q / q.sum()

class ToyLM:
    def __init__(self, V, seed=0): r = np.random.RandomState(seed); self.M = r.dirichlet(np.ones(V) * 0.4, size=V + 1); self.V = V
    def probs(self, T): return self.M[T[-1] if T else self.V]

class TableLM:  # python view of an lm_*.bin table (bigram on last token)
    def __init__(self, path):
        f = open(path, "rb").read(); V = struct.unpack("<i", f[4:8])[0]; o = 8; self.tb = []
        for _ in range(V): (n,) = struct.unpack("<H", f[o:o + 2]); o += 2; self.tb.append(f[o:o + n]); o += n
        self.M = np.frombuffer(f[o:], np.float32).reshape(V + 1, V).astype(np.float64); self.V = V
    def probs(self, T): return self.M[T[-1] if T else self.V]

def cmd_selftest(a):
    tb = [bytes([c]) for c in b"abc"] + [b"ab", b"bc", b"abc", b"ca", b"aa", b"cab"]; seen = {}; tb2 = []
    for t in tb:
        if t not in seen: seen[t] = len(tb2); tb2.append(t)
    for c in range(256):
        if bytes([c]) not in seen: seen[bytes([c])] = len(tb2); tb2.append(bytes([c]))
    lm = ToyLM(len(tb2), seed=3)
    for i in range(lm.M.shape[0]): w = np.ones(len(tb2)) * 1e-4; w[:9] = 1.0; lm.M[i] = lm.M[i] * w; lm.M[i] /= lm.M[i].sum()
    bl = Byte_LM(lm, tb2, beam=10000); worst = 0.0
    for s in [b"", b"a", b"ab", b"abc", b"abca", b"cabab", b"bcaab", b"abcabcab", b"aabbcc"]:
        worst = max(worst, np.abs(bl.dists(s + b"a")[len(s)] - brute_dist(lm, tb2, s)).max())
    print("[1] python converter vs brute force (toy vocab, unpruned): max err %.2e" % worst); assert worst < 1e-9
    W = lambda n: os.path.join(a.work, n)
    if not os.path.exists(W("lm_sub.bin")): print("run `prep` first for checks [2],[3]"); return
    exe = ensure_bin(a); ws, _ = read_windows(W("win.bin")); sample = b"".join(ws[:4])[:600]; open(W("chk.bin"), "wb").write(sample)
    for idx, nm in ((0, "subword"), (1, "word")):
        lmp = W("lm_sub.bin" if idx == 0 else "lm_word.bin")
        def cdump(K):
            subprocess.check_call([exe, "-lmsub" if idx == 0 else "-lmword", lmp, "-lmk", str(K), "-lmcheck", str(idx), W("chk.bin"), str(len(sample)), W("chk_%d.f64" % K)])
            return np.fromfile(W("chk_%d.f64" % K), np.float64).reshape(-1, 256)
        ref = cdump(128); tl = TableLM(lmp); py = Byte_LM(tl, tl.tb, beam=400).dists(sample[:200])
        print("[2] C engine (%s, K=128) vs python reference (K=400) on 200 bytes: max |diff| = %.2e" % (nm, np.abs(ref[:200] - py).max()))
        for K in (2, 4, 8, 16, 32):
            d = cdump(K); kl = np.sum(ref * (np.log(ref + 1e-12) - np.log(d + 1e-12)), 1)
            print("[3] %-7s beam %2d vs 128: mean KL %.5f nats/byte, max |dp| %.4f" % (nm, K, kl.mean(), np.abs(ref - d).max()))

def cmd_sidecar(a):  # external token LM (e.g. hf:gpt2) -> sidecars for -lmtr/-lmev. UNTESTED with real weights in the build sandbox.
    W = lambda n: os.path.join(a.work, n); import_hf = __import__
    from_hf = a.lm.startswith("hf:")
    if not from_hf: raise SystemExit("only hf:<model> supported here")
    lm = HFLM(a.lm[3:]); bl = Byte_LM(lm, lm.tokbytes, beam=a.beam)
    if a.what == "eval": ws, _ = read_windows(W("win.bin")); n = write_sidecar(a.out, 1, ((w, bl.dists(w)) for w in ws))
    else:
        seg = open(W("train.bin"), "rb").read()[a.start:a.start + a.len]; n = write_sidecar(a.out, 0, ((c, bl.dists(c)) for c in (seg[i:i + 420] for i in range(0, len(seg), 420))))
    print("wrote", a.out, n, "bytes of LM probs")

class HFLM:
    """GPT-2-style byte-level BPE HF model -> probs(T). Llama/SentencePiece vocabularies need a different token->bytes map (UNTESTED)."""
    def __init__(self, name, device=None):
        import torch; from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch; device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(name); self.dev = device
        self.m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16 if device != "cpu" else torch.float32).to(device).eval()
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)); cs = bs[:]; n = 0
        for b in range(256):
            if b not in bs: bs.append(b); cs.append(256 + n); n += 1
        u2b = {chr(c): b for b, c in zip(bs, cs)}; V = len(self.tok); inv = {i: t for t, i in self.tok.get_vocab().items()}; sp = set(self.tok.all_special_ids)
        self.V = V; self.mask = np.ones(V); self.tokbytes = []
        for i in range(V):
            t = inv.get(i, "")
            try: tb = bytes(u2b[ch] for ch in t); ok = len(tb) > 0 and i not in sp
            except KeyError: ok = False
            if not ok: tb = b"\x00SPECIAL%d\x00" % i; self.mask[i] = 0
            self.tokbytes.append(tb)
        self.bos = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id; self.cache = {}
    def probs(self, T):
        torch = self.torch
        with torch.no_grad():
            if T in self.cache: return self.cache[T][0]
            par = self.cache.get(T[:-1]) if T else None
            if par is not None and par[1] is not None: out = self.m(torch.tensor([[T[-1]]], device=self.dev), past_key_values=par[1], use_cache=True)
            else: out = self.m(torch.tensor([[self.bos] + list(T)], device=self.dev), use_cache=True)
            p = torch.softmax(out.logits[0, -1, :self.V].float(), -1).cpu().numpy().astype(np.float64) * self.mask; p /= p.sum()
            self.cache[T] = (p, out.past_key_values)
            if len(self.cache) > 2048: self.cache.pop(next(iter(self.cache)))
            return p

def main():
    ap = argparse.ArgumentParser(description="NaNCM single-file system (bytewise CM + exact token/subword/word->byte LMs)")
    ap.add_argument("cmd", choices=["prep", "run", "ablate", "selftest", "sidecar"]); ap.add_argument("--work", default="nancm_work")
    ap.add_argument("--corpus", nargs="*"); ap.add_argument("--wordlists", nargs="*"); ap.add_argument("--wl-max", dest="wl_max", type=int, default=400000)
    ap.add_argument("--maxb", type=int, default=3000000); ap.add_argument("--holdout", type=int, default=8); ap.add_argument("--nwin", type=int, default=60); ap.add_argument("--wlen", type=int, default=512); ap.add_argument("--skip", type=int, default=32)
    ap.add_argument("--merges", type=int, default=1500); ap.add_argument("--words", type=int, default=3000); ap.add_argument("--lmfrac", type=float, default=0.6); ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--tshift", type=int, default=0); ap.add_argument("--gc", type=int, default=0); ap.add_argument("--bm", type=int, default=0); ap.add_argument("--nosub", action="store_true"); ap.add_argument("--noword", action="store_true")
    ap.add_argument("--flags", nargs="*", default=[]); ap.add_argument("--only", nargs="*"); ap.add_argument("--tag", default="a")
    ap.add_argument("--what", default="eval"); ap.add_argument("--lm", default="hf:gpt2"); ap.add_argument("--out", default="lm.lmb"); ap.add_argument("--start", type=int, default=0); ap.add_argument("--len", type=int, default=1000000)
    a = ap.parse_args(); {"prep": cmd_prep, "run": cmd_run, "ablate": cmd_ablate, "selftest": cmd_selftest, "sidecar": cmd_sidecar}[a.cmd](a)

if __name__ == "__main__": main()
