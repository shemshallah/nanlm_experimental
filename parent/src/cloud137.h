#ifndef NANCM_CLOUD137_H
#define NANCM_CLOUD137_H
/* v137: 8D signed concept cloud. Exactly C(256,2)*2 addressable 8D
   vectors; 128 lower points provide 8D input to 128 upper MoE points.
   The old v136 GE pair scalars and geometrical checkpoint remain intact. */
#include "geom256.h"
#define CLOUD_D GE_D
#define CLOUD_VERSION 137u
#define CLOUD_MAGIC "C137D8PM"
#define CL_LOW 0.13f
#define CL_TOP 0.17f
#define CL_UP 1.2f
#define CL_DOWN 0.45f
#define CL_UP2 0.80f
#define CL_HEAD 20.f

typedef struct {
  float plus[GE_EDGES][CLOUD_D];
  float minus[GE_EDGES][CLOUD_D];
  float bscale[GE_BOTTOM][CLOUD_D];
  float tbias[GE_TOP][CLOUD_D];
  float head[GE_TOP][CLOUD_D][26];
  float bias[26];
} Cloud137;

typedef struct {
  GeCache g;
  float h0[GE_BOTTOM][CLOUD_D],h1[GE_BOTTOM][CLOUD_D];
  float u0[GE_TOP][CLOUD_D],u1[GE_TOP][CLOUD_D],r[GE_BOTTOM][CLOUD_D],u2[GE_TOP][CLOUD_D];
  float prob[26];
} CloudCache;

static inline void cloud_init(Cloud137 *m){
  memset(m,0,sizeof(*m));uint32_t s=0x137d0c1du;
  for(int e=0;e<GE_EDGES;e++)for(int d=0;d<CLOUD_D;d++){
    s^=s<<13;s^=s>>17;s^=s<<5;
    m->plus[e][d]=((int)(s&65535)-32768)/262144.f;
    s^=s<<13;s^=s>>17;s^=s<<5;
    m->minus[e][d]=((int)(s&65535)-32768)/262144.f;
  }
  for(int b=0;b<GE_BOTTOM;b++)for(int d=0;d<CLOUD_D;d++)m->bscale[b][d]=1.f;
  for(int t=0;t<GE_TOP;t++)for(int d=0;d<CLOUD_D;d++){
    s^=s<<13;s^=s>>17;s^=s<<5;
    m->tbias[t][d]=((int)(s&65535)-32768)/262144.f;
    for(int c=0;c<26;c++){
      s^=s<<13;s^=s>>17;s^=s<<5;
      m->head[t][d][c]=((int)(s&65535)-32768)/524288.f;
    }
  }
}

/* Choose oriented channels explicitly, including upper-upper/lower-lower. */
static inline float cloud_oriented(const Cloud137*m,int a,int b,int d){
  int k=ge_pair(a,b);return a<b?m->plus[k][d]:m->minus[k][d];
}

static inline void cloud_forward(const GeModel*base,const Cloud137*m,const GeEvents*ev,CloudCache*q,int edges,int feedback){
  memset(q,0,sizeof(*q));ge_forward(base,ev,&q->g,1,1);
  const GeCache*g=&q->g;
  float nB=1.f/sqrtf((float)(g->nb?g->nb:1));
  float nT=1.f/sqrtf((float)(g->nt?g->nt:1));
  for(int i=0;i<g->nb;i++){
    int b=g->bi[i];for(int d=0;d<CLOUD_D;d++)
      q->h0[b][d]=tanhf(g->x[b][d]*m->bscale[b][d]+0.5f*g->bf[b]*base->intake[b][d]);
  }
  for(int i=0;i<g->nb;i++){
    int b=g->bi[i];for(int d=0;d<CLOUD_D;d++){
      float z=q->h0[b][d];
      if(edges)for(int j=0;j<g->nb;j++){
        int a=g->bi[j];if(a!=b)z+=CL_LOW*nB*q->h0[a][d]*cloud_oriented(m,128+a,128+b,d);
      }
      q->h1[b][d]=tanhf(z);
    }
  }
  for(int i=0;i<g->nt;i++){
    int t=g->ti[i];for(int d=0;d<CLOUD_D;d++){
      float z=m->tbias[t][d];
      if(edges)for(int j=0;j<g->nb;j++){
        int b=g->bi[j],k=ge_pair(t,128+b);z+=CL_UP*nB*q->h1[b][d]*m->plus[k][d];
      }
      q->u0[t][d]=tanhf(z);
    }
  }
  for(int i=0;i<g->nt;i++){
    int t=g->ti[i];for(int d=0;d<CLOUD_D;d++){
      float z=q->u0[t][d];
      if(edges)for(int j=0;j<g->nt;j++){
        int a=g->ti[j];if(a!=t)z+=CL_TOP*nT*q->u0[a][d]*cloud_oriented(m,a,t,d);
      }
      q->u1[t][d]=tanhf(z);
    }
  }
  for(int i=0;i<g->nb;i++){
    int b=g->bi[i];for(int d=0;d<CLOUD_D;d++){
      float z=q->h1[b][d];
      if(edges&&feedback)for(int j=0;j<g->nt;j++){
        int t=g->ti[j],k=ge_pair(t,128+b);z+=CL_DOWN*nT*q->u1[t][d]*m->minus[k][d];
      }
      q->r[b][d]=tanhf(z);
    }
  }
  for(int i=0;i<g->nt;i++){
    int t=g->ti[i];for(int d=0;d<CLOUD_D;d++){
      float z=q->u1[t][d];
      if(edges&&feedback)for(int j=0;j<g->nb;j++){
        int b=g->bi[j],k=ge_pair(t,128+b);z+=CL_UP2*nB*q->r[b][d]*m->plus[k][d];
      }
      q->u2[t][d]=tanhf(z);
    }
  }
  float vmax=-1e30f,denom=0.f;
  for(int c=0;c<26;c++){
    float z=logf(fmaxf(1e-9f,g->prob[c]))+m->bias[c];
    for(int i=0;i<g->nt;i++){
      int t=g->ti[i];float v=0;
      for(int d=0;d<CLOUD_D;d++)v+=q->u2[t][d]*m->head[t][d][c];
      z+=CL_HEAD*g->gate[t]*v;
    }
    q->prob[c]=z;if(z>vmax)vmax=z;
  }
  for(int c=0;c<26;c++){q->prob[c]=expf(q->prob[c]-vmax);denom+=q->prob[c];}
  if(denom<=0.f)denom=1.f;
  for(int c=0;c<26;c++)q->prob[c]/=denom;
}

static inline int cloud_save(const char*path,const Cloud137*m,const Cloud137*opt,uint64_t step){
  FILE*f=fopen(path,"wb");if(!f)return -1;
  uint32_t meta[5]={CLOUD_VERSION,GE_POINTS,GE_EDGES,CLOUD_D,(uint32_t)sizeof(*m)};
  int ok=fwrite(CLOUD_MAGIC,1,8,f)==8 && fwrite(meta,4,5,f)==5 &&
    fwrite(&step,8,1,f)==1 && fwrite(m,sizeof(*m),1,f)==1 &&
    (!opt || fwrite(opt,sizeof(*opt),1,f)==1);
  if(fclose(f))ok=0;return ok?0:-1;
}
static inline int cloud_load(const char*path,Cloud137*m,Cloud137*opt,uint64_t*step){
  FILE*f=fopen(path,"rb");if(!f)return -1;
  char magic[8];uint32_t meta[5];uint64_t st=0;
  int ok=fread(magic,1,8,f)==8&&!memcmp(magic,CLOUD_MAGIC,8)&&
    fread(meta,4,5,f)==5&&meta[0]==CLOUD_VERSION&&meta[1]==GE_POINTS&&
    meta[2]==GE_EDGES&&meta[3]==CLOUD_D&&meta[4]==sizeof(*m)&&
    fread(&st,8,1,f)==1&&fread(m,sizeof(*m),1,f)==1&&
    (!opt||fread(opt,sizeof(*opt),1,f)==1)&&fgetc(f)==EOF;
  fclose(f);if(ok&&step)*step=st;return ok?0:-1;
}
#endif
