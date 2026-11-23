#ifndef NANCM_GEOM256_H
#define NANCM_GEOM256_H
/* v136: 128 top MoE nodes, 128 bottom eight-dimensional intake nodes.
   Pair addresses: C(256,2)=32640, two independent oriented signed rails.
   This C/C++ shared source is compiled into both training and native inference. */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#define GE_TOP 128
#define GE_BOTTOM 128
#define GE_POINTS 256
#define GE_EDGES 32640
#define GE_D 8
#define GE_BUCKETS 8192
#define GE_MAXEVENT 64
#define GE_MAXB 48
#define GE_MAXT 128
#define GE_SCALE 6.f
#define GE_VERSION 136u
#define GE_MAGIC "G256PAIR"

typedef struct {
  float embed[GE_BUCKETS][GE_D];
  float intake[GE_BOTTOM][GE_D];
  float plus[GE_EDGES];
  float minus[GE_EDGES];
  float topbias[GE_TOP];
  float anchor[GE_TOP];
  float expert[GE_TOP][26];
  float outbias[26];
} GeModel;
typedef struct {
  int num,idx[GE_MAXEVENT],b[GE_MAXEVENT];
  float val[GE_MAXEVENT];
} GeEvents;
typedef struct {
  float x[GE_BOTTOM][GE_D], b0[GE_BOTTOM], b1[GE_BOTTOM], bf[GE_BOTTOM];
  float t0[GE_TOP], t1[GE_TOP], t2[GE_TOP],gate[GE_TOP];
  float prob[26];
  int nb,bi[GE_MAXB],nt,ti[GE_MAXT];
} GeCache;

static inline int ge_pair(int a,int b){
  if(a>b){int t=a;a=b;b=t;}
  if(a==b||a<0||b>255)return -1;
  return a*(511-a)/2+(b-a-1);
}
static inline uint32_t ge_hash(const unsigned char *p, size_t n, uint32_t seed){
  uint32_t h=2166136261u ^ (seed*0x9E3779B9u);
  for(size_t i=0;i<n;i++){h^=p[i];h*=16777619u;}
  h ^=h>>16;h*=0x7feb352du;h^=h>>15;h*=0x846ca68bu;h^=h>>16; return h;
}
static inline int ge_word(unsigned char c){return(c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>='0'&&c<='9')||c>=128;}
static inline void ge_add(GeEvents *e,uint32_t h,float scale){
  if(e->num>=GE_MAXEVENT)return;
  int n=e->num++,idx=(int)(h&(GE_BUCKETS-1));
  e->idx[n]=idx;e->b[n]=(int)((h>>13)&127);
  e->val[n]=((h>>24)&1)?-scale:scale;
}
static inline void ge_events(const unsigned char *p,size_t n,GeEvents *ev){
  memset(ev,0,sizeof(*ev));
  /* All features use observed bytes ONLY; no target or future prefix. */
  static const int lens[]={2,4,7,11,17,25,41,65,97,161};
  for(int j=0;j<10;j++)if(n>=(size_t)lens[j])ge_add(ev,ge_hash(p+n-lens[j],lens[j],101u+j),1.f);
  /* Previous completed words, skip whitespace and punctuation between words. */
  size_t z=n;uint32_t wordh[5]={0};int nw=0;
  while(z>0 && nw<5){
    while(z>0&&!ge_word(p[z-1]))z--;
    if(!z)break;
    size_t end=z;while(z>0&&ge_word(p[z-1]))z--;
    size_t L=end-z;if(L>64){z=end-64;L=64;}
    wordh[nw]=ge_hash(p+z,L,200+nw);
    ge_add(ev,wordh[nw],1.4f/(1+.2f*nw));
    if(L>=3)ge_add(ev,ge_hash(p+end-3,3,220+nw),.65f);
    if(L>=5)ge_add(ev,ge_hash(p+end-5,5,240+nw),.45f);
    nw++;
  }
  if(nw>=2){uint32_t v[2]={wordh[0],wordh[1]};ge_add(ev,ge_hash((const unsigned char*)v,sizeof(v),311),1.2f);}
  if(nw>=3){uint32_t v[3]={wordh[0],wordh[1],wordh[2]};ge_add(ev,ge_hash((const unsigned char*)v,sizeof(v),312),1.25f);}
  if(nw>=4){uint32_t v[4]={wordh[0],wordh[1],wordh[2],wordh[3]};ge_add(ev,ge_hash((const unsigned char*)v,sizeof(v),313),1.0f);}
  /* Layout features: row position, final delimiter and local word cardinality. */
  int col=0;for(size_t j=n;j>0&&col<160;j--){if(p[j-1]=='\n')break;col++;}
  unsigned char layout[4]={(unsigned char)(col/8),(unsigned char)(col%8),(unsigned char)(n?p[n-1]:0),(unsigned char)nw};
  ge_add(ev,ge_hash(layout,4,601),.8f);
  if(n)ge_add(ev,ge_hash(p+n-1,1,602),.5f);
}
static inline float ge_tanh(float x){return tanhf(x);}
static inline void ge_init(GeModel *m){
  memset(m,0,sizeof(*m));uint32_t s=0x7b24acf1u;
  /* Deterministic sparse-friendly initialization, no pretraining leakage. */
  for(int k=0;k<GE_BUCKETS;k++)for(int d=0;d<GE_D;d++){
    s^=s<<13;s^=s>>17;s^=s<<5;m->embed[k][d]=((int)(s&65535)-32768)/327680.f;
  }
  for(int b=0;b<GE_BOTTOM;b++)for(int d=0;d<GE_D;d++){
    s^=s<<13;s^=s>>17;s^=s<<5;m->intake[b][d]=((int)(s&65535)-32768)/81920.f;
  }
  for(int e=0;e<GE_EDGES;e++){
    s^=s<<13;s^=s>>17;s^=s<<5;m->plus[e]=((int)(s&65535)-32768)/327680.f;
    s^=s<<13;s^=s>>17;s^=s<<5;m->minus[e]=((int)(s&65535)-32768)/327680.f;
  }
  for(int t=0;t<GE_TOP;t++){
    m->anchor[t]=.40f;
    for(int c=0;c<26;c++){
      s^=s<<13;s^=s>>17;s^=s<<5;m->expert[t][c]=((int)(s&65535)-32768)/655360.f;
    }
  }
}
static inline void ge_forward(const GeModel *m,const GeEvents *ev,GeCache *q,int edges,int feedback){
  memset(q,0,sizeof(*q));int seenb[128]={0},seent[128]={0};
  for(int i=0;i<ev->num;i++){
    int b=ev->b[i],id=ev->idx[i];float w=ev->val[i];
    if(!seenb[b] && q->nb<GE_MAXB){seenb[b]=1;q->bi[q->nb++]=b;}
    for(int d=0;d<GE_D;d++)q->x[b][d]+=w*m->embed[id][d];
  }
  for(int i=0;i<q->nb;i++){
    int b=q->bi[i];float z=0;
    for(int d=0;d<GE_D;d++)z+=q->x[b][d]*m->intake[b][d];
    q->b0[b]=ge_tanh(z);
    int candidates[]={b,(b*73+19)&127,(b*29+47)&127};
    for(int j=0;j<3;j++){int t=candidates[j];if(!seent[t]){seent[t]=1;q->ti[q->nt++]=t;}}
  }
  /* lower/lower abstraction: multidimensional intake is refined by shared concepts. */
  for(int i=0;i<q->nb;i++){
    int b=q->bi[i];float z=q->b0[b];
    if(edges)for(int j=0;j<q->nb;j++){int a=q->bi[j];if(a!=b)z+=.08f*q->b0[a]*(a<b?m->plus[ge_pair(128+a,128+b)]:m->minus[ge_pair(128+a,128+b)]);}
    q->b1[b]=ge_tanh(z);
  }
  /* up path: all 128 top experts have their own lower point anchor. */
  for(int i=0;i<q->nt;i++){
    int t=q->ti[i];float z=m->topbias[t]+m->anchor[t]*q->b1[t];
    if(edges)for(int j=0;j<q->nb;j++){int b=q->bi[j];z+=q->b1[b]*m->minus[ge_pair(t,128+b)];}
    q->t0[t]=ge_tanh(z);
  }
  /* Top/top expert cooperation is an actual sparse pair pass. */
  for(int i=0;i<q->nt;i++){
    int t=q->ti[i];float z=q->t0[t];
    if(edges)for(int j=0;j<q->nt;j++){int a=q->ti[j];if(a!=t)z+=.10f*q->t0[a]*(a<t?m->plus[ge_pair(a,t)]:m->minus[ge_pair(a,t)]);}
    q->t1[t]=ge_tanh(z);
  }
  /* Downward feedback traverses S+; second intake ascends through S-. */
  for(int i=0;i<q->nb;i++){
    int b=q->bi[i];float z=q->b1[b];
    if(edges&&feedback)for(int j=0;j<q->nt;j++){int t=q->ti[j];z+=.25f*q->t1[t]*m->plus[ge_pair(t,128+b)];}
    q->bf[b]=ge_tanh(z);
  }
  float vmax=-1e20f,gs=0;
  for(int i=0;i<q->nt;i++){
    int t=q->ti[i];float z=q->t1[t];
    if(edges&&feedback)for(int j=0;j<q->nb;j++){int b=q->bi[j];z+=.50f*q->bf[b]*m->minus[ge_pair(t,128+b)];}
    q->t2[t]=ge_tanh(z);q->gate[t]=expf(2*q->t2[t]);gs+=q->gate[t];
  }
  if(gs<1e-20f)gs=1.f;
  for(int i=0;i<q->nt;i++){int t=q->ti[i];q->gate[t]/=gs;}
  float zsum=0;
  for(int c=0;c<26;c++){
    float logit=m->outbias[c];
    for(int i=0;i<q->nt;i++){int t=q->ti[i];logit+=GE_SCALE*q->gate[t]*m->expert[t][c];}
    q->prob[c]=logit;if(logit>vmax)vmax=logit;
  }
  for(int c=0;c<26;c++){q->prob[c]=expf(q->prob[c]-vmax);zsum+=q->prob[c];}
  for(int c=0;c<26;c++)q->prob[c]/=zsum;
}
static inline int ge_save(const char *path,const GeModel *m,const GeModel *opt,uint64_t step){
  FILE*f=fopen(path,"wb");if(!f)return -1;
  uint32_t h[6]={GE_VERSION,GE_POINTS,GE_EDGES,GE_D,GE_BUCKETS,(uint32_t)sizeof(*m)};
  int ok=fwrite(GE_MAGIC,1,8,f)==8&&fwrite(h,4,6,f)==6&&fwrite(&step,8,1,f)==1&&fwrite(m,sizeof(*m),1,f)==1;
  if(opt)ok=ok&&fwrite(opt,sizeof(*opt),1,f)==1;
  if(fclose(f))ok=0;return ok?0:-1;
}
static inline int ge_load(const char *path,GeModel *m,GeModel *opt,uint64_t *step){
  FILE*f=fopen(path,"rb");if(!f)return -1;char magic[8];uint32_t h[6];uint64_t s;
  int ok=fread(magic,1,8,f)==8 && !memcmp(magic,GE_MAGIC,8) && fread(h,4,6,f)==6 &&
    h[0]==GE_VERSION&&h[1]==GE_POINTS&&h[2]==GE_EDGES&&h[3]==GE_D&&h[4]==GE_BUCKETS&&h[5]==sizeof(*m) &&
    fread(&s,8,1,f)==1&&fread(m,sizeof(*m),1,f)==1;
  if(opt)ok=ok && fread(opt,sizeof(*opt),1,f)==1;
  if(ok && fgetc(f)!=EOF)ok=0;
  fclose(f);if(ok&&step)*step=s;return ok?0:-1;
}
#endif
