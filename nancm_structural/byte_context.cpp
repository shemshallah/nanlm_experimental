// v16 bounded native binary-context model, corpus-count fitting only.
// No donor code copied: PAQ-inspired causal binary trees and hash backoff.
#include <array>
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <new>
using U=uint64_t;
static constexpr int K=10, BITS=18, SIZE=1<<BITS;
static constexpr int orders[K]={0,1,2,3,4,6,8,12,16,24};
struct Cell { U key=0; uint32_t a=0,b=0; };
static U hash(U x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}
struct Model{
 std::array<std::vector<Cell>,K> t;
 Model(){for(auto &v:t)v.resize(SIZE);}
 void contexts(const uint8_t*h,int n,U *keys) const{
   U k=0x9e3779b97f4a7c15ULL;int j=0;
   keys[0]=hash(k);
   for(int i=1;i<=24;++i){
     k=hash(k^U(i<=n?h[n-i]+1:257));
     if(j+1<K&&i==orders[j+1])keys[++j]=k;
   }
 }
 U id(U k,int node)const{return hash(k^U(node)*0x9e3779b97f4a7c15ULL)|1ULL;}
 const Cell* get(int j,U k)const{
   size_t i=k&(SIZE-1),q=i^1;auto &v=t[j];
   if(v[i].key==k)return &v[i];if(v[q].key==k)return &v[q];return nullptr;
 }
 Cell& insert(int j,U k){
   size_t i=k&(SIZE-1),q=i^1;auto&v=t[j];
   if(v[i].key==k)return v[i];if(v[q].key==k)return v[q];
   size_t s=(U(v[i].a)+v[i].b<=U(v[q].a)+v[q].b)?i:q;
   v[s]=Cell{k,0,0};return v[s];
 }
 void fit(const uint8_t*d,int n){
   for(int i=0;i<n;++i){
     U keys[K];contexts(d,i,keys);int node=1;
     for(int bit=7;bit>=0;--bit){
       int y=(d[i]>>bit)&1;
       for(int j=0;j<K;++j){auto &c=insert(j,id(keys[j],node));
         if(U(c.a)+c.b>=65536){c.a=(c.a+1)/2;c.b=(c.b+1)/2;}
         if(y)++c.b;else ++c.a;
       }node=2*node+y;
     }
   }
 }
 void predict(const uint8_t*h,int n,double*out)const{
   U keys[K];contexts(h,n,keys);double p[512]={};p[1]=1.;
   for(int node=1;node<256;++node){
     double z=.5;
     for(int j=0;j<K;++j){auto c=get(j,id(keys[j],node));if(c){
       double count=double(c->a)+c->b;
       double strength=(j==0?2.:12.+2*j);
       z=(double(c->b)+strength*z)/(count+strength);
     }}
     z=std::clamp(z,1e-8,1.-1e-8);
     p[2*node]=p[node]*(1-z);p[2*node+1]=p[node]*z;
   }
   for(int i=0;i<256;++i)out[i]=p[256+i];
 }
};
extern "C" {
 void* cm_new(){try{return new Model();}catch(...){return nullptr;}}
 void cm_free(void*p){delete static_cast<Model*>(p);}
 void cm_fit(void*p,const uint8_t*d,int n){if(p&&n>=0)static_cast<Model*>(p)->fit(d,n);}
 void cm_predict(void*p,const uint8_t*h,int n,double*out){if(p&&n>=0)static_cast<Model*>(p)->predict(h,n,out);}
}
