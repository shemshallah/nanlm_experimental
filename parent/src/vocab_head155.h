// NaNCM v155: full-vocabulary C++ inference for the trained causal signed-cloud head.
// Include after wordfield149.cpp/softbridge154.cpp, which define Model/hbytes.
#pragma once
#include <cmath>
#include <fstream>
#include <vector>
#include <array>
#include <stdexcept>
#include <cstdint>
#include <algorithm>

struct VocabHead155 {
 static constexpr int V=5296, E=40, D=32;
 std::vector<float> emb, key, val, query, ln1g,ln1b,up1,up1b,up2,up2b,
                   ln2g,ln2b,res1,res1b,res2,res2b,out,outb,recency;
 static void read(std::ifstream& f,std::vector<float>& a,size_t n) {
   a.resize(n);f.read(reinterpret_cast<char*>(a.data()),n*4);
   if(!f)throw std::runtime_error("v155 head truncated");
   for(float x:a)if(!std::isfinite(x))throw std::runtime_error("nonfinite v155 head parameter");
 }
 explicit VocabHead155(const std::string&path,const Model&m) {
   std::ifstream f(path,std::ios::binary);char magic[8]={};f.read(magic,8);
   if(!f||memcmp(magic,"N155VOCB",8))throw std::runtime_error("v155 head magic mismatch");
   auto vers=ru32(f),voc=ru32(f);auto tr=ru64(f),ge=ru64(f),cl=ru64(f);auto count=ru32(f);
   if(vers!=155||voc!=V||count!=19||tr!=m.trainhash||ge!=m.geomhash||cl!=m.cloudhash||m.words.size()!=V)
     throw std::runtime_error("v155 head/lexical/cloud fingerprint mismatch");
   read(f,emb,5297*40);read(f,key,40*40);read(f,val,40*40);read(f,query,40*32);
   read(f,ln1g,232);read(f,ln1b,232);read(f,up1,320*232);read(f,up1b,320);
   read(f,up2,160*320);read(f,up2b,160);
   read(f,ln2g,160);read(f,ln2b,160);read(f,res1,160*160);read(f,res1b,160);
   read(f,res2,160*160);read(f,res2b,160);read(f,out,5296*160);read(f,outb,5296);read(f,recency,4);
   if(f.peek()!=EOF)throw std::runtime_error("v155 head trailing bytes");
 }
 static std::vector<float> dense(const std::vector<float>&in,const std::vector<float>&w,const std::vector<float>&bias,int output){
    int inDim=(int)in.size();std::vector<float>z(output);
    for(int j=0;j<output;j++){float a=bias[j];const float*wr=&w[(size_t)j*inDim];
      for(int k=0;k<inDim;k++)a+=wr[k]*in[k];z[j]=a;}return z;
 }
 static std::vector<float> linear(const std::vector<float>&in,const std::vector<float>&w,int output){
   int dim=(int)in.size();std::vector<float>z(output);
   for(int j=0;j<output;j++){float a=0;for(int k=0;k<dim;k++)a+=w[(size_t)j*dim+k]*in[k];z[j]=a;}return z;
 }
 static void layernorm(std::vector<float>&x,const std::vector<float>&gamma,const std::vector<float>&bias){
   double avg=0,sq=0;for(float a:x)avg+=a;avg/=x.size();for(float a:x)sq+=(a-avg)*(a-avg);sq/=x.size();
   float inv=1.f/std::sqrt(float(sq)+1.e-5f);for(size_t i=0;i<x.size();i++)x[i]=(x[i]-float(avg))*inv*gamma[i]+bias[i];
 }
 static void gelu(std::vector<float>&v){for(float&x:v)x=.5f*x*(1.f+std::erf(x*0.7071067811865475f));}
 std::vector<float> forward(const std::array<float,32>&z,const std::array<int,4>&prev)const {
   auto q=linear(std::vector<float>(z.begin(),z.end()),query,40);
   std::vector<float>words(160),att(4),states(40);
   for(int i=0;i<4;i++){
     int id=prev[i];if(id<0||id>V)throw std::runtime_error("v155 invalid causal previous id");
     for(int d=0;d<40;d++)words[40*i+d]=emb[id*40+d];
     float dot=0;
     for(int j=0;j<40;j++){float a=0;for(int d=0;d<40;d++)a+=key[j*40+d]*words[40*i+d];dot+=a*q[j];}
     att[i]=id==0?-1.e4f:dot/std::sqrt(40.f)+recency[i];
   }
   float mx=*std::max_element(att.begin(),att.end()),sum=0;
   for(float&a:att){a=std::exp(a-mx);sum+=a;}
   for(float&a:att)a/=sum;
   for(int i=0;i<4;i++)for(int j=0;j<40;j++){
     float a=0;for(int d=0;d<40;d++)a+=val[j*40+d]*words[40*i+d];states[j]+=att[i]*a;
   }
   std::vector<float>hidden;hidden.reserve(232);hidden.insert(hidden.end(),z.begin(),z.end());
   hidden.insert(hidden.end(),words.begin(),words.end());hidden.insert(hidden.end(),states.begin(),states.end());
   layernorm(hidden,ln1g,ln1b);hidden=dense(hidden,up1,up1b,320);gelu(hidden);
   hidden=dense(hidden,up2,up2b,160);gelu(hidden);
   auto res=hidden;layernorm(res,ln2g,ln2b);res=dense(res,res1,res1b,160);gelu(res);
   res=dense(res,res2,res2b,160);for(int j=0;j<160;j++)hidden[j]+=.25f*res[j];
   auto logits=dense(hidden,out,outb,V);
   for(float x:logits)if(!std::isfinite(x))throw std::runtime_error("v155 invalid prediction");
   return logits;
 }
};

static std::array<int,4> v155_prev(const Context&c,const Model&m){
 std::array<int,4>p{};
 for(int k=0;k<4;k++){
  int j=int(c.prev.size())-4+k;
  auto it=j<0?m.id.end():m.id.find(c.prev[j]);p[k]=it==m.id.end()?0:int(it->second)+1;
 }return p;
}
// Fuse native word scores with a full-vocabulary head; strictly maintain parent top-1
// when protect=true. No target labels enter inference.
static Prediction v155_fuse(const Prediction&parent,const std::vector<float>&logits,const Model&m,float neural_share,bool protect){
 if(neural_share<=0||parent.top.empty())return parent;
 if(neural_share>1||!std::isfinite(neural_share))throw std::runtime_error("v155 blend out of range");
 float mx=*std::max_element(logits.begin(),logits.end());std::vector<float>nprob(logits.size());double total=0;
 for(int i=0;i<int(logits.size());i++){nprob[i]=std::exp((logits[i]-mx)*.5f);total+=nprob[i];}
 for(auto&x:nprob)x/=total;
 float pmin=parent.top.back().second;double denom=0;
 for(auto e:parent.top)denom+=std::max(0.f,e.second-pmin)+1.e-8;
 std::vector<std::pair<uint16_t,float>>r;r.reserve(logits.size());
 for(int i=0;i<int(logits.size());i++)r.emplace_back((uint16_t)i,float(neural_share*nprob[i]));
 for(auto e:parent.top)r[e.first].second+=(1.f-neural_share)*float((std::max(0.f,e.second-pmin)+1.e-8)/denom);
 std::partial_sort(r.begin(),r.begin()+32,r.end(),[&](auto a,auto b){return a.second!=b.second?a.second>b.second:a.first<b.first;});r.resize(32);
 if(protect && r[0].first!=parent.top[0].first){
  auto it=std::find_if(r.begin(),r.end(),[&](auto t){return t.first==parent.top[0].first;});
  if(it==r.end())r.back()={parent.top[0].first,0};
  else r.erase(it);
  r.insert(r.begin(),{parent.top[0].first,std::max(r[0].second,1e-8f)*1.01f});
 }
 Prediction o=parent;o.top=std::move(r);return o;
}
