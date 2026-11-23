// NaNCM v177: private-handle, bounded prefix-DAG memoization, batched verifier,
// exact speculative acceptance. The inherited v176/v170 probability laws are
// NOT changed. This is *not* a parallel verifier: batch execution is fused into
// a single native call and shares only causally identical prefix computations.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

namespace lattice177 {
struct Node {std::array<float,256> q{}, p{};};
struct State {
 void* inherited=nullptr;
 bool active=false;
 size_t misses=0,hits=0;
 std::unordered_map<std::string,Node> nodes;
 ~State(){if(inherited)wf176_hybrid_close(inherited);}
};
static bool valid(const State*s){return s&&s->active&&s->inherited;}
static int next(State*s,const unsigned char*prefix,size_t n,float*q,float*p){
 if(!valid(s)||(!prefix&&n)||n>40||!q||!p)return -1;
 const std::string key(n?std::string(reinterpret_cast<const char*>(prefix),n):std::string());
 auto it=s->nodes.find(key);
 if(it!=s->nodes.end()){
  std::memcpy(q,it->second.q.data(),256*sizeof(float));
  std::memcpy(p,it->second.p.data(),256*sizeof(float));
  s->hits++;return 0;
 }
 Node node;
 int rc=wf176_hybrid_next(s->inherited,prefix,n,node.q.data(),node.p.data());
 if(rc!=0)return rc;
 s->misses++;
 if(s->nodes.size()<256){
  auto pos=s->nodes.emplace(key,node).first;
  std::memcpy(q,pos->second.q.data(),256*sizeof(float));
  std::memcpy(p,pos->second.p.data(),256*sizeof(float));
 }else{
  std::memcpy(q,node.q.data(),256*sizeof(float));
  std::memcpy(p,node.p.data(),256*sizeof(float));
 }
 return 0;
}
static int draw(std::mt19937_64&rng,const std::array<float,256>&p){
 std::uniform_real_distribution<double>u(0.,1.);
 const double target=u(rng);
 double sum=0.;for(int j=0;j<256;j++){sum+=p[j];if(target<sum)return j;}
 return 255;
}
}
extern "C" void*wf177_open_full(const char*teacher,const char*m,const char*g,const char*c,
                                  const char*bytepack,float escape,const char*calib,const char*bpe){
 try{
  std::unique_ptr<lattice177::State> s(new lattice177::State());
  s->inherited=wf176_hybrid_open_full(teacher,m,g,c,bytepack,escape,calib,bpe);
  return s->inherited?s.release():nullptr;
 }catch(...){return nullptr;}
}
extern "C" void wf177_close(void*state){delete (lattice177::State*)state;}
extern "C" int wf177_begin(void*state,const unsigned char*past,size_t n){
 auto*s=(lattice177::State*)state;
 if(!s||!s->inherited)return -1;
 s->active=false;
 s->nodes.clear();s->hits=s->misses=0;
 const int rc=wf176_hybrid_begin(s->inherited,past,n);
 s->active=rc==0;return rc;
}
extern "C" int wf177_next(void*state,const unsigned char*prefix,size_t n,float q[256],float p[256]){
 return lattice177::next((lattice177::State*)state,prefix,n,q,p);
}
extern "C" int wf177_top5(void*state,int rank,int*index,char*word,size_t capacity,float*prob){
 auto*s=(lattice177::State*)state;
 return lattice177::valid(s)?wf176_hybrid_top5(s->inherited,rank,index,word,capacity,prob):-1;
}
// Prefixes are concatenated in `bytes` and pointed to by 32-bit byte offsets.
// outq/outp are 256*count row-major. Full batch parity with scalar v176 is
// exact because every unique prefix uses the *identical* inherited oracle.
extern "C" int wf177_verify_batch(void*state,const unsigned char*bytes,size_t bytes_len,
                                   const uint32_t*offsets,const uint32_t*lengths,size_t count,
                                   float*outq,float*outp){
 auto*s=(lattice177::State*)state;
 if(!lattice177::valid(s)||count>512||(!offsets&&count)||(!lengths&&count)||
    (!outq&&count)||(!outp&&count)||(!bytes&&bytes_len))return -1;
 for(size_t i=0;i<count;i++){
  const size_t off=offsets[i],n=lengths[i];
  if(n>40||off>bytes_len||n>bytes_len-off)return -2;
  const unsigned char*prefix=n?(bytes+off):nullptr;
  int rc=lattice177::next(s,prefix,n,outq+256*i,outp+256*i);
  if(rc)return -3;
 }
 return 0;
}
// Evaluate five complete-word candidates; share P(prefix) on a trie.
// Compute top-five verifier log scores without changing the ranking or law.
extern "C" int wf177_prepare_top5(void*state,float*avg_log_byte,int*raw_calls,int*unique_calls){
 auto*s=(lattice177::State*)state;
 if(!lattice177::valid(s)||!avg_log_byte||!raw_calls||!unique_calls)return -1;
 const size_t before=s->misses;
 int raw=0;
 for(int k=0;k<5;k++){
  int idx=-1;char word[48]{};float prob=0.;
  if(wf176_hybrid_top5(s->inherited,k,&idx,word,sizeof(word),&prob))return -2;
  const std::string w=std::string(word)+" ";
  if(w.size()>40)return -3;
  std::string pref;
  double score=0.;
  std::array<float,256>q{},p{};
  for(const unsigned char ch:w){
   const auto*ptr=(const unsigned char*)pref.data();
   if(lattice177::next(s,ptr,pref.size(),q.data(),p.data())!=0)return -4;
   score+=std::log(std::max(1e-20f,p[ch]));
   pref.push_back((char)ch);raw++;
  }
  avg_log_byte[k]=(float)(score/w.size());
 }
 *raw_calls=raw;
 *unique_calls=(int)(s->misses-before);
 return 0;
}
extern "C" int wf177_stats(void*state,int*misses,int*hits,int*cache_nodes){
 auto*s=(lattice177::State*)state;if(!s||!misses||!hits||!cache_nodes)return -1;
 *misses=(int)s->misses;*hits=(int)s->hits;*cache_nodes=(int)s->nodes.size();return 0;
}
// Exact P/Q rejection sampling, with P and Q from the previously verified
// shared prefix DAG when available; off-trie continuation remains live P.
extern "C" int wf177_sample(void*state,uint64_t seed,int maxbytes,unsigned char*out,size_t cap,
                              int*written,int*accepted,int*rejected){
 auto*s=(lattice177::State*)state;
 if(!lattice177::valid(s)||!out||!written||!accepted||!rejected||
    maxbytes<1||maxbytes>40||(size_t)maxbytes>cap)return -1;
 std::mt19937_64 rng(seed);
 std::uniform_real_distribution<double>u(0.,1.);
 int n=0,a=0,r=0;
 std::array<float,256>q{},p{},res{};
 for(int k=0;k<maxbytes;k++){
  if(lattice177::next(s,out,(size_t)n,q.data(),p.data())!=0)return -2;
  const int proposed=lattice177::draw(rng,q);
  float chance=0.f;
  if(wf176_spec_accept(p.data(),q.data(),proposed,&chance,res.data())!=0)return -3;
  const bool ok=u(rng)<(double)chance;
  const int value=ok?proposed:lattice177::draw(rng,res);
  out[n++]=(unsigned char)value;
  a+=ok;
  if(!ok){r=1;break;}
  if(value==' ')break;
 }
 *written=n;*accepted=a;*rejected=r;
 return 0;
}