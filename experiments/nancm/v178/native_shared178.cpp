// NaNCM v178: clone-free multi-session verifier. One inherited model instance,
// request-private causal histories and prefix caches. Cache misses are serialized
// because WF148->WF170 carries mutable inference state; no false parallel claim.
// Inherited P and Q are never approximated, quantized, or independently rebuilt.
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <limits>

namespace shared178 {
struct Node {std::array<float,256> q{},p{};};
struct Label {int index=-1;std::string word;float prob=0.;};
struct Hub {
 std::mutex lock;
 void* oracle=nullptr;
 uint64_t next_id=1,active_id=0;
 std::atomic<int> sessions{0};
 explicit Hub(void*handle):oracle(handle){}
 ~Hub(){if(oracle)wf176_hybrid_close(oracle);}
 Hub(const Hub&)=delete;Hub&operator=(const Hub&)=delete;
};
struct Session {
 std::shared_ptr<Hub> hub;
 std::mutex lock;
 uint64_t id=0;
 bool ready=false;
 std::vector<unsigned char> history;
 std::array<Label,5> top{};
 std::unordered_map<std::string,Node> cache;
 size_t hits=0,misses=0;
 explicit Session(std::shared_ptr<Hub>owner):hub(std::move(owner)){
   hub->sessions.fetch_add(1);
 }
 ~Session(){hub->sessions.fetch_sub(1);}
};
static int begin_locked(Session*s,const unsigned char*past,size_t n){
 if(!s||(!past&&n)||n>1000000)return -1;
 s->ready=false;s->cache.clear();s->hits=s->misses=0;
 try{
  s->history.clear();
  if(n)s->history.assign(past,past+n);
  std::lock_guard<std::mutex>g(s->hub->lock);
  auto*h=s->hub.get();
  if(wf176_hybrid_begin(h->oracle,past,n)!=0){h->active_id=0;return -2;}
  for(int i=0;i<5;i++){
   int ix=-1;char word[48]{};float p=0.;
   if(wf176_hybrid_top5(h->oracle,i,&ix,word,sizeof(word),&p)!=0){
    h->active_id=0;return -3;
   }
   s->top[i]={ix,std::string(word),p};
  }
  h->active_id=s->id;
  s->ready=true;
  return 0;
 }catch(...){s->hub->active_id=0;return -4;}
}
static int next_locked(Session*s,const unsigned char*prefix,size_t n,float*q,float*p){
 if(!s||!s->ready||(!prefix&&n)||n>40||!q||!p)return -1;
 try{
  const std::string key(n?std::string((const char*)prefix,n):std::string());
  const auto it=s->cache.find(key);
  if(it!=s->cache.end()){
   std::memcpy(q,it->second.q.data(),256*sizeof(float));
   std::memcpy(p,it->second.p.data(),256*sizeof(float));
   ++s->hits;return 0;
  }
  Node node;
  {
   std::lock_guard<std::mutex>g(s->hub->lock);
   auto*h=s->hub.get();
   if(h->active_id!=s->id){
    const auto*data=s->history.empty()?nullptr:s->history.data();
    if(wf176_hybrid_begin(h->oracle,data,s->history.size())!=0){
     h->active_id=0;return -2;
    }
    h->active_id=s->id;
   }
   if(wf176_hybrid_next(h->oracle,prefix,n,node.q.data(),node.p.data())!=0)return -3;
  }
  ++s->misses;
  if(s->cache.size()<256){
   const auto inserted=s->cache.emplace(key,std::move(node)).first;
   std::memcpy(q,inserted->second.q.data(),256*sizeof(float));
   std::memcpy(p,inserted->second.p.data(),256*sizeof(float));
  }else{
   std::memcpy(q,node.q.data(),256*sizeof(float));
   std::memcpy(p,node.p.data(),256*sizeof(float));
  }
  return 0;
 }catch(...){return -4;}
}
static int draw(std::mt19937_64&r,std::uniform_real_distribution<double>&uniform,const float*p){
 const double u=uniform(r);double z=0.;
 for(int i=0;i<256;i++){z+=p[i];if(u<z)return i;}
 return 255;
}
}
extern "C" void*wf178_hub_open_full(const char*teacher,const char*m,const char*g,const char*c,
 const char*bytepack,float escape,const char*cross,const char*bpe){
 try{
  void*model=wf176_hybrid_open_full(teacher,m,g,c,bytepack,escape,cross,bpe);
  if(!model)return nullptr;
  try{return new std::shared_ptr<shared178::Hub>(std::make_shared<shared178::Hub>(model));}
  catch(...){wf176_hybrid_close(model);throw;}
 }catch(...){return nullptr;}
}
extern "C" void wf178_hub_close(void*h){delete (std::shared_ptr<shared178::Hub>*)h;}
extern "C" int wf178_hub_stats(void*h,int*model_instances,int*live_sessions){
 if(!h||!model_instances||!live_sessions)return -1;
 const auto&hub=*(std::shared_ptr<shared178::Hub>*)h;
 *model_instances=1;*live_sessions=hub->sessions.load();return 0;
}
extern "C" void*wf178_session_open(void*h,const unsigned char*past,size_t n){
 if(!h||(!past&&n)||n>1000000)return nullptr;
 try{
  auto hub=*(std::shared_ptr<shared178::Hub>*)h;
  auto s=std::make_unique<shared178::Session>(hub);
  {
   std::lock_guard<std::mutex>g(hub->lock);
   s->id=hub->next_id++;
   if(!s->id||!hub->next_id)return nullptr; // fail closed on 64-bit wrap
  }
  if(shared178::begin_locked(s.get(),past,n)!=0)return nullptr;
  return s.release();
 }catch(...){return nullptr;}
}
extern "C" void wf178_session_close(void*h){delete (shared178::Session*)h;}
extern "C" int wf178_session_begin(void*h,const unsigned char*past,size_t n){
 auto*s=(shared178::Session*)h;if(!s)return -1;
 std::lock_guard<std::mutex>g(s->lock);
 return shared178::begin_locked(s,past,n);
}
extern "C" int wf178_session_next(void*h,const unsigned char*prefix,size_t n,float*q,float*p){
 auto*s=(shared178::Session*)h;if(!s)return -1;
 std::lock_guard<std::mutex>g(s->lock);
 return shared178::next_locked(s,prefix,n,q,p);
}
extern "C" int wf178_session_top5(void*h,int rank,int*index,char*word,size_t cap,float*prob){
 auto*s=(shared178::Session*)h;
 if(!s||rank<0||rank>=5||!index||!word||!prob)return -1;
 std::lock_guard<std::mutex>g(s->lock);
 if(!s->ready||cap<=s->top[rank].word.size())return -2;
 const auto &t=s->top[rank];
 *index=t.index;*prob=t.prob;std::memcpy(word,t.word.c_str(),t.word.size()+1);
 return 0;
}
extern "C" int wf178_session_stats(void*h,int*misses,int*hits,int*nodes){
 auto*s=(shared178::Session*)h;if(!s||!misses||!hits||!nodes)return -1;
 std::lock_guard<std::mutex>g(s->lock);
 *misses=(int)s->misses;*hits=(int)s->hits;*nodes=(int)s->cache.size();return 0;
}
extern "C" int wf178_session_batch(void*h,const unsigned char*bytes,size_t byte_count,
 const uint32_t*offset,const uint32_t*length,size_t count,float*outq,float*outp){
 auto*s=(shared178::Session*)h;
 if(!s||count>512||(!bytes&&byte_count)||(!offset&&count)||(!length&&count)||
    (!outq&&count)||(!outp&&count))return -1;
 // Preflight the entire batch: invalid pointers/lengths must not partially advance state.
 for(size_t i=0;i<count;i++)
  if(length[i]>40||offset[i]>byte_count||length[i]>byte_count-offset[i])return -2;
 std::lock_guard<std::mutex>g(s->lock);
 for(size_t i=0;i<count;i++){
  const unsigned char*p=length[i]?bytes+offset[i]:nullptr;
  if(shared178::next_locked(s,p,length[i],outq+256*i,outp+256*i)!=0)return -3;
 }
 return 0;
}
extern "C" int wf178_session_prepare_top5(void*h,float scores[5],int*raw,int*unique){
 auto*s=(shared178::Session*)h;if(!s||!scores||!raw||!unique)return -1;
 std::lock_guard<std::mutex>g(s->lock);
 if(!s->ready)return -2;
 size_t before=s->misses;int total=0;
 for(int k=0;k<5;k++){
  const std::string w=s->top[k].word+" ";
  if(w.size()>40)return -3;
  std::string pre;double score=0.;
  std::array<float,256>q{},p{};
  for(unsigned char ch:w){
   if(shared178::next_locked(s,(const unsigned char*)pre.data(),pre.size(),q.data(),p.data()))return -4;
   score+=std::log(std::max(1e-20f,p[ch]));pre.push_back((char)ch);++total;
  }
  scores[k]=(float)(score/w.size());
 }
 *raw=total;*unique=(int)(s->misses-before);return 0;
}
extern "C" int wf178_session_sample(void*h,uint64_t seed,int maxbytes,unsigned char*out,
 size_t capacity,int*written,int*accepted,int*rejected){
 auto*s=(shared178::Session*)h;
 if(!s||!out||!written||!accepted||!rejected||maxbytes<1||maxbytes>40||
    (size_t)maxbytes>capacity)return -1;
 std::lock_guard<std::mutex>g(s->lock);
 if(!s->ready)return -2;
 std::mt19937_64 rng(seed);std::uniform_real_distribution<double>uniform(0.,1.);
 int n=0,a=0,r=0;std::array<float,256>q{},p{},res{};
 for(int k=0;k<maxbytes;k++){
  if(shared178::next_locked(s,out,n,q.data(),p.data())!=0)return -3;
  int proposed=shared178::draw(rng,uniform,q.data());float chance=0.f;
  if(wf176_spec_accept(p.data(),q.data(),proposed,&chance,res.data())!=0)return -4;
  const bool ok=uniform(rng)<(double)chance;
  const int value=ok?proposed:shared178::draw(rng,uniform,res.data());
  out[n++]=(unsigned char)value;a+=(int)ok;
  if(!ok){r=1;break;}
  if(value==' ')break;
 }
 *written=n;*accepted=a;*rejected=r;return 0;
}
