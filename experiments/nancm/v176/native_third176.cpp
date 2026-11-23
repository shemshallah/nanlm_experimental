// NaNCM v176: top-five word -> lexical trie -> byte-verified third coding.
// Reversible speculative draft. NEVER modifies the protected v175, v170,
// v167 pretrained model, optimizer, or byte probability law.
// C API is linked into the inherited native v175 implementation.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <limits>
#include <random>
#include <unordered_map>
#include <cstddef>
extern "C" int wf176_cross_predict(void*,const unsigned char*,size_t,float*,size_t);
extern "C" void wf176_cross_close(void*);
namespace third176 {
struct Cand {int idx=-1;float probability=0.f;std::string bytes;double normalized=0.;};
struct State {
 void *words=nullptr,*bytes=nullptr,*cross=nullptr;
 float escape=.35f;
 std::vector<unsigned char> history;
 std::array<Cand,5> best;
 bool ready=false;
 ~State(){if(cross)wf176_cross_close(cross);if(words)wf175_now_close(words);if(bytes)wf170_byte_close(bytes);}
};
static bool finite_simplex(const float*p,int n){
 double z=0.;for(int i=0;i<n;i++){if(!std::isfinite(p[i])||p[i]<0)return false;z+=p[i];}
 return z>0 && std::abs(z-1.)<2e-4;
}
}
extern "C" void* wf176_hybrid_open(const char*teacher,const char*m,const char*g,const char*c,
                                      const char*bytepack,float escape){
 if(!std::isfinite(escape)||escape<=0.f||escape>=1.f)return nullptr;
 try{
  auto s=std::make_unique<third176::State>();s->escape=escape;
  s->words=wf175_now_open(teacher);if(!s->words)return nullptr;
  s->bytes=wf170_byte_open(m,g,c,bytepack);if(!s->bytes)return nullptr;
  if(wf170_byte_set_adaptive(s->bytes,0)!=0)return nullptr;
  return s.release();
 }catch(...){return nullptr;}
}
extern "C" void wf176_hybrid_close(void*h){delete (third176::State*)h;}
extern "C" int wf176_hybrid_begin(void*h,const unsigned char*past,size_t n){
 auto*s=(third176::State*)h;
 if(!s||(!past&&n)||n>1000000)return -1;
 s->ready=false;
 try{
  s->history.clear();if(n)s->history.assign(past,past+n);
  std::vector<float>probs(12000);
  int rc=s->cross?wf176_cross_predict(s->cross,past,n,probs.data(),probs.size()):
               wf175_now_predict(s->words,past,n,probs.data(),probs.size());
  if(rc!=0||!third176::finite_simplex(probs.data(),12000))return -2;
  std::array<int,5>top{{-1,-1,-1,-1,-1}};
  for(int i=0;i<12000;i++){
   for(int k=0;k<5;k++)if(top[k]<0 || probs[i]>probs[top[k]]){
    for(int a=4;a>k;a--)top[a]=top[a-1];
    top[k]=i;break;
   }
  }
  double z=0.;
  for(int k=0;k<5;k++){
   char word[48]={};if(top[k]<0||wf175_now_word_name(s->words,top[k],word,48)!=0)return -3;
   const size_t len=std::strlen(word);
   if(len==0||len>40)return -3;
   s->best[k]={top[k],probs[top[k]],std::string(word)+" ",0.};
   z+=probs[top[k]];
  }
  if(!(z>0.)||!std::isfinite(z))return -4;
  for(auto &cand:s->best)cand.normalized=cand.probability/z;
  wf170_byte_reset(s->bytes);s->ready=true;return 0;
 }catch(...){return -5;}
}
extern "C" int wf176_hybrid_top5(void*h,int rank,int*index,char*word,size_t capacity,float*p){
 auto*s=(third176::State*)h;if(!s||!s->ready||rank<0||rank>=5||!index||!word||!p)return -1;
 const auto&c=s->best[rank];if(c.bytes.size()>capacity)return -2;
 *index=c.idx;*p=c.probability;std::memcpy(word,c.bytes.data(),c.bytes.size());
 word[c.bytes.size()-1]='\0';return 0;
}
extern "C" int wf176_hybrid_next(void*h,const unsigned char*prefix,size_t n,float proposal[256],float verified[256]){
 auto*s=(third176::State*)h;
 if(!s||!s->ready||(!prefix&&n)||n>40||!proposal||!verified)return -1;
 try{
  std::vector<unsigned char> past=s->history;
  if(n)past.insert(past.end(),prefix,prefix+n);
  if(wf170_byte_next(s->bytes,past.empty()?nullptr:past.data(),past.size(),verified)!=0 ||
     !third176::finite_simplex(verified,256))return -2;
  double weights[256]{};double mass=0.;
  for(const auto&c:s->best){
   if(c.bytes.size()>n&&(n==0||std::memcmp(c.bytes.data(),prefix,n)==0)){
    auto b=(unsigned char)c.bytes[n];weights[b]+=c.normalized;mass+=c.normalized;
   }
  }
  double z=0.;
  for(int j=0;j<256;j++){
   double q=(mass>0.)?(s->escape*verified[j]+(1.-s->escape)*weights[j]/mass):verified[j];
   proposal[j]=(float)q;z+=proposal[j];
  }
  if(!(z>0.)||!std::isfinite(z))return -3;
  for(int j=0;j<256;j++)proposal[j]/=(float)z;
  return third176::finite_simplex(proposal,256)?0:-4;
 }catch(...){return -5;}
}
extern "C" int wf176_spec_accept(const float target[256],const float draft[256],
                                   int proposed,float*u_accept,float residual[256]){
 if(!target||!draft||!u_accept||!residual||proposed<0||proposed>=256||
    !third176::finite_simplex(target,256)||!third176::finite_simplex(draft,256))return -1;
 *u_accept=draft[proposed]>0?std::min(1.f,target[proposed]/draft[proposed]):0.f;
 double z=0.;for(int j=0;j<256;j++){residual[j]=std::max(0.f,target[j]-draft[j]);z+=residual[j];}
 if(z>1e-12){for(int j=0;j<256;j++)residual[j]/=(float)z;}
 else for(int j=0;j<256;j++)residual[j]=target[j];
 return 0;
}

// A genuinely distribution-preserving sequential speculative sampler. The
// proposed byte is drawn from Q at the CURRENT prefix, accepted with P/Q, and
// rejected draws from positive residual (P-Q)+. A *fixed* word top-1 draft
// would not satisfy this law. On rejection the caller continues with P alone.
extern "C" int wf176_hybrid_sample(void*h,uint64_t seed,int maxbytes,
                                     unsigned char*out,size_t capacity,
                                     int*bytes_written,int*accepted,int*rejected){
 auto*s=(third176::State*)h;
 if(!s||!s->ready||!out||!bytes_written||!accepted||!rejected||
    maxbytes<1||maxbytes>40||(size_t)maxbytes>capacity)return -1;
 std::mt19937_64 rng(seed);
 std::uniform_real_distribution<double>uniform(0.,1.);
 auto draw=[&](const float*p)->int{
  double u=uniform(rng),z=0.;
  for(int i=0;i<256;i++){z+=p[i];if(u<z)return i;}
  return 255;
 };
 int n=0,a=0,reject=0;float q[256]{},p[256]{},res[256]{};
 for(int k=0;k<maxbytes;k++){
  if(wf176_hybrid_next(s,out,(size_t)n,q,p)!=0)return -2;
  const int proposed=draw(q);float prob=0.f;
  if(wf176_spec_accept(p,q,proposed,&prob,res)!=0)return -3;
  bool ok=uniform(rng)<(double)prob;
  const int value=ok?proposed:draw(res);
  out[n++]=(unsigned char)value;
  a+=(int)ok;
  if(!ok){reject=1;break;}
  if(value==' ')break;
 }
 *bytes_written=n;*accepted=a;*rejected=reject;return 0;
}

// Native five-proposal verifier. Scores are true normalized v170 byte-log
// likelihood per byte, using exactly one byte oracle call per unique prefix.
// Ranking with these scores is an *experiment*, not the target distribution.
extern "C" int wf176_hybrid_verify_top5(void*h,float avg_log_byte[5],int*oracle_calls){
 auto*s=(third176::State*)h;if(!s||!s->ready||!avg_log_byte||!oracle_calls)return -1;
 std::unordered_map<std::string,std::array<float,256>>cache;
 int calls=0;
 for(int k=0;k<5;k++){
  const std::string &word=s->best[k].bytes;
  std::string pref;double sum=0.;
  for(const auto c:word){
   auto it=cache.find(pref);
   if(it==cache.end()){
    std::array<float,256>q{},p{};
    if(wf176_hybrid_next(s,(const unsigned char*)pref.data(),pref.size(),q.data(),p.data())!=0)return -2;
    it=cache.emplace(pref,p).first;calls++;
   }
   sum+=std::log(std::max(1e-20f,it->second[(unsigned char)c]));
   pref.push_back(c);
  }
  avg_log_byte[k]=(float)(sum/word.size());
 }
 *oracle_calls=calls;return 0;
}