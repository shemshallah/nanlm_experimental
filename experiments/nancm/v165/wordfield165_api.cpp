// v165 experimental TRAINED 128+128 signed geometric cloud: S+/S- oriented
// edge energies of COMPLETE-WORD continuations. Derives from v164 but
// leaves original native core/cloud checkpoints and optimizer files untouched.
#include <memory>
#include <array>
#include <cstdint>
#include <limits>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include "wordfield164_embed165.cpp"
namespace {
struct Signed165 {
 float lower[128][8]{};
 float upper[128][8]{};
 float plus[128][128][8]{};
 float minus[128][128][8]{};
};
struct State165 {
 Native164*base=nullptr;
 std::unique_ptr<Signed165>signed_cloud;
 double strength=0.;
};
static uint32_t lex_hash165(const std::string&word){
 uint32_t h=2166136261u;
 for(unsigned char b:word)h=(h^b)*16777619u;
 return h&127u;
}
static bool cloud_read165(const char*path,Signed165&g,Native164*base){
 std::ifstream f(path,std::ios::binary); if(!f)return false;
 char magic[8];uint64_t fp[3];uint32_t dims[2];
 if(!f.read(magic,8)||std::string(magic,8)!="N165SCL1"||
    !f.read((char*)fp,sizeof(fp))||!f.read((char*)dims,sizeof(dims))||
    dims[0]!=128||dims[1]!=8)return false;
 if(!base||!base->parent)return false;
 auto*m=(State159*)base->parent->base;
 auto*b=m?(State158*)m->base:nullptr;
 auto*r=b?(WF148State*)b->base:nullptr;
 if(!r||fp[0]!=r->model.trainhash||fp[1]!=r->model.geomhash||fp[2]!=r->model.cloudhash)return false;
 if(!f.read((char*)&g.lower,sizeof(g.lower))||!f.read((char*)&g.upper,sizeof(g.upper))||
    !f.read((char*)&g.plus,sizeof(g.plus))||!f.read((char*)&g.minus,sizeof(g.minus)))return false;
 char extra; if(f.read(&extra,1))return false;
 if(!f.eof())return false;
 const float*raw=(const float*)&g;
 for(size_t i=0;i<sizeof(g)/sizeof(float);i++)if(!std::isfinite(raw[i])||std::abs(raw[i])>100.f)return false;
 return true;
}
static double energy165(const Signed165&g,const std::vector<std::string>&words,const std::string&target){
 if(words.empty())return 0.;
 const uint32_t t=lex_hash165(target);
 const size_t count=std::min((size_t)3,words.size());
 const double recency[3]={.2,.3,.5};
 double sum=0.,den=0.;
 for(size_t j=0;j<count;j++){
  const uint32_t b=lex_hash165(words[words.size()-count+j]);
  const double temporal=recency[3-count+j];
  double v=0.;
  for(int d=0;d<8;d++){
   const double l=g.lower[b][d],u=g.upper[t][d];
   v+=l*u*.3535533905932737622 +
      (double)g.plus[t][b][d]*std::tanh(l+u) -
      (double)g.minus[t][b][d]*std::tanh(l-u);
  }
  sum+=temporal*v;den+=temporal;
 }
 return den>0?sum/den:0.;
}
static bool parse_strength165(double &v){
 auto*e=std::getenv("V165_SIGNED_ENERGY"); if(!e){v=0.;return true;}
 char*end=nullptr;v=std::strtod(e,&end);
 return end!=e&&!*end&&std::isfinite(v)&&v>=0.&&v<=4.;
}
}
extern "C" void*wf148_open(const char*m,const char*g,const char*c){
 std::unique_ptr<State165>s(new State165());
 if(!parse_strength165(s->strength))return nullptr;
 s->base=(Native164*)wf164_open_impl(m,g,c);
 if(!s->base)return nullptr;
 if(s->strength>0){
  const char*path=std::getenv("V165_CLOUD_INDEX");
  s->signed_cloud.reset(new Signed165);
  if(!path||!cloud_read165(path,*s->signed_cloud,s->base)){
   wf164_close_impl(s->base);return nullptr;
  }
 }
 return s.release();
}
extern "C" void wf148_close(void*h){
 auto*s=(State165*)h;if(!s)return;wf164_close_impl(s->base);delete s;
}
extern "C" int wf148_begin(void*h,const unsigned char*past,size_t n,float c){
 auto*s=(State165*)h;if(!s)return -1;
 const int ret=wf164_begin_impl(s->base,past,n,c);if(ret<0)return ret;
 if(s->strength<=0||!s->signed_cloud||s->base->alpha<=0||s->base->candidates.size()<2)return ret;
 const auto context=complete_words158(past,n);
 if(context.empty())return ret;
 auto&votes=s->base->candidates;
 std::vector<double>logits(votes.size());double mx=-1e300;
 for(size_t i=0;i<votes.size();i++){
  const double e=energy165(*s->signed_cloud,context,votes[i].first);
  logits[i]=std::log(std::max(1e-40,votes[i].second))+std::max(-12.,std::min(12.,s->strength*e));
  mx=std::max(mx,logits[i]);
 }
 double denom=0.;for(auto &v:logits){v=std::exp(v-mx);denom+=v;}
 if(!(denom>0.)||!std::isfinite(denom))return -5;
 for(size_t i=0;i<votes.size();i++)votes[i].second=logits[i]/denom;
 // Parent exact 27-way whole-word mixture code now consumes refined Q165.
 return ret;
}
extern "C" int wf148_next(void*h,const char*p,size_t n,float o[27],float*e){
 auto*s=(State165*)h;return s?wf164_next_impl(s->base,p,n,o,e):-1;
}
extern "C" int wf165_candidate_count(void*h){auto*s=(State165*)h;return s?(int)s->base->candidates.size():-1;}
extern "C" int wf165_candidate(void*h,int i,char*out,size_t cap,double*w){
 auto*s=(State165*)h;
 if(!s||i<0||i>=(int)s->base->candidates.size()||!out||!w)return -1;
 const auto &v=s->base->candidates[i];if(cap<=v.first.size())return -2;
 std::copy(v.first.begin(),v.first.end(),out);out[v.first.size()]=0;*w=v.second;return 0;
}
extern "C" int wf148_fuse(const float*p,const float*w,float e,float v,int pr,float*out){return wf164_fuse_impl(p,w,e,v,pr,out);}
extern "C" int wf148_fuse_surgical(const float*p,const float*w,float e,float g,float*out){return wf164_fuse_surgical_impl(p,w,e,g,out);}
extern "C" int wf157_logits(void*h,float*out,size_t n){auto*s=(State165*)h;return s?wf164_logits_impl(s->base,out,n):-1;}
extern "C" int wf157_chars(void*h,const char*p,size_t n,float*out){auto*s=(State165*)h;return s?wf164_chars_impl(s->base,p,n,out):-1;}
extern "C" int wf158_pointer(void*h,char*w,size_t c,double*a){auto*s=(State165*)h;return s?wf164_pointer_impl(s->base,w,c,a):-1;}
extern "C" int wf158_candidate_count(void*h){auto*s=(State165*)h;return s?wf164_old_candidate_count_impl(s->base):-1;}
extern "C" int wf159_count(void*h){auto*s=(State165*)h;return s?wf164_old159_count_impl(s->base):-1;}
extern "C" int wf159_candidate(void*h,int i,char*w,size_t c,double*p,int*m){auto*s=(State165*)h;return s?wf164_old159_candidate_impl(s->base,i,w,c,p,m):-1;}
