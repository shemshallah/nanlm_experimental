// NaNCM v163 — trainable signed-cloud confidence gate over native v162.
// Parent model and its optimizer tensors are never edited. When no gate is
// loaded, wf148_* is bit-identical to v162; gate uses only causal history.
#include <memory>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include "wordfield162_embed163.cpp"

namespace {
static constexpr int NF163=38;
struct Gate163{
  float weights[NF163]{};
  uint64_t parent_train=0,parent_geom=0,parent_cloud=0;
  double intercept_scale=1.0,cutoff=.5,boost=1.;
};
struct State163{
 State160*base=nullptr;
 std::unique_ptr<Gate163> gate;
 float features[NF163]{};
 double gate_score=0.,original_alpha=0.;
 bool had_candidate=false;
};
static WF148State* root163(State160*s){
 if(!s||!s->base)return nullptr;
 auto *m=(State159*)s->base;
 if(!m||!m->base)return nullptr;
 auto *p=(State158*)m->base;
 return p?(WF148State*)p->base:nullptr;
}
static bool gate_read163(const char*path,Gate163 &g,WF148State*root){
 std::ifstream f(path);std::string magic;int num=0;
 if(!f||!(f>>magic>>g.parent_train>>g.parent_geom>>g.parent_cloud>>num))return false;
 if(magic!="N163GATE1"||num!=NF163||g.parent_train!=root->model.trainhash||
    g.parent_geom!=root->model.geomhash||g.parent_cloud!=root->model.cloudhash)return false;
 for(float &w:g.weights)if(!(f>>w)||!std::isfinite(w)||std::abs(w)>1000.f)return false;
 if(!(f>>g.cutoff>>g.boost)||!std::isfinite(g.cutoff)||g.cutoff<0||g.cutoff>1||
    !std::isfinite(g.boost)||g.boost<0||g.boost>3)return false;
 std::string extra;return !(f>>extra);
}
static void features163(State163*s,const unsigned char*past,size_t n){
 std::fill(std::begin(s->features),std::end(s->features),0.f);
 s->features[37]=1.f;
 State160*base=s->base;
 s->original_alpha=base->alpha;s->had_candidate=(base->alpha>0&&!base->word.empty());
 if(!s->had_candidate||!past)return;
 auto*root=root163(base);if(!root)return;
 Context c{};c.p=past;c.pos=n;c.word="";
 auto words=complete_words158(past,n);
 if(words.size()>6)c.prev.assign(words.end()-6,words.end());else c.prev=words;
 const auto signed_state=causal_signed_state(c,root->geom,root->cloud);
 for(int j=0;j<32;j++)s->features[j]=(float)std::tanh((double)signed_state[j]);
 s->features[32]=(float)s->original_alpha;
 s->features[36]=(float)std::min(1.,base->word.size()/20.);
 for(int k=4;k>=base->min_order;k--){
  if(words.size()<(size_t)k)continue;
  std::string key=std::to_string(k)+":";
  for(int j=k;j>0;--j){if(j!=k)key.push_back(' ');key+=words[words.size()-j];}
  auto it=base->index.find(key);if(it==base->index.end()||it->second.target!=base->word)continue;
  const auto&entry=it->second;
  if(entry.total>0){
   s->features[33]=(float)entry.count/entry.total;
   s->features[34]=(float)std::min(1.0,std::log1p((double)entry.count)/5.);
   s->features[35]=k/4.f;
  }
  break;
 }
}
static double sigmoid163(double x){return x>30?1.0:(x< -30?0.0:1.0/(1.0+std::exp(-x)));}
}
extern "C" void*wf148_open(const char*m,const char*g,const char*c){
 std::unique_ptr<State163>s(new State163);
 s->base=(State160*)wf162_wf148_open_impl(m,g,c);
 if(!s->base)return nullptr;
 auto env=std::getenv("V163_GATE_FILE");
 if(env&&*env){
  auto*root=root163(s->base);if(!root){wf162_wf148_close_impl(s->base);return nullptr;}
  s->gate.reset(new Gate163);
  if(!gate_read163(env,*s->gate,root)){wf162_wf148_close_impl(s->base);return nullptr;}
 }
 return s.release();
}
extern "C" void wf148_close(void*h){auto*s=(State163*)h;if(!s)return;wf162_wf148_close_impl(s->base);delete s;}
extern "C" int wf148_begin(void*h,const unsigned char*past,size_t n,float cloud){
 auto*s=(State163*)h;if(!s)return -1;
 int r=wf162_wf148_begin_impl(s->base,past,n,cloud);if(r<0)return r;
 features163(s,past,n);
 s->gate_score=0.;
 if(s->gate&&s->had_candidate){
  double v=0.;for(int j=0;j<NF163;j++)v+=(double)s->features[j]*s->gate->weights[j];
  s->gate_score=sigmoid163(v);
  if(s->gate_score<s->gate->cutoff)s->base->alpha=0.;
  else s->base->alpha=std::min(.999,s->original_alpha*s->gate->boost);
 }
 return r;
}
extern "C" int wf148_next(void*h,const char*p,size_t n,float o[27],float*e){auto*s=(State163*)h;return s?wf162_wf148_next_impl(s->base,p,n,o,e):-1;}
extern "C" int wf163_gate_features(void*h,float*out,int cap,double*alpha,int*active){
 auto*s=(State163*)h;if(!s||!out||cap<NF163||!alpha||!active)return -1;
 std::copy(std::begin(s->features),std::end(s->features),out);
 *alpha=s->original_alpha;*active=s->had_candidate?1:0;return NF163;
}
extern "C" double wf163_gate_score(void*h){auto*s=(State163*)h;return s?s->gate_score:0.;}
extern "C" int wf160_association(void*h,char*out,size_t cap,double*alpha){auto*s=(State163*)h;return s?wf162_wf160_association_impl(s->base,out,cap,alpha):-1;}
extern "C" double wf162_domain_score(void*h){auto*s=(State163*)h;return s?wf162_wf162_domain_score_impl(s->base):-1e9;}
extern "C" int wf148_fuse(const float*p,const float*w,float e,float v,int prot,float*out){return wf162_wf148_fuse_impl(p,w,e,v,prot,out);}
extern "C" int wf148_fuse_surgical(const float*p,const float*w,float e,float g,float*out){return wf162_wf148_fuse_surgical_impl(p,w,e,g,out);}
extern "C" int wf157_logits(void*h,float*out,size_t n){auto*s=(State163*)h;return s?wf162_wf157_logits_impl(s->base,out,n):-1;}
extern "C" int wf157_chars(void*h,const char*p,size_t n,float*out){auto*s=(State163*)h;return s?wf162_wf157_chars_impl(s->base,p,n,out):-1;}
extern "C" int wf158_pointer(void*h,char*w,size_t c,double*a){auto*s=(State163*)h;return s?wf162_wf158_pointer_impl(s->base,w,c,a):-1;}
extern "C" int wf158_candidate_count(void*h){auto*s=(State163*)h;return s?wf162_wf158_candidate_count_impl(s->base):-1;}
extern "C" int wf159_count(void*h){auto*s=(State163*)h;return s?wf162_wf159_count_impl(s->base):-1;}
extern "C" int wf159_candidate(void*h,int i,char*w,size_t c,double*p,int*m){auto*s=(State163*)h;return s?wf162_wf159_candidate_impl(s->base,i,w,c,p,m):-1;}
extern "C" int wf163_parent_fingerprints(void*h,uint64_t out[3]){
 auto*s=(State163*)h;if(!s||!out)return -1;
 auto*r=root163(s->base);if(!r)return -1;
 out[0]=r->model.trainhash;out[1]=r->model.geomhash;out[2]=r->model.cloudhash;
 return 0;
}
