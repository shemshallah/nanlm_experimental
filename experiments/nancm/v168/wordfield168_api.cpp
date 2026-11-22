// NaNCM v168: native book-trained confidence arbitration candidate, exact v167 opt-out
// Derived from v166.
// NaNCM v166: lexical-birth world-state generator, native WHOLE-WORD
// probability composition. 128 lower and 128 upper oriented signed concept
// points inherited, plus distinct learned context/output embeddings for up to 12000
// complete words. Retains v164's exact native ABI as a protected parent.
// A context's original v164 candidate list is NOT used to generate v166 words.
#include <memory>
#include <array>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <cstring>
#include <limits>
#include <cstdint>
#include "wordfield164_embed165.cpp"
namespace {
constexpr int DIM166=32;
struct Lex166 {std::string word;float bias=0;float ctx[DIM166]{};float out[DIM166]{};uint32_t hash=0;int leaf=-1;};
struct TrieNode166 {std::array<int,26> edge;int terminal=-1;TrieNode166(){edge.fill(-1);}};
struct Cloud166 {float plus[128][128][8]{};float minus[128][128][8]{};float lower[128][8]{};float upper[128][8]{};};
struct Model166 {
 std::unique_ptr<Cloud166> cloud;
 std::vector<Lex166> lex;
 std::vector<TrieNode166> trie;
 std::unordered_map<std::string,size_t> lookup;
 float lex_gain=0,cloud_gain=0;
};
struct State166 {
 Native164*parent=nullptr;
 std::unique_ptr<Model166>birth;
 double alpha_max=0,alpha=0;
 double domain_min=1.5, high_alpha=.5, trust_threshold=.25, top_q=0, parent_ratio=0;
 int gate_mode=0;
 std::array<double,7> calibration{}; bool calibration_active=false;
 std::vector<double> p;
 std::vector<double> trie_mass;
 std::vector<size_t> order;
 std::unordered_map<std::string,double> parent_prefix;
};
static uint32_t hash166(const std::string&w){uint32_t h=2166136261u;for(unsigned char b:w)h=(h^b)*16777619u;return h&127u;}
static bool read_birth166(const char*path,Model166&g,Native164*base){
 std::ifstream f(path,std::ios::binary);if(!f)return false;
 char magic[8];uint64_t fp[3];uint32_t shape[2];
 if(!f.read(magic,8)||std::memcmp(magic,"N166BRT1",8)||!f.read((char*)fp,sizeof(fp))||!f.read((char*)shape,sizeof(shape)))return false;
 if(shape[0]<100||shape[0]>30000||shape[1]!=DIM166||!base||!base->parent)return false;
 auto*m=(State159*)base->parent->base;auto*b=m?(State158*)m->base:nullptr;auto*r=b?(WF148State*)b->base:nullptr;
 if(!r||fp[0]!=r->model.trainhash||fp[1]!=r->model.geomhash||fp[2]!=r->model.cloudhash)return false;
 g.cloud.reset(new Cloud166);
 auto&z=*g.cloud;
 if(!f.read((char*)z.plus,sizeof(z.plus))||!f.read((char*)z.minus,sizeof(z.minus))||!f.read((char*)z.lower,sizeof(z.lower))||!f.read((char*)z.upper,sizeof(z.upper))||
   !f.read((char*)&g.lex_gain,4)||!f.read((char*)&g.cloud_gain,4))return false;
 if(!std::isfinite(g.lex_gain)||!std::isfinite(g.cloud_gain)||g.lex_gain<0||g.lex_gain>50||g.cloud_gain<0||g.cloud_gain>50)return false;
 const float*raw=(const float*)&z;
 for(size_t i=0;i<sizeof(z)/4;i++)if(!std::isfinite(raw[i])||std::abs(raw[i])>100)return false;
 g.lex.reserve(shape[0]);
 for(uint32_t i=0;i<shape[0];i++){
  uint8_t n=0;if(!f.read((char*)&n,1)||n<1||n>20)return false;
  Lex166 w;w.word.resize(n);
  if(!f.read(&w.word[0],n)||!f.read((char*)&w.bias,4)||!f.read((char*)w.ctx,sizeof(w.ctx))||!f.read((char*)w.out,sizeof(w.out)))return false;
  for(char c:w.word)if(c<'a'||c>'z')return false;
  w.hash=hash166(w.word);
  if(!std::isfinite(w.bias)||std::abs(w.bias)>100)return false;
  for(float a:w.ctx)if(!std::isfinite(a)||std::abs(a)>100)return false;
  for(float a:w.out)if(!std::isfinite(a)||std::abs(a)>100)return false;
  if(!g.lookup.emplace(w.word,i).second)return false;
  g.lex.emplace_back(std::move(w));
 }
 char tail;if(f.read(&tail,1)||!f.eof())return false;
 // Immutable vocabulary trie: nodes are allocated in parent-before-child order.
 g.trie.emplace_back();
 for(size_t i=0;i<g.lex.size();i++){
  int node=0;
  for(char chr:g.lex[i].word){
   int j=chr-'a';int next=g.trie[node].edge[j];
   if(next<0){next=(int)g.trie.size();g.trie[node].edge[j]=next;g.trie.emplace_back();}
   node=next;
  }
  if(g.trie[node].terminal>=0)return false;
  g.trie[node].terminal=(int)i;g.lex[i].leaf=node;
 }
 return true;
}
static bool setting166(double&v){
 const char*s=std::getenv("V166_LEXICAL_BIRTH");if(!s){v=0;return true;}
 char*end=nullptr;v=std::strtod(s,&end);return end!=s&&!*end&&std::isfinite(v)&&v>=0&&v<=.5;
}
// Strict, fail-closed configuration; mode 0 and domain=1.5 are exactly v166.
static bool settings167(State166&s){
 const char*env=std::getenv("V167_GATE_MODE");
 if(env){ char*end=nullptr; long v=std::strtol(env,&end,10);
  if(end==env||*end||v<0||v>6)return false; s.gate_mode=(int)v; }
 auto parse=[](const char*key,double&out,double lo,double hi)->bool{
  const char*v=std::getenv(key);if(!v)return true;
  char*end=nullptr;double x=std::strtod(v,&end);
  if(end==v||*end||!std::isfinite(x)||x<lo||x>hi)return false;
  out=x;return true;
 };
 return parse("V167_DOMAIN_MIN",s.domain_min,-5.,5.)&&
        parse("V167_HIGH_ALPHA",s.high_alpha,0.,.5)&&
        parse("V167_TRUST_THRESHOLD",s.trust_threshold,0.,1.);
}
static bool load_gate168(State166&state,const char*path){
 std::ifstream f(path);if(!f)return false;
 std::string magic; if(!(f>>magic)||magic!="N168GATE1")return false;
 double w=0;
 for(int j=0;j<7;j++){if(!(f>>w)||!std::isfinite(w)||std::abs(w)>100.)return false;state.calibration[j]=w;}
 std::string tail; if(f>>tail)return false;
 if(!f.eof())return false;
 state.calibration_active=true;return true;
}
static double feature168(const State166&s,int j){
 switch(j){
 case 0:return 1.;
 case 1:return std::log(std::max(s.top_q,1e-12));
 case 2:return std::log(std::max(s.parent_ratio,1e-12));
 case 3:return std::min(5.,std::max(-5.,s.parent->domain_score-1.5));
 case 4:return s.parent->alpha;
 case 5:return s.parent_ratio>=.5?1.:0.;
 case 6:return (s.parent->candidates.empty()?0.:1.);
 }
 return 0.;
}
static double pair166(const Cloud166&c,const std::string&history,const uint32_t target){
 uint32_t b=hash166(history);double score=0;
 for(int d=0;d<8;d++){
  double l=c.lower[b][d],u=c.upper[target][d];
  score+=(double)c.plus[target][b][d]*std::tanh(l+u)-(double)c.minus[target][b][d]*std::tanh(l-u)+l*u/std::sqrt(8.);
 }
 return score;
}
static void birth166(State166*s,const unsigned char*past,size_t n){
 s->alpha=0;s->top_q=0;s->parent_ratio=0;s->p.clear();s->trie_mass.clear();s->order.clear();s->parent_prefix.clear();s->parent_prefix[""]=1.;
 if(s->alpha_max<=0||!s->birth||!past||!n)return;
 // The inherited v164 causal book/domain gate protects documentation settings.
 if(s->parent->domain_gate&&s->parent->domain_score<s->domain_min)return;
 auto words=complete_words158(past,n);if(words.empty())return;
 const auto&g=*s->birth;const auto&c=*g.cloud;
 double q[DIM166]={};double temporal=0;
 const size_t take=std::min((size_t)3,words.size());
 static constexpr double recency[]={.2,.3,.5};
 double signedfield[128]={};
 for(size_t j=0;j<take;j++){
  const std::string&w=words[words.size()-take+j];const double r=recency[3-take+j];temporal+=r;
  auto it=g.lookup.find(w);
  if(it!=g.lookup.end())for(int d=0;d<DIM166;d++)q[d]+=r*g.lex[it->second].ctx[d];
  for(uint32_t u=0;u<128;u++)signedfield[u]+=r*pair166(c,w,u);
 }
 if(!(temporal>0))return;
 for(int d=0;d<DIM166;d++)q[d]/=temporal;
 for(double&v:signedfield)v/=temporal;
 s->p.resize(g.lex.size());s->order.resize(g.lex.size());
 std::vector<double>logits(g.lex.size());double mx=-std::numeric_limits<double>::infinity();
 for(size_t i=0;i<g.lex.size();i++){
  const auto&w=g.lex[i];double lex=0;
  for(int d=0;d<DIM166;d++)lex+=q[d]*w.out[d];
  double val=w.bias+g.lex_gain*lex/std::sqrt((double)DIM166)+g.cloud_gain*signedfield[w.hash];
  if(!std::isfinite(val))return;
  logits[i]=val;mx=std::max(mx,val);s->order[i]=i;
 }
 double sum=0;for(auto x:logits)sum+=std::exp(x-mx);
 if(!(sum>0)||!std::isfinite(sum))return;
 for(size_t i=0;i<logits.size();i++)s->p[i]=std::exp(logits[i]-mx)/sum;
 s->trie_mass.assign(g.trie.size(),0.);
 for(size_t i=0;i<g.lex.size();i++)s->trie_mass[g.lex[i].leaf]=s->p[i];
 // Bottom-up subtree mass. Each child was created after its parent; do not
 // renormalize away the mass of words outside the top-96 diagnostics.
 for(size_t j=g.trie.size();j-- >0;){
  for(int child:g.trie[j].edge)if(child>=0)s->trie_mass[j]+=s->trie_mass[child];
 }
 const size_t limit=std::min((size_t)96,s->order.size());
 std::partial_sort(s->order.begin(),s->order.begin()+limit,s->order.end(),[&](size_t a,size_t b){return s->p[a]!=s->p[b]?s->p[a]>s->p[b]:g.lex[a].word<g.lex[b].word;});
 s->order.resize(limit);
 // The parent table supplies one candidate family, not the entire decoder.
 // We compare its strongest proposal with the neural winner only using past
 // context and model probabilities; no future word, labels, or decoder output.
 if(!s->order.empty())s->top_q=s->p[s->order.front()];
 if(!s->parent->candidates.empty()&&s->top_q>0){
  const std::string&pw=s->parent->candidates.front().first;
  auto it=g.lookup.find(pw);
  if(it!=g.lookup.end())s->parent_ratio=s->p[it->second]/s->top_q;
 }
 s->alpha=s->alpha_max;
 bool boost=false;
 switch(s->gate_mode){
  case 1:boost=s->parent_ratio>=s->trust_threshold;break;
  case 2:boost=s->parent_ratio<s->trust_threshold;break;
  case 3:boost=s->parent->domain_score>=1.5+0.5*s->trust_threshold;break;
  case 4:boost=s->parent->alpha<=s->trust_threshold;break;
  case 5:boost=s->top_q>=s->trust_threshold;break;
  case 6:boost=s->top_q<s->trust_threshold;break;
 }
 if(s->gate_mode&&boost)s->alpha=std::max(s->alpha,s->high_alpha);
 if(s->calibration_active){
  double z=0.;for(int j=0;j<7;j++)z+=s->calibration[j]*feature168(*s,j);
  s->alpha=.5/(1.+std::exp(-std::max(-30.,std::min(30.,z))));
 }

}
static double mass166(const State166*s,const std::string&pre,double out[27]){
 for(int j=0;j<27;j++)out[j]=0.;
 int node=0;
 for(char c:pre){int next=s->birth->trie[node].edge[c-'a'];if(next<0)return 0.;node=next;}
 const auto&t=s->birth->trie[node];
 if(t.terminal>=0)out[26]=s->p[t.terminal];
 for(int j=0;j<26;j++)if(t.edge[j]>=0)out[j]=s->trie_mass[t.edge[j]];
 return s->trie_mass[node];
}
static double parent_mass166(State166*s,const std::string&pre){
 auto it=s->parent_prefix.find(pre);if(it!=s->parent_prefix.end())return it->second;
 double mass=1.;std::string prefix;
 for(char c:pre){
  auto key=prefix+c;auto iter=s->parent_prefix.find(key);
  if(iter!=s->parent_prefix.end()){mass=iter->second;prefix=key;continue;}
  float out[27]={},esc=0;
  if(wf164_next_impl(s->parent,prefix.data(),prefix.size(),out,&esc)!=0)return 0.;
  mass*=std::max(0.,(double)out[c-'a']+(double)esc/27.);
  s->parent_prefix[key]=mass;prefix=key;
 }
 return mass;
}
}
extern "C" void*wf148_open(const char*m,const char*g,const char*c){
 std::unique_ptr<State166>s(new State166);
 if(!setting166(s->alpha_max)||!settings167(*s))return nullptr;
 if(const char*gate=getenv("V168_GATE_FILE")){if(!load_gate168(*s,gate))return nullptr;}

 s->parent=(Native164*)wf164_open_impl(m,g,c);if(!s->parent)return nullptr;
 if(s->alpha_max>0){
  const char*path=std::getenv("V166_BIRTH_MODEL");s->birth.reset(new Model166);
  if(!path||!read_birth166(path,*s->birth,s->parent)){
   wf164_close_impl(s->parent);return nullptr;
  }
 }
 return s.release();
}
extern "C" void wf148_close(void*h){auto*s=(State166*)h;if(!s)return;wf164_close_impl(s->parent);delete s;}
extern "C" int wf148_begin(void*h,const unsigned char*past,size_t n,float cloud){
 auto*s=(State166*)h;if(!s)return -1;
 int ret=wf164_begin_impl(s->parent,past,n,cloud);if(ret<0)return ret;
 birth166(s,past,n);return ret+(s->alpha>0);
}
extern "C" int wf148_next(void*h,const char*pre,size_t n,float out[27],float*escape){
 auto*s=(State166*)h;if(!s||(!pre&&n)||!out||!escape)return -1;
 int ok=wf164_next_impl(s->parent,pre,n,out,escape);
 if(ok||s->alpha<=0)return ok;
 if(n>64)return -2;
 std::string prefix(pre,n);for(char c:prefix)if(c<'a'||c>'z')return -2;
 double cm[27];const double mass=mass166(s,prefix,cm);
 if(mass<=0)return ok;
 const double pmass=parent_mass166(s,prefix);
 const double denominator=(1.-s->alpha)*pmass+s->alpha*mass;
 if(!(denominator>0)||!std::isfinite(denominator))return ok;
 double total=0.;for(int j=0;j<27;j++){
  const double p=out[j]+(double)(*escape)/27.;
  out[j]=(float)(((1.-s->alpha)*pmass*p+s->alpha*cm[j])/denominator);
  total+=out[j];
 }
 if(!(total>0)||!std::isfinite(total))return -4;
 for(int j=0;j<27;j++)out[j]/=(float)total;
 *escape=0;return 0;
}
extern "C" double wf168_neural_word_prob(void*h,const char*w){
 auto*s=(State166*)h;if(!s||!s->birth||!w)return -1.;
 auto it=s->birth->lookup.find(w);return it==s->birth->lookup.end()?0.:s->p.empty()?0.:s->p[it->second];
}
extern "C" double wf168_parent_word_prob(void*h,const char*word){
 auto*s=(State166*)h;if(!s||!word)return -1.;
 std::string w(word);if(w.empty()||w.size()>20)return -1.;
 for(char c:w)if(c<'a'||c>'z')return -1.;
 double p=1.;
 for(size_t i=0;i<=w.size();i++){
  float out[27]={},e=0.;if(wf164_next_impl(s->parent,w.data(),i,out,&e))return -1.;
  int k=i==w.size()?26:(w[i]-'a');p*=std::max(0.,(double)out[k]+(double)e/27.);
 }
 return p;
}
extern "C" int wf168_features(void*h,double*out,int cap){
 auto*s=(State166*)h;if(!s||!out||cap<7)return -1;
 for(int i=0;i<7;i++)out[i]=feature168(*s,i);
 return 7;
}
extern "C" double wf167_effective_alpha(void*h){auto*s=(State166*)h;return s?s->alpha:-1.;}
extern "C" double wf167_route_score(void*h){auto*s=(State166*)h;return s?s->parent->domain_score:-1e9;}
extern "C" double wf167_parent_ratio(void*h){auto*s=(State166*)h;return s?s->parent_ratio:-1.;}
extern "C" double wf167_parent_alpha(void*h){auto*s=(State166*)h;return s?s->parent->alpha:-1.;}
extern "C" double wf167_neural_top_q(void*h){auto*s=(State166*)h;return s?s->top_q:-1.;}
extern "C" int wf166_candidate_count(void*h){auto*s=(State166*)h;return s?(int)s->order.size():-1;}
extern "C" int wf166_candidate(void*h,int i,char*out,size_t cap,double*prob){
 auto*s=(State166*)h;if(!s||i<0||i>=(int)s->order.size()||!out||!prob)return -1;
 const auto idx=s->order[i];auto&w=s->birth->lex[idx].word;
 if(cap<=w.size())return -2;std::copy(w.begin(),w.end(),out);out[w.size()]=0;*prob=s->p[idx];return 0;
}
extern "C" int wf148_fuse(const float*p,const float*w,float e,float v,int pr,float*out){return wf164_fuse_impl(p,w,e,v,pr,out);}
extern "C" int wf148_fuse_surgical(const float*p,const float*w,float e,float g,float*out){return wf164_fuse_surgical_impl(p,w,e,g,out);}
extern "C" int wf157_logits(void*h,float*out,size_t n){auto*s=(State166*)h;return s?wf164_logits_impl(s->parent,out,n):-1;}
extern "C" int wf157_chars(void*h,const char*p,size_t n,float*out){auto*s=(State166*)h;return s?wf164_chars_impl(s->parent,p,n,out):-1;}
extern "C" int wf158_pointer(void*h,char*w,size_t c,double*a){auto*s=(State166*)h;return s?wf164_pointer_impl(s->parent,w,c,a):-1;}
extern "C" int wf158_candidate_count(void*h){auto*s=(State166*)h;return s?wf164_old_candidate_count_impl(s->parent):-1;}
extern "C" int wf159_count(void*h){auto*s=(State166*)h;return s?wf164_old159_count_impl(s->parent):-1;}
extern "C" int wf159_candidate(void*h,int i,char*w,size_t c,double*p,int*m){auto*s=(State166*)h;return s?wf164_old159_candidate_impl(s->parent,i,w,c,p,m):-1;}