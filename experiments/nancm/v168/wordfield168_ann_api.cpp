// NaNCM v168 — TRAIN-only signed-cloud analogical neighbor birth (native)
// Exact v167 ABI rollback when V168_ANN_STRENGTH=0; no inherited weight edits.
#include "wordfield168_embed_ann.cpp"
#include <queue>
#include <unordered_map>
#include <fstream>
#include <utility>
#include <cstring>
#include <cmath>
#include <memory>
#include <array>
#include <algorithm>
#include <string>
#include <limits>
namespace {
struct Neighbor168 {
 std::array<float,32> lex{};
 std::array<float,16> sign{};
 uint8_t k=0,n=0;
 std::array<uint32_t,8> target{};
 std::array<float,8> p{};
};
struct Ann168 {
 std::vector<std::string> words;
 std::array<std::array<float,16>,128> signed_hash{};
 std::vector<Neighbor168> addresses;
};
struct State168 {
 State166 *parent=nullptr;
 std::unique_ptr<Ann168> memory;
 double strength=0.,signed_share=.66,domain_min=1.4,alpha=0.;
 std::vector<std::pair<std::string,double>> candidates;
 std::unordered_map<std::string,double> prefix_cache;
};
static bool get_number168(const char*name,double&val,double min,double max){
 const char*env=getenv(name);if(!env)return true;
 char*e=nullptr;val=strtod(env,&e);
 return e!=env&&!*e&&std::isfinite(val)&&val>=min&&val<=max;
}
static bool load_index168(const char*path,Ann168&ann,const State166*base){
 if(!base||!base->parent||!base->parent->parent)return false;
 const auto*a=(const State159*)base->parent->parent->base;
 const auto*b=a?(const State158*)a->base:nullptr;
 const auto*r=b?(const WF148State*)b->base:nullptr;
 if(!r)return false;
 std::ifstream f(path,std::ios::binary);if(!f)return false;
 char magic[8];uint64_t fp[3];uint32_t sizes[4];
 if(!f.read(magic,8)||std::memcmp(magic,"N168ANN1",8)||
    !f.read((char*)fp,sizeof(fp))||!f.read((char*)sizes,sizeof(sizes)))return false;
 if(fp[0]!=r->model.trainhash||fp[1]!=r->model.geomhash||fp[2]!=r->model.cloudhash)return false;
 if(sizes[0]<100||sizes[0]>300000||sizes[1]<100||sizes[1]>100000||sizes[2]!=32||sizes[3]!=16)return false;
 ann.words.reserve(sizes[1]);
 for(uint32_t i=0;i<sizes[1];i++){
  uint8_t n=0;if(!f.read((char*)&n,1)||n<1||n>20)return false;
  std::string w(n,' ');if(!f.read(&w[0],n))return false;
  for(char c:w)if(c<'a'||c>'z')return false;
  ann.words.emplace_back(std::move(w));
 }
 if(!f.read((char*)&ann.signed_hash,sizeof(ann.signed_hash)))return false;
 ann.addresses.resize(sizes[0]);
 for(auto&v:ann.addresses){
  if(!f.read((char*)v.lex.data(),32*sizeof(float))||!f.read((char*)v.sign.data(),16*sizeof(float))||
     !f.read((char*)&v.k,1)||!f.read((char*)&v.n,1))return false;
  if(v.k<1||v.k>3||v.n<1||v.n>8)return false;
  for(int i=0;i<8;i++){
   if(!f.read((char*)&v.target[i],4)||!f.read((char*)&v.p[i],4))return false;
   if(v.target[i]>=sizes[1]||!std::isfinite(v.p[i])||v.p[i]<0||v.p[i]>1)return false;
  }
 }
 char tail;if(f.read(&tail,1)||!f.eof())return false;
 // Check all vector values for finite safe magnitudes.
 for(const auto&v:ann.addresses){
  for(float x:v.lex)if(!std::isfinite(x)||std::abs(x)>2.)return false;
  for(float x:v.sign)if(!std::isfinite(x)||std::abs(x)>2.)return false;
 }
 for(const auto&v:ann.signed_hash)for(float x:v)if(!std::isfinite(x)||std::abs(x)>2.)return false;
 return true;
}
template<size_t D> static void normalize168(std::array<double,D>&v){
 double n=0.;for(auto x:v)n+=x*x;n=std::sqrt(n);
 if(n>1e-12)for(auto&x:v)x/=n;
}
static void propose168(State168*s,const unsigned char*past,size_t n){
 s->alpha=0.;s->candidates.clear();s->prefix_cache.clear();s->prefix_cache[""]=1.;
 if(s->strength<=0||!s->memory||!s->parent||!past||!n)return;
 if(s->parent->parent->domain_gate && s->parent->parent->domain_score<s->domain_min)return;
 if(!s->parent->birth)return;
 const auto past_words=complete_words158(past,n);if(past_words.empty())return;
 const auto&g=*s->parent->birth;
 const auto&mem=*s->memory;
 std::array<double,32>q{};std::array<double,16>z{};
 const size_t take=std::min((size_t)3,past_words.size());static constexpr double rec[]={.2,.3,.5};double denom=0.;
 for(size_t j=0;j<take;j++){
  const auto&w=past_words[past_words.size()-take+j];double r=rec[3-take+j];denom+=r;
  auto it=g.lookup.find(w);
  if(it!=g.lookup.end())for(int d=0;d<32;d++)q[d]+=r*g.lex[it->second].ctx[d];
  const auto&c=mem.signed_hash[hash166(w)];for(int d=0;d<16;d++)z[d]+=r*c[d];
 }
 if(denom<=0)return;
 for(auto&x:q)x/=denom;for(auto&x:z)x/=denom;
 normalize168(q);normalize168(z);
 using Ranked=std::pair<double,size_t>;
 std::priority_queue<Ranked,std::vector<Ranked>,std::greater<Ranked>>best;
 const auto&all=mem.addresses;
 for(size_t i=0;i<all.size();i++){
  const auto&a=all[i];double l=0.,t=0.;
  for(int d=0;d<32;d++)l+=q[d]*a.lex[d];
  for(int d=0;d<16;d++)t+=z[d]*a.sign[d];
  double sim=(1.-s->signed_share)*l+s->signed_share*t;
  if(best.size()<32)best.push({sim,i});
  else if(sim>best.top().first){best.pop();best.push({sim,i});}
 }
 if(best.empty())return;
 std::vector<Ranked>rank;
 while(!best.empty()){rank.push_back(best.top());best.pop();}
 std::sort(rank.begin(),rank.end(),[](auto a,auto b){return a.first>b.first|| (a.first==b.first&&a.second<b.second);});
 std::unordered_map<uint32_t,double>counts;
 for(size_t j=0;j<rank.size();j++){
  const auto &ref=all[rank[j].second];
  double score=std::pow(std::max(.01,rank[j].first),3)*
     (1./(1.+j/3.))*((ref.k==3)?1.5:(ref.k==2)?1.2:1.);
  for(int k=0;k<ref.n;k++)counts[ref.target[k]]+=score*ref.p[k];
 }
 if(counts.empty())return;
 std::vector<std::pair<uint32_t,double>>sorted(counts.begin(),counts.end());
 std::sort(sorted.begin(),sorted.end(),[&](auto a,auto b){return a.second>b.second||(a.second==b.second&&mem.words[a.first]<mem.words[b.first]);});
 if(sorted.size()>64)sorted.resize(64);
 double mass=0.;for(const auto&x:sorted)mass+=x.second;
 if(!(mass>0)||!std::isfinite(mass))return;
 s->candidates.reserve(sorted.size());
 for(auto &x:sorted)s->candidates.emplace_back(mem.words[x.first],x.second/mass);
 s->alpha=s->strength;
}
static double parentprefix168(State168*s,const std::string&pre){
 auto found=s->prefix_cache.find(pre);if(found!=s->prefix_cache.end())return found->second;
 double mass=1.;std::string prefix;
 for(char c:pre){
  auto nxt=prefix+c;auto it=s->prefix_cache.find(nxt);
  if(it!=s->prefix_cache.end()){mass=it->second;prefix=nxt;continue;}
  float o[27]={},e=0.;if(wf167_next_impl(s->parent,prefix.data(),prefix.size(),o,&e))return 0.;
  mass*=std::max(0.,(double)o[c-'a']+e/27.);s->prefix_cache[nxt]=mass;prefix=nxt;
 }
 return mass;
}
}
extern "C" void*wf148_open(const char*m,const char*g,const char*c){
 auto s=std::unique_ptr<State168>(new State168);
 if(!get_number168("V168_ANN_STRENGTH",s->strength,0.,.5)||
    !get_number168("V168_ANN_SIGNED_SHARE",s->signed_share,0.,1.)||
    !get_number168("V168_ANN_DOMAIN_MIN",s->domain_min,-5.,5.))return nullptr;
 s->parent=(State166*)wf167_open_impl(m,g,c);if(!s->parent)return nullptr;
 if(s->strength>0){
  const char*path=getenv("V168_ANN_INDEX");s->memory.reset(new Ann168());
  if(!path||!load_index168(path,*s->memory,s->parent)){
   wf167_close_impl(s->parent);return nullptr;
  }
 }
 return s.release();
}
extern "C" void wf148_close(void*h){auto*s=(State168*)h;if(!s)return;wf167_close_impl(s->parent);delete s;}
extern "C" int wf148_begin(void*h,const unsigned char*past,size_t n,float cloud){
 auto*s=(State168*)h;if(!s)return -1;
 int ret=wf167_begin_impl(s->parent,past,n,cloud);if(ret<0)return ret;
 propose168(s,past,n);return ret+(s->alpha>0);
}
extern "C" int wf148_next(void*h,const char*pre,size_t n,float out[27],float*escape){
 auto*s=(State168*)h;if(!s||(!pre&&n)||!out||!escape)return -1;
 int ok=wf167_next_impl(s->parent,pre,n,out,escape);
 if(ok||s->alpha<=0)return ok;
 if(n>64)return -2;
 std::string prefix(pre,n);for(char c:prefix)if(c<'a'||c>'z')return -2;
 double cm[27]={},sum=0.;
 for(const auto&w:s->candidates){
  if(w.first.size()<n||w.first.compare(0,n,prefix)!=0)continue;
  sum+=w.second;
  const int j=w.first.size()==n?26:w.first[n]-'a';cm[j]+=w.second;
 }
 if(sum<=0)return ok;
 double p=parentprefix168(s,prefix);
 double d=(1.-s->alpha)*p+s->alpha*sum;
 if(!(d>0)||!std::isfinite(d))return ok;
 double norm=0.;
 for(int j=0;j<27;j++){
  out[j]=(float)(((1.-s->alpha)*p*(out[j]+*escape/27.)+s->alpha*cm[j])/d);
  norm+=out[j];
 }
 if(!(norm>0)||!std::isfinite(norm))return -3;
 for(int j=0;j<27;j++)out[j]/=norm;
 *escape=0.;return 0;
}
extern "C" double wf168_ann_alpha(void*h){auto*s=(State168*)h;return s?s->alpha:-1.;}
extern "C" int wf168_ann_candidate_count(void*h){auto*s=(State168*)h;return s?(int)s->candidates.size():-1;}
extern "C" int wf168_ann_candidate(void*h,int i,char*out,size_t cap,double*w){
 auto*s=(State168*)h;if(!s||i<0||i>=(int)s->candidates.size()||!out||!w)return -1;
 const auto&v=s->candidates[i];if(cap<=v.first.size())return -2;
 std::copy(v.first.begin(),v.first.end(),out);out[v.first.size()]=0;*w=v.second;return 0;
}
extern "C" double wf167_effective_alpha(void*h){auto*s=(State168*)h;return s?wf167_effective_alpha_impl(s->parent):-1.;}
extern "C" double wf167_route_score(void*h){auto*s=(State168*)h;return s?wf167_route_score_impl(s->parent):-1.;}
extern "C" int wf148_fuse(const float*p,const float*w,float e,float v,int pr,float*out){return wf167_fuse_impl(p,w,e,v,pr,out);}
extern "C" int wf148_fuse_surgical(const float*p,const float*w,float e,float g,float*out){return wf167_fuse_surgical_impl(p,w,e,g,out);}
extern "C" int wf157_logits(void*h,float*out,size_t n){auto*s=(State168*)h;return s?wf167_logits_impl(s->parent,out,n):-1;}
extern "C" int wf157_chars(void*h,const char*p,size_t n,float*out){auto*s=(State168*)h;return s?wf167_chars_impl(s->parent,p,n,out):-1;}
extern "C" int wf158_pointer(void*h,char*w,size_t c,double*a){auto*s=(State168*)h;return s?wf167_pointer_impl(s->parent,w,c,a):-1;}
extern "C" int wf158_candidate_count(void*h){auto*s=(State168*)h;return s?wf167_parent_candidate_count_impl(s->parent):-1;}
extern "C" int wf159_count(void*h){auto*s=(State168*)h;return s?wf167_159_count_impl(s->parent):-1;}
extern "C" int wf159_candidate(void*h,int i,char*w,size_t c,double*p,int*m){auto*s=(State168*)h;return s?wf167_159_candidate_impl(s->parent,i,w,c,p,m):-1;}
extern "C" double wf168_neural_word_prob(void*h,const char*w){auto*s=(State168*)h;return s?wf167_wf168_neural_word_prob_impl(s->parent,w):-1.;}
extern "C" int wf166_candidate_count(void*h){auto*s=(State168*)h;return s?wf167_wf166_candidate_count_impl(s->parent):-1;}
extern "C" int wf166_candidate(void*h,int i,char*w,size_t c,double*p){auto*s=(State168*)h;return s?wf167_wf166_candidate_impl(s->parent,i,w,c,p):-1;}