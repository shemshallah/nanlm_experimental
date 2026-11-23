// NaNCM v158: mathematically normalized causal whole-word pointer mixture.
// Retains every v157 path; optional feature has exact opt-out parity.
// Copyright (c) Justin Howard-Stanley. Research-only candidate.
#include <unordered_map>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>
#include <limits>
#define wf148_open wf157_open_impl
#define wf148_begin wf157_begin_impl
#define wf148_next wf157_next_impl
#define wf148_close wf157_close_impl
#define wf148_fuse wf157_fuse_impl
#define wf148_fuse_surgical wf157_fuse_surgical_impl
#define wf157_logits wf157_logits_impl
#define wf157_chars wf157_chars_impl
#include "wordfield157_api.cpp"
#undef wf148_open
#undef wf148_begin
#undef wf148_next
#undef wf148_close
#undef wf148_fuse
#undef wf148_fuse_surgical
#undef wf157_logits
#undef wf157_chars

struct Pointer158 {
 std::string word;
 double score=0;
 double prior=0;
 int matches=0;
};
struct State158 {
 void* base=nullptr;
 float threshold=1.f, budget=0.3f, topic_budget=0.f;
 std::vector<Pointer158> candidates;
 std::unordered_map<std::string,double> base_prefix_probability;
 double branch_mass=0;
};
static bool isletter158(unsigned char c){return (c>='a'&&c<='z')||(c>='A'&&c<='Z');}
static std::vector<std::string> complete_words158(const unsigned char*past,size_t n){
 std::vector<std::string> words;
 for(size_t i=0;i<n;){
  if(!isletter158(past[i])){++i;continue;}
  size_t j=i+1;while(j<n&&isletter158(past[j]))++j;
  // Last word is incomplete without a following delimiter.
  if(j==n)break;
  std::string w;w.reserve(j-i);
  for(size_t k=i;k<j;k++)w.push_back((char)std::tolower(past[k]));
  if(w.size()<=64)words.push_back(std::move(w));
  i=j;
 }
 return words;
}
static bool prior_suffix158(const std::vector<std::string>&w,size_t i,size_t count){
 const size_t n=w.size();
 if(i+1<count||n<count)return false;
 for(size_t k=0;k<count;k++)if(w[i-k]!=w[n-1-k])return false;
 return true;
}
static std::vector<Pointer158> propose158(const std::vector<std::string>&w,float threshold,float budget){
 std::unordered_map<std::string,Pointer158> votes;
 const size_t n=w.size();if(n<2)return {};
 for(size_t i=0;i+1<n;i++){
  int agreement=0;
  for(int k=1;k<=6&&k<=(int)i+1&&k<=(int)n;k++){
   if(w[i+1-k]!=w[n-k])break;
   agreement=k;
  }
  if(!agreement)continue;
  const auto&word=w[i+1];
  double distance=(double)(n-i-1);
  double weight=std::pow(std::log1p((double)agreement),2)/(1.+0.0001*distance);
  auto&candidate=votes[word];candidate.word=word;candidate.score+=weight;candidate.matches++;
 }
 std::vector<Pointer158> v;
 for(auto&kv:votes)if(kv.second.score>0)v.push_back(kv.second);
 std::sort(v.begin(),v.end(),[](const auto&a,const auto&b){if(a.score!=b.score)return a.score>b.score;return a.word<b.word;});
 if(v.empty()||v[0].score<threshold)return {};
 // Only the strongest causal successor is admissible. It is a *whole word*
 // branch, not a repeated per-character reward that kills long OOV strings.
 double confidence=v[0].score;
 double next=v.size()>1?v[1].score:0.;
 if(next>confidence*.9)return {}; // suppress contested continuations
 // Lower bounds ensure a useful prior at known recurrence; higher evidence
 // increases it smoothly without ever removing the inherited distribution.
 double alpha=budget*std::min(1.,std::max(.15,confidence/(confidence+1.)));
 v[0].prior=alpha;
 v.resize(1);
 return v;
}
static inline bool starts158(const std::string&w,const std::string&p){return w.size()>=p.size()&&w.compare(0,p.size(),p)==0;}
static bool base_symbol158(State158*s,const std::string&pre,int symbol,double &p){
 float a[27]={},escape=0;int ok=wf157_next_impl(s->base,pre.c_str(),pre.size(),a,&escape);
 if(ok!=0)return false;
 if(symbol<0||symbol>=27)return false;
 p=(double)a[symbol]+(double)escape/27.;
 return std::isfinite(p)&&p>=0&&p<=1.00001;
}
static double prefix_mass158(State158*s,const std::string&prefix){
 auto existing=s->base_prefix_probability.find(prefix);
 if(existing!=s->base_prefix_probability.end())return existing->second;
 double pb=1.;std::string pre;
 for(char c:prefix){
  pre.push_back(c);
  auto found=s->base_prefix_probability.find(pre);
  if(found!=s->base_prefix_probability.end()){pb=found->second;continue;}
  double p=0.;std::string prev=pre.substr(0,pre.size()-1);
  if(!base_symbol158(s,prev,c-'a',p))return 0.;
  pb*=p;s->base_prefix_probability[pre]=pb;
 }
 return pb;
}
extern "C" void*wf148_open(const char*m,const char*g,const char*c){
 std::unique_ptr<State158> s(new State158);
 const char*thr=std::getenv("V158_POINTER_THRESHOLD");
 const char*bdg=std::getenv("V158_POINTER_BUDGET");
 const char*topic=std::getenv("V158_TOPIC_BUDGET");
 if(thr)s->threshold=std::atof(thr);if(bdg)s->budget=std::atof(bdg);if(topic)s->topic_budget=std::atof(topic);
 if(!std::isfinite(s->threshold)||s->threshold<0||s->threshold>50||
    !std::isfinite(s->budget)||s->budget<0||s->budget>1||
 !std::isfinite(s->topic_budget)||s->topic_budget<0||s->topic_budget>0.3||
 s->topic_budget+s->budget>1)return nullptr;
 s->base=wf157_open_impl(m,g,c);if(!s->base)return nullptr;
 return s.release();
}
extern "C" void wf148_close(void*h){if(!h)return;auto*s=(State158*)h;wf157_close_impl(s->base);delete s;}
extern "C" int wf148_begin(void*h,const unsigned char*past,size_t n,float cloud_strength){
 auto*s=(State158*)h;if(!s||(!past&&n))return -1;
 s->candidates.clear();s->branch_mass=0;s->base_prefix_probability.clear();s->base_prefix_probability[""]=1.;
 int status=wf157_begin_impl(s->base,past,n,cloud_strength);
 if(status<0)return status;
 if(s->budget>0){
  const auto w=complete_words158(past,n);
  s->candidates=propose158(w,s->threshold,s->budget);
  if(s->topic_budget>0){
   // A tiny topic-cache branch only for tokens absent from the trained word
   // vocabulary; it cannot invent future tokens or access target suffix.
   const auto &model=((WF148State*)s->base)->model;
   struct Local { int count=0; size_t nearest=999999; double score=0;};
   std::unordered_map<std::string,Local> local;
   for(size_t i=w.size()>384?w.size()-384:0;i<w.size();i++){
    const auto&token=w[i];
    if(token.size()<4||token.size()>24||model.id.count(token))continue;
    auto&st=local[token];st.count++;
    st.nearest=std::min(st.nearest,w.size()-i);
   }
   std::vector<Pointer158> proposals;
   for(const auto&kv:local){
    bool exists=false;for(const auto&c:s->candidates)if(c.word==kv.first)exists=true;
    if(exists)continue;
    Pointer158 p;p.word=kv.first;p.matches=kv.second.count;
    p.score=(1.+std::log1p((double)kv.second.count)) / (1.+0.009*kv.second.nearest);
    proposals.push_back(std::move(p));
   }
   std::sort(proposals.begin(),proposals.end(),[](const auto&a,const auto&b){
    if(a.score!=b.score)return a.score>b.score;return a.word<b.word;});
   if(proposals.size()>12)proposals.resize(12);
   double z=0;for(const auto&p:proposals)z+=p.score;
   for(auto&p:proposals){p.prior=s->topic_budget*p.score/std::max(1.e-15,z);s->candidates.push_back(std::move(p));}
  }
  for(const auto&c:s->candidates)s->branch_mass+=c.prior;
 }
 return status+(int)s->candidates.size();
}
extern "C" int wf148_next(void*h,const char*prefix,size_t n,float out[27],float*escape){
 auto*s=(State158*)h;if(!s||(!prefix&&n)||!out||!escape)return -1;
 int ok=wf157_next_impl(s->base,prefix,n,out,escape);
 if(ok!=0||s->candidates.empty())return ok;
 if(n>64)return -2;
 std::string pre(prefix,n);
 for(char c:pre)if(c<'a'||c>'z')return -2;
 double pbase=prefix_mass158(s,pre),branch=0.;
 for(const auto&c:s->candidates)if(starts158(c.word,pre))branch+=c.prior;
 double denom=(1.-s->branch_mass)*pbase+branch;
 if(denom<=0||!std::isfinite(denom))return ok;
 double basecoef=(1.-s->branch_mass)*pbase/denom;
 double q[27]={};
 for(const auto&c:s->candidates){
  if(!starts158(c.word,pre))continue;
  int symbol=c.word.size()==n?26:(int)(c.word[n]-'a');
  if(symbol>=0&&symbol<27)q[symbol]+=c.prior/denom;
 }
 double total=0.;
 for(int j=0;j<27;j++){
  double base=(double)out[j]+(double)(*escape)/27.;
  out[j]=(float)(basecoef*base+q[j]);total+=out[j];
 }
 if(total<=0||!std::isfinite(total))return -3;
 for(int j=0;j<27;j++)out[j]/=(float)total;
 *escape=0.f;return 0;
}
extern "C" int wf158_pointer(void*h,char*word,size_t capacity,double*alpha){
 auto*s=(State158*)h;if(!s||!alpha)return -1;
 if(s->candidates.empty()){*alpha=0.;if(word&&capacity)word[0]=0;return 0;}
 auto&c=s->candidates[0];*alpha=c.prior;
 if(!word||capacity<=c.word.size())return -2;
 std::copy(c.word.begin(),c.word.end(),word);word[c.word.size()]=0;
 return 1;
}
extern "C" int wf148_fuse(const float*p,const float*w,float e,float s,int prot,float*out){return wf157_fuse_impl(p,w,e,s,prot,out);}
extern "C" int wf148_fuse_surgical(const float*p,const float*w,float e,float g,float*out){return wf157_fuse_surgical_impl(p,w,e,g,out);}

// Diagnostic bridges must unwrap the actual v157 handle.
extern "C" int wf157_logits(void*h,float*out,size_t n){
 auto*s=(State158*)h;return s?wf157_logits_impl(s->base,out,n):-1;
}
extern "C" int wf157_chars(void*h,const char*p,size_t n,float*out){
 auto*s=(State158*)h;return s?wf157_chars_impl(s->base,p,n,out):-1;
}
extern "C" int wf158_candidate_count(void*h){
 auto*s=(State158*)h;return s?(int)s->candidates.size():-1;
}
