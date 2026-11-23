// NaNCM v159 candidate: byte-context worldline pointer, composed *after* v158.
// All matching is strictly before the target position. v159 disabled => exact v158.
#include <unordered_map>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>
#include <limits>
#include <cctype>
#include <array>
#include <unordered_set>
#include "wordfield158_embed159.cpp"

struct ByteWorldline159 {std::string word; double evidence=0,weight=0; int matches=0, max_suffix=0;};
struct State159 {
 void *base=nullptr;
 double budget=0.0, spell_budget=0.0;
 bool unknown_only=true;
 std::unordered_set<std::string> known_prefixes;
 int min_match=12, max_candidates=4;
 double mass=0;
 std::vector<ByteWorldline159> words;
 std::unordered_map<std::string,double> prefix_p;
 std::unordered_map<std::string,std::array<unsigned int,27>> spell;
};
static bool letter159(unsigned char c){return (c>='a'&&c<='z')||(c>='A'&&c<='Z');}
static std::vector<ByteWorldline159> byteworldlines159(const unsigned char*ctx,size_t n,int minmatch,int maxcand,double budget){
 if(!ctx||n<(size_t)minmatch+3||!budget)return {};
 std::unordered_map<std::string,ByteWorldline159> votes;
 // Current observed byte suffix matches the past ending just before an
 // earlier, fully observed word. Compare only *causal* bytes; never the target.
 // Bounded to last 32 KiB and last 48 suffix bytes to control CPU cost.
 const size_t start=n>32768?n-32768:0;
 for(size_t i=start;i+1<n;){
  if(!letter159(ctx[i]) || (i>0&&letter159(ctx[i-1]))){i++;continue;}
  size_t j=i+1;while(j<n&&letter159(ctx[j]))j++;
  if(j>=n){break;}
  if(i>= (size_t)minmatch){
   size_t m=0,lim=std::min((size_t)48,std::min(i,n));
   while(m<lim && ctx[i-m-1]==ctx[n-m-1])++m;
   if((int)m>=minmatch && j-i>=2 && j-i<=32){
    std::string word;word.reserve(j-i);
    for(size_t z=i;z<j;z++)word.push_back((char)std::tolower(ctx[z]));
    // A longer byte-context match is a *different* source of evidence from
    // word-level suffix agreement used in v158.
    const double recency=1./(1.+.0003*(n-i));
    double score=std::log1p((double)m)*std::log1p((double)m)*recency;
    auto &vote=votes[word]; vote.word=word;vote.evidence+=score;vote.matches++;
    vote.max_suffix=std::max(vote.max_suffix,(int)m);
   }
  }
  i=j;
 }
 std::vector<ByteWorldline159> v;
 for(auto &kv:votes)v.push_back(kv.second);
 std::sort(v.begin(),v.end(),[](const auto&a,const auto&b){return a.evidence!=b.evidence?a.evidence>b.evidence:a.word<b.word;});
 if(v.empty())return {};
 if((int)v.size()>maxcand)v.resize(maxcand);
 // Evidence confidence and disagreement-aware gate. A contested memory
 // gets less branch probability than an unambiguous long match.
 double total=0.;for(const auto&x:v)total+=x.evidence;
 const double clarity=v[0].evidence/total;
 const double support=std::min(1.,(v[0].max_suffix-minmatch+1.)/20.);
 const double alpha=budget*(.35+.65*support)*(.3+.7*clarity);
 for(auto&x:v)x.weight=alpha*x.evidence/total;
 return v;
}

// Local, causal, word-internal Markov statistics. Each complete *past* word
// contributes suffix->next-character observations including EOS. All counts
// rebuild at wf148_begin; there is no target word or future lookahead.
static void learn_spelling159(State159*s,const unsigned char*past,size_t n){
 s->spell.clear();if(!s->spell_budget || !past || n==0)return;
 size_t start=n>8192?n-8192:0;
 auto words=complete_words158(past+start,n-start);
 for(const auto&w:words){
  if(w.size()>32)continue;
  for(size_t k=1;k<=w.size();k++){
   int symbol=k==w.size()?26:(int)(w[k]-'a');
   if(symbol<0||symbol>=27)continue;
   for(size_t order=1;order<=4 && order<=k;order++){
    std::string key=std::to_string(order)+":"+w.substr(k-order,order);
    s->spell[key][symbol]++;
   }
  }
 }
}
static void adapt_spelling159(State159*s,const std::string&pre,float out[27],float*escape){
 if(!s->spell_budget || pre.empty() || pre.size()>64)return;
 if(s->unknown_only && s->known_prefixes.count(pre))return;
 const std::array<unsigned int,27>*counts=nullptr;unsigned int total=0;
 for(size_t order=std::min((size_t)4,pre.size());order>=1;order--){
  std::string key=std::to_string(order)+":"+pre.substr(pre.size()-order,order);
  auto it=s->spell.find(key);
  if(it!=s->spell.end()){
   unsigned int c=0;for(auto n:it->second)c+=n;
   if(c>=2 || order==1){counts=&it->second;total=c;break;}
  }
  if(order==1)break;
 }
 if(!counts||!total)return;
 const double alpha=s->spell_budget*(double)total/(total+8.);
 double norm=0.;
 // Additive parent-informed smoothing gives an independent productive char
 // route while still allowing the global speller to retain rare symbols.
 for(int j=0;j<27;j++){
  const double base=(double)out[j]+(double)(*escape)/27.;
  const double local=((double)(*counts)[j]+.5*base)/(total+.5);
  out[j]=(float)((1.-alpha)*base+alpha*local);norm+=out[j];
 }
 for(int j=0;j<27;j++)out[j]/=(float)norm;
 *escape=0.;
}
static inline bool starts159(const std::string&w,const std::string&p){return w.size()>=p.size()&&w.compare(0,p.size(),p)==0;}
static bool base_symbol159(State159*s,const std::string&p,int symbol,double&prob){
 float a[27]={},escape=0.f;
 int ok=wf158_next_impl(s->base,p.c_str(),p.size(),a,&escape);
 if(ok!=0||symbol<0||symbol>=27)return false;
 prob=(double)a[symbol]+(double)escape/27.;
 return std::isfinite(prob)&&prob>=0.;
}
static double prefix_mass159(State159*s,const std::string&p){
 auto it=s->prefix_p.find(p);if(it!=s->prefix_p.end())return it->second;
 double pb=1;std::string prev;
 for(char c:p){
  std::string cur=prev+c;
  auto got=s->prefix_p.find(cur);
  if(got!=s->prefix_p.end()){pb=got->second;prev=cur;continue;}
  double q=0;if(!base_symbol159(s,prev,c-'a',q))return 0.;
  pb*=q;s->prefix_p[cur]=pb;prev=cur;
 }
 return pb;
}
extern "C" void*wf148_open(const char*m,const char*g,const char*c){
 std::unique_ptr<State159> s(new State159);
 auto read=[](const char*k,double fallback){const char*v=std::getenv(k);return v?std::atof(v):fallback;};
 s->budget=read("V159_BYTE_BUDGET",0.0);
 s->spell_budget=read("V159_CHAR_BUDGET",0.0);
 double unknown=read("V159_CHAR_ONLY_UNKNOWN",1.);
 if(!std::isfinite(unknown) || (unknown!=0. && unknown!=1.))return nullptr;
 s->unknown_only=(unknown==1.);
 double len=read("V159_MIN_MATCH",12),count=read("V159_MAX_CANDIDATES",4);
 if(!std::isfinite(s->budget)||s->budget<0||s->budget>.5||
 !std::isfinite(s->spell_budget)||s->spell_budget<0||s->spell_budget>.5||
 !std::isfinite(len)||len!=std::floor(len)||len<4||len>48||
 !std::isfinite(count)||count!=std::floor(count)||count<1||count>16)return nullptr;
 s->min_match=(int)len;s->max_candidates=(int)count;
 s->base=wf158_open_impl(m,g,c);if(!s->base)return nullptr;
 if(s->spell_budget && s->unknown_only){
  const auto &id=((WF148State*)((State158*)s->base)->base)->model.id;
  for(const auto& kv:id){
   const auto&w=kv.first;
   for(size_t k=1;k<=w.size()&&k<=64;k++)s->known_prefixes.insert(w.substr(0,k));
  }
 }
 return s.release();
}
extern "C" void wf148_close(void*h){if(!h)return;auto*s=(State159*)h;wf158_close_impl(s->base);delete s;}
extern "C" int wf148_begin(void*h,const unsigned char*past,size_t n,float cloud){
 auto*s=(State159*)h;if(!s||(!past&&n))return -1;
 s->words.clear();s->mass=0;s->prefix_p.clear();s->prefix_p[""]=1.;
 int status=wf158_begin_impl(s->base,past,n,cloud);if(status<0)return status;
 learn_spelling159(s,past,n);
 if(s->budget){s->words=byteworldlines159(past,n,s->min_match,s->max_candidates,s->budget);
  for(const auto&p:s->words)s->mass+=p.weight;
 }
 return status+(int)s->words.size();
}
extern "C" int wf148_next(void*h,const char*prefix,size_t n,float out[27],float*escape){
 auto*s=(State159*)h;if(!s||(!prefix&&n)||!out||!escape)return -1;
 int ok=wf158_next_impl(s->base,prefix,n,out,escape);if(ok)return ok;
 if(s->words.empty()){if(s->spell_budget&&n){std::string p(prefix,n);adapt_spelling159(s,p,out,escape);}return ok;}
 if(n>64)return -2;
 std::string pre(prefix,n);for(char c:pre)if(c<'a'||c>'z')return -2;
 const double baseprefix=prefix_mass159(s,pre);
 double branch=0;for(const auto&w:s->words)if(starts159(w.word,pre))branch+=w.weight;
 double denom=(1.-s->mass)*baseprefix+branch;
 if(!(denom>0.)||!std::isfinite(denom))return ok;
 double q[27]={};
 for(const auto&w:s->words)if(starts159(w.word,pre)){
  int symbol=(w.word.size()==n)?26:(int)(w.word[n]-'a');
  if(symbol>=0&&symbol<27)q[symbol]+=w.weight/denom;
 }
 const double baseline=(1.-s->mass)*baseprefix/denom;
 double total=0.;for(int k=0;k<27;k++){
  double old=(double)out[k]+(double)(*escape)/27.;
  out[k]=(float)(baseline*old+q[k]);total+=out[k];
 }
 if(!(total>0.)||!std::isfinite(total))return -3;
 for(int k=0;k<27;k++)out[k]/=(float)total;
 *escape=0.f;adapt_spelling159(s,pre,out,escape);return 0;
}
extern "C" int wf159_count(void*h){auto*s=(State159*)h;return s?(int)s->words.size():-1;}
extern "C" int wf159_candidate(void*h,int index,char*word,size_t capacity,double*prob,int*match){
 auto*s=(State159*)h;if(!s||index<0||index>=(int)s->words.size()||!word||!prob||!match)return -1;
 auto&w=s->words[index];if(capacity<=w.word.size())return -2;
 std::copy(w.word.begin(),w.word.end(),word);word[w.word.size()]=0;
 *prob=w.weight;*match=w.max_suffix;return 0;
}
extern "C" int wf148_fuse(const float*p,const float*w,float e,float v,int prot,float*out){return wf158_fuse_impl(p,w,e,v,prot,out);}
extern "C" int wf148_fuse_surgical(const float*p,const float*w,float e,float g,float*out){return wf158_fuse_surgical_impl(p,w,e,g,out);}
extern "C" int wf157_logits(void*h,float*out,size_t n){auto*s=(State159*)h;return s?wf158_logits_impl(s->base,out,n):-1;}
extern "C" int wf157_chars(void*h,const char*p,size_t n,float*out){auto*s=(State159*)h;return s?wf158_chars_impl(s->base,p,n,out):-1;}
extern "C" int wf158_pointer(void*h,char*w,size_t cap,double*a){auto*s=(State159*)h;return s?wf158_pointer_impl(s->base,w,cap,a):-1;}
extern "C" int wf158_candidate_count(void*h){auto*s=(State159*)h;return s?wf158_candidate_count_impl(s->base):-1;}
