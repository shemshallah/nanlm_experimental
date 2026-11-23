// NaNCM v160 — causal whole-word associative worldline; optional and reversible.
// Maintains exact v159 parity at zero budget. Never reads target/future bytes.
#include <unordered_map>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <memory>
#include <cmath>
#include <cstdlib>
#include "wordfield159_embed160.cpp"
#undef wf148_open
#undef wf148_close
#undef wf148_begin
#undef wf148_next
#undef wf148_fuse
#undef wf148_fuse_surgical
#undef wf157_logits
#undef wf157_chars
#undef wf158_pointer
#undef wf158_candidate_count
#undef wf159_count
#undef wf159_candidate

struct Assoc160{std::string target;int count=0,total=0,order=0;};
struct State160{
 void* base=nullptr;
 double budget=0,alpha=0;
 bool domain_gate=false;
 double domain_threshold=0.25,domain_score=-1e9;
 double book_total=0,doc_total=0;
 size_t domain_vocab=0;
 std::unordered_map<std::string,int> book_freq,doc_freq;
 int min_order=3;
 std::unordered_map<std::string,Assoc160> index;
 std::string word;
 std::unordered_map<std::string,double> prefix_mass;
};
static bool read_freq162(const char*path,std::unordered_map<std::string,int>&counts,double&total){
 std::ifstream in(path);if(!in)return false;std::string line;size_t lines=0;
 while(std::getline(in,line)){
  auto tab=line.find('\t');if(tab==std::string::npos)return false;
  std::string word=line.substr(0,tab);if(word.empty()||word.size()>64)return false;
  for(char c:word)if(c<'a'||c>'z')return false;
  int n;try{n=std::stoi(line.substr(tab+1));}catch(...){return false;}
  if(n<=0||counts.count(word))return false;
  counts.emplace(word,n);total+=n;lines++;
 }
 return lines>0&&(in.good()||in.eof());
}
static double domain_score162(State160*s,const unsigned char*past,size_t n){
 if(!past || n==0)return -1e9;
 auto words=complete_words158(past,n);
 if(words.size()<4)return -1e9;
 size_t start=words.size()>64?words.size()-64:0;
 double v=(double)s->domain_vocab,a=.5;
 double bd=s->book_total+a*v,dd=s->doc_total+a*v;
 double score=0.;size_t tokens=0;
 for(size_t i=start;i<words.size();++i){
  const auto&w=words[i];auto b=s->book_freq.find(w),d=s->doc_freq.find(w);
  double bn=b==s->book_freq.end()?0.:b->second;
  double dn=d==s->doc_freq.end()?0.:d->second;
  double q=std::log((bn+a)/bd)-std::log((dn+a)/dd);
  score+=std::max(-5.,std::min(5.,q));tokens++;
 }
 return tokens>0?score/tokens:-1e9;
}
static bool source160(State160*s,const char *path){
 std::ifstream in(path);if(!in.good())return false;
 std::string line;
 while(std::getline(in,line)){
  std::vector<std::string>p;std::stringstream ss(line);std::string part;
  while(std::getline(ss,part,'\t'))p.push_back(part);
  if(p.size()!=5) return false;
  int k,n,c;
  try{k=std::stoi(p[0]);c=std::stoi(p[3]);n=std::stoi(p[4]);}catch(...){return false;}
  if(k<1||k>4||c<1||n<c||p[2].size()>64)return false;
  std::string key=std::to_string(k)+":"+p[1];
  if(s->index.count(key))return false;
  s->index.emplace(key,Assoc160{p[2],c,n,k});
 }
 return in.good()||in.eof();
}
static double base_prefix_mass160(State160*s,const std::string&p){
 auto it=s->prefix_mass.find(p);if(it!=s->prefix_mass.end())return it->second;
 double prob=1.0;std::string prefix;
 for(char ch:p){
  std::string next=prefix+ch;auto saved=s->prefix_mass.find(next);
  if(saved!=s->prefix_mass.end()) {prob=saved->second;prefix=next;continue;}
  float a[27]={},esc=0.;if(wf159_next_impl(s->base,prefix.c_str(),prefix.size(),a,&esc)!=0)return 0.;
  prob*=std::max(0.,(double)a[ch-'a']+(double)esc/27.0);
  s->prefix_mass[next]=prob;prefix=next;
 }
 return prob;
}
extern "C" void* wf148_open(const char*m,const char*g,const char*c){
 std::unique_ptr<State160> s(new State160());
 const char*env=std::getenv("V160_ASSOC_BUDGET");
 if(env){char *end=nullptr;s->budget=std::strtod(env,&end);if(end==env||*end||!std::isfinite(s->budget)||s->budget<0.||s->budget>1.)return nullptr;}
 const char*minorder=std::getenv("V161_MIN_ORDER");if(minorder){char*end=nullptr;long n=std::strtol(minorder,&end,10);if(end==minorder||*end||n<1||n>3)return nullptr;s->min_order=(int)n;}
 const char*dg=std::getenv("V162_DOMAIN_GATE");
 if(dg){if(std::string(dg)!="0"&&std::string(dg)!="1")return nullptr;s->domain_gate=std::string(dg)=="1";}
 const char*th=std::getenv("V162_DOMAIN_THRESHOLD");
 if(th){char*end=nullptr;s->domain_threshold=std::strtod(th,&end);if(end==th||*end||!std::isfinite(s->domain_threshold)||s->domain_threshold< -5||s->domain_threshold>5)return nullptr;}
 if(s->budget>0){
  const char*path=std::getenv("V160_ASSOC_INDEX");if(!path||!source160(s.get(),path))return nullptr;
  if(s->domain_gate){
   const char*b=std::getenv("V162_BOOK_FREQ"),*d=std::getenv("V162_DOC_FREQ");
   if(!b||!d||!read_freq162(b,s->book_freq,s->book_total)||!read_freq162(d,s->doc_freq,s->doc_total))return nullptr;
   s->domain_vocab=s->book_freq.size();
   for(const auto&kv:s->doc_freq)if(!s->book_freq.count(kv.first))s->domain_vocab++;
  }
 }
 s->base=wf159_open_impl(m,g,c);if(!s->base)return nullptr;
 return s.release();
}
extern "C" void wf148_close(void*h){if(!h)return;auto*s=(State160*)h;wf159_close_impl(s->base);delete s;}
extern "C" int wf148_begin(void*h,const unsigned char*past,size_t n,float cloud){
 auto*s=(State160*)h;if(!s||(!past&&n))return -1;
 int ret=wf159_begin_impl(s->base,past,n,cloud);if(ret<0)return ret;
 s->word.clear();s->alpha=0;s->domain_score=-1e9;s->prefix_mass.clear();s->prefix_mass[""]=1.;
 if(s->budget==0||!past)return ret;
 auto words=complete_words158(past,n);
 for(int k=4;k>=s->min_order;k--){
  if(words.size()<(size_t)k)continue;
  std::string key=std::to_string(k)+":";
  for(int j=k;j>0;--j){if(j!=k)key.push_back(' ');key+=words[words.size()-j];}
  auto it=s->index.find(key);if(it==s->index.end())continue;
  const auto&z=it->second;
  const double purity=(double)z.count/z.total;
  const int support=k==1?4:(k==2?2:3);
  const double threshold=k==1?.08:(k==2?.2:.7);
  if(z.count>=support && z.total>=z.count && purity>=threshold){
   s->word=z.target;
   // Gate with frequency, purity and unique long-context evidence.
   double trust=std::min(1.0,z.count/(k<=2?10.0:5.0))*purity;
   s->alpha=std::min(.999,s->budget*trust);
   break;
  }
 }
 if(s->domain_gate){
  s->domain_score=domain_score162(s,past,n);
  if(s->domain_score<s->domain_threshold){s->alpha=0;s->word.clear();}
 }
 return ret+(s->alpha>0);
}
extern "C" int wf148_next(void*h,const char*prefix,size_t n,float out[27],float*escape){
 auto*s=(State160*)h;if(!s||(!prefix&&n)||!out||!escape)return -1;
 int ok=wf159_next_impl(s->base,prefix,n,out,escape);if(ok||s->alpha==0)return ok;
 if(n>64)return -2;
 std::string pre(prefix,n);
 for(char c:pre)if(c<'a'||c>'z')return -2;
 const bool match=s->word.size()>=n&&s->word.compare(0,n,pre)==0;
 const double bm=(1.0-s->alpha)*base_prefix_mass160(s,pre);
 const double pm=match?s->alpha:0.0;
 const double denom=bm+pm;
 if(!(denom>0.)||!std::isfinite(denom))return 0; // base backoff
 int hit=match?(s->word.size()==n?26:s->word[n]-'a'):-1;
 if(hit>26||hit<0){if(match)return -3;}
 double norm=0.;for(int j=0;j<27;j++){
  double base=(double)out[j]+(double)(*escape)/27.;
  out[j]=(float)((bm*base+(j==hit?pm:0.0))/denom);norm+=out[j];
 }
 if(!(norm>0.)||!std::isfinite(norm))return -4;
 for(int j=0;j<27;j++)out[j]/=(float)norm;
 *escape=0.;return 0;
}
extern "C" double wf162_domain_score(void*h){auto*s=(State160*)h;return s?s->domain_score:-1e9;}
extern "C" int wf160_association(void*h,char*out,size_t cap,double*alpha){
 auto*s=(State160*)h;if(!s||!out||!alpha)return -1;
 if(cap<=s->word.size())return -2;
 std::copy(s->word.begin(),s->word.end(),out);out[s->word.size()]='\0';*alpha=s->alpha;return s->alpha>0?1:0;
}
extern "C" int wf148_fuse(const float*p,const float*w,float e,float v,int prot,float*out){return wf159_fuse_impl(p,w,e,v,prot,out);}
extern "C" int wf148_fuse_surgical(const float*p,const float*w,float e,float g,float*out){return wf159_fuse_surgical_impl(p,w,e,g,out);}
extern "C" int wf157_logits(void*h,float*out,size_t n){auto*s=(State160*)h;return s?wf159_logits_impl(s->base,out,n):-1;}
extern "C" int wf157_chars(void*h,const char*p,size_t n,float*out){auto*s=(State160*)h;return s?wf159_chars_impl(s->base,p,n,out):-1;}
extern "C" int wf158_pointer(void*h,char*w,size_t cap,double*a){auto*s=(State160*)h;return s?wf159_pointer_impl(s->base,w,cap,a):-1;}
extern "C" int wf158_candidate_count(void*h){auto*s=(State160*)h;return s?wf159_candidate_count_impl(s->base):-1;}
extern "C" int wf159_count(void*h){auto*s=(State160*)h;return s?wf159_count_impl(s->base):-1;}
extern "C" int wf159_candidate(void*h,int i,char*w,size_t c,double*p,int*m){auto*s=(State160*)h;return s?wf159_candidate_impl(s->base,i,w,c,p,m):-1;}
