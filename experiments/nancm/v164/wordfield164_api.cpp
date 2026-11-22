// NaNCM v164: a normalized, whole-word distribution over multiple conditional
// candidates; trained only from independent NaNROUTER TRAIN books.
// Exact v162 parity when V164_DISTRIBUTION_BUDGET=0. Original cloud frozen.
#include <memory>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include "wordfield162_embed163.cpp"
struct DistWord164{std::string word;int count=0,total=0;};
struct Cond164{std::vector<DistWord164> words;int total=0;};
struct Native164{
 State160* parent=nullptr;
 double budget=0,alpha=0,threshold=1.5,domain_score=-1e9;
 bool domain_gate=true;
 State160 domain_book;
 std::unordered_map<std::string,Cond164> model;
 std::vector<std::pair<std::string,double>> candidates;
 std::unordered_map<std::string,double> parent_prefix;
};
static bool read_distribution164(const char*path,Native164*s,WF148State*root){
 std::ifstream in(path);if(!in)return false;
 std::string line;size_t nrows=0;
 if(!std::getline(in,line))return false;
 {
  std::vector<std::string> meta;std::stringstream ss(line);std::string cell;
  while(std::getline(ss,cell,'\t'))meta.push_back(cell);
  if(meta.size()!=4||meta[0]!="N164DIST1")return false;
  uint64_t train=0,geom=0,cloud=0;
  try{train=std::stoull(meta[1]);geom=std::stoull(meta[2]);cloud=std::stoull(meta[3]);}catch(...){return false;}
  if(!root||train!=root->model.trainhash||geom!=root->model.geomhash||cloud!=root->model.cloudhash)return false;
 }
 while(std::getline(in,line)){
  std::vector<std::string>v;std::string cell;
  std::stringstream ss(line);while(std::getline(ss,cell,'\t'))v.push_back(cell);
  if(v.size()!=5)return false;
  int k,n,total;
  try{k=std::stoi(v[0]);n=std::stoi(v[3]);total=std::stoi(v[4]);}catch(...){return false;}
  if(k<1||k>3||n<=0||total<n||v[2].empty()||v[2].size()>20)return false;
  for(char c:v[2])if(c<'a'||c>'z')return false;
  std::string key=std::to_string(k)+":"+v[1];
  auto &dist=s->model[key];
  if(!dist.words.empty() && dist.total!=total)return false;
  if(dist.words.size()>=8)return false;
  if(std::any_of(dist.words.begin(),dist.words.end(),[&](const DistWord164&w){return w.word==v[2];}))return false;
  dist.total=total;dist.words.push_back({v[2],n,total});
  if(++nrows>1800000)return false;
 }
 return (in.good()||in.eof())&&nrows>0;
}
static double parse164(const char*key,double fallback,double low,double high,bool &ok){
 const char*p=getenv(key);if(!p)return fallback;
 char*end=nullptr;double x=strtod(p,&end);
 if(end==p||*end||!std::isfinite(x)||x<low||x>high){ok=false;return fallback;}
 return x;
}
static void propose164(Native164*s,const unsigned char*past,size_t n){
 s->candidates.clear();s->alpha=0;s->domain_score=-1e9;
 s->parent_prefix.clear();s->parent_prefix[""]=1.;
 if(!past||!n||s->budget<=0)return;
 auto w=complete_words158(past,n);
 if(w.empty())return;
 if(s->domain_gate){
  s->domain_score=domain_score162(&s->domain_book,past,n);
  if(s->domain_score<s->threshold)return;
 }
 std::unordered_map<std::string,double>votes;
 double rtotal=0;int maxorder=0;
 static constexpr double wt[]={0,.18,.85,1.30};
 static constexpr double prior[]={0,40.,8.,4.};
 for(int k=1;k<=3;k++){
  if(w.size()<(size_t)k)continue;
  std::string key=std::to_string(k)+":";
  for(int i=k;i>=1;i--){if(i!=k)key+=' ';key+=w[w.size()-i];}
  auto got=s->model.find(key);if(got==s->model.end())continue;
  const auto &z=got->second;
  if(z.total<=0)continue;
  const double reliability=wt[k]*z.total/(z.total+prior[k]);
  for(auto&a:z.words){votes[a.word]+=reliability*((double)a.count/z.total);}
  rtotal+=reliability;maxorder=std::max(maxorder,k);
 }
 if(votes.empty())return;
 for(const auto&v:votes)s->candidates.push_back({v.first,v.second});
 std::sort(s->candidates.begin(),s->candidates.end(),[](const auto&a,const auto&b){return a.second!=b.second?a.second>b.second:a.first<b.first;});
 if(s->candidates.size()>16)s->candidates.resize(16);
 double keep=0;for(auto&p:s->candidates)keep+=p.second;
 if(!(keep>0)) {s->candidates.clear();return;}
 for(auto&p:s->candidates)p.second/=keep;
 // More confident when there is independent context agreement and longer
 // histories; conservative for diffuse 1-gram candidates.
 const double maxp=s->candidates.front().second;
 s->alpha=std::min(.75,s->budget*std::min(1.,rtotal/1.2)*(.25+.75*std::min(1.,maxp*2)));
 if(maxorder==1)s->alpha*=.4;
}
static double parent_mass164(Native164*s,const std::string&pre){
 auto found=s->parent_prefix.find(pre);if(found!=s->parent_prefix.end())return found->second;
 double mass=1.;std::string prefix;
 for(char c:pre){
  std::string nxt=prefix+c;auto it=s->parent_prefix.find(nxt);
  if(it!=s->parent_prefix.end()){mass=it->second;prefix=nxt;continue;}
  float out[27]={},e=0.;
  if(wf162_wf148_next_impl(s->parent,prefix.c_str(),prefix.size(),out,&e)!=0)return 0.;
  mass*=std::max(0.,(double)out[c-'a']+(double)e/27.);
  s->parent_prefix[nxt]=mass;prefix=nxt;
 }
 return mass;
}
extern "C" void* wf148_open(const char*model,const char*geom,const char*cloud){
 std::unique_ptr<Native164>s(new Native164);
 bool ok=true;
 s->budget=parse164("V164_DISTRIBUTION_BUDGET",0.,0.,1.,ok);
 s->threshold=parse164("V164_DOMAIN_THRESHOLD",1.5,-5.,5.,ok);
 if(const char*g=getenv("V164_DOMAIN_GATE")){if(std::string(g)!="0"&&std::string(g)!="1")ok=false;else s->domain_gate=std::string(g)=="1";}
 if(!ok)return nullptr;
 // Mathematically normalized composition of a v162 single-word parent
 // and a separate v164 multiword distribution is allowed.
 s->parent=(State160*)wf162_wf148_open_impl(model,geom,cloud);
 if(!s->parent)return nullptr;
 if(s->budget>0){
  auto *m=(State159*)s->parent->base;
  auto *b=m?(State158*)m->base:nullptr;
  auto *root=b?(WF148State*)b->base:nullptr;
  const char*path=getenv("V164_DISTRIBUTION_INDEX");
  if(!path||!read_distribution164(path,s.get(),root)){
   wf162_wf148_close_impl(s->parent);return nullptr;
  }
  if(s->domain_gate){
   const char*book=getenv("V162_BOOK_FREQ"),*docs=getenv("V162_DOC_FREQ");
   if(!book||!docs||!read_freq162(book,s->domain_book.book_freq,s->domain_book.book_total)||
      !read_freq162(docs,s->domain_book.doc_freq,s->domain_book.doc_total)){
    wf162_wf148_close_impl(s->parent);return nullptr;
   }
   s->domain_book.domain_vocab=s->domain_book.book_freq.size();
   for(auto&p:s->domain_book.doc_freq)if(!s->domain_book.book_freq.count(p.first))s->domain_book.domain_vocab++;
  }
 }
 return s.release();
}
extern "C" void wf148_close(void*h){auto*s=(Native164*)h;if(!s)return;wf162_wf148_close_impl(s->parent);delete s;}
extern "C" int wf148_begin(void*h,const unsigned char*past,size_t n,float cloud){
 auto*s=(Native164*)h;if(!s)return -1;
 int r=wf162_wf148_begin_impl(s->parent,past,n,cloud);if(r<0)return r;
 propose164(s,past,n);return r+(s->alpha>0);
}
extern "C" int wf148_next(void*h,const char*prefix,size_t n,float out[27],float*escape){
 auto*s=(Native164*)h;if(!s||(!prefix&&n)||!out||!escape)return -1;
 int ok=wf162_wf148_next_impl(s->parent,prefix,n,out,escape);
 if(ok||s->alpha<=0)return ok;
 if(n>64)return -2;
 std::string pre(prefix,n);
 for(char c:pre)if(c<'a'||c>'z')return -2;
 double cmass=0.,cm[27]={};
 for(const auto&w:s->candidates){
  if(w.first.size()<n||w.first.compare(0,n,pre)!=0)continue;
  cmass+=w.second;
  int j=w.first.size()==n?26:w.first[n]-'a';
  if(j<0||j>26)return -3;
  cm[j]+=w.second;
 }
 if(cmass==0)return ok;
 const double pmass=parent_mass164(s,pre);
 const double d=(1.-s->alpha)*pmass+s->alpha*cmass;
 if(!(d>0.)||!std::isfinite(d))return ok;
 double sum=0.;for(int j=0;j<27;j++){
  double p=(double)out[j]+(double)(*escape)/27.;
  out[j]=(float)(((1.-s->alpha)*pmass*p+s->alpha*cm[j])/d);
  sum+=out[j];
 }
 if(!(sum>0.)||!std::isfinite(sum))return -4;
 for(int j=0;j<27;j++)out[j]/=(float)sum;
 *escape=0.;return 0;
}
extern "C" int wf164_candidates(void*h,int index,char*word,size_t cap,double*weight,double*alpha){
 auto*s=(Native164*)h;if(!s||index<0||index>=(int)s->candidates.size()||!word||!weight||!alpha)return -1;
 auto&w=s->candidates[index];if(cap<=w.first.size())return -2;
 std::copy(w.first.begin(),w.first.end(),word);word[w.first.size()]='\0';
 *weight=w.second;*alpha=s->alpha;return 0;
}
extern "C" int wf164_candidate_count(void*h){auto*s=(Native164*)h;return s?(int)s->candidates.size():-1;}
extern "C" double wf164_domain_score(void*h){auto*s=(Native164*)h;return s?s->domain_score:-1e9;}
extern "C" int wf148_fuse(const float*p,const float*w,float e,float v,int pr,float*out){return wf162_wf148_fuse_impl(p,w,e,v,pr,out);}
extern "C" int wf148_fuse_surgical(const float*p,const float*w,float e,float g,float*out){return wf162_wf148_fuse_surgical_impl(p,w,e,g,out);}
extern "C" int wf157_logits(void*h,float*out,size_t n){auto*s=(Native164*)h;return s?wf162_wf157_logits_impl(s->parent,out,n):-1;}
extern "C" int wf157_chars(void*h,const char*p,size_t n,float*out){auto*s=(Native164*)h;return s?wf162_wf157_chars_impl(s->parent,p,n,out):-1;}
extern "C" int wf158_pointer(void*h,char*w,size_t c,double*a){auto*s=(Native164*)h;return s?wf162_wf158_pointer_impl(s->parent,w,c,a):-1;}
extern "C" int wf158_candidate_count(void*h){auto*s=(Native164*)h;return s?wf162_wf158_candidate_count_impl(s->parent):-1;}
extern "C" int wf159_count(void*h){auto*s=(Native164*)h;return s?wf162_wf159_count_impl(s->parent):-1;}
extern "C" int wf159_candidate(void*h,int i,char*w,size_t c,double*p,int*m){auto*s=(Native164*)h;return s?wf162_wf159_candidate_impl(s->parent,i,w,c,p,m):-1;}
