// Native NaNCM v176 trained 14-feature counterfactual copy gate and
// v133-inspired TRAIN-only causal word-start letter expert.
// This is opt-in: wf176_hybrid_open (without sidecar) preserves exact v175 top5.
// OOV byte handling remains the original v170 verifier in native_third176.cpp.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
namespace cross176 {
struct Pair {int a,b;bool operator==(const Pair&o)const{return a==o.a&&b==o.b;}};
struct PairHash {size_t operator()(const Pair&x)const{return (uint64_t)(uint32_t)x.a*0x9e3779b185ebca87ull^(uint32_t)x.b;}};
struct BPE {
 uint32_t V=1856;float prior=24.f,beta=.35f;
 std::array<uint16_t,12000>first{};
 std::vector<uint32_t>global;
 std::vector<double>global_p;
 using Sparse=std::vector<std::pair<uint16_t,uint32_t>>;
 std::unordered_map<int,Sparse>c1;
 std::unordered_map<Pair,Sparse,PairHash>c2;
};
struct Native {
 n175::Now *parent=nullptr;
 std::unique_ptr<BPE> bpe;
 n175::Now noecho;
 std::array<float,14> weight{};float bias=0.f,prior=24.f,beta=.5f;
 std::array<uint32_t,26> global{};
 std::vector<uint32_t> c1;
 std::unordered_map<Pair,std::array<uint32_t,26>,PairHash> c2;
 std::array<float,26> glob_p{};
};
template<class T>bool read(std::ifstream&f,T&x){return bool(f.read((char*)&x,sizeof(x)));}
static bool load(const char*path,Native&n){
 if(!path||!n.parent)return false;
 std::ifstream f(path,std::ios::binary);if(!f)return false;
 char magic[8];if(!f.read(magic,8)||std::memcmp(magic,"N176HYB1",8))return false;
 for(int i=0;i<3;i++){uint64_t fp;if(!read(f,fp)||fp!=n.parent->fingerprints[i])return false;}
 char checksum[32];if(!f.read(checksum,32))return false;
 uint32_t vocab=0,features=0,keys=0;
 if(!read(f,vocab)||!read(f,n.prior)||!read(f,features)||!read(f,keys))return false;
 if(vocab!=12000||features!=14||keys>95040||n.prior!=24.f)return false;
 if(!f.read((char*)n.weight.data(),14*sizeof(float))||!read(f,n.bias))return false;
 for(auto x:n.weight)if(!std::isfinite(x)||std::fabs(x)>100)return false;
 if(!std::isfinite(n.bias)||std::fabs(n.bias)>100)return false;
 if(!f.read((char*)n.global.data(),26*sizeof(uint32_t)))return false;
 n.c1.resize((size_t)12001*26);
 if(!f.read((char*)n.c1.data(),n.c1.size()*sizeof(uint32_t)))return false;
 n.c2.reserve(keys*2);
 for(uint32_t i=0;i<keys;i++){
  int32_t a,b;std::array<uint32_t,26>counts{};
  if(!read(f,a)||!read(f,b)||!f.read((char*)counts.data(),26*sizeof(uint32_t)))return false;
  if(a < -1 || a>=12000 || b < -1||b>=12000 || !n.c2.emplace(Pair{a,b},counts).second)return false;
 }
 if(f.peek()!=EOF)return false;
 double sum=26.;for(auto c:n.global)sum+=c;
 for(int k=0;k<26;k++)n.glob_p[k]=(n.global[k]+1.)/sum;
 // The inherited model's exact parameter tensor is cloned only once. The
 // noecho branch sets copy strength softplus(-100) ~0 and leaves ALL trained
 // temporal and signed-cloud parameters unchanged.
 n.noecho=*n.parent;
 n.noecho.t.at("copy_log_strength").a.at(0)=-100.f;
 return true;
}
static bool load_bpe(const char*path,Native&n){
 if(!path||!n.parent)return false;
 try{
  std::ifstream f(path,std::ios::binary);if(!f)return false;
  char magic[8];if(!f.read(magic,8)||std::memcmp(magic,"N176BPE1",8))return false;
  for(int i=0;i<3;i++){uint64_t fp;if(!read(f,fp)||fp!=n.parent->fingerprints[i])return false;}
  char crosssha[32];if(!f.read(crosssha,32))return false;
  uint32_t vocab=0,V=0,n1=0,n2=0;float prior=0.,beta=0.;
  if(!read(f,vocab)||!read(f,V)||!read(f,prior)||!read(f,beta)||!read(f,n1)||!read(f,n2))return false;
  if(vocab!=12000||V!=1856||prior!=24.f||beta!=.35f||n1>12001||n2>95040)return false;
  auto b=std::make_unique<BPE>();b->prior=prior;b->beta=beta;
  if(!f.read((char*)b->first.data(),b->first.size()*sizeof(uint16_t)))return false;
  for(auto x:b->first)if(x>=V)return false;
  b->global.resize(V);b->global_p.resize(V);
  if(!f.read((char*)b->global.data(),V*sizeof(uint32_t)))return false;
  double total=V;for(auto x:b->global)total+=x;
  for(uint32_t j=0;j<V;j++)b->global_p[j]=(b->global[j]+1.)/total;
  auto sparse=[&](BPE::Sparse&out)->bool{
   uint16_t count=0;if(!read(f,count)||count>V)return false;
   out.reserve(count);int last=-1;
   for(uint16_t k=0;k<count;k++){
    uint16_t idx=0;uint32_t n=0;
    if(!read(f,idx)||!read(f,n)||idx>=V||idx<=last||n==0)return false;
    out.emplace_back(idx,n);last=idx;
   }
   return true;
  };
  for(uint32_t j=0;j<n1;j++){
   int32_t a=0;if(!read(f,a)||a < -1||a>=12000)return false;
   BPE::Sparse v;if(!sparse(v)||!b->c1.emplace(a,std::move(v)).second)return false;
  }
  for(uint32_t j=0;j<n2;j++){
   int32_t a=0,c=0;if(!read(f,a)||!read(f,c)||a < -1||a>=12000||c < -1||c>=12000)return false;
   BPE::Sparse v;if(!sparse(v)||!b->c2.emplace(Pair{a,c},std::move(v)).second)return false;
  }
  if(f.peek()!=EOF)return false;
  n.bpe=std::move(b);return true;
 }catch(...){return false;}
}
static std::array<int,48> parse(const n175::Now&s,const unsigned char*past,size_t n){
 std::array<int,48>out{};out.fill(-1);
 std::vector<int>ids;ids.reserve(std::min<size_t>(n/4,10000));std::string word;
 auto flush=[&](){if(word.empty())return;auto it=s.ids.find(word);ids.push_back(it==s.ids.end()?-1:it->second);word.clear();};
 for(size_t i=0;i<n;i++){
  const unsigned char c=past[i];
  if((c>='a'&&c<='z')||(c>='A'&&c<='Z')){
   if(word.size()<64)word.push_back(c>='A'&&c<='Z'?(char)(c+32):(char)c);
  }else flush();
 }flush();
 size_t count=std::min<size_t>(48,ids.size());
 for(size_t j=0;j<count;j++)out[48-count+j]=ids[ids.size()-count+j];
 return out;
}
static std::pair<int,int> top2(const float*p,int n){
 int a=-1,b=-1;
 for(int i=0;i<n;i++)if(a<0||p[i]>p[a]){b=a;a=i;}else if(b<0||p[i]>p[b])b=i;
 return {a,b};
}
inline double logprob(float x){return std::log(std::max(1e-20f,x));}
static int predict(Native&st,const unsigned char*past,size_t n,float*out,size_t cap){
 if(!out||cap<12000||(!past&&n)||n>1000000)return -1;
 try{
  float full[12000],base[12000];
  n175::predict(*st.parent,past,n,full);
  n175::predict(st.noecho,past,n,base);
  const auto h=parse(*st.parent,past,n);
  double f[14]{};int match=0,pair=0,triple=0,latest=-1,seen=0;std::unordered_map<int,int>hist;
  for(int j=0;j<48;j++)seen+=(h[j]>=0);
  if(h[47]>=0)for(int j=0;j<47;j++){
   if(h[j]!=h[47]||h[j+1]<0)continue;
   match++;latest=j;
   pair+=(j>0&&h[j-1]==h[46]&&h[46]>=0);
   triple+=(j>1&&h[j-1]==h[46]&&h[j-2]==h[45]&&h[45]>=0);
   hist[h[j+1]]++;
  }
  int maxcount=0;for(const auto&x:hist)maxcount=std::max(maxcount,x.second);
  f[0]=std::log1p(match);f[1]=std::log1p(pair);f[2]=std::log1p(triple);
  f[3]=latest>=0?(latest+1)/48.:0.;
  f[4]=(double)hist.size()/std::max(1,match);
  f[5]=(double)maxcount/std::max(1,match);f[6]=seen/48.;
  auto ap=top2(base,12000),bp=top2(full,12000);
  f[7]=logprob(base[ap.first])-logprob(base[ap.second]);
  f[8]=logprob(full[bp.first])-logprob(full[bp.second]);
  f[9]=base[ap.first];f[10]=full[bp.first];f[11]=ap.first==bp.first;
  f[12]=logprob(full[ap.first])-logprob(base[ap.first]);
  f[13]=logprob(base[bp.first])-logprob(full[bp.first]);
  double logit=st.bias;for(int i=0;i<14;i++)logit+=st.weight[i]*std::max(-15.,std::min(15.,f[i]));
  logit=std::max(-80.,std::min(80.,logit));
  const double gate=.01+.98/(1.+std::exp(-logit));
  double marginal[26]{};int letters[12000];
  for(int i=0;i<12000;i++){
   auto word=st.parent->names[i];int l=(int)(word[0]-'a');
   if(l<0||l>=26)return -2;
   letters[i]=l;out[i]=(float)((1.-gate)*base[i]+gate*full[i]);
   marginal[l]+=out[i];
  }
  const int a=h[47],b=h[46];
  const uint32_t*row=st.c1.data()+(a+1)*26;
  double s1=0.;for(int j=0;j<26;j++)s1+=row[j];
  double p1[26],letter[26];
  for(int j=0;j<26;j++)p1[j]=(row[j]+st.prior*st.glob_p[j])/(s1+st.prior);
  auto item=st.c2.find(Pair{a,b});
  if(item!=st.c2.end()){
   double s2=0.;for(int j=0;j<26;j++)s2+=item->second[j];
   for(int j=0;j<26;j++)letter[j]=(item->second[j]+st.prior*p1[j])/(s2+st.prior);
  }else for(int j=0;j<26;j++)letter[j]=p1[j];
  double factors[26];
  for(int j=0;j<26;j++)factors[j]=std::pow(std::max(1e-8,letter[j])/std::max(1e-8,marginal[j]),st.beta);
  double z=0.;for(int i=0;i<12000;i++){out[i]*=(float)factors[letters[i]];z+=out[i];}
  if(!(z>0.)||!std::isfinite(z))return -3;
  for(int i=0;i<12000;i++){out[i]=(float)(out[i]/z);if(!std::isfinite(out[i])||out[i]<0)return -4;}
  // Optional third coding: normalized conditional first learned subword.
  // Counts were constructed on TRAIN only. Backoff mirrors Python exactly:
  // global -> preceding word -> preceding two words, each Dirichlet smoothed.
  if(st.bpe){
   const auto&bt=*st.bpe;
   std::vector<double>mass(bt.V,0.),p1=bt.global_p,probs(bt.V);
   for(int i=0;i<12000;i++)mass[bt.first[i]]+=out[i];
   auto v1=bt.c1.find(a);
   if(v1!=bt.c1.end()){
    double cnt=0.;for(const auto&q:v1->second)cnt+=q.second;
    for(auto &q:p1)q*=bt.prior;
    for(const auto&q:v1->second)p1[q.first]+=q.second;
    for(auto &q:p1)q/=(bt.prior+cnt);
   }
   probs=p1;
   auto v2=bt.c2.find(Pair{a,b});
   if(v2!=bt.c2.end()){
    double cnt=0.;for(const auto&q:v2->second)cnt+=q.second;
    for(auto &q:probs)q*=bt.prior;
    for(const auto&q:v2->second)probs[q.first]+=q.second;
    for(auto &q:probs)q/=(bt.prior+cnt);
   }
   std::vector<double>weight(bt.V);
   for(size_t j=0;j<weight.size();j++)weight[j]=std::pow(std::max(1e-8,probs[j])/std::max(1e-8,mass[j]),bt.beta);
   double zp=0.;for(int i=0;i<12000;i++){out[i]*=(float)weight[bt.first[i]];zp+=out[i];}
   if(!(zp>0.)||!std::isfinite(zp))return -5;
   for(int i=0;i<12000;i++)out[i]=(float)(out[i]/zp);
  }
  return 0;
 }catch(...){return -5;}
}
}
extern "C" void* wf176_cross_open(void*parent,const char*sidecar){
 if(!parent)return nullptr;
 try{
  auto n=std::make_unique<cross176::Native>();n->parent=(n175::Now*)parent;
  if(!cross176::load(sidecar,*n))return nullptr;
  return n.release();
 }catch(...){return nullptr;}
}
extern "C" void wf176_cross_close(void*state){delete (cross176::Native*)state;}
extern "C" int wf176_cross_predict(void*state,const unsigned char*past,size_t n,float*out,size_t cap){
 auto*s=(cross176::Native*)state;return s?cross176::predict(*s,past,n,out,cap):-1;
}
extern "C" void* wf176_hybrid_open_cross(const char*teacher,const char*m,const char*g,const char*c,
                                            const char*bytepack,float escape,const char*calib){
 auto*s=(third176::State*)wf176_hybrid_open(teacher,m,g,c,bytepack,escape);
 if(!s)return nullptr;
 s->cross=wf176_cross_open(s->words,calib);
 if(!s->cross){wf176_hybrid_close(s);return nullptr;}
 return s;
}

extern "C" int wf176_cross_add_bpe(void*state,const char*sidecar){
 auto*s=(cross176::Native*)state;
 return s&&cross176::load_bpe(sidecar,*s)?0:-1;
}
extern "C" void* wf176_hybrid_open_full(const char*teacher,const char*m,const char*g,const char*c,
                                         const char*bytepack,float escape,const char*calib,const char*bpe){
 auto*s=(third176::State*)wf176_hybrid_open_cross(teacher,m,g,c,bytepack,escape,calib);
 if(!s)return nullptr;
 if(wf176_cross_add_bpe(s->cross,bpe)!=0){wf176_hybrid_close(s);return nullptr;}
 return s;
}