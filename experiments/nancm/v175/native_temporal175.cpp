// Native float32 causal signed-cloud 12k-word 48-history episodic expert.
// Inherits v174 weights and adds a learnable causal repeated-prefix pointer. This opt-in API
// does not change v167 default word predictions or v170 byte predictions.
#include <fstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
namespace n175 {
struct Tensor {std::vector<uint32_t> shape;std::vector<float> a;};
struct Now {
 std::unordered_map<std::string,Tensor> t;
 std::vector<std::string> names;
 std::unordered_map<std::string,int> ids;
 uint64_t fingerprints[3]{};
 const std::vector<float>& at(const char*name)const{return t.at(name).a;}
};
template<class T> bool readv(std::ifstream &f,T&v){return bool(f.read((char*)&v,sizeof(v)));}
bool load(const char*path,Now&s){
 if(!path)return false;
 std::ifstream f(path,std::ios::binary);if(!f)return false;
 char magic[8]{};if(!f.read(magic,8)||memcmp(magic,"N175NOW1",8))return false;
 for(auto &q:s.fingerprints)if(!readv(f,q))return false;
 uint32_t vocab=0,length=0,d=0,hidden=0;
 if(!readv(f,vocab)||!readv(f,length)||!readv(f,d)||!readv(f,hidden))return false;
 if(vocab!=12000||length!=48||d!=32||hidden!=64)return false;
 for(uint32_t i=0;i<vocab;i++){
  uint8_t len=0;if(!readv(f,len)||len==0||len>40)return false;
  std::string w(len,'\0');if(!f.read(&w[0],len))return false;
  for(char c:w)if(c<'a'||c>'z')return false;
  if(!s.ids.emplace(w,(int)i).second)return false;
  s.names.push_back(w);
 }
 uint32_t count=0;if(!readv(f,count)||count!=28)return false;
 for(uint32_t i=0;i<count;i++){
  uint8_t len=0,dim=0;if(!readv(f,len)||len==0||len>64)return false;
  std::string key(len,'\0');if(!f.read(&key[0],len)||!readv(f,dim)||dim>3)return false;
  Tensor item;size_t n=1;
  for(int j=0;j<dim;j++){uint32_t x=0;if(!readv(f,x)||x==0||x>10000000)return false;item.shape.push_back(x);n*=x;if(n>5000000)return false;}
  item.a.resize(n);
  if(!f.read((char*)item.a.data(),n*sizeof(float)))return false;
  for(float q:item.a)if(!std::isfinite(q))return false;
  if(!s.t.emplace(key,std::move(item)).second)return false;
 }
 if(f.peek()!=EOF)return false;
 auto shape=[&](const char*n,std::initializer_list<uint32_t>v){auto i=s.t.find(n);return i!=s.t.end()&&i->second.shape==std::vector<uint32_t>(v);};
 if(!shape("embed",{12000,32})||!shape("out",{12000,32})||!shape("bias",{12000})||
   !shape("geo",{12000,8})||!shape("projection",{32,8})||!shape("gain",{})||
   !shape("gpos",{12000,8})||!shape("gneg",{12000,8})||!shape("trust",{})||
   !shape("signed_scale",{})||!shape("signed_norm.weight",{192})||
   !shape("signed_norm.bias",{192})||!shape("to_lex.0.weight",{96,192})||
   !shape("to_lex.0.bias",{96})||!shape("to_lex.2.weight",{32,96})||
   !shape("to_lex.2.bias",{32}))return false;
 for(auto prefix:{"gru_pos","gru_neg"}){
  std::string g=prefix;
  if(!shape((g+".weight_ih_l0").c_str(),{192,48})||
     !shape((g+".weight_hh_l0").c_str(),{192,64})||
     !shape((g+".bias_ih_l0").c_str(),{192})||
     !shape((g+".bias_hh_l0").c_str(),{192}))return false;
 }
 return true;
}
inline float sigmoid(float x){return x>=0?1.f/(1.f+std::exp(-x)):std::exp(x)/(1.f+std::exp(x));}
void gru(const Now&s,const char*prefix,const float x[48],float h[64]){
 std::string name=prefix;
 const auto&wi=s.at((name+".weight_ih_l0").c_str());const auto&wh=s.at((name+".weight_hh_l0").c_str());
 const auto&bi=s.at((name+".bias_ih_l0").c_str());const auto&bh=s.at((name+".bias_hh_l0").c_str());
 float input[192],recurrent[192];
 for(int row=0;row<192;row++){
  float a=bi[row],b=bh[row];const float*ri=wi.data()+row*48,*rh=wh.data()+row*64;
  for(int i=0;i<48;i++)a+=ri[i]*x[i];
  for(int i=0;i<64;i++)b+=rh[i]*h[i];
  input[row]=a;recurrent[row]=b;
 }
 for(int i=0;i<64;i++){
  float reset=sigmoid(input[i]+recurrent[i]);
  float update=sigmoid(input[64+i]+recurrent[64+i]);
  float news=std::tanh(input[128+i]+reset*recurrent[128+i]);
  h[i]=(1.f-update)*news+update*h[i];
 }
}
void predict(const Now&s,const unsigned char*raw,size_t n,float*out){
 std::array<int,48>hist{};hist.fill(-1);
 std::vector<int>id_seq;id_seq.reserve(std::min<size_t>(n/4,10000));
 std::string w;w.reserve(64);
 auto flush=[&](){if(w.empty())return;auto it=s.ids.find(w);id_seq.push_back(it==s.ids.end()?-1:it->second);w.clear();};
 for(size_t i=0;i<n;i++){
  unsigned char c=raw[i];if((c>='a'&&c<='z')||(c>='A'&&c<='Z')){
   if(w.size()<64)w.push_back(c>='A'&&c<='Z'?(char)(c+32):(char)c);
  }else flush();
 }flush();
 for(size_t i=0;i<std::min<size_t>(48,id_seq.size());i++)hist[48-std::min<size_t>(48,id_seq.size())+i]=id_seq[id_seq.size()-std::min<size_t>(48,id_seq.size())+i];
 const auto&emb=s.at("embed"),&op=s.at("out"),&bias=s.at("bias"),&geo=s.at("geo"),&proj=s.at("projection");
 const auto&gp=s.at("gpos"),&gn=s.at("gneg");
 float vec[32]={},geom[8]={};double denom=0;
 for(int k=0;k<3;k++)if(hist[45+k]>=0){float wt[]={.2f,.3f,.5f};float a=wt[k];denom+=a;int t=hist[45+k];
  for(int i=0;i<32;i++)vec[i]+=a*emb[t*32+i];
  for(int i=0;i<8;i++)geom[i]+=a*geo[t*8+i];
 }
 for(int i=0;i<32;i++){
  if(denom>0)vec[i]/=(float)denom;
  float g=0;for(int j=0;j<8;j++)g+=proj[i*8+j]*(denom>0?geom[j]/(float)denom:0.f);
  vec[i]+=g;
 }
 auto normalize=[](float*x,int n){double z=0.;for(int i=0;i<n;i++)z+=(double)x[i]*x[i];float d=(float)std::sqrt(z);if(d<1e-8f)d=1e-8f;for(int i=0;i<n;i++)x[i]/=d;};
 normalize(vec,32);
 float hp[64]={},hn[64]={};
 for(int ix:hist){
  float pos[48]={},neg[48]={};
  if(ix>=0){for(int j=0;j<32;j++)pos[j]=neg[j]=emb[ix*32+j];
   for(int j=0;j<8;j++){
    pos[32+j]=gp[ix*8+j];pos[40+j]=gn[ix*8+j];
    neg[32+j]=gn[ix*8+j];neg[40+j]=gp[ix*8+j];
   }
  }
  gru(s,"gru_pos",pos,hp);gru(s,"gru_neg",neg,hn);
 }
 float signedx[192];for(int j=0;j<64;j++){signedx[j]=hp[j];signedx[64+j]=hn[j];signedx[128+j]=hp[j]*hn[j];}
 double mean=0.,variance=0.;for(float x:signedx)mean+=x;mean/=192.;
 for(float x:signedx)variance+=(x-mean)*(x-mean);
 variance/=192.;
 float inv=1.f/std::sqrt((float)variance+1e-5f);
 const auto&lnw=s.at("signed_norm.weight"),&lnb=s.at("signed_norm.bias");
 for(int j=0;j<192;j++)signedx[j]=(signedx[j]-(float)mean)*inv*lnw[j]+lnb[j];
 float hid[96]={},res[32]={};
 const auto&W0=s.at("to_lex.0.weight"),&B0=s.at("to_lex.0.bias");
 for(int j=0;j<96;j++){
  float y=B0[j];for(int k=0;k<192;k++)y+=W0[j*192+k]*signedx[k];
  hid[j]=0.5f*y*(1.f+std::erf(y*0.7071067811865475f));
 }
 const auto&W2=s.at("to_lex.2.weight"),&B2=s.at("to_lex.2.bias");
 for(int j=0;j<32;j++){
  float y=B2[j];for(int k=0;k<96;k++)y+=W2[j*96+k]*hid[k];res[j]=y;
 }
 normalize(res,32);
 float trust=s.at("trust")[0],alpha=2.f*(trust>0?trust+std::log1p(std::exp(-trust)):std::log1p(std::exp(trust)));
 float gain=s.at("gain")[0];gain=std::min(25.f,std::max(0.f,gain));
 float mx=-1e30f;
 for(int i=0;i<12000;i++){
  float v=bias[i],r=0;for(int j=0;j<32;j++){v+=gain*vec[j]*op[i*32+j];r+=res[j]*op[i*32+j];}
  out[i]=v+alpha*gain*r;mx=std::max(mx,out[i]);
 }
 double mass=0.;for(int i=0;i<12000;i++){out[i]=std::exp(out[i]-mx);mass+=out[i];}
 if(mass>0.)for(int i=0;i<12000;i++)out[i]/=(float)mass;
 // v175 causal signed episodic pointer, applied as logit energy after the
 // exact inherited normalized 12k-word distribution. Candidate y at i+1
 // may be copied only from already observed history; no future tokens.
 float scores[47]={};float score_sum=0.f;
 int key=hist[47];
 if(key>=0)for(int i=0;i<47;i++){
   int tgt=hist[i+1];if(hist[i]<0||tgt<0||hist[i]!=key)continue;
   bool bi=i>=1&&hist[46]>=0&&hist[i-1]==hist[46];
   bool tri=i>=2&&hist[45]>=0&&hist[i-2]==hist[45]&&bi;
   float tau=std::exp(s.at("copy_log_tau")[0]);tau=std::min(128.f,std::max(1.f,tau));
   float v=std::exp(-(float)(47-i)/tau);
   v*=1.f+2.f*sigmoid(s.at("copy_bigram")[0])*(float)bi
          +4.f*sigmoid(s.at("copy_trigram")[0])*(float)tri;
   scores[i]=v;score_sum+=v;
 }
 float copy_trust=s.at("copy_log_strength")[0];
 float boost=8.f*(copy_trust>0?copy_trust+std::log1p(std::exp(-copy_trust)):std::log1p(std::exp(copy_trust)));
 if(score_sum>0&&std::isfinite(boost)){
   float factor=boost/(score_sum+1.f);
   for(int i=0;i<47;i++)if(scores[i]>0)out[hist[i+1]]*=std::exp(factor*scores[i]);
   double z=0.;for(int i=0;i<12000;i++)z+=out[i];
   if(z>0)for(int i=0;i<12000;i++)out[i]/=(float)z;
 }

}
}
extern "C" void*wf175_now_open(const char*sidecar){
 try{auto v=std::make_unique<n175::Now>();if(!n175::load(sidecar,*v))return nullptr;return v.release();}
 catch(...){return nullptr;}
}
extern "C" void wf175_now_close(void*h){delete (n175::Now*)h;}
extern "C" int wf175_now_word_count(void*h){return h?12000:-1;}
extern "C" int wf175_now_word_name(void*h,int idx,char*dst,size_t cap){
 auto*s=(n175::Now*)h;if(!s||idx<0||idx>=12000||!dst||s->names[idx].size()+1>cap)return -1;
 memcpy(dst,s->names[idx].c_str(),s->names[idx].size()+1);return 0;
}
extern "C" int wf175_now_predict(void*h,const unsigned char*past,size_t n,float*probs,size_t cap){
 auto*s=(n175::Now*)h;
 if(!s||(!past&&n)||n>1000000||!probs||cap<12000)return -1;
 try{n175::predict(*s,past,n,probs);for(int i=0;i<12000;i++)if(!std::isfinite(probs[i])||probs[i]<0)return -2;return 0;}
 catch(...){return -3;}
}