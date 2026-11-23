#pragma once
#include <queue>
constexpr int SOFT_D=32;
struct SoftEntry {uint16_t word; float z[SOFT_D];};
struct SoftIndex {uint64_t trainhash,cloudhash,geomhash;vector<SoftEntry> entries;};
struct SoftVote {uint16_t word;float mass;};
struct SoftMetric {array<float,SOFT_D*SOFT_D> weight;};
static SoftMetric load_metric(const string&path,const SoftIndex&s){
 ifstream f(path,ios::binary);char tag[8];f.read(tag,8);
 if(!f||memcmp(tag,"N154METR",8)||ru32(f)!=154||ru32(f)!=SOFT_D)throw runtime_error("soft metric header mismatch");
 if(ru64(f)!=s.trainhash||ru64(f)!=s.geomhash||ru64(f)!=s.cloudhash)throw runtime_error("soft metric fingerprint mismatch");
 SoftMetric m{};f.read((char*)m.weight.data(),m.weight.size()*sizeof(float));
 if(!f||f.peek()!=EOF)throw runtime_error("soft metric length mismatch");
 for(float x:m.weight)if(!isfinite(x))throw runtime_error("nonfinite metric parameter");
 return m;
}
static array<float,SOFT_D> transformed(const array<float,SOFT_D>&x,const SoftMetric&metric){
 array<float,SOFT_D>z{};float sq=1.e-14f;
 for(int j=0;j<SOFT_D;j++){
  float v=0;for(int k=0;k<SOFT_D;k++)v+=metric.weight[j*SOFT_D+k]*x[k];
  z[j]=v;sq+=v*v;
 }
 float inv=1.f/sqrtf(sq);for(float&v:z)v*=inv;return z;
}
static void mask_channels(array<float,SOFT_D>&x,const string&name){
 int which=-1;
 if(name.find("nosigned")!=string::npos){for(int i=16;i<32;i++)x[i]=0;return;}
 if(name.find("noupper")!=string::npos)which=0;
 if(name.find("nolower")!=string::npos)which=1;
 if(name.find("noplus")!=string::npos)which=2;
 if(name.find("nominus")!=string::npos)which=3;
 if(which>=0)for(int j=8*which;j<8*(which+1);j++)x[j]=0;
}
static void transform_index(SoftIndex&s,const SoftMetric&m,const string&mask=""){

 for(auto &e:s.entries){array<float,SOFT_D>x{};copy(begin(e.z),end(e.z),x.begin());
  mask_channels(x,mask);auto z=transformed(x,m);copy(z.begin(),z.end(),e.z);
 }
}


// Four normalized views of the same live C(256,2) signed cloud:
// upper expert, lower intake, signed S+ support, signed S- refutation.
// All read only input bytes before c.pos, and prior completed words.
static array<float,SOFT_D> causal_signed_state(const Context&c,const GeModel*gm,const Cloud137*cl){
 GeEvents ev;ge_events(c.p,c.pos,&ev);
 auto q=new CloudCache;cloud_forward(gm,cl,&ev,q,1,1);
 array<float,SOFT_D>z{};
 double nt=0,nb=0,edge=0;
 for(int i=0;i<q->g.nt;i++){
  int t=q->g.ti[i];float wt=max(1.e-5f,q->g.gate[t]);nt+=wt;
  for(int d=0;d<8;d++)z[d]+=wt*q->u2[t][d];
 }
 for(int j=0;j<q->g.nb;j++){
  int b=q->g.bi[j];float norm=0;
  for(int d=0;d<8;d++)norm+=q->r[b][d]*q->r[b][d];
  float wb=sqrtf(norm)+.01f;nb+=wb;
  for(int d=0;d<8;d++)z[8+d]+=wb*q->r[b][d];
 }
 for(int i=0;i<q->g.nt;i++)for(int j=0;j<q->g.nb;j++){
  int t=q->g.ti[i],b=q->g.bi[j],e=ge_pair(t,128+b);
  float w=max(.01f,q->g.gate[t]);edge+=w;
  for(int d=0;d<8;d++){
   z[16+d]+=w*cl->plus[e][d]*q->r[b][d];
   z[24+d]+=w*cl->minus[e][d]*q->u2[t][d];
  }
 }
 delete q;
 for(int d=0;d<8;d++){z[d]/=max(1.,nt);z[8+d]/=max(1.,nb);z[16+d]/=max(1.,edge);z[24+d]/=max(1.,edge);}
 // Scale-invariant directions preserve sign rather than 8-bit phase hashes.
 for(int g=0;g<4;g++){
  float norm=1.e-8f;for(int d=0;d<8;d++)norm+=z[8*g+d]*z[8*g+d];
  float inv=1.f/sqrtf(norm);for(int d=0;d<8;d++)z[8*g+d]*=inv;
 }
 return z;
}
static void save_soft(const SoftIndex&m,const string&path){
 ofstream f(path,ios::binary);if(!f)throw runtime_error("soft sidecar write failed");
 f.write("N154SOFT",8);write_u32(f,154);write_u32(f,SOFT_D);
 write_u64(f,m.trainhash);write_u64(f,m.geomhash);write_u64(f,m.cloudhash);write_u64(f,m.entries.size());
 for(auto &e:m.entries){uint32_t word=e.word;write_u32(f,word);f.write((const char*)e.z,sizeof(e.z));}
 if(!f)throw runtime_error("short soft sidecar write");
}
static SoftIndex load_soft(const string&p,const Model&m,const GeModel*gm,const Cloud137*cl){
 ifstream f(p,ios::binary);char tag[8];f.read(tag,8);
 if(!f||memcmp(tag,"N154SOFT",8)||ru32(f)!=154||ru32(f)!=SOFT_D)throw runtime_error("soft sidecar magic/version mismatch");
 SoftIndex s;s.trainhash=ru64(f);s.geomhash=ru64(f);s.cloudhash=ru64(f);u64 n=ru64(f);
 if(s.trainhash!=m.trainhash||s.geomhash!=hbytes((const unsigned char*)gm,sizeof(*gm))||s.cloudhash!=hbytes((const unsigned char*)cl,sizeof(*cl)))throw runtime_error("soft sidecar fingerprint mismatch");
 if(n>1000000)throw runtime_error("soft sidecar too large");s.entries.reserve(n);
 for(u64 i=0;i<n;i++){
  uint32_t word=ru32(f);if(word>=m.words.size())throw runtime_error("soft sidecar bad word id");
  SoftEntry e{};e.word=(uint16_t)word;f.read((char*)e.z,sizeof(e.z));if(!f)throw runtime_error("soft sidecar truncated");
  for(float v:e.z)if(!isfinite(v))throw runtime_error("soft sidecar nonfinite vector");
  s.entries.push_back(e);
 }
 if(f.peek()!=EOF)throw runtime_error("soft sidecar trailing data");return s;
}
static vector<SoftVote> neighbors(const SoftIndex&s,const array<float,SOFT_D>&q,int k,float temperature,const array<float,4>&view){
 using Pair=pair<float,uint32_t>;
 auto comp=[](const Pair&a,const Pair&b){return a.first>b.first;};
 priority_queue<Pair,vector<Pair>,decltype(comp)>heap(comp);
 for(uint32_t i=0;i<s.entries.size();i++){
  const auto &v=s.entries[i];float sim=0;
  for(int g=0;g<4;g++){
   float dot=0;for(int d=0;d<8;d++)dot+=q[8*g+d]*v.z[8*g+d];
   sim+=view[g]*dot;
  }
  if(!isfinite(sim))continue;
  if((int)heap.size()<k)heap.push({sim,i});
  else if(sim>heap.top().first){heap.pop();heap.push({sim,i});}
 }
 vector<Pair>tops;while(!heap.empty()){tops.push_back(heap.top());heap.pop();}
 if(tops.empty())return {};
 float high=-1.e10f;for(auto &p:tops)high=max(high,p.first);
 unordered_map<uint16_t,float>mass;float denom=0;
 for(auto &p:tops){float w=expf(max(-50.f,min(0.f,temperature*(p.first-high))));denom+=w;mass[s.entries[p.second].word]+=w;}
 vector<SoftVote>votes;for(auto [id,score]:mass)votes.push_back({id,score/denom});
 sort(votes.begin(),votes.end(),[](auto a,auto b){return a.mass!=b.mass?a.mass>b.mass:a.word<b.word;});return votes;
}
static Prediction fuse(const Prediction&old,const vector<SoftVote>&n,const Model&m,float beta,bool protect=false){
 if(beta==0)return old; // byte-for-byte existing ranking and scores
 unordered_map<uint16_t,float>score;
 for(auto &it:old.top)score[it.first]=it.second;
 for(auto it:n)score[it.word]+=beta*it.mass;
 Prediction fused=old;fused.top.clear();fused.top.reserve(score.size());
 for(auto [id,s]:score)fused.top.push_back({id,s});
 size_t count=min<size_t>(32,fused.top.size());
 partial_sort(fused.top.begin(),fused.top.begin()+count,fused.top.end(),[&](auto a,auto b){
  return a.second!=b.second?a.second>b.second:m.unigrams[a.first]>m.unigrams[b.first];
 });fused.top.resize(count);
 if(protect&&!old.top.empty()&&!fused.top.empty()&&fused.top[0].first!=old.top[0].first){
  fused.top.erase(remove_if(fused.top.begin(),fused.top.end(),[&](auto x){return x.first==old.top[0].first;}),fused.top.end());
  fused.top.insert(fused.top.begin(),old.top[0]);
  fused.top.resize(min<size_t>(32,fused.top.size()));
 }
 return fused;
}
