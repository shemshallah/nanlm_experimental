// NaNCM v157: native signed-state extraction. Training source unchanged.
// Only causal bytes c.p[:c.pos] enter the cloud; c.word is used for labels only.
#define WF148_NO_MAIN
#include "wordfield148.cpp"
#include "softbridge154_core.h"
#include <memory>
#include <cstdlib>
struct Record157 {
  float z[32];
  uint32_t prev[4];
  int32_t target;
  uint32_t offset;
};
static_assert(sizeof(Record157)==152,"unexpected alignment");
int main(int argc,char**argv) {
 try {
  if(argc!=7)throw runtime_error("usage: export_context157 MODEL GEOM CLOUD TRAIN|WINDOWS INPUT OUTPUT");
  Model m=restore(argv[1]);auto gm=make_unique<GeModel>();auto cl=make_unique<Cloud137>();
  if(ge_load(argv[2],gm.get(),nullptr,nullptr)||cloud_load(argv[3],cl.get(),nullptr,nullptr))throw runtime_error("cloud load");
  if(m.cloudhash!=hbytes((const unsigned char*)cl.get(),sizeof(*cl))||m.geomhash!=hbytes((const unsigned char*)gm.get(),sizeof(*gm)))throw runtime_error("fingerprint mismatch");
  ofstream o(argv[6],ios::binary);if(!o)throw runtime_error("output open error");
  unsigned long count=0;
  auto consume=[&](const Context&c){
   Record157 r{};auto z=causal_signed_state(c,gm.get(),cl.get());copy(z.begin(),z.end(),r.z);
   for(int k=0;k<4;k++){int j=(int)c.prev.size()-4+k;if(j>=0){auto it=m.id.find(c.prev[j]);if(it!=m.id.end())r.prev[k]=it->second+1;}}
   auto it=m.id.find(c.word);r.target=it==m.id.end()?-1:it->second;r.offset=(uint32_t)c.pos;
   o.write((const char*)&r,sizeof(r));count++;
  };
  if(string(argv[4])=="TRAIN"){auto b=load(argv[5]);if(m.trainhash!=hbytes(b.data(),b.size()))throw runtime_error("TRAIN data differs from parent fingerprint");traverse(b,0,consume);}
  else if(string(argv[4])=="WINDOWS"){
   auto b=windows(argv[5]);for(const auto&win:b)traverse(win.first,win.second,consume);
  }else throw runtime_error("input type must be TRAIN or WINDOWS");
  if(!o)throw runtime_error("incomplete output write");
  cout<<"{\"records\":"<<count<<",\"record_bytes\":"<<sizeof(Record157)<<"}"<<endl;
  return 0;
 }catch(const exception&e){cerr<<"v157 export: "<<e.what()<<endl;return 1;}
}
