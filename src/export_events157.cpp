// Exact native GeCache capture for differentiable S+/S- adjoint learning.
// No current word or later bytes are fed to GeEvents.
#define WF148_NO_MAIN
#include "wordfield148.cpp"
#include "softbridge154_core.h"
#include <memory>
struct EvRow157 {
  uint16_t nb,nt;
  uint8_t b[48],t[128];
  float x[48][8],bf[48],gate[128];
};
static_assert(sizeof(EvRow157)==2420,"bad GE event record size");
int main(int argc,char**argv){
 try{
  if(argc!=7)throw runtime_error("usage: export_events157 MODEL GEOM CLOUD STRIDE TRAIN OUTPUT");
  Model m=restore(argv[1]);auto gm=make_unique<GeModel>();auto cl=make_unique<Cloud137>();
  if(ge_load(argv[2],gm.get(),nullptr,nullptr)||cloud_load(argv[3],cl.get(),nullptr,nullptr))throw runtime_error("load model");
  if(m.cloudhash!=hbytes((const unsigned char*)cl.get(),sizeof(*cl))||m.geomhash!=hbytes((const unsigned char*)gm.get(),sizeof(*gm)))throw runtime_error("checkpoint mismatch");
  int stride=atoi(argv[4]);if(stride<1||stride>64)throw runtime_error("invalid stride");
  auto bytes=load(argv[5]);if(m.trainhash!=hbytes(bytes.data(),bytes.size()))throw runtime_error("wrong TRAIN fingerprint");
  ofstream o(argv[6],ios::binary);o.write("N155GEO1",8);write_u32(o,155);write_u32(o,(uint32_t)stride);
  write_u64(o,m.trainhash);write_u64(o,m.geomhash);write_u64(o,m.cloudhash);write_u64(o,0);
  uint64_t n=0,all=0;
  traverse(bytes,0,[&](const Context&c){
   if(all++%stride)return;
   GeEvents ev{};ge_events(c.p,c.pos,&ev);CloudCache q{};cloud_forward(gm.get(),cl.get(),&ev,&q,1,1);
   EvRow157 r{};r.nb=q.g.nb;r.nt=q.g.nt;
   for(int i=0;i<q.g.nb;i++){
    int b=q.g.bi[i];r.b[i]=b;r.bf[i]=q.g.bf[b];for(int d=0;d<8;d++)r.x[i][d]=q.g.x[b][d];
   }
   for(int i=0;i<q.g.nt;i++){int t=q.g.ti[i];r.t[i]=t;r.gate[i]=q.g.gate[t];}
   o.write((char*)&r,sizeof(r));n++;
  });
  o.seekp(40);o.write((char*)&n,8);o.close();if(!o)throw runtime_error("failed output");
  cout<<"{\"rows\":"<<n<<",\"all_words\":"<<all<<",\"stride\":"<<stride<<"}"<<endl;
  return 0;
 }catch(const exception&e){cerr<<"v157 cache: "<<e.what()<<endl;return 1;}
}
