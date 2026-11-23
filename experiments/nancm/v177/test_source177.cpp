// Independent deterministic source test of the inherited P/Q law, trie sharing,
// exact cached speculation, complete batch parity and cache invalidation.
#define main v176_source_smoke
#include "../v176/test_source176.cpp"
#undef main
extern "C" void* wf176_hybrid_open_full(const char*t,const char*m,const char*g,
 const char*c,const char*b,float escape,const char*,const char*){
 return wf176_hybrid_open(t,m,g,c,b,escape);
}
#include "native_lattice177.cpp"
int main(){
 void*h=wf177_open_full("x",nullptr,nullptr,nullptr,nullptr,.35f,"x","x");
 assert(h);
 int calls=0,hits=0,nodes=0;
 assert(wf177_stats(h,&calls,&hits,&nodes)==0&&calls==0&&nodes==0);
 assert(wf177_begin(h,(const unsigned char*)"hello ",6)==0);
 std::array<float,256>q{},p{};
 assert(wf177_next(h,nullptr,0,q.data(),p.data())==0);
 assert(wf177_stats(h,&calls,&hits,&nodes)==0&&calls==1&&nodes==1);
 assert(wf177_next(h,nullptr,0,q.data(),p.data())==0);
 assert(wf177_stats(h,&calls,&hits,&nodes)==0&&calls==1&&hits==1);
 const char prefixes[]="thetheirxyzthe";
 const uint32_t o[]={0,0,3,8,11,11};
 const uint32_t n[]={0,1,2,3,0,3};
 std::vector<float>Q(6*256),P(6*256);
 assert(wf177_verify_batch(h,(const unsigned char*)prefixes,sizeof(prefixes)-1,o,n,6,Q.data(),P.data())==0);
 for(int i=0;i<6;i++){
  std::array<float,256>x{},y{};
  const auto*pre=n[i]?(const unsigned char*)prefixes+o[i]:nullptr;
  assert(wf176_hybrid_next(((lattice177::State*)h)->inherited,pre,n[i],x.data(),y.data())==0);
  for(int j=0;j<256;j++){
   assert(Q[i*256+j]==x[j]&&P[i*256+j]==y[j]);
  }
 }
 const uint32_t bad[]={1000000};
 assert(wf177_verify_batch(h,(const unsigned char*)prefixes,strlen(prefixes),bad,n,1,Q.data(),P.data())<0);
 float score[5]{};int raw=0,uniq=0;
 assert(wf177_prepare_top5(h,score,&raw,&uniq)==0);
 assert(uniq<raw&&raw>16&&uniq>0);
 int before=0;assert(wf177_stats(h,&before,&hits,&nodes)==0);
 assert(wf177_prepare_top5(h,score,&raw,&uniq)==0&&uniq==0);
 assert(wf177_stats(h,&calls,&hits,&nodes)==0&&before==calls);
 for(uint64_t seed=1;seed<256;seed++){
  unsigned char a[40]{},b[40]{};int na=0,nb=0,ac=0,bc=0,ra=0,rb=0;
  assert(wf177_sample(h,seed,16,a,40,&na,&ac,&ra)==0);
  assert(wf176_hybrid_sample(((lattice177::State*)h)->inherited,seed,16,b,40,&nb,&bc,&rb)==0);
  assert(na==nb&&ac==bc&&ra==rb&&std::memcmp(a,b,na)==0);
 }
 assert(wf177_begin(h,(const unsigned char*)"different ",10)==0);
 assert(wf177_stats(h,&calls,&hits,&nodes)==0&&calls==0&&hits==0&&nodes==0);
 assert(wf177_sample(h,1,16,(unsigned char*)prefixes,0,&calls,&hits,&nodes)<0);
 wf177_close(h);
 puts("PASS v177 native DAG: byte-exact batched vectors, unique-prefix call reduction, cached exact speculation (255 seeds), reset invariants");
}