#define main inherited_v176_source_main
#include "../v176/test_source176.cpp"
#undef main
extern "C" void* wf176_hybrid_open_full(const char*t,const char*m,const char*g,const char*c,
 const char*b,float e,const char*,const char*){
 return wf176_hybrid_open(t,m,g,c,b,e);
}
#include "../v177/native_lattice177.cpp"
#include "../v177/native_parallel177.cpp"
#include "native_fork179.cpp"
#include <cassert>
int main(){
 void*h=wf179_fork_open_full("x","x","x","x","x",.35f,"x","x",2);
 assert(h);
 void*old=wf176_hybrid_open("x","x","x","x","x",.35f);assert(old);
 for(const std::string history:{std::string("hello "),std::string("another context ")}){
  assert(wf179_fork_begin(h,(const unsigned char*)history.data(),history.size())==0);
  assert(wf176_hybrid_begin(old,(const unsigned char*)history.data(),history.size())==0);
  float scores[5]{},ref[5]{};int raw=0,unique=0,calls=0;
  assert(wf179_fork_prefill(h,scores,&raw,&unique)==0);
  assert(wf176_hybrid_verify_top5(old,ref,&calls)==0);
  assert(raw>=unique&&unique==calls&&unique>0);
  for(int i=0;i<5;i++)assert(scores[i]==ref[i]);
  for(const auto&pre:{std::string(""),std::string("t"),std::string("th"),std::string("the"),std::string("xyz")}){
   float q[256]{},p[256]{},a[256]{},b[256]{};
   assert(wf179_fork_next(h,(const unsigned char*)pre.data(),pre.size(),q,p)==0);
   assert(wf176_hybrid_next(old,(const unsigned char*)pre.data(),pre.size(),a,b)==0);
   for(int j=0;j<256;j++)assert(q[j]==a[j]&&p[j]==b[j]);
  }
  assert(wf179_fork_prefill(h,scores,&raw,&unique)==0&&unique==0);
 }
 int workers=0,dispatched=0,nodes=0,hits=0;
 assert(wf179_fork_stats(h,&workers,&dispatched,&nodes,&hits)==0);
 assert(workers==2&&dispatched>0&&nodes>0);
 assert(wf179_fork_pid(h,0)>0&&wf179_fork_pid(h,1)>0);
 // v179 speculative sampling is exactly the inherited RNG law.
 assert(wf179_fork_begin(h,(const unsigned char*)"hello ",6)==0);
 assert(wf176_hybrid_begin(old,(const unsigned char*)"hello ",6)==0);
 for(uint64_t seed=1;seed<=256;seed++){
  unsigned char a[40]{},b[40]{};int na=0,nb=0,ca=0,cb=0,ra=0,rb=0;
  assert(wf179_fork_sample(h,seed,16,a,40,&na,&ca,&ra)==0);
  assert(wf176_hybrid_sample(old,seed,16,b,40,&nb,&cb,&rb)==0);
  assert(na==nb&&ca==cb&&ra==rb&&memcmp(a,b,na)==0);
 }
 // Different contexts are delegated to separate COW workers simultaneously.
 const std::string ha="first book ",hb="other book ";
 const std::string buf=ha+hb;
 uint32_t offsets[2]={0,(uint32_t)ha.size()};
 uint32_t lengths[2]={(uint32_t)ha.size(),(uint32_t)hb.size()};
 float many[10]{},one[5]{};uint32_t counts[2]{};int calls=0;
 assert(wf179_fork_verify_contexts(h,(const unsigned char*)buf.data(),buf.size(),offsets,lengths,2,many,counts)==0);
 assert(counts[0]>0&&counts[1]>0);
 assert(wf176_hybrid_begin(old,(const unsigned char*)ha.data(),ha.size())==0);
 assert(wf176_hybrid_verify_top5(old,one,&calls)==0);
 for(int k=0;k<5;k++)assert(one[k]==many[k]);
 assert(wf176_hybrid_begin(old,(const unsigned char*)hb.data(),hb.size())==0);
 assert(wf176_hybrid_verify_top5(old,one,&calls)==0);
 for(int k=0;k<5;k++)assert(one[k]==many[5+k]);
 uint32_t bad[2]={0,UINT32_MAX};
 assert(wf179_fork_verify_contexts(h,(const unsigned char*)buf.data(),buf.size(),bad,lengths,2,many,counts)<0);
 // Entire unchanged word+byte model runs in COW children, not in parent.
 FullResult179 full[2]{};
 assert(wf179_fork_full_contexts(h,(const unsigned char*)buf.data(),buf.size(),offsets,lengths,2,full)==0);
 for(int idx=0;idx<2;idx++){
  const auto&history=idx==0?ha:hb;
  assert(wf176_hybrid_begin(old,(const unsigned char*)history.data(),history.size())==0);
  assert(wf176_hybrid_verify_top5(old,one,&calls)==0);
  assert(full[idx].verifier_calls==(uint32_t)calls);
  for(int k=0;k<5;k++){
   assert(full[idx].avg_log_byte[k]==one[k]);
   int id=0;char word[48]{};float prob=0.f;
   assert(wf176_hybrid_top5(old,k,&id,word,sizeof(word),&prob)==0);
   assert(full[idx].indices[k]==id&&full[idx].word_probabilities[k]==prob&&
          std::string(full[idx].words[k])==word);
  }
  unsigned char bytes[40]{};int nw=0,na=0,nr=0;
  assert(wf176_hybrid_sample(old,1790000+idx,12,bytes,sizeof(bytes),&nw,&na,&nr)==0);
  assert(full[idx].sampled_bytes==(uint32_t)nw&&full[idx].sampled_accepted==(uint32_t)na&&
         full[idx].sampled_rejected==(uint32_t)nr&&memcmp(bytes,full[idx].sample,nw)==0);
 }
 wf176_hybrid_close(old);wf179_fork_close(h);
 puts("PASS v179 COW workers: simultaneous child processes, exact P/Q, shared-parent weights, cache, resets");
}
