// Source-only synthetic parity, session isolation, cache reset and lifetime.
#define main inherited_v176_source_main
#include "../v176/test_source176.cpp"
#undef main
extern "C" void*wf176_hybrid_open_full(const char*t,const char*m,const char*g,
 const char*c,const char*b,float e,const char*,const char*){
 return wf176_hybrid_open(t,m,g,c,b,e);
}
#include "native_shared178.cpp"
#include <thread>
#include <atomic>
int main(){
 void*hub=wf178_hub_open_full("x","x","x","x","x",.35f,"x","x");assert(hub);
 const std::string h1="first book context ",h2="different book context ";
 void*s1=wf178_session_open(hub,(const unsigned char*)h1.data(),h1.size());
 void*s2=wf178_session_open(hub,(const unsigned char*)h2.data(),h2.size());
 assert(s1&&s2);
 int models=0,live=0;
 assert(wf178_hub_stats(hub,&models,&live)==0&&models==1&&live==2);
 void*old=wf176_hybrid_open("x","x","x","x","x",.35f);assert(old);
 const std::vector<std::string> prefixes={"","t","th","the","their","xyz"};
 auto check=[&](void*s,const std::string&history){
  assert(wf176_hybrid_begin(old,(const unsigned char*)history.data(),history.size())==0);
  for(const auto&pre:prefixes){
   float q[256]{},p[256]{},rq[256]{},rp[256]{};
   assert(wf178_session_next(s,(const unsigned char*)pre.data(),pre.size(),q,p)==0);
   assert(wf176_hybrid_next(old,(const unsigned char*)pre.data(),pre.size(),rq,rp)==0);
   for(int j=0;j<256;j++)assert(q[j]==rq[j]&&p[j]==rp[j]);
  }
  for(int k=0;k<5;k++){
   char w[48]{},r[48]{};int ix=0,jx=0;float p=0.,rp=0.;
   assert(wf178_session_top5(s,k,&ix,w,48,&p)==0);
   assert(wf176_hybrid_top5(old,k,&jx,r,48,&rp)==0);
   assert(ix==jx&&std::string(w)==r&&p==rp);
  }
  for(uint64_t seed=1;seed<=100;seed++){
   unsigned char a[40]{},b[40]{};int n=0,m=0,acc=0,bcc=0,rej=0,rj=0;
   assert(wf178_session_sample(s,seed,12,a,40,&n,&acc,&rej)==0);
   assert(wf176_hybrid_sample(old,seed,12,b,40,&m,&bcc,&rj)==0);
   assert(n==m&&acc==bcc&&rej==rj&&memcmp(a,b,n)==0);
  }
 };
 check(s1,h1);check(s2,h2);check(s1,h1);
 std::atomic<int>fail{0};
 auto concurrent=[&](void*s){
  for(int t=0;t<200;t++){
   float q[256]{},p[256]{};
   if(wf178_session_next(s,(const unsigned char*)"the",3,q,p)!=0||
      !(q[(unsigned char)' ']>=p[(unsigned char)' ']))fail.fetch_add(1);
  }
 };
 std::thread a(concurrent,s1),b(concurrent,s2);a.join();b.join();assert(fail.load()==0);
 assert(wf178_session_begin(s1,(const unsigned char*)h1.data(),h1.size())==0);
 float scores[5]{};int raw=0,unique=0;
 assert(wf178_session_prepare_top5(s1,scores,&raw,&unique)==0);
 assert(raw>unique);
 assert(wf178_session_prepare_top5(s1,scores,&raw,&unique)==0&&unique==0);
 const unsigned char data[]={'t','h','e','x','y','z'};
 uint32_t offs[4]={0,0,3,0},lens[4]={0,3,3,3};
 float q[4*256]{},p[4*256]{};
 assert(wf178_session_batch(s2,data,sizeof(data),offs,lens,4,q,p)==0);
 uint32_t invalid[4]={0,0,1000000,0};
 int m=0,h=0,n=0;
 assert(wf178_session_stats(s2,&m,&h,&n)==0);
 assert(wf178_session_batch(s2,data,sizeof(data),invalid,lens,4,q,p)==-2);
 int mm=0,hh=0,nn=0;
 assert(wf178_session_stats(s2,&mm,&hh,&nn)==0&&m==mm&&h==hh&&n==nn);
 assert(wf178_session_begin(s1,(const unsigned char*)h2.data(),h2.size())==0);
 assert(wf178_session_stats(s1,&m,&h,&n)==0&&m==0&&h==0&&n==0);
 check(s1,h2);
 wf178_hub_close(hub);
 float qq[256]{},pp[256]{};
 assert(wf178_session_next(s1,nullptr,0,qq,pp)==0);
 wf178_session_close(s1);wf178_session_close(s2);wf176_hybrid_close(old);
 puts("PASS v178 shared-model lifetime, concurrent sessions, exact P/Q, sampling, resets");
}
