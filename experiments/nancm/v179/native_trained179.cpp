// Standalone SINGLE-THREADED native launcher: fork is performed before any
// user application threads. Never embed this entry point inside Torch/Python.
#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>
#include "full179_api.h"
extern "C" {
void* wf176_hybrid_open_full(const char*,const char*,const char*,const char*,const char*,float,const char*,const char*);
void wf176_hybrid_close(void*);
int wf176_hybrid_begin(void*,const unsigned char*,size_t);
int wf176_hybrid_next(void*,const unsigned char*,size_t,float*,float*);
int wf176_hybrid_top5(void*,int,int*,char*,size_t,float*);
int wf176_hybrid_verify_top5(void*,float*,int*);
int wf176_hybrid_sample(void*,uint64_t,int,unsigned char*,size_t,int*,int*,int*);
void* wf179_fork_open_full(const char*,const char*,const char*,const char*,const char*,float,const char*,const char*,int);
void wf179_fork_close(void*);
int wf179_fork_begin(void*,const unsigned char*,size_t);
int wf179_fork_prefill(void*,float*,int*,int*);
int wf179_fork_next(void*,const unsigned char*,size_t,float*,float*);
int wf179_fork_sample(void*,uint64_t,int,unsigned char*,size_t,int*,int*,int*);
int wf179_fork_stats(void*,int*,int*,int*,int*);
int wf179_fork_pid(void*,int);
int wf179_fork_verify_contexts(void*,const unsigned char*,size_t,const uint32_t*,const uint32_t*,size_t,float*,uint32_t*);
int wf179_fork_full_contexts(void*,const unsigned char*,size_t,const uint32_t*,const uint32_t*,size_t,FullResult179*);
}
using Clock=std::chrono::steady_clock;
static double now(){return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();}
static long read_kb(pid_t pid,const std::string&field){
 std::ifstream f("/proc/"+std::to_string(pid)+"/smaps_rollup");
 std::string line;
 while(std::getline(f,line)){
  const std::string key=field+":";
  if(line.compare(0,key.size(),key)!=0)continue;
  const size_t pos=line.find_first_of("0123456789");
  if(pos==std::string::npos)return -1;
  try{return std::stol(line.substr(pos));}catch(...){return -1;}
 }
 return -1;
}
static long sum_pss(const std::vector<pid_t>&pids){
 long x=0;
 for(pid_t pid:pids){const long a=read_kb(pid,"Pss");if(a<0)return -1;x+=a;}
 return x;
}
static std::vector<std::string>read_cases(const std::string&path){
 std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("contexts absent");
 uint32_t n=0;in.read((char*)&n,4);
 if(n<1||n>256)throw std::runtime_error("bad count");
 std::vector<std::string>v;
 for(uint32_t i=0;i<n;i++){
  uint32_t sz=0;in.read((char*)&sz,4);
  if(sz<2||sz>1000000)throw std::runtime_error("bad history length");
  std::string s(sz,' ');in.read(&s[0],sz);if(!in)throw std::runtime_error("truncated history");
  v.push_back(std::move(s));
 }
 char c=0;if(in.read(&c,1))throw std::runtime_error("trailing history bytes");
 return v;
}
static bool sample(void*h,bool fork,uint64_t seed,std::string&out,int&a,int&r){
 unsigned char buf[40]{};int n=0;a=r=0;
 const int rc=fork?wf179_fork_sample(h,seed,12,buf,40,&n,&a,&r):
  wf176_hybrid_sample(h,seed,12,buf,40,&n,&a,&r);
 if(rc!=0)return false;out.assign((char*)buf,(size_t)n);return true;
}
static void fail(const std::string&s){throw std::runtime_error(s);}
int main(int argc,char**argv){
 if(argc!=11){std::cerr<<"usage: runner teacher model geom cloud bytepack cross bpe histories.bin workers output.json\n";return 2;}
 try{
  int nw=std::stoi(argv[9]);if(nw<1||nw>4)fail("invalid workers");
  std::vector<std::string>histories=read_cases(argv[8]);
  double t=now();void*fork=wf179_fork_open_full(argv[1],argv[2],argv[3],argv[4],argv[5],.35f,argv[6],argv[7],nw);
  const double fork_open=now()-t;if(!fork)fail("failed to load trained fork verifier");
  std::vector<pid_t>pids{getpid()};for(int k=0;k<nw;k++)pids.push_back(wf179_fork_pid(fork,k));
  // Proportional set size accounts for shared pages rather than counting RSS
  // once per fork. Capture before loading a separate reference baseline.
  const long pool_pss_kb=sum_pss(pids);
  std::vector<unsigned char>fully_packed;
  std::vector<uint32_t>fully_offsets,fully_lengths;
  for(const auto&h:histories){
   fully_offsets.push_back((uint32_t)fully_packed.size());
   fully_lengths.push_back((uint32_t)h.size());
   fully_packed.insert(fully_packed.end(),h.begin(),h.end());
  }
  std::vector<FullResult179>full_results(histories.size());
  t=now();
  const int full_rc=wf179_fork_full_contexts(fork,fully_packed.data(),fully_packed.size(),
    fully_offsets.data(),fully_lengths.data(),histories.size(),full_results.data());
  const double full_cold_seconds=now()-t;
  if(full_rc)fail("full independent-context COW evaluation failed: "+std::to_string(full_rc));
  const long full_pool_pss_kb=sum_pss(pids);
  t=now();void*old=wf176_hybrid_open_full(argv[1],argv[2],argv[3],argv[4],argv[5],.35f,argv[6],argv[7]);
  const double reference_open=now()-t;
  if(!old){wf179_fork_close(fork);fail("failed to open independent reference verifier");}
  const std::vector<std::string>prefixes={"","t","th","the","xyz"};
  double serial_seconds=0.,parallel_seconds=0.,begin_old=0.,begin_fork=0.;
  size_t vectors=0,samples=0,score_checks=0,contexts=0;
  int raw_calls=0,unique_calls=0,workers_used=0;bool okay=true;
  std::vector<float>old_scores(histories.size()*5);
  for(size_t i=0;i<histories.size();i++){
   const std::string&h=histories[i];
   t=now();int rc=wf176_hybrid_begin(old,(const unsigned char*)h.data(),h.size());begin_old+=now()-t;
   if(rc)fail("reference begin failed: "+std::to_string(rc));
   t=now();rc=wf179_fork_begin(fork,(const unsigned char*)h.data(),h.size());begin_fork+=now()-t;
   if(rc)fail("fork begin failed: "+std::to_string(rc));
   float scores[5]{},ref[5]{};int raw=0,unique=0,oracle=0;
   if(i%2){
    t=now();rc=wf179_fork_prefill(fork,scores,&raw,&unique);parallel_seconds+=now()-t;
    if(rc)fail("parallel prefill failed: "+std::to_string(rc));
    t=now();rc=wf176_hybrid_verify_top5(old,ref,&oracle);serial_seconds+=now()-t;
    if(rc)fail("reference prefill failed");
   }else{
    t=now();rc=wf176_hybrid_verify_top5(old,ref,&oracle);serial_seconds+=now()-t;
    if(rc)fail("reference prefill failed");
    t=now();rc=wf179_fork_prefill(fork,scores,&raw,&unique);parallel_seconds+=now()-t;
    if(rc)fail("parallel prefill failed: "+std::to_string(rc));
   }
   if(raw<unique||unique!=oracle)fail("unique-prefix mismatch");
   std::copy(ref,ref+5,old_scores.begin()+i*5);
   raw_calls+=raw;unique_calls+=unique;score_checks+=5;
   for(int k=0;k<5;k++){
    if(std::abs(scores[k]-ref[k])>1e-7f)fail("exact word verifier log-score mismatch");
    if(full_results[i].avg_log_byte[k]!=ref[k])fail("full COW likelihood mismatch");
    int ix=-1;char word[48]{};float prob=0.f;
    if(wf176_hybrid_top5(old,k,&ix,word,sizeof(word),&prob)!=0)fail("topfive reference failed");
    if(full_results[i].indices[k]!=ix||full_results[i].word_probabilities[k]!=prob||
       strcmp(full_results[i].words[k],word)!=0)fail("full COW word/probability mismatch");
   }
   if(full_results[i].verifier_calls!=(uint32_t)oracle)fail("full COW unique-prefix mismatch");
   {
    unsigned char b[40]{};int nb=0,acc=0,rej=0;
    if(wf176_hybrid_sample(old,1790000u+i,12,b,sizeof(b),&nb,&acc,&rej)!=0)fail("full COW sample baseline failed");
    if(full_results[i].sampled_bytes!=(uint32_t)nb||full_results[i].sampled_accepted!=(uint32_t)acc||
       full_results[i].sampled_rejected!=(uint32_t)rej||memcmp(full_results[i].sample,b,nb)!=0)
       fail("full COW complete-context seeded sampler mismatch");
   }
   for(const auto&p:prefixes){
    float q[256]{},v[256]{},a[256]{},b[256]{};
    if(wf179_fork_next(fork,(const unsigned char*)p.data(),p.size(),q,v)||
       wf176_hybrid_next(old,(const unsigned char*)p.data(),p.size(),a,b))fail("next failed");
    if(memcmp(q,a,sizeof(q))||memcmp(v,b,sizeof(v)))fail("P/Q 256-way mismatch");
    ++vectors;
   }
   for(int k=0;k<8;k++){
    const uint64_t seed=179000+(uint64_t)i*8+k;
    std::string a,b;int aa=0,ab=0,ra=0,rb=0;
    if(!sample(old,false,seed,a,aa,ra)||!sample(fork,true,seed,b,ab,rb)||
       a!=b||aa!=ab||ra!=rb)fail("seeded byte sample mismatch");
    ++samples;
   }
   float warm_scores[5]{};int rr=0,uu=0;
   if(wf179_fork_prefill(fork,warm_scores,&rr,&uu)!=0||uu!=0)fail("warm prefix cache is not exact");
   ++contexts;
  }
  std::vector<unsigned char>packed;
  std::vector<uint32_t>starts,lens;
  for(const auto&h:histories){
   starts.push_back((uint32_t)packed.size());lens.push_back((uint32_t)h.size());
   packed.insert(packed.end(),h.begin(),h.end());
  }
  std::vector<float>multi_scores(histories.size()*5);
  std::vector<uint32_t>multi_prefixes(histories.size());
  t=now();
  const int multi_rc=wf179_fork_verify_contexts(fork,packed.data(),packed.size(),
    starts.data(),lens.data(),histories.size(),multi_scores.data(),multi_prefixes.data());
  const double multi_seconds=now()-t;
  if(multi_rc)fail("multi-context prefill failed: "+std::to_string(multi_rc));
  for(size_t i=0;i<old_scores.size();i++)if(std::abs(old_scores[i]-multi_scores[i])>1e-7f)
    fail("multi-context trained top-five log-score mismatch");
  size_t multi_unique=0;
  for(auto n:multi_prefixes)multi_unique+=n;
  if(multi_unique!=(size_t)unique_calls)fail("multi-context unique-prefix mismatch");
  const double serial_total=serial_seconds+begin_old;
  const double multi_ratio=serial_total/multi_seconds;
  // Fully independent trained request benchmark: includes begin, top-five
  // verifier AND one seeded sampler. Alternating the invocation order helps
  // control cache ordering; no TRAIN tuning or future labels are used.
  double full_parallel_s=full_cold_seconds,full_serial_s=0.;
  auto serial_full=[&](){
   const double start=now();
   for(size_t i=0;i<histories.size();i++){
    const auto&ctx=histories[i];float sc[5]{};int calls=0;
    if(wf176_hybrid_begin(old,(const unsigned char*)ctx.data(),ctx.size())!=0||
       wf176_hybrid_verify_top5(old,sc,&calls)!=0)fail("full serial baseline failed");
    unsigned char bytes[40]{};int written=0,accepted=0,rejected=0;
    if(wf176_hybrid_sample(old,1790000u+i,12,bytes,sizeof(bytes),&written,&accepted,&rejected)!=0)
       fail("full serial sampler failed");
   }
   return now()-start;
  };
  auto parallel_full=[&](){
   const double start=now();
   const int rc=wf179_fork_full_contexts(fork,fully_packed.data(),fully_packed.size(),
       fully_offsets.data(),fully_lengths.data(),histories.size(),full_results.data());
   if(rc)fail("full COW repeated benchmark failed");
   return now()-start;
  };
  full_serial_s+=serial_full();
  full_parallel_s+=parallel_full();
  full_parallel_s+=parallel_full();
  full_serial_s+=serial_full();
  const double full_parallel_speedup=(full_serial_s/2.)/(full_parallel_s/3.);
  int count=0,dispatched=0,nodes=0,hits=0;
  if(wf179_fork_stats(fork,&count,&dispatched,&nodes,&hits)!=0)fail("stats");
  const long after_pss_kb=sum_pss(pids);
  const double speed=parallel_seconds>0?serial_seconds/parallel_seconds:0.;
  std::ofstream out(argv[10]);if(!out)fail("results file unavailable");
  out<<"{\n  \"status\":\"PASS\",\n  \"architecture\":\"fork COW separate causal state, identical inherited byte law\",\n"
     <<"  \"workers\":"<<count<<",\n  \"books\":"<<contexts<<",\n  \"exact_full_256way_PQ_vectors\":"<<vectors
     <<",\n  \"seeded_samples_exact\":"<<samples<<",\n  \"topfive_score_checks\":"<<score_checks
     <<",\n  \"raw_prefix_calls\":"<<raw_calls<<",\n  \"unique_prefix_calls\":"<<unique_calls
     <<",\n  \"serial_prefill_seconds\":"<<serial_seconds<<",\n  \"parallel_prefill_seconds\":"<<parallel_seconds
     <<",\n  \"parallel_speedup\":"<<speed<<",\n  \"many_context_seconds\":"<<multi_seconds
     <<",\n  \"many_context_serial_including_begin_seconds\":"<<serial_total
     <<",\n  \"many_context_throughput_ratio\":"<<multi_ratio
     <<",\n  \"many_context_exact_scores\":"<<old_scores.size()
     <<",\n  \"full_contexts_all_outputs_exact\":"<<contexts
     <<",\n  \"full_contexts_parallel_seconds_three_trials\":"<<full_parallel_s
     <<",\n  \"full_contexts_serial_seconds_two_trials\":"<<full_serial_s
     <<",\n  \"full_contexts_speedup_mean_per_trial\":"<<full_parallel_speedup
     <<",\n  \"full_contexts_first_pass_seconds\":"<<full_cold_seconds
     <<",\n  \"full_pool_pss_kib_before_reference\":"<<full_pool_pss_kb
     <<",\n  \"worker_open_seconds\":"<<fork_open
     <<",\n  \"baseline_open_seconds\":"<<reference_open<<",\n  \"pool_pss_kib_before_reference\":"<<pool_pss_kb
     <<",\n  \"pool_pss_kib_after\":"<<after_pss_kb
     <<",\n  \"last_context_dispatched\":"<<dispatched<<",\n  \"last_context_cache_nodes\":"<<nodes
     <<",\n  \"methodology\":\"standalone single-threaded parent pre-fork; same trained weights, alternating AB/BA context order; one-stream prefill excludes begin; multi-context includes begin; full-context includes begin, verifier and seeded sample. Full-context mean of 2 reference and 3 parallel passes; initial PSS measured before separate baseline load\"\n}\n";
  std::cout<<"v179 "<<nw<<" workers: "<<contexts<<" contexts, "<<vectors<<" exact vectors, "<<samples<<" seeded samples; parallel vs serial "<<speed<<"x, many-context "<<multi_ratio<<"x, full-context "<<full_parallel_speedup<<"x, PSS pool "<<pool_pss_kb<<"KiB\n";
  wf176_hybrid_close(old);wf179_fork_close(fork);return 0;
 }catch(const std::exception&e){std::cerr<<"v179 FAILED: "<<e.what()<<"\n";return 1;}
}
