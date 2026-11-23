// v177 experimental *real* CPU parallel independent-prefix verifier.
// Optional, separate-handle mode only. Each worker owns its own frozen v170
// byte-model inference state. No two threads touch the same WF148/ANN state.
// Deterministic results are checked against the serial byte verifier.
#include <atomic>
#include <thread>
#include <mutex>
#include <unordered_set>
namespace parallel177 {
struct State {
 void* memo=nullptr;
 std::vector<void*> workers;
 std::string model,geometry,cloud,bytepack;
 ~State(){for(void*w:workers)wf170_byte_close(w);if(memo)wf177_close(memo);}
};
struct Task {
 std::string prefix;
 lattice177::Node node;
 int error=0;
};
static bool fill_proposal(const third176::State&core,const std::string&prefix,lattice177::Node&node){
 double weights[256]{},mass=0.;
 for(const auto&cand:core.best){
  if(cand.bytes.size()>prefix.size() && cand.bytes.compare(0,prefix.size(),prefix)==0){
   const unsigned char b=(unsigned char)cand.bytes[prefix.size()];
   weights[b]+=cand.normalized;mass+=cand.normalized;
  }
 }
 double z=0.;
 for(int j=0;j<256;j++){
  const double q=mass>0?(core.escape*node.p[j]+(1.-core.escape)*weights[j]/mass):node.p[j];
  node.q[j]=(float)q;z+=node.q[j];
 }
 if(!(z>0.)||!std::isfinite(z))return false;
 for(float &p:node.q)p/=(float)z;
 return third176::finite_simplex(node.q.data(),256);
}
}
extern "C" void* wf177_parallel_open_full(const char*teacher,const char*m,const char*g,const char*c,
                                             const char*bytepack,float escape,
                                             const char*cross,const char*bpe,int threads){
 if(threads<1||threads>4||!m||!g||!c||!bytepack)return nullptr;
 try{
  auto s=std::make_unique<parallel177::State>();
  s->model=m;s->geometry=g;s->cloud=c;s->bytepack=bytepack;
  s->memo=wf177_open_full(teacher,m,g,c,bytepack,escape,cross,bpe);
  if(!s->memo)return nullptr;
  for(int t=0;t<threads;t++){
   void* h=wf170_byte_open(m,g,c,bytepack);
   if(!h)return nullptr;
   if(wf170_byte_set_adaptive(h,0)!=0){wf170_byte_close(h);return nullptr;}
   s->workers.push_back(h);
  }
  return s.release();
 }catch(...){return nullptr;}
}
extern "C" void wf177_parallel_close(void*h){delete (parallel177::State*)h;}
extern "C" int wf177_parallel_begin(void*h,const unsigned char*past,size_t n){
 auto*s=(parallel177::State*)h;
 if(!s||!s->memo)return -1;
 int rc=wf177_begin(s->memo,past,n);
 if(rc)return rc;
 for(void*worker:s->workers)if(wf170_byte_reset(worker)!=0)return -2;
 return 0;
}
extern "C" int wf177_parallel_prefill_top5(void*h,float scores[5],int*new_calls,int*workers_used){
 auto*s=(parallel177::State*)h;
 if(!s||!s->memo||!scores||!new_calls||!workers_used||s->workers.empty())return -1;
 auto*mem=(lattice177::State*)s->memo;
 if(!lattice177::valid(mem))return -2;
 auto*core=(third176::State*)mem->inherited;
 std::unordered_set<std::string>unique;
 for(const auto&cand:core->best){
  if(cand.bytes.empty()||cand.bytes.size()>40)return -3;
  std::string prefix;
  for(const unsigned char c:cand.bytes){
   if(mem->nodes.find(prefix)==mem->nodes.end())unique.insert(prefix);
   prefix.push_back((char)c);
  }
 }
 std::vector<parallel177::Task>tasks;
 tasks.reserve(unique.size());
 for(const auto&p:unique){parallel177::Task t;t.prefix=p;tasks.push_back(std::move(t));}
 std::sort(tasks.begin(),tasks.end(),[](const auto&a,const auto&b){return a.prefix<b.prefix;});
 std::atomic<size_t>index{0};std::atomic<int>err{0};
 const int threads=(int)std::min(tasks.size(),s->workers.size());
 std::vector<std::thread>jobs;
 auto work=[&](int w){
  while(true){
   const size_t i=index.fetch_add(1);
   if(i>=tasks.size())break;
   std::vector<unsigned char>past=core->history;
   const auto&pref=tasks[i].prefix;
   past.insert(past.end(),pref.begin(),pref.end());
   const int rc=wf170_byte_next(s->workers[w],past.empty()?nullptr:past.data(),past.size(),tasks[i].node.p.data());
   if(rc!=0||!parallel177::fill_proposal(*core,pref,tasks[i].node)){tasks[i].error=rc?rc:-10;err.store(1);}
  }
 };
 try{
  for(int w=0;w<threads;w++)jobs.emplace_back(work,w);
  for(auto&job:jobs)job.join();
 }catch(...){for(auto&job:jobs)if(job.joinable())job.join();return -4;}
 if(err.load()!=0)return -5;
 // Commit as one transaction after workers join. No cache mutation inside threads.
 for(auto&t:tasks){
  if(mem->nodes.size()>=256)return -6;
  mem->nodes.emplace(std::move(t.prefix),std::move(t.node));
  mem->misses++;
 }
 *new_calls=(int)tasks.size();*workers_used=threads;
 int raw=0,need=0;
 return wf177_prepare_top5(s->memo,scores,&raw,&need)==0&&need==0?0:-7;
}
extern "C" int wf177_parallel_next(void*h,const unsigned char*prefix,size_t n,float*q,float*p){
 auto*s=(parallel177::State*)h;return s?wf177_next(s->memo,prefix,n,q,p):-1;
}
extern "C" int wf177_parallel_sample(void*h,uint64_t seed,int maxbytes,unsigned char*out,size_t cap,
                                        int*written,int*accepted,int*rejected){
 auto*s=(parallel177::State*)h;
 return s?wf177_sample(s->memo,seed,maxbytes,out,cap,written,accepted,rejected):-1;
}
extern "C" int wf177_parallel_stats(void*h,int*misses,int*hits,int*nodes){
 auto*s=(parallel177::State*)h;
 return s?wf177_stats(s->memo,misses,hits,nodes):-1;
}