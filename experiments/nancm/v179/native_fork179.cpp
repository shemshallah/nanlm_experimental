// NaNCM v179: *opt-in* copy-on-write POSIX worker pool.
// One trained v176 model is loaded in a SINGLE-THREADED dedicated native
// process, then fork() makes read-only model pages physically shared (COW).
// Each child has an independent byte-inference state. No WF148 object is
// accessed concurrently by multiple threads; the byte law remains unchanged.
// IMPORTANT: never call fork_open from a multi-threaded embedded interpreter.
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <limits>
#include "full179_api.h"
#ifdef __linux__
#include <poll.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <dirent.h>
#endif
namespace fork179 {
#ifdef __linux__
static bool single_threaded_parent(){
 DIR*d=::opendir("/proc/self/task");if(!d)return false;
 int tasks=0;struct dirent*entry=nullptr;
 while((entry=::readdir(d))){
  if(entry->d_name[0]=='.')continue;
  ++tasks;
  if(tasks>1)break;
 }
 ::closedir(d);return tasks==1;
}
struct Request {uint32_t magic=0x179ca11u, history=0, prefix=0, sequence=0;};
struct Response {uint32_t magic=0x1790ce1u, sequence=0;int32_t rc=-1;float p[256]{};};
struct FullResponse {uint32_t magic=0x1790f11u, sequence=0;int32_t rc=-1;FullResult179 result{};};
static bool write_all(int fd,const void*p,size_t n){
 const unsigned char*b=(const unsigned char*)p;
 while(n){ssize_t z=::write(fd,b,n);if(z<0&&errno==EINTR)continue;if(z<=0)return false;b+=z;n-=(size_t)z;}return true;
}
static bool read_all(int fd,void*p,size_t n){
 unsigned char*b=(unsigned char*)p;
 while(n){ssize_t z=::read(fd,b,n);if(z<0&&errno==EINTR)continue;if(z<=0)return false;b+=z;n-=(size_t)z;}return true;
}
static void worker(int input,int output,void*byte_state,void*full_model){
 // Child is single-threaded and touches ONLY its COW causal state.
 // 0x179ca11: one prefix (compatibility); 0x179ca12: batch of <=32.
 for(;;){
  Request req{};
  if(!read_all(input,&req,sizeof(req)))break;
  const bool batch=req.magic==0x179ca12u;
  const bool full=req.magic==0x179ca13u;
  if(!(batch||full||req.magic==0x179ca11u)||req.history>1000000||
     (batch?(req.prefix>32):(full?(req.prefix!=0):(req.prefix>40))))break;
  std::vector<unsigned char>past(req.history);
  if(!past.empty()&&!read_all(input,past.data(),past.size()))break;
  if(full){
   FullResponse rep{};rep.sequence=req.sequence;
   rep.rc=wf176_hybrid_begin(full_model,past.empty()?nullptr:past.data(),past.size());
   if(rep.rc==0)rep.rc=wf176_hybrid_verify_top5(full_model,
       rep.result.avg_log_byte,(int*)&rep.result.verifier_calls);
   if(rep.rc==0)for(int k=0;k<5;k++){
    int ix=-1;float p=0.f;char name[48]{};
    rep.rc=wf176_hybrid_top5(full_model,k,&ix,name,48,&p);
    if(rep.rc)break;
    rep.result.indices[k]=ix;rep.result.word_probabilities[k]=p;
    std::memcpy(rep.result.words[k],name,sizeof(name));
   }
   if(rep.rc==0){
    int written=0,accepted=0,rejected=0;
    rep.rc=wf176_hybrid_sample(full_model,1790000ull+req.sequence,12,
       rep.result.sample,sizeof(rep.result.sample),&written,&accepted,&rejected);
    rep.result.sampled_bytes=(uint32_t)written;
    rep.result.sampled_accepted=(uint32_t)accepted;
    rep.result.sampled_rejected=(uint32_t)rejected;
   }
   if(!write_all(output,&rep,sizeof(rep)))break;
   continue;
  }
  const uint32_t jobs=batch?req.prefix:1;
  for(uint32_t j=0;j<jobs;j++){
   uint32_t sequence=req.sequence,length=req.prefix;
   if(batch && (!read_all(input,&sequence,4)||!read_all(input,&length,4)))_exit(12);
   if(length>40)_exit(13);
   std::vector<unsigned char>input_bytes=past;
   input_bytes.resize(past.size()+length);
   if(length&&!read_all(input,input_bytes.data()+past.size(),length))_exit(14);
   Response rep{};rep.sequence=sequence;
   rep.rc=wf170_byte_next(byte_state,input_bytes.empty()?nullptr:input_bytes.data(),input_bytes.size(),rep.p);
   if(!write_all(output,&rep,sizeof(rep)))_exit(15);
  }
 }
 ::close(input);::close(output);_exit(0);
}
struct Worker {pid_t pid=-1;int writefd=-1,readfd=-1;};
struct State {
 void*model=nullptr;
 std::vector<Worker>workers;
 std::unordered_map<std::string,lattice177::Node> cache;
 size_t misses=0,hits=0;
 bool ready=false;
 int dispatched=0,workers_touched=0;
 ~State(){
  // Close in reverse responsibility order; EOF is a graceful child shutdown.
  for(auto&w:workers){if(w.writefd>=0)::close(w.writefd);if(w.readfd>=0)::close(w.readfd);}
  for(auto&w:workers){if(w.pid>0){int status=0;while(::waitpid(w.pid,&status,0)<0&&errno==EINTR){}}}
  if(model)wf176_hybrid_close(model);
 }
};
struct Task {std::string prefix;lattice177::Node node;int rc=0;};
static bool dispatch(const Worker&w,const std::vector<unsigned char>&history,
                     const std::string&prefix,uint32_t seq){
 Request q{};q.sequence=seq;q.history=(uint32_t)history.size();q.prefix=(uint32_t)prefix.size();
 return write_all(w.writefd,&q,sizeof(q))&&
  (history.empty()||write_all(w.writefd,history.data(),history.size()))&&
  (prefix.empty()||write_all(w.writefd,prefix.data(),prefix.size()));
}
static bool spawn(State&s,int number){
 auto*core=(third176::State*)s.model;
 if(!core||!core->bytes)return false;
 for(int i=0;i<number;i++){
  int to[2]{-1,-1},from[2]{-1,-1};
  if(::pipe(to)!=0)return false;
  if(::pipe(from)!=0){::close(to[0]);::close(to[1]);return false;}
  const pid_t pid=::fork();
  if(pid<0){::close(to[0]);::close(to[1]);::close(from[0]);::close(from[1]);return false;}
  if(pid==0){
   ::close(to[1]);::close(from[0]);
   for(auto&w:s.workers){::close(w.writefd);::close(w.readfd);}
   worker(to[0],from[1],core->bytes,s.model);
  }
  ::close(to[0]);::close(from[1]);
  s.workers.push_back({pid,to[1],from[0]});
 }
 return true;
}
static int fill(State*s,const std::vector<std::string>&prefixes){
 if(!s||!s->ready||s->workers.empty())return -1;
 auto*core=(third176::State*)s->model;
 std::vector<Task>tasks;tasks.reserve(prefixes.size());
 std::unordered_set<std::string>seen;
 for(const auto&p:prefixes){
  if(p.size()>40)return -2;
  if(s->cache.count(p)||!seen.insert(p).second)continue;
  tasks.push_back({p,{},{}});
 }
 if(tasks.empty())return 0;
 if(s->cache.size()+tasks.size()>256)return -3;
 std::sort(tasks.begin(),tasks.end(),[](const Task&a,const Task&b){return a.prefix<b.prefix;});
 // One history transfer and <=32 fixed-size replies per worker per wave.
 // 32 * sizeof(Response) fits a 64 KiB pipe even while parent dispatches peers;
 // no thread, busy waiting, or individual-prefix parent/child handshake.
 constexpr size_t MAX_BATCH=32;
 const size_t nw=s->workers.size();
 int workers_used=0;
 for(size_t offset=0;offset<tasks.size();offset+=nw*MAX_BATCH){
  std::vector<std::vector<size_t>>assign(nw);
  for(size_t j=offset;j<std::min(tasks.size(),offset+nw*MAX_BATCH);j++){
   const size_t w=((j-offset)/MAX_BATCH);
   assign[w].push_back(j);
  }
  for(size_t w=0;w<nw;w++)if(!assign[w].empty()){
   Request q{};q.magic=0x179ca12u;q.history=(uint32_t)core->history.size();
   q.prefix=(uint32_t)assign[w].size();q.sequence=0;
   if(!write_all(s->workers[w].writefd,&q,sizeof(q))||
     (!core->history.empty()&&!write_all(s->workers[w].writefd,core->history.data(),core->history.size())))return -4;
   for(const auto j:assign[w]){
    const uint32_t seq=(uint32_t)j,len=(uint32_t)tasks[j].prefix.size();
    if(!write_all(s->workers[w].writefd,&seq,4)||!write_all(s->workers[w].writefd,&len,4)||
       (len&&!write_all(s->workers[w].writefd,tasks[j].prefix.data(),len)))return -5;
   }
   ++workers_used;
  }
  for(size_t w=0;w<nw;w++)for(const auto j:assign[w]){
   Response resp{};
   if(!read_all(s->workers[w].readfd,&resp,sizeof(resp)))return -6;
   if(resp.magic!=0x1790ce1u||resp.sequence!=(uint32_t)j||resp.rc!=0)return -7;
   std::memcpy(tasks[j].node.p.data(),resp.p,sizeof(resp.p));
   if(!third176::finite_simplex(tasks[j].node.p.data(),256)||
       !parallel177::fill_proposal(*core,tasks[j].prefix,tasks[j].node))return -8;
   ++s->dispatched;
  }
 }
 for(auto&t:tasks)s->cache.emplace(std::move(t.prefix),std::move(t.node));
 s->misses+=tasks.size();s->workers_touched=std::min((size_t)workers_used,nw);
 return (int)tasks.size();
}
#endif
}
extern "C" void*wf179_fork_open_full(const char*teacher,const char*m,const char*g,const char*c,
 const char*bytepack,float escape,const char*cross,const char*bpe,int workers){
#ifdef __linux__
 if(workers<1||workers>4||!fork179::single_threaded_parent())return nullptr;
 try{
  std::unique_ptr<fork179::State>s(new fork179::State);
  s->model=wf176_hybrid_open_full(teacher,m,g,c,bytepack,escape,cross,bpe);
  if(!s->model||!fork179::spawn(*s,workers))return nullptr;
  return s.release();
 }catch(...){return nullptr;}
#else
 (void)teacher;(void)m;(void)g;(void)c;(void)bytepack;(void)escape;(void)cross;(void)bpe;(void)workers;
 return nullptr;
#endif
}
extern "C" void wf179_fork_close(void*h){
#ifdef __linux__
 delete (fork179::State*)h;
#else
 (void)h;
#endif
}
extern "C" int wf179_fork_begin(void*h,const unsigned char*history,size_t n){
#ifdef __linux__
 auto*s=(fork179::State*)h;
 if(!s||(!history&&n)||n>1000000)return -1;
 s->ready=false;s->cache.clear();s->hits=s->misses=0;s->dispatched=s->workers_touched=0;
 int rc=wf176_hybrid_begin(s->model,history,n);
 s->ready=rc==0;return rc;
#else
 return -1;
#endif
}
extern "C" int wf179_fork_prefill(void*h,float scores[5],int*raw,int*unique){
#ifdef __linux__
 auto*s=(fork179::State*)h;if(!s||!s->ready||!scores||!raw||!unique)return -1;
 auto*core=(third176::State*)s->model;
 std::vector<std::string>prefixes;
 int sumcalls=0;
 for(const auto&cand:core->best){
  if(cand.bytes.empty()||cand.bytes.size()>40)return -2;
  std::string prefix;
  for(char c:cand.bytes){prefixes.push_back(prefix);prefix+=c;++sumcalls;}
 }
 const int n=fork179::fill(s,prefixes);
 if(n<0)return n;
 for(int k=0;k<5;k++){
  const auto&w=core->best[k].bytes;
  std::string prefix;double logscore=0.;
  for(unsigned char c:w){
   auto it=s->cache.find(prefix);if(it==s->cache.end())return -3;
   logscore+=std::log(std::max(1e-20f,it->second.p[c]));prefix.push_back((char)c);
  }
  scores[k]=(float)(logscore/w.size());
 }
 *raw=sumcalls;*unique=n;return 0;
#else
 return -1;
#endif
}
extern "C" int wf179_fork_next(void*h,const unsigned char*prefix,size_t n,float*q,float*p){
#ifdef __linux__
 auto*s=(fork179::State*)h;
 if(!s||!s->ready||(!prefix&&n)||n>40||!q||!p)return -1;
 const std::string key(n?std::string((const char*)prefix,n):std::string());
 auto it=s->cache.find(key);
 if(it!=s->cache.end()){
  std::memcpy(q,it->second.q.data(),256*sizeof(float));
  std::memcpy(p,it->second.p.data(),256*sizeof(float));++s->hits;return 0;
 }
 // Fallback is the exact inherited oracle, not an approximation. For non-top5
 // prefixes, this remains serial. Fork workers only accelerate planned batches.
 return wf176_hybrid_next(s->model,prefix,n,q,p);
#else
 return -1;
#endif
}
extern "C" int wf179_fork_stats(void*h,int*workers,int*dispatched,int*cache_nodes,int*hits){
#ifdef __linux__
 auto*s=(fork179::State*)h;
 if(!s||!workers||!dispatched||!cache_nodes||!hits)return -1;
 *workers=(int)s->workers.size();*dispatched=s->dispatched;*cache_nodes=(int)s->cache.size();*hits=(int)s->hits;return 0;
#else
 return -1;
#endif
}
extern "C" int wf179_fork_pid(void*h,int index){
#ifdef __linux__
 auto*s=(fork179::State*)h;
 if(!s||index<0||index>=(int)s->workers.size())return -1;
 return (int)s->workers[index].pid;
#else
 return -1;
#endif
}
// Exact seeded speculative decoder, with P/Q from the COW-shared prefix cache
// when available and the original v176 oracle otherwise.
extern "C" int wf179_fork_sample(void*h,uint64_t seed,int maxbytes,unsigned char*out,
 size_t cap,int*written,int*accepted,int*rejected){
#ifdef __linux__
 auto*s=(fork179::State*)h;
 if(!s||!s->ready||!out||!written||!accepted||!rejected||maxbytes<1||maxbytes>40||(size_t)maxbytes>cap)return -1;
 std::mt19937_64 rng(seed);
 std::uniform_real_distribution<double>uniform(0.,1.);
 int n=0,a=0,r=0;
 std::array<float,256>q{},p{},res{};
 auto draw=[&](const float*law)->int{
  const double x=uniform(rng);double total=0.;
  for(int j=0;j<256;j++){total+=law[j];if(x<total)return j;}
  return 255;
 };
 for(int k=0;k<maxbytes;k++){
  if(wf179_fork_next(h,out,n,q.data(),p.data())!=0)return -2;
  int proposed=draw(q.data());float chance=0.f;
  if(wf176_spec_accept(p.data(),q.data(),proposed,&chance,res.data())!=0)return -3;
  const bool yes=uniform(rng)<(double)chance;
  const int byte=yes?proposed:draw(res.data());
  out[n++]=(unsigned char)byte;a+=(int)yes;
  if(!yes){r=1;break;}
  if(byte==' ')break;
 }
 *written=n;*accepted=a;*rejected=r;return 0;
#else
 return -1;
#endif
}
// Throughput-oriented multi-context prefill. Each worker is assigned a distinct
// history, so it can reuse causal anchor state through that history's prefixes.
// The parent computes word proposals in exactly the inherited v176 order;
// workers compute only independent BYTE verifier probabilities. Contexts do
// not exchange predictions, labels, or hidden states.
extern "C" int wf179_fork_verify_contexts(void*h,const unsigned char*bytes,size_t total,
 const uint32_t*offset,const uint32_t*length,size_t contexts,float*out_scores,
 uint32_t*out_unique){
#ifdef __linux__
 auto*s=(fork179::State*)h;
 if(!s||contexts<1||contexts>128||(!bytes&&total)||!offset||!length||!out_scores||!out_unique)return -1;
 struct Item {
  std::vector<unsigned char>history;
  std::array<third176::Cand,5>best;
  std::vector<std::string>prefixes;
  std::unordered_map<std::string,lattice177::Node>nodes;
 };
 std::vector<Item>items(contexts);
 auto*core=(third176::State*)s->model;
 // All history sizes and offsets are validated before touching the model.
 for(size_t i=0;i<contexts;i++)if(length[i]>1000000||offset[i]>total||length[i]>total-offset[i])return -2;
 s->ready=false;s->cache.clear();
 for(size_t i=0;i<contexts;i++){
  auto&item=items[i];const auto*past=length[i]?bytes+offset[i]:nullptr;
  if(length[i])item.history.assign(past,past+length[i]);
  if(wf176_hybrid_begin(s->model,past,length[i])!=0)return -3;
  item.best=core->best;
  std::unordered_set<std::string>seen;
  for(const auto&cand:item.best){
   if(cand.bytes.empty()||cand.bytes.size()>40)return -4;
   std::string pre;
   for(unsigned char c:cand.bytes){
    if(seen.insert(pre).second)item.prefixes.push_back(pre);
    pre.push_back((char)c);
   }
  }
  if(item.prefixes.size()>256)return -5;
  std::sort(item.prefixes.begin(),item.prefixes.end());
  out_unique[i]=(uint32_t)item.prefixes.size();
 }
 constexpr size_t CHUNK=32;
 const size_t nw=s->workers.size();
 for(size_t first=0;first<contexts;first+=nw){
  // Each wave assigns at most one independent causal history per worker.
  size_t max_prefix=0;
  for(size_t w=0;w<nw&&first+w<contexts;w++)
   max_prefix=std::max(max_prefix,items[first+w].prefixes.size());
  for(size_t start=0;start<max_prefix;start+=CHUNK){
   for(size_t w=0;w<nw&&first+w<contexts;w++){
    const auto&item=items[first+w];
    if(start>=item.prefixes.size())continue;
    const size_t count=std::min(CHUNK,item.prefixes.size()-start);
    fork179::Request req{};req.magic=0x179ca12u;
    req.history=(uint32_t)item.history.size();req.prefix=(uint32_t)count;
    if(!fork179::write_all(s->workers[w].writefd,&req,sizeof(req))||
      (!item.history.empty()&&!fork179::write_all(s->workers[w].writefd,item.history.data(),item.history.size())))return -6;
    for(size_t j=start;j<start+count;j++){
     const uint32_t seq=(uint32_t)j,len=(uint32_t)item.prefixes[j].size();
     if(!fork179::write_all(s->workers[w].writefd,&seq,4)||
        !fork179::write_all(s->workers[w].writefd,&len,4)||
        (len&&!fork179::write_all(s->workers[w].writefd,item.prefixes[j].data(),len)))return -7;
    }
   }
   for(size_t w=0;w<nw&&first+w<contexts;w++){
    auto&item=items[first+w];
    if(start>=item.prefixes.size())continue;
    const size_t count=std::min(CHUNK,item.prefixes.size()-start);
    // A scratch core recreates v176's normalized top-five proposal exactly.
    third176::State scratch;scratch.best=item.best;scratch.escape=core->escape;
    for(size_t j=start;j<start+count;j++){
     fork179::Response resp{};
     if(!fork179::read_all(s->workers[w].readfd,&resp,sizeof(resp)))return -8;
     if(resp.magic!=0x1790ce1u||resp.sequence!=j||resp.rc!=0)return -9;
     lattice177::Node node;
     std::memcpy(node.p.data(),resp.p,sizeof(resp.p));
     if(!third176::finite_simplex(node.p.data(),256)||
       !parallel177::fill_proposal(scratch,item.prefixes[j],node))return -10;
     item.nodes.emplace(item.prefixes[j],std::move(node));
    }
   }
  }
 }
 // Summarize conditional P(log word). No trained weights are changed.
 for(size_t i=0;i<contexts;i++){
  auto&item=items[i];
  for(int k=0;k<5;k++){
   const auto&word=item.best[k].bytes;std::string pre;double ll=0.;
   for(unsigned char c:word){
    auto it=item.nodes.find(pre);if(it==item.nodes.end())return -11;
    ll+=std::log(std::max(1e-20f,it->second.p[c]));pre.push_back((char)c);
   }
   out_scores[i*5+k]=(float)(ll/word.size());
  }
 }
 return 0;
#else
 return -1;
#endif
}
// Full parallel model execution: COW workers run inherited word prediction,
// byte verification and speculative sampling independently for each context.
// Unlike verify_contexts(), parent no longer performs any neural prediction.
// The shared pre-fork immutable tables are not cloned or deserialized.
extern "C" int wf179_fork_full_contexts(void*h,const unsigned char*bytes,size_t total,
 const uint32_t*offset,const uint32_t*length,size_t contexts,FullResult179*out){
#ifdef __linux__
 auto*s=(fork179::State*)h;
 if(!s||contexts<1||contexts>128||(!bytes&&total)||!offset||!length||!out||s->workers.empty())return -1;
 for(size_t i=0;i<contexts;i++)if(length[i]>1000000||offset[i]>total||length[i]>total-offset[i])return -2;
 // A disabled state prevents accidental reuse of a previously cached context.
 s->ready=false;s->cache.clear();
 const size_t nw=s->workers.size();
 std::vector<int>active(nw,-1);
 size_t next=0,done=0;
 auto dispatch=[&](size_t w)->bool{
  if(next>=contexts){active[w]=-1;return true;}
  const size_t id=next++;
  active[w]=(int)id;
  fork179::Request req{};req.magic=0x179ca13u;req.history=length[id];req.prefix=0;req.sequence=(uint32_t)id;
  return fork179::write_all(s->workers[w].writefd,&req,sizeof(req))&&
      (req.history==0||fork179::write_all(s->workers[w].writefd,bytes+offset[id],req.history));
 };
 for(size_t w=0;w<nw;w++)if(!dispatch(w))return -3;
 while(done<contexts){
  std::vector<pollfd>fds;std::vector<size_t>who;
  for(size_t w=0;w<nw;w++)if(active[w]>=0){fds.push_back({s->workers[w].readfd,POLLIN,0});who.push_back(w);}
  if(fds.empty())return -4;
  int rc=::poll(fds.data(),fds.size(),30000);
  if(rc<0&&errno==EINTR)continue;
  if(rc<=0)return -5;
  for(size_t i=0;i<fds.size();i++)if(fds[i].revents){
   const size_t w=who[i];const int task=active[w];
   if(!(fds[i].revents&POLLIN)||task<0)return -6;
   fork179::FullResponse rep{};
   if(!fork179::read_all(s->workers[w].readfd,&rep,sizeof(rep)))return -7;
   if(rep.magic!=0x1790f11u||rep.sequence!=(uint32_t)task||rep.rc!=0)return -8;
   out[task]=rep.result;
   if(!dispatch(w))return -9;
   ++done;
  }
 }
 return 0;
#else
 return -1;
#endif
}
