// NaNCM v180: LOSSLESS train-book-only stream recovery -> exact count n-gram memory.
// S+/S- cloud remains the inherited model; this is an opt-in candidate specialist.
// Version 1: no held-out book's targets enter the index; no cross-book leakage.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <cassert>
struct Book {int id;std::vector<int16_t> w;};
struct Row {int id,pos;std::array<int16_t,49> w;};
struct Pair {uint64_t key;uint16_t target; bool operator<(const Pair&x) const {return key!=x.key?key<x.key:target<x.target;}};
struct Entry {uint64_t key;uint32_t count;uint32_t best_count;uint32_t second_count;uint16_t best,second;uint32_t distinct;};
static std::vector<Book>read_books(const std::string&path){
 std::ifstream f(path,std::ios::binary);char m[8];f.read(m,8);if(std::string(m,8)!="NC180OBS")throw std::runtime_error("bad stream magic");
 uint32_t count;f.read((char*)&count,4);std::vector<Book>r;r.reserve(count);
 for(uint32_t i=0;i<count;i++){int32_t id;uint32_t n;f.read((char*)&id,4);f.read((char*)&n,4);if(n>1000000)throw std::runtime_error("long book");Book b; b.id=id;b.w.resize(n);f.read((char*)b.w.data(),n*2);r.push_back(std::move(b));}
 if(!f ||f.peek()!=EOF)throw std::runtime_error("truncated stream");return r;
}
static std::vector<Row>read_rows(const std::string&path){
 std::ifstream f(path,std::ios::binary);char m[8];f.read(m,8);if(std::string(m,8)!="NC180ROW")throw std::runtime_error("bad rows magic");uint32_t n;f.read((char*)&n,4);
 std::vector<Row>r(n);for(auto&x:r){f.read((char*)&x.id,4);f.read((char*)&x.pos,4);f.read((char*)x.w.data(),98);}if(!f||f.peek()!=EOF)throw std::runtime_error("truncated rows");return r;
}
static uint64_t makekey(const int16_t*hist,int n){uint64_t v=0;for(int i=n;i>0;i--){int w=hist[-i];if(w<0||w>=12000)return UINT64_MAX;v=v*16384+(uint16_t)w+1;}return v;}
static bool fit_id(int id){uint32_t x=(uint32_t)id*2654435761u+0x9e3779b9u;return x%5!=0;}
struct Memory{
 std::array<std::vector<Entry>,5>index;
 std::array<uint32_t,12000>unigram{};
 void train(const std::vector<Book>&books,bool fit_only){
  for(const auto&b:books)if(!fit_only||fit_id(b.id))for(int16_t w:b.w)if(w>=0&&w<12000)unigram[w]++;
  for(int n=1;n<=4;n++){
   std::vector<Pair>pairs;
   for(const auto&b:books){if(fit_only&&!fit_id(b.id))continue;
    const auto &w=b.w;for(size_t j=n;j<w.size();j++){
     if(w[j]<0||w[j]>=12000)continue;
     uint64_t key=makekey(w.data()+j,n);if(key==UINT64_MAX)continue;
     pairs.push_back({key,(uint16_t)w[j]});
    }
   }
   std::sort(pairs.begin(),pairs.end());auto&dest=index[n];dest.reserve(pairs.size());
   for(size_t i=0;i<pairs.size();){
    const uint64_t key=pairs[i].key;Entry e{};e.key=key;e.best=65535;e.second=65535;
    while(i<pairs.size()&&pairs[i].key==key){
     uint16_t token=pairs[i].target;uint32_t c=0;
     while(i<pairs.size()&&pairs[i].key==key&&pairs[i].target==token){i++;c++;}
     e.count+=c;e.distinct++;
     if(c>e.best_count){e.second=e.best;e.second_count=e.best_count;e.best=token;e.best_count=c;}
     else if(c>e.second_count){e.second=token;e.second_count=c;}
    }
    dest.push_back(e);
   }
   std::cerr<<"ngram "<<n<<" contexts="<<dest.size()<<" pairs="<<pairs.size()<<"\n";
  }
 }
 const Entry*query(const int16_t*end,int n)const{
  auto key=makekey(end,n);if(key==UINT64_MAX)return nullptr;
  const auto&arr=index[n];auto it=std::lower_bound(arr.begin(),arr.end(),key,[](const Entry&e,uint64_t k){return e.key<k;});
  return it!=arr.end()&&it->key==key?&*it:nullptr;
 }
};
struct Totals {uint64_t n=0,correct=0,coverage=0,base=0;};
struct Pred {int word=-1;int ngram=0,count=0,majority=0;};
static Pred predict(const Memory&mem,const Row&r,int threshold=1){
  auto x=r.w.data()+48;
  for(int n=4;n>=1;n--){const auto *p=mem.query(x,n);if(!p)continue;
   if(p->count>= (uint32_t)threshold && (p->best_count>=2 || (n<=2&&p->count>=3)))return {(int)p->best,n,(int)p->count,(int)p->best_count};
  }
  return {-1,0,0,0};
}
static void eval(const Memory&mem,const std::vector<Row>&rows,const std::string&split,bool fit_calib){
 std::array<uint64_t,5>hit{},correct{},top2{},nchosen{};uint64_t used=0,coverage=0;
 uint64_t oracle8=0,cnt1=0,cnt2=0,cnt4=0,all=0;
 std::array<uint64_t,5>fixedHit{},fixedCorrect{};
 for(const auto&r:rows){if(fit_calib?fit_id(r.id):false)continue;
  int gold=r.w[48];all++;std::vector<int>cand;const int16_t*end=r.w.data()+48;
  for(int n=1;n<=4;n++){const Entry *e=mem.query(end,n);if(!e)continue;
   hit[n]++;correct[n]+=(e->best==gold);top2[n]+=(e->best==gold||e->second==gold);
   cand.push_back(e->best);if(e->second!=65535)cand.push_back(e->second);
  }
  oracle8+=std::find(cand.begin(),cand.end(),gold)!=cand.end();
  for(int th:{1,2,3,4}){int found=-1;for(int n=4;n>=1;n--){const Entry*e=mem.query(end,n);if(!e)continue;
    if(e->best_count>=(uint32_t)th){found=e->best;break;}}
    if(th==1){cnt1+=found==gold;}if(th==2)cnt2+=found==gold;if(th==4)cnt4+=found==gold;
  }
  Pred p=predict(mem,r);used+=p.word>=0;coverage+=p.word==gold;
 }
 std::cout<<"EVAL "<<split<<" rows="<<all<<" exact_suffix_corpus_retrieval="<<coverage<<"/"<<all<<" rate="<<(double)coverage/all<<" coverage="<<used<<" oracle_top2_union="<<oracle8<<"/"<<all<<" rate="<<(double)oracle8/all<<" thresholds="<<cnt1<<","<<cnt2<<","<<cnt4<<"\n";
 for(int n=1;n<=4;n++)std::cout<<"ORDER "<<n<<" hit="<<hit[n]<<" rate="<<(double)hit[n]/all<<" top1="<<correct[n]<<" top2="<<top2[n]<<" conditional_top1="<<(hit[n]?(double)correct[n]/hit[n]:0)<<"\n";
}
static void export_candidates(const Memory&mem,const std::vector<Row>&rows,const std::string&path,bool only_calib){
 std::ofstream f(path,std::ios::binary);if(!f)throw std::runtime_error("open output");
 f.write("NC180CAN",8);uint32_t n=0;
 for(const auto &r:rows)if(!only_calib||!fit_id(r.id))n++;
 f.write((char*)&n,4);
 for(const auto &r:rows){if(only_calib&&fit_id(r.id))continue;
  f.write((char*)&r.id,4);f.write((char*)&r.pos,4);int32_t gold=r.w[48];f.write((char*)&gold,4);
  const int16_t *x=r.w.data()+48;
  for(int k=1;k<=4;k++){
    const auto *entry=mem.query(x,k);
    uint32_t d[6];if(entry){d[0]=entry->best;d[1]=entry->second;d[2]=entry->best_count;d[3]=entry->second_count;d[4]=entry->count;d[5]=entry->distinct;}
    else {d[0]=d[1]=65535;d[2]=d[3]=d[4]=d[5]=0;}
    f.write((char*)d,sizeof(d));
  }
 }
 if(!f)throw std::runtime_error("write failed");
 std::cerr<<"export "<<path<<" "<<n<<" rows\n";
}

#ifndef NANCM_MEMORY180_LIBRARY
int main(int argc,char**argv){try{if(argc!=4&&argc!=6){std::cerr<<"usage: program TRAIN_OBS TRAIN_ROWS VAL_ROWS [TRAIN_CANDIDATES VAL_CANDIDATES]\n";return 2;}
 auto t0=std::chrono::steady_clock::now();auto books=read_books(argv[1]);auto train=read_rows(argv[2]);auto val=read_rows(argv[3]);
 Memory mem;mem.train(books,true);eval(mem,train,"TRAIN_BOOK_HELDOUT",true);
 if(argc==6)export_candidates(mem,train,argv[4],true);
 auto t1=std::chrono::steady_clock::now();std::cerr<<"FIT elapsed="<<std::chrono::duration<double>(t1-t0).count()<<"s\n";
 // Final fit includes ALL TRAIN books before the already-explored VAL evaluation.
 Memory full;full.train(books,false);eval(full,val,"REUSED_VAL",false);
 if(argc==6)export_candidates(full,val,argv[5],false);
 auto t2=std::chrono::steady_clock::now();std::cerr<<"TOTAL elapsed="<<std::chrono::duration<double>(t2-t0).count()<<"s\n";
 return 0;
 }catch(const std::exception&e){std::cerr<<"ERROR "<<e.what()<<"\n";return 1;}}

#endif // NANCM_MEMORY180_LIBRARY
