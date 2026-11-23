// v180: train-only reconstructed prefix suffixes participate in the SAME
// signed-cloud whole-word simplex via an exact normalized sparse residual.
// No byte verifier modifications, no target lookup, no TEST use.
#define NANCM_MEMORY180_LIBRARY
#include "native_memory180.cpp"
#undef NANCM_MEMORY180_LIBRARY
#include <array>
#include <memory>
namespace symbiotic180 {
struct State {
 Memory memory;
 cross176::Native* cross = nullptr; // borrowed; caller retains ownership
 std::array<double,4> gamma{{.1,.1,.1,.1}}; // frozen on TRAIN-only calibration
};
static int apply(State&state,const int16_t*history,float*prob,size_t cap){
 if(!history||!prob||cap!=12000||!state.cross)return -1;
 std::array<const Entry*,4>hit{};double scale=1.;
 for(int k=0;k<4;k++){
  hit[k]=state.memory.query(history+48,k+1);
  const Entry*e=hit[k];if(!e||!e->count)continue;
  if(e->best>=12000||e->second>=12000&&e->second!=65535)return -2;
  double m=(double)(e->best_count+e->second_count)/e->count;
  if(m>1.+1e-10||m<0)return -3;
  scale-=state.gamma[k]*m;
 }
 if(scale<0.||scale>1.000001||!std::isfinite(scale))return -4;
 double total=0.;for(int i=0;i<12000;i++){
  if(!std::isfinite(prob[i])||prob[i]<0)return -5;
  prob[i]=(float)(scale*prob[i]);total+=prob[i];
 }
 for(int k=0;k<4;k++){
  auto*e=hit[k];if(!e)continue;
  double factor=state.gamma[k]/e->count;
  prob[e->best]+=(float)(factor*e->best_count);
  if(e->second!=65535)prob[e->second]+=(float)(factor*e->second_count);
 }
 total=0.;for(int i=0;i<12000;i++)total+=prob[i];
 return std::abs(total-1.)<=5e-4?0:-6;
}
}
extern "C" void* wf180_memory_open(void*cross,const char*train_stream){
 if(!cross||!train_stream)return nullptr;
 try{
  auto s=std::make_unique<symbiotic180::State>();
  s->cross=(cross176::Native*)cross;
  if(!s->cross->parent)return nullptr;
  auto books=read_books(train_stream);
  if(books.size()!=396)return nullptr;
  s->memory.train(books,false);
  return s.release();
 }catch(...){return nullptr;}
}
extern "C" void wf180_memory_close(void*h){delete (symbiotic180::State*)h;}
extern "C" int wf180_memory_predict(void*h,const unsigned char*past,size_t n,float*out,size_t cap){
 if(!h||(!past&&n)||!out||cap!=12000||n>1000000)return -1;
 auto*s=(symbiotic180::State*)h;
 if(wf176_cross_predict(s->cross,past,n,out,cap)!=0)return -2;
 auto hist=cross176::parse(*s->cross->parent,past,n);
 std::array<int16_t,48> short_history{};
 for(size_t k=0;k<48;k++)short_history[k]=(int16_t)hist[k];
 return symbiotic180::apply(*s,short_history.data(),out,cap);
}
extern "C" int wf180_memory_apply(void*h,const int16_t*history48,float*out,size_t cap){
 if(!h)return -1;
 return symbiotic180::apply(*(symbiotic180::State*)h,history48,out,cap);
}
extern "C" int wf180_memory_stats(void*h,uint32_t counts[4]){
 if(!h||!counts)return -1;
 auto*s=(symbiotic180::State*)h;
 for(int k=0;k<4;k++)counts[k]=(uint32_t)s->memory.index[k+1].size();
 return 0;
}
