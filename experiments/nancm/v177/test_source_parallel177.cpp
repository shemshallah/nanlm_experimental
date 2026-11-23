// Source-only two-thread smoke test with independent synthetic byte states.
#define main inherited_v176_source_main
#include "../v176/test_source176.cpp"
#undef main
extern "C" void* wf176_hybrid_open_full(const char*t,const char*m,const char*g,
 const char*c,const char*b,float escape,const char*,const char*){
 return wf176_hybrid_open(t,m,g,c,b,escape);
}
#include "native_lattice177.cpp"
#include "native_parallel177.cpp"
int main(){
 void*h=wf177_parallel_open_full("x","x","x","x","x",.35f,"x","x",2);
 assert(h);
 assert(wf177_parallel_begin(h,(const unsigned char*)"hello ",6)==0);
 float scores[5]{};int calls=0,workers=0;
 assert(wf177_parallel_prefill_top5(h,scores,&calls,&workers)==0);
 assert(calls>0&&workers==2);
 for(const std::string&prefix:{std::string(),std::string("t"),std::string("th"),std::string("the"),std::string("xyz")}){
  float a[256]{},b[256]{};
  assert(wf177_parallel_next(h,(const unsigned char*)prefix.data(),prefix.size(),a,b)==0);
  assert(std::abs(b[(int)'t']-1./256.)<1e-8);
 }
 assert(wf177_parallel_begin(h,(const unsigned char*)"other ",6)==0);
 int n=0,a=0,r=0;unsigned char out[40]{};
 assert(wf177_parallel_sample(h,177,16,out,sizeof(out),&n,&a,&r)==0&&n>0);
 wf177_parallel_close(h);
 puts("PASS two-worker independent causal byte state, exact uniform native P, bounded API and cache reset");
}