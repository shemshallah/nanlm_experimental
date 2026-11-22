// Standalone CPU smoke for the v170 online trust module in source-only GitHub.
// Stub experts do NOT reproduce the inherited trained NaNCM; those are checked
// by test_native170.py against the full recovery archive.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <cstdio>
#include <string>
struct BytePack169{};
struct ByteState169{BytePack169*pack=nullptr;};
extern "C" void*wf169_byte_open(const char*,const char*,const char*,const char*){
 auto*s=new ByteState169; s->pack=new BytePack169;return s;
}
extern "C" void wf169_byte_close(void*p){
 auto*s=(ByteState169*)p;if(s){delete s->pack;delete s;}
}
static void ngram169(const BytePack169&,const unsigned char*,size_t,
                     std::array<double,256>&out){
 out.fill(1./512.);out[(int)'e']+=.5;
}
static bool wordbridge169(ByteState169*,const unsigned char*p,size_t n,
                          std::array<double,256>&out,bool&inside){
 inside=n && ((p[n-1]>='a'&&p[n-1]<='z') || (p[n-1]>='A'&&p[n-1]<='Z'));
 out.fill(1./512.);out[(int)'t']+=.5;
 return true;
}
#include "native_online170.cpp"
static std::array<float,256> pred(void*h,const std::string&history){
 std::array<float,256>p{};
 assert(wf170_byte_next(h,(const unsigned char*)history.data(),history.size(),p.data())==0);
 double total=0.;for(float v:p){assert(v>0&&std::isfinite(v));total+=v;}
 assert(std::abs(total-1.)<1e-5);return p;
}
int main(){
 void*h=wf170_byte_open(nullptr,nullptr,nullptr,nullptr);assert(h);
 const std::string history="The village was ";
 auto a=pred(h,history),b=pred(h,history);
 assert(a==b&&std::abs(wf170_byte_last_eta(h)-.1)<1e-14);
 // The previous outcome 't' strongly favors the word bridge, but its
 // first update belongs to the outside-word category, not inside-word.
 auto c=pred(h,history+"t");
 assert(std::abs(wf170_byte_last_eta(h)-.5)<1e-14 && c['t']>0);
 pred(h,history+"tt");
 assert(wf170_byte_last_eta(h)>.5);
 // An unrelated sequence must reset the online evidence.
 pred(h,"Different story ");assert(std::abs(wf170_byte_last_eta(h)-.1)<1e-14);
 assert(wf170_byte_reset(h)==0);
 assert(wf170_byte_set_adaptive(h,0)==0);
 auto d=pred(h,history+"abc");assert(wf170_byte_last_eta(h)==.5);
 assert(d['t']>0&&d['e']>0);
 assert(wf170_byte_set_adaptive(h,1)==0);
 assert(wf170_byte_set_adaptive(h,-2)<0);
 wf170_byte_close(h);
 std::puts("PASS v170 synthetic source-only causal trust, norm, reset, ABI smoke");
}
