// NaNCM v170 -- prequential, per-book causal expert trust.
// The 128+128 S+/S- cloud, trained lexical head, ANN memory, native 256-way
// experts and optimizer state are inherited unchanged from v169.
// This file is included AFTER native_byte169.cpp in wordfield170_api.cpp.
#include <cmath>
#include <algorithm>
#include <array>
#include <string>
#include <memory>

namespace {
struct OnlineState170 {
 ByteState169 *base=nullptr;
 std::array<double,256> previous_ngram{},previous_word{};
 std::string previous_past;
 double log_evidence[2]{0.,0.};
 double last_eta=0.;
 bool pending=false, prior_inside=false, adaptive=true;
};
static void reset170(OnlineState170*s){
 s->pending=false;s->previous_past.clear();s->log_evidence[0]=s->log_evidence[1]=0.;
 s->last_eta=0.;
}
static double trust170(bool inside,double evidence,bool adaptive){
 const double prior=inside?.5:.1;
 if(!adaptive)return prior;
 const double logit=std::log(prior/(1.-prior))+.2*evidence;
 const double bound=std::max(-80.,std::min(80.,logit));
 return std::max(.01,std::min(.95,1./(1.+std::exp(-bound))));
}
}

extern "C" void*wf170_byte_open(const char*m,const char*g,const char*c,const char*bytepack){
 std::unique_ptr<OnlineState170>s(new OnlineState170);
 s->base=(ByteState169*)wf169_byte_open(m,g,c,bytepack);
 return s->base?s.release():nullptr;
}
extern "C" void wf170_byte_close(void*h){
 auto*s=(OnlineState170*)h;
 if(!s)return;
 wf169_byte_close(s->base);delete s;
}
extern "C" int wf170_byte_reset(void*h){
 auto*s=(OnlineState170*)h;if(!s)return -1;
 reset170(s);return 0;
}
extern "C" int wf170_byte_set_adaptive(void*h,int enabled){
 auto*s=(OnlineState170*)h;
 if(!s||!(enabled==0||enabled==1))return -1;
 s->adaptive=enabled!=0;reset170(s);return 0;
}
extern "C" double wf170_byte_last_eta(void*h){
 auto*s=(OnlineState170*)h;return s?s->last_eta:-1.;
}
extern "C" int wf170_byte_next(void*h,const unsigned char*past,size_t n,float out[256]){
 auto*s=(OnlineState170*)h;
 if(!s||!s->base||(!past&&n)||!out)return -1;
 static const unsigned char empty[1]={0};if(!past)past=empty;
 // Prior prediction is settled ONLY if a single new byte is now observed.
 // Repeated identical queries are idempotent; discontinuous/random queries
 // reset trust rather than leaking information from an unrelated sequence.
 if(s->pending){
  bool step=(n==s->previous_past.size()+1) &&
            !std::memcmp(past,s->previous_past.data(),s->previous_past.size());
  bool same=(n==s->previous_past.size()) &&
            !std::memcmp(past,s->previous_past.data(),n);
  if(step){
   const int category=(int)s->prior_inside;
   const unsigned char observed=past[n-1];
   const double pn=s->previous_ngram[observed],pw=s->previous_word[observed];
   if(!(pn>0&&pw>0&&std::isfinite(pn)&&std::isfinite(pw))){reset170(s);return -2;}
   // Memory decay occurs per observation in the same regime, not per byte
   // in the other regime. This is exactly the archived reference replay.
   s->log_evidence[category]=.9*s->log_evidence[category]+std::log(pw/pn);
  } else if(!same){reset170(s);}
 }
 std::array<double,256>gram{},word{};
 ngram169(*s->base->pack,past,n,gram);
 bool inside=false;
 if(!wordbridge169(s->base,past,n,word,inside))return -3;
 const double eta=trust170(inside,s->log_evidence[(int)inside],s->adaptive);
 double total=0.;
 for(int j=0;j<256;j++){
  const double q=(1.-eta)*gram[j]+eta*word[j];
  if(!(q>0.)||!std::isfinite(q))return -4;
  out[j]=(float)q;total+=out[j];
 }
 if(!(total>0.)||!std::isfinite(total)||std::abs(total-1.)>.0005)return -5;
 for(int j=0;j<256;j++)out[j]=(float)(out[j]/total);
 s->previous_ngram=gram;s->previous_word=word;
 s->previous_past.assign((const char*)past,n);
 s->previous_past.reserve(n+2);
 s->prior_inside=inside;s->pending=true;s->last_eta=eta;
 return 0;
}