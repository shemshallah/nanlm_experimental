// NaNCM v173: opt-in, causal *normalized* byte-to-word-start product of experts.
// The native v172 ABI, v170 byte model and inherited checkpoints are protected.
// This source is deliberately a thin layer over v172, NOT a reconstructed model.
#include "../v172/wordfield172_api.cpp"
#include <array>
#include <cmath>
#include <algorithm>
#include <memory>
#include <limits>
namespace {
struct Joint173 {
 void *words=nullptr,*bytes=nullptr;
 double gamma=0.;
 std::array<float,27> root{};
 bool ready=false;
};
}
extern "C" void*wf173_joint_open(const char*m,const char*g,const char*c,const char*bytepack,double gamma){
 if(!std::isfinite(gamma)||gamma<0.||gamma>1.)return nullptr;
 std::unique_ptr<Joint173>s(new Joint173);
 s->gamma=gamma;
 s->words=wf148_open(m,g,c);
 if(!s->words)return nullptr;
 // Root-gate disabled is *identical* to v172 and does not need a byte pack.
 if(gamma>0.){
  s->bytes=wf170_byte_open(m,g,c,bytepack);
  if(!s->bytes||wf170_byte_set_adaptive(s->bytes,0)!=0){
   if(s->bytes)wf170_byte_close(s->bytes);
   wf148_close(s->words);return nullptr;
  }
 }
 return s.release();
}
extern "C" void wf173_joint_close(void*h){
 auto*s=(Joint173*)h;if(!s)return;
 if(s->bytes)wf170_byte_close(s->bytes);
 if(s->words)wf148_close(s->words);
 delete s;
}
extern "C" double wf173_joint_gamma(void*h){auto*s=(Joint173*)h;return s?s->gamma:-1.;}
extern "C" int wf173_joint_begin(void*h,const unsigned char*past,size_t n,float cloud){
 auto*s=(Joint173*)h;if(!s||(!past&&n))return -1;
 s->ready=false;
 int rc=wf148_begin(s->words,past,n,cloud);
 if(rc<0)return rc;
 if(s->gamma==0.){s->ready=true;return rc;}
 float out[27]={},escape=0.f,byte[256]={};
 if(wf148_next(s->words,"",0,out,&escape)!=0||
    wf170_byte_next(s->bytes,past,n,byte)!=0)return -2;
 std::array<double,26>root{},alpha{},fused{};
 double lexical_alpha=0.,byte_alpha=0.;
 for(int j=0;j<26;j++){
  root[j]=(double)out[j]+(double)escape/27.;
  alpha[j]=(double)byte['a'+j]+(double)byte['A'+j];
  if(!(root[j]>0.)||!(alpha[j]>0.)||!std::isfinite(root[j])||!std::isfinite(alpha[j]))return -3;
  lexical_alpha+=root[j];byte_alpha+=alpha[j];
 }
 const double eos=(double)out[26]+(double)escape/27.;
 if(!(lexical_alpha>0.)||!(byte_alpha>0.)||!std::isfinite(eos)||eos<0.)return -4;
 // Preserve parent probability of an empty/EOS word, while geometrically
 // fusing the categorical first-*letter* distributions. Exact simplex.
 double sum=0.;
 for(int j=0;j<26;j++){
  fused[j]=std::exp((1.-s->gamma)*std::log(root[j]/lexical_alpha)+
                    s->gamma*std::log(alpha[j]/byte_alpha));
  sum+=fused[j];
 }
 if(!(sum>0.)||!std::isfinite(sum))return -5;
 double renorm=0.;
 for(int j=0;j<26;j++){
  s->root[j]=(float)(lexical_alpha*fused[j]/sum);renorm+=s->root[j];
 }
 s->root[26]=(float)eos;renorm+=s->root[26];
 if(!(renorm>0.)||!std::isfinite(renorm)||std::abs(renorm-1.)>1e-4)return -6;
 for(auto&v:s->root)v=(float)(v/renorm);
 s->ready=true;return rc;
}
extern "C" int wf173_joint_next(void*h,const char*prefix,size_t n,float out[27],float*escape){
 auto*s=(Joint173*)h;
 if(!s||!s->ready||(!prefix&&n)||!out||!escape)return -1;
 if(s->gamma==0.||n>0)return wf148_next(s->words,prefix,n,out,escape);
 for(int j=0;j<27;j++)out[j]=s->root[j];
 *escape=0.f;return 0;
}