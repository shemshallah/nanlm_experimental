// Native FP32 v157 signed-cloud conditioned word and character head.
// Exact PyTorch forward mapping (inference mode). Never reads a future character.
#pragma once
#include <cmath>
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include <limits>
#include <cstring>
struct SignedJoint157Native {
 static constexpr int V=5296,H=96;
 struct Linear{int in,out;std::vector<float>w,b;
  Linear(int i,int o):in(i),out(o){}
  std::vector<float>operator()(const std::vector<float>&x)const{
   if(int(x.size())!=in)throw std::runtime_error("v157 linear wrong dimension");
   std::vector<float>z(out,0);for(int j=0;j<out;j++){
    float v=b.empty()?0:b[j];const float*p=w.data()+size_t(j)*in;
    for(int k=0;k<in;k++)v+=p[k]*x[k];z[j]=v;
   }return z;
  }
 };
 std::vector<float>prev,view1w[4],view1b[4],view2w[4],view2b[4];
 std::vector<float>braid1w,braid1b,braid2w,braid2b,gatew,gateb,
      ln1w,ln1b,c1w,c1b,c2w,c2b,ln2w,ln2b,
      outw,outb,charemb,pos,ce1w,ce1b,ce2w,ce2b,cgw,cgb;
 static void read(std::ifstream&f,std::vector<float>&v,size_t n){
  v.resize(n);f.read((char*)v.data(),n*4);if(!f)throw std::runtime_error("v157 checkpoint truncated");
  for(float q:v)if(!std::isfinite(q))throw std::runtime_error("v157 checkpoint nonfinite");
 }
 static uint32_t u32(std::ifstream&f){uint32_t v;f.read((char*)&v,4);if(!f)throw std::runtime_error("v157 truncated uint32");return v;}
 static uint64_t u64(std::ifstream&f){uint64_t v;f.read((char*)&v,8);if(!f)throw std::runtime_error("v157 truncated uint64");return v;}
 SignedJoint157Native(const std::string&path,uint64_t trainhash,uint64_t geomhash,uint64_t cloudhash){
  std::ifstream f(path,std::ios::binary);char tag[8]={};f.read(tag,8);
  if(!f||std::memcmp(tag,"N157JOI1",8)||u32(f)!=157||u32(f)!=V||u32(f)!=41)
    throw std::runtime_error("v157 header mismatch");
  if(u64(f)!=trainhash||u64(f)!=geomhash||u64(f)!=cloudhash)
    throw std::runtime_error("v157 source/cloud fingerprint mismatch");
  read(f,prev,5297*32);
  for(int i=0;i<4;i++){read(f,view1w[i],32*8);read(f,view1b[i],32);read(f,view2w[i],32*32);read(f,view2b[i],32);}
  read(f,braid1w,48*24);read(f,braid1b,48);read(f,braid2w,32*48);read(f,braid2b,32);
  read(f,gatew,4*160);read(f,gateb,4);read(f,ln1w,320);read(f,ln1b,320);
  read(f,c1w,192*320);read(f,c1b,192);read(f,c2w,96*192);read(f,c2b,96);
  read(f,ln2w,96);read(f,ln2b,96);read(f,outw,V*H);read(f,outb,V);
  read(f,charemb,28*40);read(f,pos,26*16);read(f,ce1w,128*192);read(f,ce1b,128);
  read(f,ce2w,27*128);read(f,ce2b,27);read(f,cgw,27*H);read(f,cgb,27);
  if(f.peek()!=EOF)throw std::runtime_error("v157 unexpected trailing bytes");
 }
 static std::vector<float> dense(const std::vector<float>&x,const std::vector<float>&w,const std::vector<float>&b,int rows){
  const int cols=x.size();std::vector<float>out(rows);
  for(int j=0;j<rows;j++){float z=b[j];const float*p=w.data()+size_t(j)*cols;for(int i=0;i<cols;i++)z+=p[i]*x[i];out[j]=z;}return out;
 }
 static void norm(std::vector<float>&x,const std::vector<float>&g,const std::vector<float>&b){
  double s=0,var=0;for(float a:x)s+=a;s/=x.size();for(float a:x)var+=(a-s)*(a-s);var/=x.size();
  float r=1.f/std::sqrt(float(var)+1e-5f);for(size_t i=0;i<x.size();i++)x[i]=(x[i]-float(s))*r*g[i]+b[i];
 }
 static void gelu(std::vector<float>&x){for(float&q:x)q=.5f*q*(1.f+std::erf(q*0.7071067811865475f));}
 static void softmax(std::vector<float>&x){float mx=*std::max_element(x.begin(),x.end());double s=0;for(float&q:x){q=std::exp(q-mx);s+=q;}for(float&q:x)q/=float(s);}
 static void cat(std::vector<float>&x,const std::vector<float>&other){x.insert(x.end(),other.begin(),other.end());}
 std::vector<float>context(const std::array<float,32>&z,const std::array<int,4>&pr)const{
  std::vector<float>full;full.reserve(128);std::vector<float>views[4];
  for(int i=0;i<4;i++){
   std::vector<float>x(z.begin()+8*i,z.begin()+8*i+8);
   x=dense(x,view1w[i],view1b[i],32);for(float&v:x)v=std::tanh(v);
   views[i]=dense(x,view2w[i],view2b[i],32);cat(full,views[i]);
  }
  std::vector<float>q;q.reserve(24);
  for(int i=0;i<8;i++)q.push_back(z[i]*z[16+i]);
  for(int i=0;i<8;i++)q.push_back(z[8+i]*z[24+i]);
  for(int i=0;i<8;i++)q.push_back(z[16+i]*z[24+i]);
  q=dense(q,braid1w,braid1b,48);gelu(q);q=dense(q,braid2w,braid2b,32);
  std::vector<float>gg(z.begin(),z.end());cat(gg,full);gg=dense(gg,gatew,gateb,4);softmax(gg);
  std::vector<float>input(z.begin(),z.end());for(int id:pr){
   if(id<0||id>V)throw std::runtime_error("v157 bad previous word id");
   input.insert(input.end(),prev.begin()+size_t(id)*32,prev.begin()+size_t(id+1)*32);
  }cat(input,q);
  for(int i=0;i<4;i++)for(float v:views[i])input.push_back(v*gg[i]);
  norm(input,ln1w,ln1b);input=dense(input,c1w,c1b,192);gelu(input);
  input=dense(input,c2w,c2b,H);norm(input,ln2w,ln2b);gelu(input);
  return input;
 }
 std::vector<float>word_logits(const std::vector<float>&h)const{return dense(h,outw,outb,V);}
 std::array<float,27> char_probs(const std::vector<float>&h,const std::string&prefix)const{
  if(prefix.size()>24)throw std::runtime_error("v157 prefix too long");
  std::vector<float>in=h;int n=prefix.size();int a=n?prefix[n-1]-'a':26;
  int b=n>1?prefix[n-2]-'a':26;
  if(a<0||a>26||b<0||b>26)throw std::runtime_error("v157 bad prefix");
  in.insert(in.end(),charemb.begin()+a*40,charemb.begin()+(a+1)*40);
  in.insert(in.end(),charemb.begin()+b*40,charemb.begin()+(b+1)*40);
  in.insert(in.end(),pos.begin()+n*16,pos.begin()+(n+1)*16);
  in=dense(in,ce1w,ce1b,128);gelu(in);in=dense(in,ce2w,ce2b,27);
  auto gate=dense(h,cgw,cgb,27);for(int i=0;i<27;i++)in[i]+=gate[i];softmax(in);
  std::array<float,27>p{};copy(in.begin(),in.end(),p.begin());return p;
 }
};
