// Source-only self-contained native v174 ABI / fail-closed / normalized probabilities.
#include "native_temporal175.cpp"
#include <cassert>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <map>
#include <vector>
#include <cstdint>
#include <fstream>
#include <unistd.h>
using std::string;
static void u32(std::ofstream&o,uint32_t x){o.write((char*)&x,4);}
static void u64(std::ofstream&o,uint64_t x){o.write((char*)&x,8);}
static string encode(int x){string a(4,'a');for(int j=3;j>=0;j--){a[j]=char('a'+x%26);x/=26;}return a;}
static void tensor(std::ofstream&o,const string&name,std::vector<uint32_t>shape,float constant=0){
 uint8_t len=(uint8_t)name.size();o.write((char*)&len,1);o.write(name.data(),name.size());uint8_t ndim=(uint8_t)shape.size();o.write((char*)&ndim,1);
 size_t n=1;for(uint32_t v:shape){u32(o,v);n*=v;}
 std::vector<float>floats(n,constant);if(name=="bias")floats[0]=2.f;
 o.write((char*)floats.data(),floats.size()*4);
}
int main(){
 auto path=std::string("/tmp/v175_synthetic_now_")+std::to_string((long long)getpid())+".bin";
 {
  std::ofstream o(path,std::ios::binary);assert(o);
  o.write("N175NOW1",8);for(int i=0;i<3;i++)u64(o,(uint64_t)(42+i));
  for(uint32_t x:{12000u,48u,32u,64u})u32(o,x);
  for(int i=0;i<12000;i++){auto s=encode(i);uint8_t n=4;o.write((char*)&n,1);o.write(s.data(),4);}
  u32(o,28);
  tensor(o,"embed",{12000,32});tensor(o,"out",{12000,32});tensor(o,"bias",{12000});
  tensor(o,"geo",{12000,8});tensor(o,"projection",{32,8});tensor(o,"gain",{},1);
  tensor(o,"gpos",{12000,8});tensor(o,"gneg",{12000,8});
  for(string rail:{"gru_pos","gru_neg"}){
   tensor(o,rail+".weight_ih_l0",{192,48});tensor(o,rail+".weight_hh_l0",{192,64});
   tensor(o,rail+".bias_ih_l0",{192});tensor(o,rail+".bias_hh_l0",{192});
  }
  tensor(o,"signed_norm.weight",{192},1);tensor(o,"signed_norm.bias",{192});
  tensor(o,"to_lex.0.weight",{96,192});tensor(o,"to_lex.0.bias",{96});
  tensor(o,"to_lex.2.weight",{32,96});tensor(o,"to_lex.2.bias",{32});
  tensor(o,"trust",{},-3);tensor(o,"signed_scale",{});
  tensor(o,"copy_log_strength",{},-3);tensor(o,"copy_bigram",{});tensor(o,"copy_trigram",{});tensor(o,"copy_log_tau",{},3);
 }
 auto handle=wf175_now_open(path.c_str());assert(handle);
 assert(wf175_now_word_count(handle)==12000);
 std::vector<float>prob(12000);
 auto text=string("A perfect native causal sequence");
 assert(wf175_now_predict(handle,(const unsigned char*)text.data(),text.size(),prob.data(),prob.size())==0);
 assert(wf175_now_predict(handle,nullptr,4,prob.data(),prob.size())==-1);
 assert(wf175_now_predict(handle,nullptr,0,prob.data(),100)==-1);
 double sum=0;for(float p:prob){assert(std::isfinite(p)&&p>0);sum+=p;}
 assert(std::abs(sum-1)<1e-5);assert(std::abs(prob[0]-std::exp(2.)/(11999+std::exp(2.)))<1e-7);
 char name[32]={};assert(wf175_now_word_name(handle,11999,name,sizeof name)==0);
 assert(string(name)==encode(11999));
 // The causal pointer reads an earlier occurrence of the last word, then
 // boosts ONLY the already observed continuation (no future target).
 auto repeat=encode(10)+" "+encode(11)+" "+encode(10);
 assert(wf175_now_predict(handle,(const unsigned char*)repeat.data(),repeat.size(),prob.data(),prob.size())==0);
 assert(prob[11]>prob[12]);
 wf175_now_close(handle);
 assert(!wf175_now_open("/nonexistent/native174"));
 std::ofstream damaged(path,std::ios::binary|std::ios::trunc);damaged.write("badmagic",8);damaged.close();
 assert(!wf175_now_open(path.c_str()));std::remove(path.c_str());
 std::cout<<"PASS native v175 source-only 12000-normalization/ABI/parse/fail-closed"<<std::endl;
 return 0;
}