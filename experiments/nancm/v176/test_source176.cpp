// Source-only smoke uses synthetic *independently normalized* word and byte
// experts. Trained numerical validation requires full cumulative ZIP.
#include <cassert>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
extern "C" void*wf175_now_open(const char*){return new int(1);}
extern "C" void wf175_now_close(void*p){delete (int*)p;}
extern "C" int wf175_now_predict(void*h,const unsigned char*,size_t,float*out,size_t cap){
 if(!h||cap<12000)return -1;
 double s=0.;for(int i=0;i<12000;i++){out[i]=i<5?float(40-i*3):.01f;s+=out[i];}
 for(int i=0;i<12000;i++)out[i]=float(out[i]/s);
 return 0;
}
extern "C" int wf175_now_word_name(void*,int i,char*b,size_t c){
 const char*w[5]={"the","their","there","then","they"};
 std::string s=i<5?w[i]:"other";if(c<=s.size())return -1;
 memcpy(b,s.c_str(),s.size()+1);return 0;
}
extern "C" void*wf170_byte_open(const char*,const char*,const char*,const char*){return new int(2);}
extern "C" void wf170_byte_close(void*p){delete (int*)p;}
extern "C" int wf170_byte_reset(void*){return 0;}
extern "C" int wf170_byte_set_adaptive(void*,int e){return e==0?0:-1;}
extern "C" int wf170_byte_next(void*,const unsigned char*,size_t,float out[256]){
 for(int j=0;j<256;j++)out[j]=1.f/256.f;
 return 0;
}
extern "C" int wf176_cross_predict(void*,const unsigned char*,size_t,float*,size_t){return -1;}
extern "C" void wf176_cross_close(void*){}
#include "native_third176.cpp"
int main(){
 assert(!wf176_hybrid_open("x",nullptr,nullptr,nullptr,nullptr,0.f));
 void *h=wf176_hybrid_open("x",nullptr,nullptr,nullptr,nullptr,.15f);assert(h);
 float p[256]{},q[256]{};int idx=-1;char word[48]{};float prob=0.;
 assert(wf176_hybrid_begin(h,(const unsigned char*)"hello ",6)==0);
 assert(wf176_hybrid_top5(h,0,&idx,word,sizeof(word),&prob)==0);
 assert(idx==0&&std::string(word)=="the"&&prob>0);
 assert(wf176_hybrid_next(h,nullptr,0,q,p)==0);
 double a=0,b=0;for(int j=0;j<256;j++){a+=q[j];b+=p[j];}assert(std::abs(a-1)<1e-6&&std::abs(b-1)<1e-6);
 assert(q[(int)'t']>p[(int)'t'] && q[(int)'z']<p[(int)'z']);
 assert(wf176_hybrid_next(h,(const unsigned char*)"xyz",3,q,p)==0);
 for(int j=0;j<256;j++)assert(q[j]==p[j]);
 assert(wf176_hybrid_next(h,(const unsigned char*)"the",3,q,p)==0);
 assert(q[(int)' ']>p[(int)' ']);
 float alpha=0.,residual[256]{};
 assert(wf176_spec_accept(p,q,' ',&alpha,residual)==0&&alpha>=0&&alpha<=1);
 double z=0;for(float x:residual)z+=x;assert(std::abs(z-1.)<1e-6);
 assert(wf176_spec_accept(p,q,256,&alpha,residual)<0);
 assert(wf176_hybrid_next(h,(const unsigned char*)"overlong",100,q,p)<0);
 float logscore[5]{};int calls=-1;
 assert(wf176_hybrid_verify_top5(h,logscore,&calls)==0 && calls>0 && calls<40);
 assert(std::isfinite(logscore[0]) && logscore[0]<0);
 unsigned char bytes[40]={};int n=0,accepted_count=0,rejected=0;
 assert(wf176_hybrid_sample(h,176,12,bytes,40,&n,&accepted_count,&rejected)==0);
 assert(n>0 && n<=12 && accepted_count<=n && (rejected==0||rejected==1));
 // Exact root target mass under the independent uniform verifier: even if
 // draft Q strongly prefers 't', after rejection sampling the output law is
 // uniform. Use many independent seeds; no future labels are consulted.
 int sampled_t=0;
 for(uint64_t seed=1;seed<=20000;seed++){
  assert(wf176_hybrid_sample(h,seed,1,bytes,40,&n,&accepted_count,&rejected)==0 && n==1);
  sampled_t+=bytes[0]=='t';
 }
 assert(std::abs(sampled_t/20000.-1./256.)<.003);

 wf176_hybrid_close(h);
 puts("PASS native v176 5-way top words, full 256-way normalized byte escape, off-trie rollback, exact residual, bounds");
}