#define WF148_NO_MAIN
#include "wordfield148.cpp"
#include "wordfield148_api.h"
#include "echo_core.h"
#include "temporal_braid152.h"
#include "softbridge154_core.h"
#include "vocab_head155.h"
#include "speller156_core.h"
// Per-thread proposal lives only until the next observed word boundary.
// It is never populated using the target word or later bytes.
static thread_local bool v152_expand_strong=false;
static thread_local int v152_expand_target=-1;
struct WF148State{
 Model model;
 GeModel *geom=nullptr;
 Cloud137 *cloud=nullptr;
 vector<pair<uint16_t,float>> words;
 std::string echo_word; float echo_weight=0;
 SoftIndex* soft=nullptr;SoftMetric* metric=nullptr;float soft_strength=0.f;
 VocabHead155* neural=nullptr;float neural_share=0.f;bool protect_top1=true;
 Speller156* speller=nullptr; float open_share=0.f;
};
extern "C" void*wf148_open(const char*model,const char*geom,const char*cloud){
 WF148State*s=nullptr;
 try{s=new WF148State;s->model=restore(model);s->geom=new GeModel;s->cloud=new Cloud137;
  if(ge_load(geom,s->geom,nullptr,nullptr)||cloud_load(cloud,s->cloud,nullptr,nullptr)||
     s->model.cloudhash!=hbytes((const unsigned char*)s->cloud,sizeof(*s->cloud))||
     s->model.geomhash!=hbytes((const unsigned char*)s->geom,sizeof(*s->geom)))throw runtime_error("bad paired model");
  const char *str=getenv("V154_SOFT_STRENGTH");
  s->soft_strength=str?atof(str):0.f;
  if(!isfinite(s->soft_strength)||s->soft_strength<0.f||s->soft_strength>10.f)throw runtime_error("invalid soft strength");
  if(s->soft_strength>0.f){
   const char*index=getenv("V154_SOFT_SIDECAR");const char*metric=getenv("V154_SOFT_METRIC");
   if(!index||!*index||!metric||!*metric)throw runtime_error("v154 soft index and metric required");
   s->soft=new SoftIndex(load_soft(index,s->model,s->geom,s->cloud));
   s->metric=new SoftMetric(load_metric(metric,*s->soft));
   transform_index(*s->soft,*s->metric);
  }
  const char*share=getenv("V155_HEAD_STRENGTH");s->neural_share=share?atof(share):0.f;
  if(!isfinite(s->neural_share)||s->neural_share<0.f||s->neural_share>1.f)throw runtime_error("invalid v155 strength");
  if(s->neural_share>0){
    const char*path=getenv("V155_VOCAB_HEAD");if(!path||!*path)throw runtime_error("missing native v155 head");
    s->neural=new VocabHead155(path,s->model);
    const char*protect=getenv("V155_PROTECT_TOP1");s->protect_top1=(!protect||protect[0]!='0');
  }
  const char *open=getenv("V156_OPEN_SHARE");s->open_share=open?atof(open):0.f;
  if(!isfinite(s->open_share)||s->open_share<0.f||s->open_share>1.f)throw runtime_error("invalid v156 open share");
  if(s->open_share>0.f){
    const char *path=getenv("V156_SPELLER");if(!path||!*path)throw runtime_error("missing v156 speller");
    s->speller=new Speller156(path,s->model.trainhash);
  }
  return s;
 }catch(...){if(s){delete s->geom;delete s->cloud;delete s->soft;delete s->metric;delete s->neural;delete s->speller;delete s;}return nullptr;}
}
extern "C" void wf148_close(void*handle){auto*s=(WF148State*)handle;if(!s)return;delete s->geom;delete s->cloud;delete s->soft;delete s->metric;delete s->neural;delete s->speller;delete s;}
extern "C" int wf148_begin(void*handle,const unsigned char*past,size_t n,float cloud_strength){
 auto*s=(WF148State*)handle;v152_expand_strong=false;v152_expand_target=-1;if(!s||(!past&&n)||!isfinite(cloud_strength)||cloud_strength<0||cloud_strength>10)return -1;
 try{
  Context c{};c.p=past;c.pos=n;c.word="";
  ConceptEcho echo;
  TemporalBraid152 braid;
  size_t i=0;while(i<n){if(!letter(past[i])){i++;continue;}size_t j=i+1;while(j<n&&letter(past[j]))j++;
   if(j==n)return -2;
   auto w=lowercase(past,i,j);c.prev.push_back(w);if(c.prev.size()>6)c.prev.erase(c.prev.begin());echo.observe(w);braid.observe(w);i=j;}
  auto keys=make_keys(c,s->geom,s->cloud,true);
  auto parent=predict(s->model,keys,cloud_strength,65535);
  if(s->soft&&s->metric&&s->soft_strength>0.f){
    auto raw=causal_signed_state(c,s->geom,s->cloud);
    auto z=transformed(raw,*s->metric);
    auto vote=neighbors(*s->soft,z,16,12.f,{1.f,1.f,1.f,1.f});
    parent=fuse(parent,vote,s->model,s->soft_strength,true);
  }
  if(s->neural&&s->neural_share>0){
    auto z=causal_signed_state(c,s->geom,s->cloud);
    auto scores=s->neural->forward(z,v155_prev(c,s->model));
    parent=v155_fuse(parent,scores,s->model,s->neural_share,s->protect_top1);
  }
  s->words=parent.top;
  s->echo_word.clear();s->echo_weight=0;
  const char *env=getenv("V151_ECHO_STRENGTH");float strength=env?atof(env):0;
  if(isfinite(strength)&&strength>0&&strength<=10){
   EchoProposal proposal=echo.propose();
   if(ConceptEcho::permitted(proposal)){
    s->echo_word=proposal.word;
    float base=s->words.empty()?1.f:fmaxf(1e-6f,s->words.front().second);
    s->echo_weight=base*strength*fminf(1.f,proposal.purity);
   }
  }
  // New concept-commitment source: only earlier completed words are in
  // braid. S+/S− cloud votes already shape the parent candidates above.
  // The override is intentionally opt-in and does not overwrite v151.
  const char *braid_env=getenv("V152_BRAID_STRENGTH");
  const float braid_strength=braid_env?atof(braid_env):0.f;
  if(isfinite(braid_strength)&&braid_strength>0&&braid_strength<=10){
    Braid152Proposal candidate=braid.propose();
    if(TemporalBraid152::permitted(candidate)){
      s->echo_word=candidate.word;
      v152_expand_strong=(candidate.exact>=4 && candidate.purity>=.90 && candidate.support>=1.);
      const float base=s->words.empty()?1.f:fmaxf(1e-6f,s->words.front().second);
      s->echo_weight=base*braid_strength*fminf(1.f,(float)candidate.purity);
    }
  }
  return (int)s->words.size()+(s->echo_weight>0);
 }catch(...){return -3;}
}
extern "C" int wf148_next(void*handle,const char*prefix,size_t n,float out27[27],float*escape_mass){
 auto*s=(WF148State*)handle;
 if(!s||(!prefix&&n)||!out27||!escape_mass)return -1;
 for(int j=0;j<27;j++)out27[j]=0;
 v152_expand_target=-1;
 if(v152_expand_strong && s->echo_weight>0 && s->echo_word.size()>n &&
    s->echo_word.compare(0,n,prefix,n)==0){
    const unsigned char ch=(unsigned char)s->echo_word[n];
    if(ch>='a'&&ch<='z')v152_expand_target=ch;
 }
 *escape_mass=1.f;
 double sum=0;
 for(auto [id,score]:s->words){const string&w=s->model.words[id];if(w.size()<n||w.compare(0,n,prefix,n)!=0)continue;
   int symbol=w.size()==n?26:w[n]-'a';if(symbol<0||symbol>26)continue;
   out27[symbol]+=fmaxf(1e-9f,score);sum+=fmaxf(1e-9f,score);
 }
 if(s->echo_weight>0&&s->echo_word.size()>=n && s->echo_word.compare(0,n,prefix,n)==0){
  int symbol=s->echo_word.size()==n?26:s->echo_word[n]-'a';
  if(symbol>=0&&symbol<=26){out27[symbol]+=s->echo_weight;sum+=s->echo_weight;}
 }
 if(s->speller && s->open_share>0.f){
   if(n>64)return -2;
   std::string pre=n?std::string(prefix,n):std::string();
   for(char ch:pre)if(ch<'a'||ch>'z')return -2;
   auto chars=s->speller->probability(pre);
   // Dynamic escape-gate: when the lexical shortlist is absent at a legal
   // prefix, the open-vocabulary speller gets its full allowed share.
   // No future word/target information is inspected here.
   const float mix=sum>0.f?s->open_share:1.f;
   const float reserve=0.12f;
   for(int j=0;j<27;j++){
     float old=sum>0 ? float(out27[j]/sum) : 0.f;
     out27[j]=(1.f-reserve)*((1.f-mix)*old+mix*chars[j==26?0:j+1]);
   }
   *escape_mass=reserve;
   return 0;
 }
 if(sum<=0)return 0;*escape_mass=0.12f;
 for(int j=0;j<27;j++)out27[j]=float(out27[j]/sum)*(1-*escape_mass);
 return 0;
}
extern "C" int wf148_fuse(const float parent[256],const float word[27],float escape_mass,float strength,int protect,float out[256]){
 if(!parent||!word||!out||!isfinite(strength)||strength<0||strength>1||!isfinite(escape_mass)||escape_mass<0||escape_mass>1)return -1;
 double ps=0,ws=escape_mass;for(int i=0;i<256;i++){if(!isfinite(parent[i])||parent[i]<0)return -1;ps+=parent[i];}
 for(int i=0;i<27;i++){if(!isfinite(word[i])||word[i]<0)return -1;ws+=word[i];}
 if(ps<1e-12||fabs(ws-1)>0.003)return -1;
 // Ensure C ABI supports in-place caller output.
 float normed[256];for(int b=0;b<256;b++)normed[b]=parent[b]/ps;
 float raw[256];for(int b=0;b<256;b++){
  float q=(b>='a'&&b<='z')?word[b-'a']:0.f;
  if(b>='A'&&b<='Z')q=word[b-'A']*.08f;
  // The EOS mass is handled by the parent delimiter model, not guessed as a space.
  raw[b]=(1.f-strength*(1.f-escape_mass))*normed[b]+strength*q;
 }
 if(protect){
  int idx[256];for(int i=0;i<256;i++)idx[i]=i;
  partial_sort(idx,idx+5,idx+256,[&](int a,int b){return normed[a]!=normed[b]?normed[a]>normed[b]:a<b;});
  bool isTop[256]={};for(int i=0;i<5;i++)isTop[idx[i]]=true;
  // Top-five set preservation is guaranteed for any parent distribution.
  float boundary=1e30f;for(int i=0;i<5;i++)boundary=fminf(boundary,raw[idx[i]]);
  float maxOther=0;for(int i=0;i<256;i++)if(!isTop[i])maxOther=fmaxf(maxOther,raw[i]);
  if(maxOther>=boundary){
   // Drop the non-top-five tail just enough, then re-normalize.
   float factor=boundary/(maxOther+1e-20f)*.999f;
   for(int i=0;i<256;i++)if(!isTop[i])raw[i]*=factor;
  }
 }
 double total=0;for(int i=0;i<256;i++)total+=raw[i];if(total<=0)return -1;
 for(int i=0;i<256;i++)out[i]=raw[i]/total;
 return 0;
}
extern "C" int wf148_fuse_surgical(const float parent[256],const float word[27],float escape,float maxgap,float out[256]){
 if(!parent||!word||!out||!isfinite(maxgap)||maxgap<0||maxgap>2||!isfinite(escape)||escape<0||escape>1)return -1;
 double total=0;for(int i=0;i<256;i++){if(!isfinite(parent[i])||parent[i]<0)return -1;total+=parent[i];}
 if(total<=0)return -1;
 for(int i=0;i<256;i++)out[i]=parent[i]/total;
 int top[5];for(int i=0;i<5;i++){
  int best=-1;for(int k=0;k<256;k++){
   bool used=false;for(int j=0;j<i;j++)used|=top[j]==k;
   if(!used&&(best<0||out[k]>out[best]))best=k;
  }top[i]=best;
 }
 int current=top[0];if(!((current>='A'&&current<='Z')||(current>='a'&&current<='z')))return 0;
 float wsum=0.f,bestq=0.f,nextq=0.f;int bestletter=-1;
 for(int l=0;l<26;l++){
  float z=word[l];if(!isfinite(z)||z<0)return -1;
  wsum+=z;
  if(z>bestq){nextq=bestq;bestq=z;bestletter=l;}else if(z>nextq)nextq=z;
 }
 if(wsum<.2f||bestq<.1f||bestq<1.25f*(nextq+1e-12f))return 0;
 int target=-1;
 for(int k=0;k<5;k++){
  int b=top[k];if((b>='a'&&b<='z'&&b-'a'==bestletter)||
        (b>='A'&&b<='Z'&&b-'A'==bestletter)){
    if(target<0||out[b]>out[target])target=b;
  }
 }
 // Experimental candidate-set expansion: a 3+ word causal replay can
 // challenge the current top-five set. It is OFF by default, and the
 // production surgical controller is unchanged when the flag is absent.
 const char *experimental=getenv("V152_EXPAND");
 if(target<0 && experimental && experimental[0]=='1' &&
    v152_expand_strong && v152_expand_target>='a' &&
    v152_expand_target<='z' && current>='a' && current<='z'){
    target=v152_expand_target;
 }
 if(target<0||target==current)return 0;
 float gap=out[current]-out[target];
 bool outside=true;for(int i=0;i<5;i++)outside&=(target!=top[i]);
 bool strong_expand=outside && experimental && experimental[0]=='1' && v152_expand_strong;
 if(gap<0||(!strong_expand&&gap>maxgap*out[current]))return 0;
 float delta=.5f*gap+1e-6f;
 if(out[current]<=delta)return 0;
 out[current]-=delta;out[target]+=delta;
 return 0;
}
