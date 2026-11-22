// NaNCM v172 experimental signed multi-rail retrieval, single-rail exact rollback.
// Default 32 nearest addresses / 64 word candidates has exact v169 rollback.
#include <cerrno>
#include <cstdlib>
// NaNCM v168 — TRAIN-only signed-cloud analogical neighbor birth (native)
// Exact v167 ABI rollback when V168_ANN_STRENGTH=0; no inherited weight edits.
#include "../v169/wordfield168_embed_ann169.cpp"
#include <queue>
#include <unordered_map>
#include <fstream>
#include <utility>
#include <cstring>
#include <cmath>
#include <memory>
#include <array>
#include <algorithm>
#include <string>
#include <limits>
namespace {
struct Neighbor168 {
 std::array<float,32> lex{};
 std::array<float,16> sign{};
 uint8_t k=0,n=0;
 std::array<uint32_t,8> target{};
 std::array<float,8> p{};
};
struct Ann168 {
 std::vector<std::string> words;
 std::array<std::array<float,16>,128> signed_hash{};
 std::vector<Neighbor168> addresses;
};
struct Teacher172 {
 std::vector<std::array<float,32>> ctx,out;
 std::vector<std::array<float,8>> geo;
 std::vector<float> bias;
 std::array<std::array<float,8>,32> project{};
 float gain=0.;
};
static bool teacher_load172(const char*path,Teacher172&teacher,const State166*base){
 if(!path||!base||!base->parent||!base->parent->parent||!base->birth)return false;
 const auto*m=(const State159*)base->parent->parent->base;
 const auto*b=m?(const State158*)m->base:nullptr;
 const auto*r=b?(const WF148State*)b->base:nullptr;
 if(!r)return false;
 std::ifstream f(path,std::ios::binary);if(!f)return false;
 char magic[8];uint64_t fp[3];uint32_t shape[3];
 if(!f.read(magic,8)||std::memcmp(magic,"N172TCH1",8)||
    !f.read((char*)fp,sizeof(fp))||!f.read((char*)shape,sizeof(shape))||
    !f.read((char*)&teacher.gain,4))return false;
 if(fp[0]!=r->model.trainhash||fp[1]!=r->model.geomhash||fp[2]!=r->model.cloudhash||
    shape[0]!=base->birth->lex.size()||shape[0]!=12000||shape[1]!=32||shape[2]!=8||
    !std::isfinite(teacher.gain)||teacher.gain<0||teacher.gain>25)return false;
 if(!f.read((char*)teacher.project.data(),sizeof(teacher.project)))return false;
 teacher.ctx.resize(shape[0]);teacher.out.resize(shape[0]);teacher.geo.resize(shape[0]);teacher.bias.resize(shape[0]);
 for(size_t i=0;i<shape[0];i++){
  uint8_t n=0;if(!f.read((char*)&n,1)||n<1||n>20)return false;
  char word[21]={};if(!f.read(word,n))return false;
  if(std::string(word,n)!=base->birth->lex[i].word)return false;
  if(!f.read((char*)teacher.ctx[i].data(),32*sizeof(float))||
     !f.read((char*)teacher.out[i].data(),32*sizeof(float))||
     !f.read((char*)&teacher.bias[i],4)||
     !f.read((char*)teacher.geo[i].data(),8*sizeof(float)))return false;
 }
 char tail;if(f.read(&tail,1)||!f.eof())return false;
 for(const auto&row:teacher.project)for(float v:row)if(!std::isfinite(v)||std::abs(v)>100)return false;
 for(size_t i=0;i<shape[0];i++){
  if(!std::isfinite(teacher.bias[i])||std::abs(teacher.bias[i])>100)return false;
  for(float v:teacher.ctx[i])if(!std::isfinite(v)||std::abs(v)>100)return false;
  for(float v:teacher.out[i])if(!std::isfinite(v)||std::abs(v)>100)return false;
  for(float v:teacher.geo[i])if(!std::isfinite(v)||std::abs(v)>100)return false;
 }
 return true;
}
struct State168 {
 State166 *parent=nullptr;
 std::unique_ptr<Ann168> memory;
 double strength=0.,signed_share=.66,domain_min=1.4,alpha=0.,neural_rerank=0.;
 int candidate_limit=64,neighbor_limit=32,rail_count=1;
 double teacher_strength=0.;
 std::unique_ptr<Teacher172>teacher;
 std::vector<double>teacher_probs,teacher_trie_mass;
 std::unordered_map<std::string,double>teacher_parent_prefix;
 std::vector<std::pair<std::string,double>> candidates;
 std::unordered_map<std::string,double> prefix_cache;
};
static bool teacher_begin172(State168*s,const unsigned char*past,size_t n){
 if(!s||!s->teacher||!s->parent||!s->parent->birth)return false;
 const auto&g=*s->parent->birth;const auto&model=*s->teacher;
 const auto prev=complete_words158(past,n);const int k=(int)std::min((size_t)3,prev.size());
 double weights[]={.2,.3,.5};double denom=0.;
 std::array<double,32>q{};std::array<double,8>z{};
 for(int i=0;i<k;i++){
  const auto&w=prev[prev.size()-k+i];
  const auto it=g.lookup.find(w);if(it==g.lookup.end())continue;
  const size_t idx=it->second;const double weight=weights[3-k+i];denom+=weight;
  for(int d=0;d<32;d++)q[d]+=weight*model.ctx[idx][d];
  for(int d=0;d<8;d++)z[d]+=weight*model.geo[idx][d];
 }
 if(denom>0){for(auto&v:q)v/=denom;for(auto&v:z)v/=denom;}
 for(int d=0;d<32;d++)for(int j=0;j<8;j++)q[d]+=model.project[d][j]*z[j];
 double norm=0.;for(auto x:q)norm+=x*x;
 norm=std::max(1e-8,std::sqrt(norm));for(auto&v:q)v/=norm;
 const size_t count=g.lex.size();
 s->teacher_probs.assign(count,0.);
 double maxlogit=-1e100;
 for(size_t i=0;i<count;i++){
  double dot=0.;for(int d=0;d<32;d++)dot+=q[d]*model.out[i][d];
  const double logit=model.gain*dot+model.bias[i];
  if(!std::isfinite(logit))return false;
  s->teacher_probs[i]=logit;maxlogit=std::max(maxlogit,logit);
 }
 double mass=0.;for(auto&v:s->teacher_probs){v=std::exp(v-maxlogit);mass+=v;}
 if(!(mass>0)||!std::isfinite(mass))return false;
 for(auto&v:s->teacher_probs)v/=mass;
 const auto&trie=g.trie;s->teacher_trie_mass.assign(trie.size(),0.);
 for(size_t j=trie.size();j-->0;){
  const auto&node=trie[j];double m=0.;
  if(node.terminal>=0)m+=s->teacher_probs[node.terminal];
  for(int child:node.edge)if(child>=0)m+=s->teacher_trie_mass[child];
  s->teacher_trie_mass[j]=m;
 }
 s->teacher_parent_prefix.clear();s->teacher_parent_prefix[""]=1.;
 return std::abs(s->teacher_trie_mass[0]-1.)<1e-5;
}
static bool get_number168(const char*name,double&val,double min,double max){
 const char*env=getenv(name);if(!env)return true;
 char*e=nullptr;val=strtod(env,&e);
 return e!=env&&!*e&&std::isfinite(val)&&val>=min&&val<=max;
}
static bool get_integer171(const char*name,int&out,int lo,int hi){
 const char *env=getenv(name);if(!env)return true;
 char*end=nullptr;errno=0;long v=strtol(env,&end,10);
 if(env==end||*end||errno||v<lo||v>hi)return false;
 out=(int)v;return true;
}
static bool load_index168(const char*path,Ann168&ann,const State166*base){
 if(!base||!base->parent||!base->parent->parent)return false;
 const auto*a=(const State159*)base->parent->parent->base;
 const auto*b=a?(const State158*)a->base:nullptr;
 const auto*r=b?(const WF148State*)b->base:nullptr;
 if(!r)return false;
 std::ifstream f(path,std::ios::binary);if(!f)return false;
 char magic[8];uint64_t fp[3];uint32_t sizes[4];
 if(!f.read(magic,8)||std::memcmp(magic,"N168ANN1",8)||
    !f.read((char*)fp,sizeof(fp))||!f.read((char*)sizes,sizeof(sizes)))return false;
 if(fp[0]!=r->model.trainhash||fp[1]!=r->model.geomhash||fp[2]!=r->model.cloudhash)return false;
 if(sizes[0]<100||sizes[0]>300000||sizes[1]<100||sizes[1]>100000||sizes[2]!=32||sizes[3]!=16)return false;
 ann.words.reserve(sizes[1]);
 for(uint32_t i=0;i<sizes[1];i++){
  uint8_t n=0;if(!f.read((char*)&n,1)||n<1||n>20)return false;
  std::string w(n,' ');if(!f.read(&w[0],n))return false;
  for(char c:w)if(c<'a'||c>'z')return false;
  ann.words.emplace_back(std::move(w));
 }
 if(!f.read((char*)&ann.signed_hash,sizeof(ann.signed_hash)))return false;
 ann.addresses.resize(sizes[0]);
 for(auto&v:ann.addresses){
  if(!f.read((char*)v.lex.data(),32*sizeof(float))||!f.read((char*)v.sign.data(),16*sizeof(float))||
     !f.read((char*)&v.k,1)||!f.read((char*)&v.n,1))return false;
  if(v.k<1||v.k>3||v.n<1||v.n>8)return false;
  for(int i=0;i<8;i++){
   if(!f.read((char*)&v.target[i],4)||!f.read((char*)&v.p[i],4))return false;
   if(v.target[i]>=sizes[1]||!std::isfinite(v.p[i])||v.p[i]<0||v.p[i]>1)return false;
  }
 }
 char tail;if(f.read(&tail,1)||!f.eof())return false;
 // Check all vector values for finite safe magnitudes.
 for(const auto&v:ann.addresses){
  for(float x:v.lex)if(!std::isfinite(x)||std::abs(x)>2.)return false;
  for(float x:v.sign)if(!std::isfinite(x)||std::abs(x)>2.)return false;
 }
 for(const auto&v:ann.signed_hash)for(float x:v)if(!std::isfinite(x)||std::abs(x)>2.)return false;
 return true;
}
template<size_t D> static void normalize168(std::array<double,D>&v){
 double n=0.;for(auto x:v)n+=x*x;n=std::sqrt(n);
 if(n>1e-12)for(auto&x:v)x/=n;
}
static void propose168(State168*s,const unsigned char*past,size_t n){
 s->alpha=0.;s->candidates.clear();s->prefix_cache.clear();s->prefix_cache[""]=1.;
 if(s->strength<=0||!s->memory||!s->parent||!past||!n)return;
 if(s->parent->parent->domain_gate && s->parent->parent->domain_score<s->domain_min)return;
 if(!s->parent->birth)return;
 const auto past_words=complete_words158(past,n);if(past_words.empty())return;
 const auto&g=*s->parent->birth;
 const auto&mem=*s->memory;
 std::array<double,32>q{};std::array<double,16>z{};
 const size_t take=std::min((size_t)3,past_words.size());static constexpr double rec[]={.2,.3,.5};double denom=0.;
 for(size_t j=0;j<take;j++){
  const auto&w=past_words[past_words.size()-take+j];double r=rec[3-take+j];denom+=r;
  auto it=g.lookup.find(w);
  if(it!=g.lookup.end())for(int d=0;d<32;d++)q[d]+=r*g.lex[it->second].ctx[d];
  const auto&c=mem.signed_hash[hash166(w)];for(int d=0;d<16;d++)z[d]+=r*c[d];
 }
 if(denom<=0)return;
 for(auto&x:q)x/=denom;for(auto&x:z)x/=denom;
 normalize168(q);normalize168(z);
 // Orthogonal signed/lexical rails plus causal temporal world-lines.
 // Rail count 1 is EXACT v171; 1..6 are spatial projection variants;
 // 7..9 use only the most recent complete word;
 // 10..12 blend the last two words, all without future inputs.
 using Ranked=std::pair<double,size_t>;
 std::array<std::array<double,32>,3>query_lex{};
 std::array<std::array<double,16>,3>query_sign{};
 query_lex[0]=q;query_sign[0]=z;
 if(s->rail_count>6){
  const int count=past_words.size()>=2?2:1;
  static constexpr double last2w[2]={.4,.6};
  for(int step=0;step<count;step++){
   const auto&w=past_words[past_words.size()-count+step];
   const double rw=last2w[2-count+step];
   auto it=g.lookup.find(w);
   if(it!=g.lookup.end()){
    for(int d=0;d<32;d++){
     if(step==count-1)query_lex[1][d]=g.lex[it->second].ctx[d];
     query_lex[2][d]+=rw*g.lex[it->second].ctx[d];
    }
   }
   const auto&hash=mem.signed_hash[hash166(w)];
   for(int d=0;d<16;d++){
    if(step==count-1)query_sign[1][d]=hash[d];
    query_sign[2][d]+=rw*hash[d];
   }
  }
  normalize168(query_lex[1]);normalize168(query_lex[2]);
  normalize168(query_sign[1]);normalize168(query_sign[2]);
 }
 std::array<double,12>rail_share={s->signed_share,0.,.25,.5,.75,1., .66,0.,1., .66,0.,1.};
 std::array<std::priority_queue<Ranked,std::vector<Ranked>,std::greater<Ranked>>,12>best;
 const auto&all=mem.addresses;
 for(size_t i=0;i<all.size();i++){
  const auto&a=all[i];std::array<double,3>l{},t{};
  const int variants=s->rail_count<=6?1:3;
  for(int d=0;d<32;d++)for(int k=0;k<variants;k++)l[k]+=query_lex[k][d]*a.lex[d];
  for(int d=0;d<16;d++)for(int k=0;k<variants;k++)t[k]+=query_sign[k][d]*a.sign[d];
  for(int k=0;k<s->rail_count;k++){
   const int temporal=k<6?0:(k<9?1:2);
   const double share=rail_share[k];
   const double sim=(1.-share)*l[temporal]+share*t[temporal];
   auto&heap=best[k];
   if(heap.size()<(size_t)s->neighbor_limit)heap.push({sim,i});
   else if(sim>heap.top().first){heap.pop();heap.push({sim,i});}
  }
 }
 std::unordered_map<uint32_t,double>counts;
 for(int k=0;k<s->rail_count;k++){
  auto&heap=best[k];
  if(heap.empty())continue;
  std::vector<Ranked>rank;
  while(!heap.empty()){rank.push_back(heap.top());heap.pop();}
  std::sort(rank.begin(),rank.end(),[](auto a,auto b){return a.first>b.first||(a.first==b.first&&a.second<b.second);});
  // Relative position and positive similarity are causal, never target-based.
  // Each rail contributes one normalized mass. This avoids heavier channels
  // dominating simply because their dot-product variance is larger.
  double rail_total=0.;
  std::unordered_map<uint32_t,double>rail_counts;
  for(size_t j=0;j<rank.size();j++){
   const auto&ref=all[rank[j].second];
   double score=std::pow(std::max(.01,rank[j].first),3)*
      (1./(1.+j/3.))*((ref.k==3)?1.5:(ref.k==2)?1.2:1.);
   for(int a=0;a<ref.n;a++){
    const double m=score*ref.p[a];
    rail_counts[ref.target[a]]+=m;rail_total+=m;
   }
  }
  if(!(rail_total>0)||!std::isfinite(rail_total))continue;
  const double rail_share_weight=1./s->rail_count;
  for(const auto&v:rail_counts){
   counts[v.first]+=s->rail_count==1?v.second:rail_share_weight*v.second/rail_total;
  }
 }
 if(counts.empty())return;
 std::vector<std::pair<uint32_t,double>>sorted(counts.begin(),counts.end());
 std::sort(sorted.begin(),sorted.end(),[&](auto a,auto b){return a.second>b.second||(a.second==b.second&&mem.words[a.first]<mem.words[b.first]);});
 if(sorted.size()>(size_t)s->candidate_limit)sorted.resize(s->candidate_limit);
 double mass=0.;for(const auto&x:sorted)mass+=x.second;
 if(!(mass>0)||!std::isfinite(mass))return;
 s->candidates.reserve(sorted.size());
 for(auto &x:sorted)s->candidates.emplace_back(mem.words[x.first],x.second/mass);
 s->alpha=s->strength;
}
static double parentprefix168(State168*s,const std::string&pre){
 auto found=s->prefix_cache.find(pre);if(found!=s->prefix_cache.end())return found->second;
 double mass=1.;std::string prefix;
 for(char c:pre){
  auto nxt=prefix+c;auto it=s->prefix_cache.find(nxt);
  if(it!=s->prefix_cache.end()){mass=it->second;prefix=nxt;continue;}
  float o[27]={},e=0.;if(wf167_next_impl(s->parent,prefix.data(),prefix.size(),o,&e))return 0.;
  mass*=std::max(0.,(double)o[c-'a']+e/27.);s->prefix_cache[nxt]=mass;prefix=nxt;
 }
 return mass;
}
}
extern "C" void*wf148_open(const char*m,const char*g,const char*c){
 auto s=std::unique_ptr<State168>(new State168);
 if(!get_number168("V168_ANN_STRENGTH",s->strength,0.,.5)||
    !get_number168("V168_ANN_SIGNED_SHARE",s->signed_share,0.,1.)||
    !get_number168("V168_ANN_DOMAIN_MIN",s->domain_min,-5.,5.)||
    !get_number168("V169_NEURAL_RERANK",s->neural_rerank,0.,2.)||
    !get_integer171("V171_ANN_NEIGHBORS",s->neighbor_limit,16,512)||
    !get_integer171("V171_ANN_CANDIDATES",s->candidate_limit,32,1024)||
    !get_integer171("V172_ANN_RAILS",s->rail_count,1,12)||
    !get_number168("V172_TEACHER_SHARE",s->teacher_strength,0.,.9))return nullptr;
 s->parent=(State166*)wf167_open_impl(m,g,c);if(!s->parent)return nullptr;
 if(s->teacher_strength>0.){
  s->teacher.reset(new Teacher172());
  const char* path=getenv("V172_TEACHER_MODEL");
  if(!teacher_load172(path,*s->teacher,s->parent)){
   wf167_close_impl(s->parent);return nullptr;
  }
 }
 if(s->strength>0){
  const char*path=getenv("V168_ANN_INDEX");s->memory.reset(new Ann168());
  if(!path||!load_index168(path,*s->memory,s->parent)){
   wf167_close_impl(s->parent);return nullptr;
  }
 }
 return s.release();
}
extern "C" void wf148_close(void*h){auto*s=(State168*)h;if(!s)return;wf167_close_impl(s->parent);delete s;}
extern "C" int wf148_begin(void*h,const unsigned char*past,size_t n,float cloud){
 auto*s=(State168*)h;if(!s)return -1;
 int ret=wf167_begin_impl(s->parent,past,n,cloud);if(ret<0)return ret;
 propose168(s,past,n);
 // Proposal rescoring: the S+/S- ANN proposes words and the inherited
 // native lexical head evaluates them causally. Zero is exact v168 rollback.
 if(s->alpha>0 && s->neural_rerank>0 && !s->candidates.empty() &&
    s->parent && s->parent->birth){
  const auto &g=*s->parent->birth;
  double mass=0.;
  for(auto &v:s->candidates){
   auto it=g.lookup.find(v.first);
   const double q=it==g.lookup.end()?1e-12:std::max(1e-12,s->parent->p[it->second]);
   v.second*=std::exp(s->neural_rerank*std::log(q));
   mass+=v.second;
  }
  if(!(mass>0.)||!std::isfinite(mass))return -9;
  for(auto &v:s->candidates)v.second/=mass;
  std::sort(s->candidates.begin(),s->candidates.end(),
    [](const auto&a,const auto&b){return a.second!=b.second?a.second>b.second:a.first<b.first;});
 }
 if(s->teacher && !teacher_begin172(s,past,n))return -10;
 return ret+(s->alpha>0);
}
static int wordnext171_impl(void*h,const char*pre,size_t n,float out[27],float*escape){
 auto*s=(State168*)h;if(!s||(!pre&&n)||!out||!escape)return -1;
 int ok=wf167_next_impl(s->parent,pre,n,out,escape);
 if(ok||s->alpha<=0)return ok;
 if(n>64)return -2;
 std::string prefix(pre,n);for(char c:prefix)if(c<'a'||c>'z')return -2;
 double cm[27]={},sum=0.;
 for(const auto&w:s->candidates){
  if(w.first.size()<n||w.first.compare(0,n,prefix)!=0)continue;
  sum+=w.second;
  const int j=w.first.size()==n?26:w.first[n]-'a';cm[j]+=w.second;
 }
 if(sum<=0)return ok;
 double p=parentprefix168(s,prefix);
 double d=(1.-s->alpha)*p+s->alpha*sum;
 if(!(d>0)||!std::isfinite(d))return ok;
 double norm=0.;
 for(int j=0;j<27;j++){
  out[j]=(float)(((1.-s->alpha)*p*(out[j]+*escape/27.)+s->alpha*cm[j])/d);
  norm+=out[j];
 }
 if(!(norm>0)||!std::isfinite(norm))return -3;
 for(int j=0;j<27;j++)out[j]/=norm;
 *escape=0.;return 0;
}
static double parent_mass172(State168*s,const std::string&prefix){
 auto it=s->teacher_parent_prefix.find(prefix);if(it!=s->teacher_parent_prefix.end())return it->second;
 double m=1.;std::string p;
 for(char c:prefix){
  const std::string next=p+c;
  auto found=s->teacher_parent_prefix.find(next);
  if(found!=s->teacher_parent_prefix.end())m=found->second;
  else{
   float out[27]={},esc=0.;
   if(wordnext171_impl(s,p.data(),p.size(),out,&esc))return 0.;
   m*=std::max(0.,(double)out[c-'a']+(double)esc/27.);
   s->teacher_parent_prefix[next]=m;
  }
  p=next;
 }
 return m;
}
extern "C" int wf148_next(void*h,const char*pre,size_t n,float out[27],float*escape){
 auto*s=(State168*)h;if(!s||(!pre&&n)||!out||!escape)return -1;
 const int err=wordnext171_impl(h,pre,n,out,escape);
 if(err||!s->teacher||s->teacher_strength<=0.)return err;
 if(n>64)return -2;
 std::string prefix(pre,n);
 for(char c:prefix)if(c<'a'||c>'z')return -2;
 const auto&trie=s->parent->birth->trie;
 int node=0;for(char c:prefix){node=trie[node].edge[c-'a'];if(node<0)break;}
 const double p_parent=parent_mass172(s,prefix);
 const double p_teacher=node>=0?s->teacher_trie_mass[node]:0.;
 const double mix=(1.-s->teacher_strength)*p_parent+s->teacher_strength*p_teacher;
 if(!(mix>0.)||!std::isfinite(mix))return err;
 double tnext[27]={};
 if(node>=0){
  const auto&branch=trie[node];
  for(int j=0;j<26;j++)if(branch.edge[j]>=0)tnext[j]=s->teacher_trie_mass[branch.edge[j]];
  if(branch.terminal>=0)tnext[26]=s->teacher_probs[branch.terminal];
 }
 double norm=0.;
 for(int j=0;j<27;j++){
  const double base=(double)out[j]+(double)*escape/27.;
  out[j]=(float)(((1.-s->teacher_strength)*p_parent*base+s->teacher_strength*tnext[j])/mix);
  norm+=out[j];
 }
 if(!(norm>0.)||!std::isfinite(norm))return -3;
 for(int j=0;j<27;j++)out[j]=(float)(out[j]/norm);
 *escape=0.;return 0;
}
extern "C" double wf172_teacher_strength(void*h){auto*s=(State168*)h;return s?s->teacher_strength:-1.;}
extern "C" int wf172_teacher_distribution(void*h,float*out,size_t count){
 auto*s=(State168*)h;
 if(!s||!s->teacher||!out||count!=s->teacher_probs.size())return -1;
 for(size_t i=0;i<count;i++)out[i]=(float)s->teacher_probs[i];
 return 0;
}
extern "C" int wf172_rail_count(void*h){auto*s=(State168*)h;return s?s->rail_count:-1;}
extern "C" int wf171_ann_limits(void*h,int*neighbors,int*candidates){
 auto*s=(State168*)h;if(!s||!neighbors||!candidates)return -1;
 *neighbors=s->neighbor_limit;*candidates=s->candidate_limit;return 0;
}
extern "C" double wf169_neural_rerank(void*h){auto*s=(State168*)h;return s?s->neural_rerank:-1.;}
extern "C" double wf168_ann_alpha(void*h){auto*s=(State168*)h;return s?s->alpha:-1.;}
extern "C" int wf168_ann_candidate_count(void*h){auto*s=(State168*)h;return s?(int)s->candidates.size():-1;}
extern "C" int wf168_ann_candidate(void*h,int i,char*out,size_t cap,double*w){
 auto*s=(State168*)h;if(!s||i<0||i>=(int)s->candidates.size()||!out||!w)return -1;
 const auto&v=s->candidates[i];if(cap<=v.first.size())return -2;
 std::copy(v.first.begin(),v.first.end(),out);out[v.first.size()]=0;*w=v.second;return 0;
}
extern "C" double wf167_effective_alpha(void*h){auto*s=(State168*)h;return s?wf167_effective_alpha_impl(s->parent):-1.;}
extern "C" double wf167_route_score(void*h){auto*s=(State168*)h;return s?wf167_route_score_impl(s->parent):-1.;}
extern "C" int wf148_fuse(const float*p,const float*w,float e,float v,int pr,float*out){return wf167_fuse_impl(p,w,e,v,pr,out);}
extern "C" int wf148_fuse_surgical(const float*p,const float*w,float e,float g,float*out){return wf167_fuse_surgical_impl(p,w,e,g,out);}
extern "C" int wf157_logits(void*h,float*out,size_t n){auto*s=(State168*)h;return s?wf167_logits_impl(s->parent,out,n):-1;}
extern "C" int wf157_chars(void*h,const char*p,size_t n,float*out){auto*s=(State168*)h;return s?wf167_chars_impl(s->parent,p,n,out):-1;}
extern "C" int wf158_pointer(void*h,char*w,size_t c,double*a){auto*s=(State168*)h;return s?wf167_pointer_impl(s->parent,w,c,a):-1;}
extern "C" int wf158_candidate_count(void*h){auto*s=(State168*)h;return s?wf167_parent_candidate_count_impl(s->parent):-1;}
extern "C" int wf159_count(void*h){auto*s=(State168*)h;return s?wf167_159_count_impl(s->parent):-1;}
extern "C" int wf159_candidate(void*h,int i,char*w,size_t c,double*p,int*m){auto*s=(State168*)h;return s?wf167_159_candidate_impl(s->parent,i,w,c,p,m):-1;}
extern "C" double wf168_neural_word_prob(void*h,const char*w){auto*s=(State168*)h;return s?wf167_wf168_neural_word_prob_impl(s->parent,w):-1.;}
extern "C" int wf166_candidate_count(void*h){auto*s=(State168*)h;return s?wf167_wf166_candidate_count_impl(s->parent):-1;}
extern "C" int wf166_candidate(void*h,int i,char*w,size_t c,double*p){auto*s=(State168*)h;return s?wf167_wf166_candidate_impl(s->parent,i,w,c,p):-1;}

// Experimental byte-level native API layered over the inherited 27-way ABI.
#include "../v169/native_byte169.cpp"
#include "../v170/native_online170.cpp"