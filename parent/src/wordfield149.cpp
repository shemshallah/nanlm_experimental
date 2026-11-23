// NaNCM v148: Causal Multi-World Word Attractor.
// TRAIN-only variable-order phrase context + two oriented signed concept
// field signatures. Native C++ inference; no future target data in keys.
#include "cloud137.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
using namespace std;
using u64=uint64_t;
static u64 mix(u64 x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}
static u64 hbytes(const unsigned char*p,size_t n,u64 seed=0){u64 h=14695981039346656037ULL^seed;for(size_t i=0;i<n;i++){h^=p[i];h*=1099511628211ULL;}return mix(h);}
static u64 hword(const string&s){return hbytes((const unsigned char*)s.data(),s.size());}
static u64 combine(u64 h,u64 y){return mix(h^(y+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2)));}
static bool letter(unsigned char c){return (c>='A'&&c<='Z')||(c>='a'&&c<='z');}
static string lowercase(const unsigned char*p,size_t a,size_t z){string s;s.reserve(z-a);for(size_t i=a;i<z;i++)s+=(char)tolower((unsigned char)p[i]);return s;}
static vector<unsigned char>load(const string&p){ifstream f(p,ios::binary);if(!f)throw runtime_error("open "+p);return vector<unsigned char>(istreambuf_iterator<char>(f),{});}
// Single live cloud forward produces every signed-world signature. This
// replaces four redundant identical forward passes per boundary.
static array<u64,4> cloud_hashes(const GeModel*gm,const Cloud137*cl,const unsigned char*p,size_t i){
 array<u64,4>out{};if(!gm||!cl)return out;
 GeEvents ev;ge_events(p,i,&ev);CloudCache*q=new CloudCache;cloud_forward(gm,cl,&ev,q,1,1);
 int top=-1,low=-1;float mt=-1,mb=-1;
 for(int t=0;t<q->g.nt;t++){int a=q->g.ti[t];float g=q->g.gate[a];if(g>mt){top=a;mt=g;}}
 for(int b=0;b<q->g.nb;b++){int a=q->g.bi[b];float g=0;for(int d=0;d<8;d++)g+=q->r[a][d]*q->r[a][d];if(g>mb){low=a;mb=g;}}
 if(top>=0&&low>=0){
   int k=ge_pair(top,128+low);float plus=0,minus=0;
   for(int d=0;d<8;d++){
    plus+=cl->plus[k][d]*q->r[low][d];
    minus+=cl->minus[k][d]*q->u2[top][d];
   }
   int orientation=(plus>0)+2*(minus>0)+4*(fabs(plus)>fabs(minus));
   out[0]=(u64)top*128+low;
   out[1]=combine(out[0],orientation);
   for(int d=0;d<8;d++){
    out[2]|=(u64)((q->u2[top][d]+q->r[low][d])>0)<<d;
    out[3]|=(u64)((cl->plus[k][d]*q->r[low][d]-cl->minus[k][d]*q->u2[top][d])>0)<<d;
   }
 }
 delete q;return out;
}
struct Context{string word;vector<string>prev;const unsigned char*p=nullptr;size_t pos=0;};
struct Keys{vector<pair<int,u64>> values;};

// v149 noncommutative semantic worldline: quaternionic chiral binding.
// q(a)q(b) != q(b)q(a), so word-role order survives as a geometric phase.
struct Q4{double a,b,c,d;};
static Q4 qmul(Q4 x,Q4 y){return {x.a*y.a-x.b*y.b-x.c*y.c-x.d*y.d,
 x.a*y.b+x.b*y.a+x.c*y.d-x.d*y.c,
 x.a*y.c-x.b*y.d+x.c*y.a+x.d*y.b,
 x.a*y.d+x.b*y.c-x.c*y.b+x.d*y.a};}
static Q4 qnorm(Q4 x){double n=sqrt(x.a*x.a+x.b*x.b+x.c*x.c+x.d*x.d)+1e-18;
 return {x.a/n,x.b/n,x.c/n,x.d/n};}
static Q4 qhash(u64 h){double v[4];for(int d=0;d<4;d++){h=mix(h+0x9e3779b97f4a7c15ULL);v[d]=(double)(h&65535)/32767.5-1.;}return qnorm({v[0],v[1],v[2],v[3]});}
static int role(const string &s){
 if(s=="the"||s=="a"||s=="an")return 1;
 if(s=="to"||s=="for"||s=="of"||s=="with"||s=="from"||s=="by"||s=="in"||s=="on"||s=="at")return 2;
 if(s=="and"||s=="or"||s=="but"||s=="nor")return 3;
 if(s=="he"||s=="she"||s=="they"||s=="it"||s=="we"||s=="i"||s=="you")return 4;
 if(s=="is"||s=="was"||s=="are"||s=="were"||s=="be"||s=="been"||s=="has"||s=="had"||s=="have")return 5;
 if(s=="not"||s=="no"||s=="never")return 6;
 if(s=="his"||s=="her"||s=="their"||s=="our"||s=="my"||s=="your")return 7;
 if(s=="that"||s=="which"||s=="who"||s=="where"||s=="when")return 8;
 if(s.size()>4&&s.substr(s.size()-3)=="ing")return 9;
 if(s.size()>3&&s.substr(s.size()-2)=="ed")return 10;
 if(s.size()>3&&s.substr(s.size()-2)=="ly")return 11;
 if(s.size()>5&&s.substr(s.size()-4)=="tion")return 12;
 if(s.size()>4&&s.substr(s.size()-3)=="ous")return 13;
 if(s.size()>4&&s.substr(s.size()-3)=="est")return 14;
 if(s.size()>3&&s.back()=='s')return 15;
 if(s.size()<4)return 16;
 if(s.size()<7)return 17;
 if(s.size()<11)return 18;
 return 19;
}
static u64 qcode(Q4 q,int precision){
 // Quantized chiral projection. Each coordinate contributes precision bits.
 u64 h=0;int levels=1<<precision;
 for(double x:{q.a,q.b,q.c,q.d}){
  int y=max(0,min(levels-1,int((x+1.)*.5*levels)));
  h=(h<<precision)|(u64)y;
 }
 return h;
}
static void worldline_keys(Keys&out,const Context&c,const array<u64,4>&phase){
 const auto &v=c.prev;if(v.empty())return;
 const string &last=v.back();int rlast=role(last);
 out.values.emplace_back(24,(u64)rlast);
 if(v.size()>=2){
   int rprev=role(v[v.size()-2]);
   out.values.emplace_back(25,(u64)(rprev*32+rlast));
   out.values.emplace_back(28,combine(hword(last),(u64)rprev));
   out.values.emplace_back(31,combine(hword(v[v.size()-2].substr(v[v.size()-2].size()>3?v[v.size()-2].size()-3:0)),hword(last.substr(0,min<size_t>(3,last.size())))));
 }
 // Quaternion multiplication encodes position and word order, rather than a commutative bag.
 Q4 qr{1,0,0,0},qw{1,0,0,0};
 int from=max(0,(int)v.size()-4);for(size_t i=from;i<v.size();i++){
   qr=qnorm(qmul(qr,qhash(mix((u64)role(v[i])*0x47291898ULL+((u64)i-from+1)*1193))));
   qw=qnorm(qmul(qw,qhash(combine(hword(v[i]),(u64)i-from+7))));
 }
 out.values.emplace_back(26,qcode(qr,2));
 out.values.emplace_back(27,qcode(qr,3));
 out.values.emplace_back(32,qcode(qw,2));
 if(v.size()>=3){int r0=role(v[v.size()-3]),r1=role(v[v.size()-2]);
   out.values.emplace_back(29,(u64)r0*1024+r1*32+rlast);
 }
 if(c.pos>=8)out.values.emplace_back(30,combine((u64)rlast,hbytes(c.p+c.pos-8,8)&1023ULL));
 // Bidirectional signed cloud bindings are coupled with role identities.
 if(phase[0]||phase[1]||phase[2]||phase[3]){
   out.values.emplace_back(33,combine((u64)rlast,phase[1]&1023ULL));
   out.values.emplace_back(34,combine((u64)rlast,(phase[2]&63ULL)*256ULL+(phase[3]&255ULL)));
   out.values.emplace_back(35,combine(qcode(qr,2),phase[3]&255ULL));
 }
 // Long-horizon semantic gravity: a slowly evolving hologram of completed
 // content words. Unlike exact suffixes, it can recognize a document topic
 // after intervening grammatical words. Signed cloud phase bends the field.
 double moments[16]={};int count=0;
 for(int i=max(0,(int)v.size()-32);i<(int)v.size();i++){
   int rr=role(v[i]);if(rr<=8||v[i].size()<5)continue;
   int age=(int)v.size()-1-i;double decay=1.0/(1.0+.06*age);
   u64 hw=hword(v[i]);for(int b=0;b<16;b++)moments[b]+=((hw>>b)&1?decay:-decay);
   count++;
 }
 if(count){u64 topic=0;for(int b=0;b<16;b++)topic|=(u64)(moments[b]>0.)<<b;
   out.values.emplace_back(37,topic&63ULL);
   out.values.emplace_back(38,topic&1023ULL);
   out.values.emplace_back(39,combine((u64)rlast,topic&255ULL));
   if(phase[0]||phase[1])out.values.emplace_back(40,combine(topic&255ULL,phase[1]&63ULL));
   out.values.emplace_back(41,topic&65535ULL);
 }
 // Inter-word punctuation/layout is a causal role, not a future feature.
 if(c.pos>=1){unsigned char ch=c.p[c.pos-1];out.values.emplace_back(36,(u64)rlast*16+(ch==' '?1:ch=='\n'?2:ch=='\"'?3:ch==','?4:ch=='.'?5:0));}
}

static Keys make_keys(const Context&c,const GeModel*gm,const Cloud137*cl,bool useCloud){
 Keys out;const auto&v=c.prev;
 for(int n=1;n<=4;n++)if(v.size()>=(size_t)n){u64 h=0x8a531+(u64)n;for(size_t j=v.size()-n;j<v.size();j++)h=combine(h,hword(v[j]));out.values.emplace_back(n,h);}
 if(v.size()>=2){u64 h=combine(hword(v[v.size()-2]),hword(v.back().substr(0,3)));out.values.emplace_back(5,h);}
 if(v.size()>=3){u64 h=combine(hword(v[v.size()-3]),hword(v.back()));out.values.emplace_back(6,h);}
 if(!v.empty()){out.values.emplace_back(7,hword(v.back().substr(v.back().size()>3?v.back().size()-3:0)));}
 if(c.pos>=4)out.values.emplace_back(8,hbytes(c.p+c.pos-4,4));
 if(c.pos>=8)out.values.emplace_back(9,hbytes(c.p+c.pos-8,8));
 if(c.pos>=12)out.values.emplace_back(10,hbytes(c.p+c.pos-12,12));
 for(int len: {16,24,32,48,64,96,160})if(c.pos>=(size_t)len)out.values.emplace_back(17+(len==16?0:len==24?1:len==32?2:len==48?3:len==64?4:len==96?5:6),hbytes(c.p+c.pos-len,len));
 array<u64,4> phase{};
 if(useCloud){
   auto signatures=cloud_hashes(gm,cl,c.p,c.pos);
   phase=signatures;
   u64 ckey=signatures[0],signedkey=signatures[1],ck2=signatures[2],ck3=signatures[3];
   out.values.emplace_back(11,ckey);
   out.values.emplace_back(12,signedkey);
   out.values.emplace_back(13,ck2);
   out.values.emplace_back(14,ck3);
   if(v.size()){out.values.emplace_back(15,combine(ck2,hword(v.back())));out.values.emplace_back(16,combine(ck3,hword(v.back())));}
 }
 worldline_keys(out,c,phase);
 return out;
}
struct Flat{uint16_t word=0;uint32_t count=0;};
struct Bucket{uint32_t total=0;vector<Flat> top;};
struct Model{
 vector<string>words;unordered_map<string,uint16_t> id;
 unordered_map<u64,Bucket> bins;
 vector<uint32_t>unigrams;
 u64 trainhash=0,cloudhash=0,geomhash=0;
};
static u64 kkey(int rail,u64 key){return combine((u64)rail*0x9e3779b97f4a7c15ULL,key);}
static void addcontext(Model&m,const Keys&keys,uint16_t y){m.unigrams[y]++;for(auto [rail,key]:keys.values){auto &b=m.bins[kkey(rail,key)];b.total++;bool hit=false;for(auto &a:b.top)if(a.word==y){a.count++;hit=true;break;}if(!hit)b.top.push_back({y,1});}}
static void shrink(Model&m,size_t cap=24){for(auto &kv:m.bins){auto &a=kv.second.top;if(a.size()>cap){nth_element(a.begin(),a.begin()+cap,a.end(),[](const Flat&x,const Flat&y){return x.count>y.count;});a.resize(cap);}sort(a.begin(),a.end(),[](const Flat&x,const Flat&y){return x.count!=y.count?x.count>y.count:x.word<y.word;});}}
static const float WEIGHTS[]={0,1,2.8f,5.f,7.f,1.7f,2.5f,.55f,1.4f,2.9f,4.5f,.9f,1.2f,1.4f,1.4f,1.f,1.f,4.5f,6.f,7.f,9.f,12.f,15.f,18.f,
 .65f,1.3f,1.0f,.85f,1.75f,1.25f,.55f,1.2f,.2f,1.f,1.f,.7f,.4f,
  .2f,.65f,.7f,.3f,.2f};
static float wf149_beta(){const char *p=getenv("WF149_BETA");if(!p)return 0.f;float z=atof(p);return isfinite(z)?max(0.f,min(10.f,z)):0.f;}
static float wf149_topic(){const char *p=getenv("WF149_TOPIC");if(!p)return 0.f;float z=atof(p);return isfinite(z)?max(0.f,min(10.f,z)):0.f;}
static float wf149_refute(){const char *p=getenv("WF149_REFUTE");if(!p)return 0.f;float z=atof(p);return isfinite(z)?max(0.f,min(2.f,z)):0.f;}
struct Prediction{vector<pair<uint16_t,float>>top;float truth=0;bool cUsed=false;};
static Prediction predict(const Model&m,const Keys&keys,float cloudstrength,uint16_t y,float dream=0.f,float consensus=0.f,const Context*ctx=nullptr,int adaptive=0){
 unordered_map<uint16_t,float> scores;unordered_map<uint16_t,float> votes;int activeRails=0;
 // Unigram prior prevents the no-context model from failing open.
 vector<uint16_t> freq(m.words.size());iota(freq.begin(),freq.end(),0);
 size_t baseCount=min<size_t>(48,freq.size());
 partial_sort(freq.begin(),freq.begin()+baseCount,freq.end(),[&](uint16_t a,uint16_t b){return m.unigrams[a]>m.unigrams[b];});
 u64 uni=accumulate(m.unigrams.begin(),m.unigrams.end(),(u64)0);
 for(size_t i=0;i<baseCount;i++)scores[freq[i]]+=.08f*float(m.unigrams[freq[i]])/max<u64>(1,uni);
 bool used=false;
 if(adaptive){float strongest=0.f;int support=0;
   for(auto [rail,k]:keys.values)if((rail>=2&&rail<=4)||(rail>=17&&rail<=23)){
    auto it=m.bins.find(kkey(rail,k));if(it!=m.bins.end()&&it->second.total>=2){
     int hi=0;for(auto a:it->second.top)hi=max(hi,(int)a.count);
     float purity=(float)hi/it->second.total;strongest=max(strongest,purity);support++;
    }
   }
   if(strongest>.88f&&support>=2)cloudstrength*=.20f;
   else if(support==0)cloudstrength*=1.7f;
 }
 for(auto [rail,k]:keys.values){auto it=m.bins.find(kkey(rail,k));if(it==m.bins.end())continue;
  const Bucket&b=it->second;float w=WEIGHTS[rail]*(rail>=37?wf149_topic():(rail>=24?wf149_beta():(rail>=11&&rail<=16?cloudstrength:1.f)));
  // Bayesian reliability prevents singleton contexts from taking over.
  float reliability=float(b.total)/float(b.total+(rail>=17?2:(rail>=11?18:12)));
  if(rail>=11&&rail<=16)used=true;
  if(w>0)activeRails++;
  int voteCount=0;
  for(auto a:b.top){scores[a.word]+=w*reliability*(float)a.count/float(b.total);
   if(voteCount++<3&&w>0)votes[a.word]+=reliability;
  }
 }
 // Signed Bayesian counterfactual energy: a phrase can *refute* an otherwise
 // plausible word even when it never explicitly proposes that word.
 // This is a negative-evidence channel, not another count-based positive voter.
 if(wf149_refute()>0){
  double uniD=max<u64>(1,uni);float beta=wf149_refute();
  for(auto [rail,k]:keys.values)if(rail>=24&&rail<37){auto it=m.bins.find(kkey(rail,k));
   if(it==m.bins.end()||it->second.total<8)continue;
   const Bucket &b=it->second;float rel=float(b.total)/(b.total+24.f);
   for(auto &kv:scores){
     float prior=max(1e-6f,float(double(m.unigrams[kv.first])/uniD));
     float p=(4.f*prior)/(b.total+4.f);
     for(auto a:b.top)if(a.word==kv.first){p=float(a.count+4.f*prior)/(b.total+4.f);break;}
     float ratio=max(-2.f,min(2.f,logf(max(1e-8f,p)/prior)));
     kv.second+=beta*.04f*rel*ratio;
   }
  }
 }
 // Counterfactual one-word rollout: hypothetical futures are generated only
 // from a candidate, never observed target bytes. This is an ablatable path.
 if(consensus!=0.f&&activeRails)for(auto &kv:scores)kv.second+=consensus*votes[kv.first]/activeRails;
 if(dream!=0.f&&ctx){for(auto &kv:scores){
    u64 h=0x8a531+2;h=combine(h,hword(ctx->prev.empty()?string(""):ctx->prev.back()));h=combine(h,hword(m.words[kv.first]));
    auto it=m.bins.find(kkey(2,h));
    if(it!=m.bins.end()&&it->second.total){const auto&b=it->second;float maximum=0;for(auto a:b.top)maximum=max(maximum,(float)a.count/(float)b.total);
       kv.second+=dream*float(b.total)/(b.total+6.f)*maximum;}
    else{u64 k=combine((u64)0x8a532,hword(m.words[kv.first]));
      auto jt=m.bins.find(kkey(1,k));if(jt!=m.bins.end()&&jt->second.total){float maximum=0;for(auto a:jt->second.top)maximum=max(maximum,(float)a.count/(float)jt->second.total);
       kv.second+=dream*.15f*maximum;}}
 }}
 Prediction out;out.cUsed=used;
 for(auto [id,p]:scores){if(id==y)out.truth=p;out.top.push_back({id,p});}
 size_t n=min<size_t>(32,out.top.size());if(n){partial_sort(out.top.begin(),out.top.begin()+n,out.top.end(),[&](auto a,auto b){return a.second!=b.second?a.second>b.second:m.unigrams[a.first]>m.unigrams[b.first];});out.top.resize(n);}
 return out;
}
static void write_u32(ofstream&f,uint32_t x){f.write((const char*)&x,4);}static void write_u64(ofstream&f,u64 x){f.write((const char*)&x,8);}
static uint32_t ru32(ifstream&f){uint32_t v;f.read((char*)&v,4);if(!f)throw runtime_error("bad model");return v;}
static u64 ru64(ifstream&f){u64 v;f.read((char*)&v,8);if(!f)throw runtime_error("bad model");return v;}
static void save(const Model&m,const string&p){ofstream f(p,ios::binary);if(!f)throw runtime_error("write error");f.write("N148WRLD",8);write_u32(f,148);write_u64(f,m.trainhash);write_u64(f,m.cloudhash);write_u64(f,m.geomhash);write_u32(f,m.words.size());write_u32(f,m.bins.size());
 for(size_t i=0;i<m.words.size();i++){auto &w=m.words[i];write_u32(f,w.size());f.write(w.data(),w.size());write_u32(f,m.unigrams[i]);}
 vector<u64>order;order.reserve(m.bins.size());for(auto &a:m.bins)order.push_back(a.first);sort(order.begin(),order.end());
 for(u64 k:order){auto &b=m.bins.at(k);write_u64(f,k);write_u32(f,b.total);write_u32(f,b.top.size());for(auto a:b.top){write_u32(f,a.word);write_u32(f,a.count);}}
 if(!f)throw runtime_error("incomplete save");}
static Model restore(const string&p){ifstream f(p,ios::binary);char magic[8];f.read(magic,8);if(!f||memcmp(magic,"N148WRLD",8))throw runtime_error("bad model header");if(ru32(f)!=148)throw runtime_error("bad version");Model m;m.trainhash=ru64(f);m.cloudhash=ru64(f);m.geomhash=ru64(f);uint32_t n=ru32(f),nb=ru32(f);if(n>65000||nb>2000000)throw runtime_error("excessive model size");m.unigrams.reserve(n);
 for(uint32_t i=0;i<n;i++){uint32_t z=ru32(f);if(z>128)throw runtime_error("bad word");string w(z,' ');f.read(&w[0],z);m.id[w]=i;m.words.push_back(w);m.unigrams.push_back(ru32(f));}
 for(uint32_t i=0;i<nb;i++){u64 k=ru64(f);Bucket b;b.total=ru32(f);uint32_t count=ru32(f);if(count>128)throw runtime_error("too many candidates");for(uint32_t j=0;j<count;j++){uint32_t id=ru32(f),ct=ru32(f);if(id>=n)throw runtime_error("invalid word id");b.top.push_back({(uint16_t)id,ct});}m.bins.emplace(k,std::move(b));}
 if(f.peek()!=EOF)throw runtime_error("trailing bytes");return m;}
struct Stats{u64 seen=0,iv=0,t1=0,t5=0,t32=0,first1=0,first5=0,cmatches=0,cfixes=0,cerrors=0;};
static vector<pair<vector<unsigned char>,uint32_t>>windows(const string&path){auto b=load(path);if(b.size()<8)throw runtime_error("bad windows");auto r32=[&](size_t x){uint32_t v;memcpy(&v,b.data()+x,4);return v;};uint32_t nw=r32(0),skip=r32(4);if(nw>10000)throw runtime_error("bad nw");size_t off=8;vector<pair<vector<unsigned char>,uint32_t>>out;
 for(uint32_t j=0;j<nw;j++){if(off+4>b.size())throw runtime_error("truncated windows");uint32_t n=r32(off);off+=4;if(off+n>b.size())throw runtime_error("bad window");out.push_back({vector<unsigned char>(b.begin()+off,b.begin()+off+n),skip});off+=n;}return out;}
template<class Fn>static void traverse(const vector<unsigned char>&data,size_t skip,Fn cb){vector<string>prev;for(size_t i=0;i<data.size();i++)if(letter(data[i])&&(i==0||!letter(data[i-1]))){size_t j=i+1;while(j<data.size()&&letter(data[j]))j++;string word=lowercase(data.data(),i,j);if(word.size()<=64){Context c{word,prev,data.data(),i};if(i>=skip)cb(c);prev.push_back(word);if(prev.size()>32)prev.erase(prev.begin());}i=j-1;}}
static Stats evaluate(const Model&m,const vector<pair<vector<unsigned char>,uint32_t>>&wins,const GeModel*gm,const Cloud137*cl,float strength,bool ablate,float dream,float consensus,int adaptive){Stats stats;
 for(auto &win:wins){traverse(win.first,win.second,[&](const Context&c){stats.seen++;auto yit=m.id.find(c.word);if(yit!=m.id.end())stats.iv++;uint16_t y=yit==m.id.end()?65535:yit->second;
  auto keys=make_keys(c,gm,cl,!ablate);auto out=predict(m,keys,strength,y,dream,consensus,&c,adaptive);stats.cmatches+=out.cUsed;
  bool hit1=!out.top.empty()&&out.top[0].first==y;bool hit5=false,hit32=false;
  for(size_t k=0;k<out.top.size();k++){if(out.top[k].first==y){hit32=true;if(k<5)hit5=true;}}
  stats.t1+=hit1;stats.t5+=hit5;stats.t32+=hit32;
  if(!c.word.empty()&&!out.top.empty()){
   stats.first1+=(m.words[out.top[0].first][0]==c.word[0]);
   for(size_t k=0;k<min<size_t>(5,out.top.size());k++)if(m.words[out.top[k].first][0]==c.word[0]){stats.first5++;break;}
  }
 });}return stats;}
static void show(const char*name,const Stats&s){auto pr=[&](u64 x){return s.seen?100.0*x/s.seen:0.0;};cout<<"{\"set\":\""<<name<<"\",\"words\":"<<s.seen<<",\"in_vocab\":"<<s.iv<<",\"word_top1\":"<<pr(s.t1)<<",\"word_top5\":"<<pr(s.t5)<<",\"word_top32\":"<<pr(s.t32)<<",\"first_letter_top1\":"<<pr(s.first1)<<",\"first_letter_covered_by_word_top5\":"<<pr(s.first5)<<",\"cloud_hit_pct\":"<<pr(s.cmatches)<<"}"<<endl;}
#ifndef WF148_NO_MAIN
int main(int argc,char**argv){try{
 if(argc<2)throw runtime_error("train TRAIN GEOM CLOUD OUTFILE [cloud_sample_every] | eval MODEL GEOM CLOUD WINDOWS [strength] [ablate]");
 string mode=argv[1];if(mode=="train"){
  if(argc<6)throw runtime_error("train arguments");auto bytes=load(argv[2]);auto gm=new GeModel;auto cl=new Cloud137; if(ge_load(argv[3],gm,nullptr,nullptr)||cloud_load(argv[4],cl,nullptr,nullptr))throw runtime_error("cloud load");
  unordered_map<string,uint32_t>freq;traverse(bytes,0,[&](const Context&c){freq[c.word]++;});Model m;
  vector<pair<string,uint32_t>>v(freq.begin(),freq.end());sort(v.begin(),v.end(),[](auto&a,auto&b){return a.second!=b.second?a.second>b.second:a.first<b.first;});
  if(v.size()>60000)v.resize(60000);for(auto&w:v){m.id[w.first]=(uint16_t)m.words.size();m.words.push_back(w.first);}m.unigrams.resize(m.words.size(),0);m.trainhash=hbytes(bytes.data(),bytes.size());m.cloudhash=hbytes((const unsigned char*)cl,sizeof(*cl));m.geomhash=hbytes((const unsigned char*)gm,sizeof(*gm));
  uint32_t count=0,cloudCount=0;int sample=argc>6?max(1,atoi(argv[6])):4;
  traverse(bytes,0,[&](const Context&c){auto it=m.id.find(c.word);if(it==m.id.end())return;auto keys=make_keys(c,gm,cl,(count%sample)==0);addcontext(m,keys,it->second);if(count%sample==0)cloudCount++;count++;});shrink(m);
  save(m,argv[5]);cout<<"{\"train_words\":"<<count<<",\"vocab\":"<<m.words.size()<<",\"context_buckets\":"<<m.bins.size()<<",\"cloud_samples\":"<<cloudCount<<",\"model_file\":\""<<argv[5]<<"\"}"<<endl;
 } else if(mode=="augment"){
  if(argc<7)throw runtime_error("augment PARENT TRAIN GEOM CLOUD OUTPUT [cloud_every]");
  Model m=restore(argv[2]);auto bytes=load(argv[3]);auto gm=new GeModel;auto cl=new Cloud137;
  if(ge_load(argv[4],gm,nullptr,nullptr)||cloud_load(argv[5],cl,nullptr,nullptr))throw runtime_error("load paired cloud");
  if(m.cloudhash!=hbytes((const unsigned char*)cl,sizeof(*cl))||m.geomhash!=hbytes((const unsigned char*)gm,sizeof(*gm))||m.trainhash!=hbytes(bytes.data(),bytes.size()))throw runtime_error("paired parent/train fingerprint mismatch");
  u64 count=0;int every=argc>7?max(1,atoi(argv[7])):8;
  traverse(bytes,0,[&](const Context&c){auto it=m.id.find(c.word);if(it==m.id.end())return;
    auto keys=make_keys(c,gm,cl,(count%every)==0);
    for(auto [rail,k]:keys.values)if(rail>=24){auto &b=m.bins[kkey(rail,k)];b.total++;
      bool hit=false;for(auto &a:b.top)if(a.word==it->second){a.count++;hit=true;break;}
      if(!hit)b.top.push_back({it->second,1});
    }
    count++;
  });shrink(m);save(m,argv[6]);
  cout<<"{\"augment_words\":"<<count<<",\"all_context_buckets\":"<<m.bins.size()<<",\"model_file\":\""<<argv[6]<<"\"}"<<endl;
 } else if(mode=="eval"){
  if(argc<6)throw runtime_error("eval arguments");Model m=restore(argv[2]);auto gm=new GeModel;auto cl=new Cloud137;if(ge_load(argv[3],gm,nullptr,nullptr)||cloud_load(argv[4],cl,nullptr,nullptr))throw runtime_error("load geometry");if(m.cloudhash!=hbytes((const unsigned char*)cl,sizeof(*cl))||m.geomhash!=hbytes((const unsigned char*)gm,sizeof(*gm)))throw runtime_error("paired geometry fingerprint mismatch");
  float strength=argc>6?atof(argv[6]):1.f;bool ablate=argc>7?atoi(argv[7]):0;float dream=argc>8?atof(argv[8]):0.f;float consensus=argc>9?atof(argv[9]):0.f;int adaptive=argc>10?atoi(argv[10]):0;show(argv[5],evaluate(m,windows(argv[5]),gm,cl,strength,ablate,dream,consensus,adaptive));
 }else throw runtime_error("bad mode");
 }catch(const exception&e){cerr<<"v148 error: "<<e.what()<<"\n";return 1;}return 0;}

#endif /* WF148_NO_MAIN */
