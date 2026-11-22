// NaNCM v169 native 256-byte symbiotic signed-cloud + byte ngram inference.
// Included at end of wordfield169_api.cpp.  Does not change wf148_* ABI.
#include <map>
#include <unordered_map>
#include <vector>
#include <array>
#include <string>
#include <memory>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {
struct ByteRow169{
 uint64_t total=0;
 std::vector<std::pair<uint8_t,uint32_t>> counts;
 uint32_t get(int value)const{
  // At most 256 entries; usually 1, direct sparse scan is cheap.
  for(const auto&v:counts)if((int)v.first==value)return v.second;
  return 0;
 }
};
struct BytePack169 {
 std::unordered_map<uint32_t,ByteRow169> ngram,alpha,other;
 uint32_t uppercase_start[2]{},uppercase_inner[2]{};
};
struct ByteState169{
 State168*base=nullptr;
 std::unique_ptr<BytePack169>pack;
 std::string anchor;
 bool anchored=false;
};
static bool alpha169(unsigned char c){return (c>='a'&&c<='z')||(c>='A'&&c<='Z');}
static uint32_t byte_key169(const unsigned char*p,size_t n,uint8_t k){
 uint32_t v=((uint32_t)k)<<24;
 if(k>n)return 0xffffffffu;
 for(uint8_t j=0;j<k;j++)v|=(uint32_t)p[n-k+j]<<(8*j);
 return v;
}
static const ByteRow169* row169(const std::unordered_map<uint32_t,ByteRow169>&m,
                                const unsigned char*p,size_t n,uint8_t k){
 if(k>n)return nullptr;
 const auto it=m.find(byte_key169(p,n,k));
 return it==m.end()?nullptr:&it->second;
}
static bool load_byte169(const char*path,BytePack169&z,const State168*base){
 if(!path||!base||!base->parent||!base->parent->parent||!base->parent->parent->parent)return false;
 const auto*n=(const Native164*)base->parent->parent;
 const auto*m=(const State159*)n->parent->base;
 const auto*b=m?(const State158*)m->base:nullptr;
 const auto*r=b?(const WF148State*)b->base:nullptr;
 if(!r)return false;
 std::ifstream f(path,std::ios::binary);if(!f)return false;
 char magic[8];uint64_t fp[3];uint32_t nr[3];uint32_t cases[4];
 if(!f.read(magic,8)||std::memcmp(magic,"N169BYT1",8)||
    !f.read((char*)fp,sizeof(fp))||!f.read((char*)nr,sizeof(nr))||
    !f.read((char*)cases,sizeof(cases)))return false;
 if(fp[0]!=r->model.trainhash||fp[1]!=r->model.geomhash||fp[2]!=r->model.cloudhash)return false;
 if(nr[0]<100||nr[0]>200000||nr[1]<1||nr[1]>70000||nr[2]<1||nr[2]>1000)return false;
 for(uint32_t v:cases)if(v>100000000)return false;
 z.uppercase_start[0]=cases[0];z.uppercase_start[1]=cases[1];
 z.uppercase_inner[0]=cases[2];z.uppercase_inner[1]=cases[3];
 auto load_family=[&](std::unordered_map<uint32_t,ByteRow169>&dest,uint32_t count,uint8_t maxorder,bool binary)->bool{
  dest.reserve(count);
  for(uint32_t i=0;i<count;i++){
   uint8_t k=0;uint32_t packed=0;uint16_t count_values=0;
   if(!f.read((char*)&k,1)||!f.read((char*)&packed,4)||!f.read((char*)&count_values,2)||
      k>maxorder||count_values<1||count_values>256)return false;
   if(k<3 && packed >= (1u << (k*8)))return false;
   if(k==0&&packed!=0)return false;
   ByteRow169 row;row.counts.reserve(count_values);
   std::array<bool,256>used{};
   for(uint16_t j=0;j<count_values;j++){
    uint8_t c=0;uint32_t n=0;
    if(!f.read((char*)&c,1)||!f.read((char*)&n,4)||n==0||used[c]||(binary&&c>1))return false;
    used[c]=true;row.total+=n;row.counts.emplace_back(c,n);
   }
   if(row.total>1000000000)return false;
   if(!dest.emplace((((uint32_t)k)<<24)|packed,std::move(row)).second)return false;
  }
  return true;
 };
 if(!load_family(z.ngram,nr[0],3,false)||!load_family(z.alpha,nr[1],2,true)||
    !load_family(z.other,nr[2],1,false))return false;
 if(!row169(z.ngram,nullptr,0,0)||!row169(z.alpha,nullptr,0,0)||
    !row169(z.other,nullptr,0,0))return false;
 char tail=0;if(f.read(&tail,1)||!f.eof())return false;
 return true;
}
static void ngram169(const BytePack169&z,const unsigned char*p,size_t n,
                     std::array<double,256>&out){
 const auto*g=row169(z.ngram,p,n,0);
 const double den=(double)g->total+25.6;
 for(int i=0;i<256;i++)out[i]=((double)g->get(i)+.1)/den;
 static const double lambda[4]={0,10.,30.,80.};
 for(uint8_t k=1;k<=3;k++){
  const auto*r=row169(z.ngram,p,n,k);if(!r)continue;
  const double d=(double)r->total+lambda[k];
  for(double&v:out)v*=lambda[k]/d;
  for(const auto&v:r->counts)out[v.first]+=v.second/d;
 }
}
static double start_alpha169(const BytePack169&z,const unsigned char*p,size_t n){
 const auto*r=row169(z.alpha,p,n,0);
 double q=((double)r->get(1)+1.)/((double)r->total+2.);
 for(uint8_t k=1;k<=2;k++){
  const auto*s=row169(z.alpha,p,n,k);if(!s)continue;
  double l=(k==1)?40.:80.;q=((double)s->get(1)+l*q)/((double)s->total+l);
 }
 return std::min(.999999,std::max(.000001,q));
}
static double uppercase169(const BytePack169&z,bool inside){
 const uint32_t*counts=inside?z.uppercase_inner:z.uppercase_start;
 return ((double)counts[1]+.5)/((double)counts[0]+counts[1]+1.);
}
static bool wordbridge169(ByteState169*s,const unsigned char*p,size_t n,
                          std::array<double,256>&out,bool&inside){
 size_t begin=n;
 while(begin>0&&alpha169(p[begin-1]))--begin;
 if(n-begin>63)begin=n;
 const std::string anchor((const char*)p,begin);
 std::string prefix((const char*)(p+begin),n-begin);
 for(char&c:prefix)if(c>='A'&&c<='Z')c=(char)(c+('a'-'A'));
 inside=!prefix.empty();
 if(!s->anchored||s->anchor!=anchor){
  const unsigned char*base=(const unsigned char*)anchor.data();
  if(wf148_begin(s->base,base,anchor.size(),.25f)<0)return false;
  s->anchored=true;s->anchor=anchor;
 }
 float probs[27]={},escape=0.f;
 if(wf148_next(s->base,prefix.data(),prefix.size(),probs,&escape))return false;
 std::array<double,27>symbols{};
 for(int j=0;j<27;j++)symbols[j]=(double)probs[j]+(double)escape/27.;
 double letter_mass=0.;for(int i=0;i<26;i++)letter_mass+=symbols[i];
 double pnon=inside?symbols[26]:1.-start_alpha169(*s->pack,p,n);
 if(pnon<0||pnon>1||letter_mass<=0)return false;
 const double scale=inside?1.:(1.-pnon)/letter_mass;
 const double up=uppercase169(*s->pack,inside);
 out.fill(0.);
 for(int j=0;j<26;j++){
  out[(unsigned char)('a'+j)]=symbols[j]*scale*(1.-up);
  out[(unsigned char)('A'+j)]=symbols[j]*scale*up;
 }
 const ByteRow169*g=row169(s->pack->other,p,n,0);
 const ByteRow169*local=row169(s->pack->other,p,n,1);
 const double global_d=(double)g->total+20.4; // 204 non-ASCII-letter bytes
 const double local_d=local?(double)local->total+25.:25.;
 for(int j=0;j<256;j++){
  if(alpha169((unsigned char)j))continue;
  double prior=((double)g->get(j)+.1)/global_d;
  double next=local?((double)local->get(j)+25.*prior)/local_d:prior;
  out[j]=pnon*next;
 }
 return true;
}
} // namespace

extern "C" void*wf169_byte_open(const char*m,const char*g,const char*c,const char*bytepack){
 std::unique_ptr<ByteState169>s(new ByteState169);
 s->base=(State168*)wf148_open(m,g,c);
 if(!s->base)return nullptr;
 s->pack.reset(new BytePack169);
 if(!load_byte169(bytepack,*s->pack,s->base)){
  wf148_close(s->base);return nullptr;
 }
 return s.release();
}
extern "C" void wf169_byte_close(void*h){
 auto*s=(ByteState169*)h;if(!s)return;
 wf148_close(s->base);delete s;
}
extern "C" int wf169_byte_next(void*h,const unsigned char*past,size_t n,float out[256]){
 auto*s=(ByteState169*)h;
 if(!s||(!past&&n)||!out||!s->pack)return -1;
 static const unsigned char empty[1]={0};if(!past)past=empty;
 std::array<double,256>ng{},word{};
 ngram169(*s->pack,past,n,ng);
 bool inside=false;
 if(!wordbridge169(s,past,n,word,inside))return -2;
 const double eta=inside?.5:.1;
 double den=0.;
 for(int j=0;j<256;j++){
  double q=(1.-eta)*ng[j]+eta*word[j];
  if(!std::isfinite(q)||q<=0)return -3;
  out[j]=(float)q;den+=out[j];
 }
 if(!(den>0)||!std::isfinite(den)||std::abs(den-1.)>.0005)return -4;
 for(int j=0;j<256;j++)out[j]=(float)(out[j]/den);
 return 0;
}