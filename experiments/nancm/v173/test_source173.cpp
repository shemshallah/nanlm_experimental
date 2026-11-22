// Standalone numerical contract for normalized first-letter product of experts.
// CI smoke does NOT replace full trained-model replay from cumulative recovery.
#include <array>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cassert>
int main(){
 for(int seed=1;seed<=1000;seed++){
  std::array<double,27> root{};
  std::array<double,26> byte{};
  double m=0.,b=0.;
  for(int i=0;i<27;i++){root[i]=(1.+((seed*7+i*13)%71));m+=root[i];}
  for(int i=0;i<27;i++)root[i]/=m;
  for(int i=0;i<26;i++){byte[i]=1.+((seed*5+i*19)%47);b+=byte[i];}
  for(int i=0;i<26;i++)byte[i]/=b;
  const double letter=1.-root[26];
  for(double gamma:{0.,.01,.1,.5,.99,1.}){
   std::array<double,26> p{};double norm=0.;
   for(int i=0;i<26;i++){
    p[i]=std::exp((1.-gamma)*std::log(root[i]/letter)+gamma*std::log(byte[i]));norm+=p[i];
   }
   double total=root[26];
   for(int i=0;i<26;i++){
    p[i]=letter*p[i]/norm;total+=p[i];
    assert(std::isfinite(p[i])&&p[i]>0.);
    if(gamma==0.)assert(std::abs(p[i]-root[i])<1e-13);
    if(gamma==1.)assert(std::abs(p[i]-letter*byte[i])<1e-13);
   }
   assert(std::abs(total-1.)<1e-13);
  }
 }
 std::puts("PASS 6,000 normalized causal first-letter geometry checks");
}