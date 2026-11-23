#ifndef NANCM_V151_ECHO_CORE_H
#define NANCM_V151_ECHO_CORE_H
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cstdint>
// Causal wordline echo: the only observation mutation occurs AFTER prediction.
struct EchoProposal{std::string word;int pattern_length=0;float purity=0;int hits=0;float margin=0;};
class ConceptEcho {
 std::vector<std::string> history_;
 public:
 void reset(){history_.clear();}
 void observe(const std::string &w){if(history_.size()==96)history_.erase(history_.begin());history_.push_back(w);}
 EchoProposal propose()const{
  std::unordered_map<std::string,double> vote;
  std::unordered_map<std::string,int> longest,occurs;
  const int n=(int)history_.size();
  for(int k=1;k<=std::min(6,n-1);k++){
   for(int j=k;j<n;j++){
    bool same=true;
    for(int h=0;h<k;h++)if(history_[j-k+h]!=history_[n-k+h]){same=false;break;}
    if(!same)continue;
    const std::string &candidate=history_[j];
    const int age=n-j;
    vote[candidate]+=(double)(k*k*k)/std::sqrt(1.+age/20.);
    longest[candidate]=std::max(longest[candidate],k);occurs[candidate]++;
   }
  }
  EchoProposal p;double best=0,second=0,total=0;
  for(auto&kv:vote){total+=kv.second;
   if(kv.second>best||(kv.second==best&&(longest[kv.first]>p.pattern_length||(longest[kv.first]==p.pattern_length&&kv.first<p.word)))){
    second=best;best=kv.second;p.word=kv.first;p.pattern_length=longest[kv.first];p.hits=occurs[kv.first];
   }else second=std::max(second,kv.second);
  }
  if(best>0){p.purity=(float)(best/total);p.margin=(float)((best-second)/best);}
  return p;
 }
 static bool permitted(const EchoProposal&p,int minimum_len=2,float minimum_purity=.5f){
  return !p.word.empty()&&p.pattern_length>=minimum_len&&p.purity>=minimum_purity;
 }
};
#endif
