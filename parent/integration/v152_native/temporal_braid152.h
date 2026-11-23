#ifndef NANCM_TEMPORAL_BRAID152_H
#define NANCM_TEMPORAL_BRAID152_H
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <limits>

// A strictly causal episodic proposal. A candidate successor always appears
// in prior observed text; the target is never admitted before evaluation.
struct Braid152Proposal {
 std::string word;
 int exact=0, occurrences=0;
 double support=0, refutation=0, purity=0, margin=0;
};
class TemporalBraid152 {
 std::vector<std::string> history_;
 int horizon_=256;
public:
 void reset(){history_.clear();}
 void observe(const std::string &word){
  if(word.empty())return;
  if((int)history_.size()==horizon_)history_.erase(history_.begin());
  history_.push_back(word);
 }
 size_t size()const{return history_.size();}
 Braid152Proposal propose()const{
  const int n=(int)history_.size();
  std::unordered_map<std::string,double> evidence;
  std::unordered_map<std::string,int> longest,hits;
  // A past occurrence can count as evidence only when its successor has
  // ALREADY BEEN OBSERVED. For each older position j, candidate=history[j].
  for(int j=1;j<n;j++){
   int k=0;
   while(k<6 && k<j && history_[n-k-1]==history_[j-k-1])k++;
   if(k==0)continue;
   const std::string &word=history_[j];
   const int age=n-j;
   // v152 braid differs from v151: one evidence packet per occurrence,
   // cubic in the maximum coherent suffix, rather than stacked k-votes.
   const double weight=(double)(k*k*k)/std::sqrt(1.0+age/20.0);
   evidence[word]+=weight;
   longest[word]=std::max(longest[word],k);
   hits[word]++;
  }
  Braid152Proposal p;
  double total=0.,first=0.,second=0.;
  for(const auto &kv:evidence){
   total+=kv.second;
   bool better=(kv.second>first+1e-15) ||
     (std::abs(kv.second-first)<1e-15 &&
       (longest[kv.first]>p.exact ||
        (longest[kv.first]==p.exact && kv.first<p.word)));
   if(better){second=first;first=kv.second;p.word=kv.first;p.exact=longest[kv.first];p.occurrences=hits[kv.first];}
   else second=std::max(second,kv.second);
  }
  if(first>0){p.support=first;p.refutation=total-first;p.purity=first/total;p.margin=(first-second)/first;}
  return p;
 }
 static bool permitted(const Braid152Proposal&p,double min_purity=.60,double min_support=1.0,int min_match=1){
  return !p.word.empty()&&p.exact>=min_match&&p.support>=min_support&&p.purity>=min_purity;
 }
};
#endif
