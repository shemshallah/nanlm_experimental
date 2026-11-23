// NaNCM v156 open-vocabulary spelling prior.
// TRAIN-derived variable-order character counts. No target or future prefix is read.
#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <limits>

struct Speller156 {
  using Count=std::array<uint32_t,27>;
  std::unordered_map<std::string,Count> prefix;
  std::vector<std::unordered_map<std::string,Count>> backoff;
  double smoothing=12.0, prefix_smoothing=3.0;
  uint64_t trainhash=0;
  static uint32_t u32(std::ifstream&f){uint32_t n;f.read((char*)&n,4);if(!f)throw std::runtime_error("truncated v156 u32");return n;}
  static uint64_t u64(std::ifstream&f){uint64_t n;f.read((char*)&n,8);if(!f)throw std::runtime_error("truncated v156 u64");return n;}
  static void readmap(std::ifstream&f,std::unordered_map<std::string,Count>&m){
    uint32_t count=u32(f);if(count>1500000)throw std::runtime_error("oversize v156 countmap");
    for(uint32_t i=0;i<count;i++){
      uint8_t n=0;f.read((char*)&n,1);if(!f||n>64)throw std::runtime_error("invalid spelling key length");
      std::string key(n,' ');if(n)f.read(&key[0],n);if(!f)throw std::runtime_error("truncated spelling key");
      for(char c:key)if(c<'a'||c>'z')throw std::runtime_error("nonletter spelling key");
      Count arr{};f.read((char*)arr.data(),27*4);if(!f)throw std::runtime_error("truncated spelling counts");
      if(!m.emplace(std::move(key),arr).second)throw std::runtime_error("duplicate spelling key");
    }
  }
  explicit Speller156(const std::string&path,uint64_t parenthash){
    std::ifstream f(path,std::ios::binary);char magic[8]={};f.read(magic,8);
    if(!f||std::memcmp(magic,"N156SPLR",8))throw std::runtime_error("v156 spelling-model magic");
    if(u32(f)!=156)throw std::runtime_error("v156 spelling-model version");
    char digest[32];f.read(digest,32);if(!f)throw std::runtime_error("missing v156 SHA-256");
    float x,y;f.read((char*)&x,4);f.read((char*)&y,4);
    if(!f||!std::isfinite(x)||!std::isfinite(y)||x<=0||y<=0||x>100||y>100)throw std::runtime_error("invalid v156 smoothing");
    smoothing=x;prefix_smoothing=y;trainhash=u64(f);
    if(parenthash!=trainhash)throw std::runtime_error("v156 wrong parent TRAIN fingerprint");
    readmap(f,prefix);backoff.resize(6);for(auto&dic:backoff)readmap(f,dic);
    if(f.peek()!=EOF)throw std::runtime_error("unexpected v156 spelling-model trailing bytes");
  }
  static void blend(std::array<double,27>&p,const Count&count,double penalty){
    double n=0,t=0;for(auto c:count){n+=c;t+=c>0;}
    if(n<=0)return;
    const double a=n/(n+penalty*std::sqrt(t));
    for(int i=0;i<27;i++)p[i]=a*double(count[i])/n+(1.0-a)*p[i];
  }
  std::array<float,27> probability(const std::string&word_prefix)const {
    for(char c:word_prefix)if(c<'a'||c>'z')throw std::runtime_error("v156 spelling prefix must be lower ascii");
    std::array<double,27> p{};p.fill(1./27.);
    for(size_t k=0;k<backoff.size()&&k<=word_prefix.size();k++){
      std::string key=k?word_prefix.substr(word_prefix.size()-k):"";
      auto it=backoff[k].find(key);if(it!=backoff[k].end())blend(p,it->second,smoothing);
    }
    auto it=prefix.find(word_prefix);if(it!=prefix.end())blend(p,it->second,prefix_smoothing);
    std::array<float,27>r{};double sum=0;for(double x:p)sum+=x;
    for(int i=0;i<27;i++)r[i]=float(p[i]/sum);
    return r;
  }
};
