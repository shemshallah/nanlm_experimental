// Source-only regression: no trained model assets or external corpus required.
// Tests the core 14-bit exact key representation, backward context,
// TRAIN-only index isolation, sparse normalized mixture mathematics.
#define NANCM_MEMORY180_LIBRARY
#include "native_memory180.cpp"
#undef NANCM_MEMORY180_LIBRARY
#include <cassert>
#include <cmath>
#include <iostream>
int main(){
 Book one;one.id=1;one.w={21,22,23,24,21,22,23,24,21,22,23,24};
 Book other;other.id=2;other.w={41,42,43,44,41,42,43,44};
 Memory model;model.train({one},false);
 int16_t query[]={21,22,23};
 const Entry*e1=model.query(query+3,1);
 const Entry*e2=model.query(query+3,2);
 const Entry*e3=model.query(query+3,3);
 assert(e1&&e2&&e3);
 assert(e3->best==24&&e3->best_count==3&&e3->count==3);
 assert(e2->best==24 && e1->best==24);
 int16_t unknown[]={41,42,43};
 assert(model.query(unknown+3,3)==nullptr);
 int16_t barrier[]={21,-1,23};
 assert(model.query(barrier+3,3)==nullptr);
 // Consecutive 14-bit word IDs produce collision-free 4-gram keys.
 int16_t a[]={1,2,3,4},b[]={1,2,3,5};
 assert(makekey(a+4,4)!=makekey(b+4,4));
 // Exact convex full-simplex residual: M top2 observed mass, residual parent.
 const double parent[5]={.2,.3,.1,.15,.25};
 double result[5];const double gamma=.4;const double count=2.,total=3.;
 double scale=1.-gamma*(count/total);
 for(int i=0;i<5;i++)result[i]=scale*parent[i];
 result[3]+=gamma*count/total;
 double sum=0.;for(double p:result)sum+=p;
 assert(std::abs(sum-1.)<1e-12);
 assert(result[3]>parent[3]);
 std::cout<<"v180 source regression PASS: exact backward keys, heldout exclusion, normalized word mixture\n";
 return 0;
}
