// Fixed-degree-normalized symmetric sparse contraction. Forward and adjoint.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
extern "C" {
struct Edge { uint32_t i,j; double v; };
int nancm_solve(const Edge* edges,size_t count,const double* force,double* result,
                uint32_t dim,double scale,double tolerance,uint32_t max_iter) {
    if (!force || !result || !dim || !std::isfinite(scale) || scale<1 ||
        !std::isfinite(tolerance) || tolerance<=0 || max_iter==0 || (count && !edges)) return -1;
    std::vector<double> x(dim),q(dim),row(dim,0.0);
    for (uint32_t i=0;i<dim;i++) {
        if (!std::isfinite(force[i])) return -2;
        x[i]=force[i];
    }
    for (size_t k=0;k<count;k++) {
        const auto& e=edges[k];
        if(e.i>=dim || e.j>=dim || e.i==e.j || !std::isfinite(e.v) || std::abs(e.v)>1.0) return -3;
        row[e.i]+=std::abs(e.v);row[e.j]+=std::abs(e.v);
    }
    for(double s:row) if(s>scale+1e-9) return -4;
    for(uint32_t iter=0;iter<max_iter;iter++){
        std::copy(force,force+dim,q.begin());
        for(size_t k=0;k<count;k++){
            const auto& e=edges[k];
            double w=.35*e.v/scale;
            q[e.i]+=w*x[e.j];q[e.j]+=w*x[e.i];
        }
        double delta=0;
        for(uint32_t i=0;i<dim;i++){double z=q[i]-x[i];delta+=z*z;}
        if(!std::isfinite(delta))return -5;
        if(delta<=tolerance*tolerance){std::copy(q.begin(),q.end(),result);return int(iter+1);}
        x.swap(q);
    }
    std::copy(force,force+dim,result);
    return 0;
}
}
