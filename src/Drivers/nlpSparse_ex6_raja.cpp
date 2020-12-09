#include "nlpSparse_ex6_raja.hpp"

#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>
#include <RAJA/RAJA.hpp>
#include <hiopMatrixSparseTriplet.hpp>
#include <hiopMatrixRajaSparseTriplet.hpp>

#ifdef HIOP_USE_GPU
  #include "cuda.h"
  #define RAJA_CUDA_BLOCK_SIZE 128
  using ex6_raja_exec   = RAJA::cuda_exec<RAJA_CUDA_BLOCK_SIZE>;
  using ex6_raja_reduce = RAJA::cuda_reduce;
  #define RAJA_LAMBDA [=] __device__
#else
  using ex6_raja_exec   = RAJA::omp_parallel_for_exec;
  using ex6_raja_reduce = RAJA::omp_reduce;
  #define RAJA_LAMBDA [=]
#endif

/* Test with bounds and constraints of all types. For some reason this
 *  example is not very well behaved numerically.
 *  min   sum 1/4* { (x_{i}-1)^4 : i=1,...,n}
 *  s.t.
 *            4*x_1 + 2*x_2                     == 10
 *        5<= 2*x_1         + x_3
 *        1<= 2*x_1                 + 0.5*x_i   <= 2*n, for i=4,...,n
 *        x_1 free
 *        0.0 <= x_2
 *        1.5 <= x_3 <= 10
 *        x_i >=0.5, i=4,...,n
 */
Ex6::Ex6(int n, std::string mem_space)
  : n_vars_(n), 
    n_cons_{2}, 
    mem_space_(mem_space)
{
  assert(n>=3);
  n_cons_ += n-3;
  
  // Make sure mem_space_ is uppercase
  transform(mem_space_.begin(), mem_space_.end(), mem_space_.begin(), ::toupper);
}

Ex6::~Ex6()
{}

bool Ex6::get_prob_sizes(long long& n, long long& m)
{ 
  n=n_vars_; 
  m=n_cons_; 
  return true; 
}

bool Ex6::get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type)
{
  assert(n==n_vars_);
  
  RAJA::forall<ex6_raja_exec>(RAJA::RangeSegment(0, n_vars_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      type[i]=hiopNonlinear;
      if(i==0)
      { 
        xlow[i]=-1e20;
        xupp[i]=1e20;
      }
      else if(i==1)
      { 
        xlow[i]= 0.0;
        xupp[i]=1e20;
      }
      else if(i==2)
      { 
        xlow[i]= 1.5;
        xupp[i]=10.0;
      }
      else
      {
        //this is for x_4, x_5, ... , x_n (i>=3), which are bounded only from below
        xlow[i]= 0.5;
        xupp[i]=1e20;
      }
    }); 
  
  return true;
}

bool Ex6::get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type)
{
  assert(m==n_cons_);
  
  RAJA::forall<ex6_raja_exec>(RAJA::RangeSegment(0, n_cons_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      type[i]=hiopNonlinear;
      if(i==0)
      { 
        clow[i]= 10.0;
        cupp[i]= 10.0;
      }
      else if(i==1)
      { 
        clow[i]=  5.0;
        cupp[i]= 1e20;
      }
      else
      {
        clow[i]= 1.0;
        cupp[i]=2*n_vars_;
      }
    });  

  return true;
}

bool Ex6::get_sparse_blocks_info(int& nx,
                                 int& nnz_sparse_Jaceq, int& nnz_sparse_Jacineq,
					                       int& nnz_sparse_Hess_Lagr)
{
    nx = n_vars_;;
    nnz_sparse_Jaceq = 2;
    nnz_sparse_Jacineq = 2 + 2*(n_vars_-3);
    nnz_sparse_Hess_Lagr = n_vars_;
    return true;
}

bool Ex6::eval_f(const long long& n, const double* x, bool new_x, double& obj_value)
{
  assert(n==n_vars_);
  obj_value=0.;

  {
    RAJA::ReduceSum<ex6_raja_reduce, double> aux(0);
    RAJA::forall<ex6_raja_exec>(RAJA::RangeSegment(0, n_vars_),
      RAJA_LAMBDA(RAJA::Index_type i)
      {
        aux += (x[i]-1.)*(x[i]-1.)*(x[i]-1.)*(x[i]-1.);
      });
    obj_value += aux.get();
    obj_value *= 0.25;
  }  
  
  return true;
}

bool Ex6::eval_grad_f(const long long& n, const double* x, bool new_x, double* gradf)
{
  assert(n==n_vars_);

  RAJA::forall<ex6_raja_exec>(RAJA::RangeSegment(0, n_vars_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      gradf[i] = (x[i]-1.)*(x[i]-1.)*(x[i]-1.);
    });  
  
  return true;
}

bool Ex6::eval_cons(const long long& n, const long long& m,
                    const long long& num_cons, const long long* idx_cons,
			              const double* x, bool new_x, double* cons)
{
  return false;
}

/* Four constraints no matter how large n is */
bool Ex6::eval_cons(const long long& n, const long long& m,
		                const double* x, bool new_x, double* cons)
{
  assert(n==n_vars_); assert(m==n_cons_);
  assert(n_cons_==2+n-3);

  RAJA::forall<ex6_raja_exec>(RAJA::RangeSegment(0, n_cons_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      cons[i] = 0.;
      if(i==0)
      { 
        cons[i] += 4*x[0] + 2*x[1];
      }
      else if(i==1)
      { 
        cons[i] += 2*x[0] + 1*x[2];
      }
      else
      {
        cons[i] += 2*x[0] + 0.5*x[i+1];
      }
    }); 

  return true;
}

bool Ex6::eval_Jac_cons(const long long& n, const long long& m,
                        const long long& num_cons, const long long* idx_cons,
                        const double* x, bool new_x,
                        const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS)
{
  return false;
}

bool Ex6::eval_Jac_cons(const long long& n, const long long& m,
                        const double* x, bool new_x,
                        const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS)
{
    assert(n==n_vars_); assert(m==n_cons_);
    assert(nnzJacS == 4 + 2*(n_vars_-3));

    int nnzit{0};
    long long conidx{0};

    if(iJacS!=nullptr && jJacS!=nullptr){
      RAJA::forall<ex6_raja_exec>(RAJA::RangeSegment(0, n_cons_),
        RAJA_LAMBDA(RAJA::Index_type i)
        {
          if(i==0)
          { 
            iJacS[2*i] = i;
            jJacS[2*i] = 0;
            
            iJacS[2*i+1] = i;
            jJacS[2*i+1] = 1;
          }
          else if(i==1)
          { 
            iJacS[2*i] = i;
            jJacS[2*i] = 0;
            
            iJacS[2*i+1] = i;
            jJacS[2*i+1] = 2;
          }
          else
          {
            iJacS[2*i] = i;
            jJacS[2*i] = 0;
            
            iJacS[2*i+1] = i;
            jJacS[2*i+1] = i+1;
          }
        });
    }

    //values for sparse Jacobian if requested by the solver
    if(MJacS!=nullptr) {
      RAJA::forall<ex6_raja_exec>(RAJA::RangeSegment(0, n_cons_),
        RAJA_LAMBDA(RAJA::Index_type i)
        {
          if(i==0)
          { 
            // --- constraint 1 body --->  4*x_1 + 2*x_2 == 10
            MJacS[2*i] = 4;
            MJacS[2*i+1] = 2;
          }
          else if(i==1)
          { 
            // --- constraint 2 body ---> 2*x_1 + x_3
            MJacS[2*i] = 2;
            MJacS[2*i+1] = 1;
          }
          else
          {
            // --- constraint 3 body --->   2*x_1 + 0.5*x_4
            MJacS[2*i] = 2;
            MJacS[2*i+1] = 0.5;
          }
        });
    }
    return true;
}

bool Ex6::eval_Hess_Lagr(const long long& n, const long long& m,
                         const double* x, bool new_x, const double& obj_factor,
                         const double* lambda, bool new_lambda,
                         const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS)
{
    //Note: lambda is not used since all the constraints are linear and, therefore, do
    //not contribute to the Hessian of the Lagrangian
    assert(nnzHSS == n);
    assert(n_vars_ == n);

    if(iHSS!=nullptr && jHSS!=nullptr) {
      RAJA::forall<ex6_raja_exec>(RAJA::RangeSegment(0, n_vars_),
        RAJA_LAMBDA(RAJA::Index_type i)
        {
          iHSS[i] = jHSS[i] = i;
        });
    }

    if(MHSS!=nullptr) {
      RAJA::forall<ex6_raja_exec>(RAJA::RangeSegment(0, n_vars_),
        RAJA_LAMBDA(RAJA::Index_type i)
        {
          MHSS[i] = obj_factor * 3 *(x[i]-1.)*(x[i]-1.);
        });
    }
    return true;
}

bool Ex6::get_starting_point(const long long& n, double* x0)
{
  assert(n==n_vars_);
  RAJA::forall<ex6_raja_exec>(RAJA::RangeSegment(0, n_vars_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      x0[i] = 1.;
    });
  return true;
}
