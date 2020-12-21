#ifndef EX9_RECOURSE
#define EX9_RECOURSE

#include <cstring> //for memcpy
#include <cstdio>

/** This class provide an example of what a user of hiop::hiopInterfacePriDecProblem 
 * should implement in order to provide the recourse problem to 
 * hiop::hiopAlgPrimalDecomposition solver
 * 
 * For a given vector x\in R^n and \xi \in R^{n_S}, this example class implements
 *
 *     r_i(x;\xi^i) = 1/S *  min_y 0.5 || y - x ||^2 such that 
 * 
 *                   (1-y_1 + \xi^i_1)^2 + \sum_{k=2}^{n_S} (y_k+\xi^i_k)^2 
 * 
 *                                       + \sum_{k=n_S+1}^{n_y} y_k^2 >= 1   //last one in the constraint implementation 
 * 
 *                   y_k - y_{k-1} >=0, k=2, ..., n_y
 *
 *                   y_1 >=0
 *
 * Coding of the problem in MDS HiOp input: order of variables need to be [ysparse,ydense] 
 */

class PriDecRecourseProblemEx9 : public hiop::hiopInterfaceMDS
{
public:
  PriDecRecourseProblemEx9(int n, int nS): n_(n), nS_(nS),x_(nullptr),xi_(nullptr)
  {
    assert(nS_>=1);
    assert(n_>=nS_);  // ny=nx=n
    nsparse_ = n_*sparse_ratio;

  }

  PriDecRecourseProblemEx9(int n, int nS, const double* x,
		           const double* xi): n_(n), nS_(nS)
  {
    assert(nS_>=1);
    assert(n_>=nS_);  // ny=nx=n

    xi_ = new double[n_];
    //printf("n %d\n",n_); 

    memcpy(xi_,xi, n_*sizeof(double));

    x_ = new double[n_];

    //assert("for debugging" && false); //for debugging purpose
    memcpy(x_,x, n_*sizeof(double));

    nsparse_ = int(n_*sparse_ratio);
  }
  virtual ~PriDecRecourseProblemEx9()
  {
    delete[] x_;
    delete[] xi_;
  }

  // set the ratio of sparse matrices
  void set_sparse(const double ratio)
  {
    sparse_ratio = ratio;
    nsparse_ = int(ratio*n_);
    assert(nsparse_>=1 && ratio<1 && ratio>0);  
  }

  /// Set the basecase solution `x`
  void set_x(const double* x)
  {
    if(x_==NULL){
      x_ = new double[n_]; 
    }
    memcpy(x_,x, n_*sizeof(double));
  }

  /// Set the "sample" vector \xi
  void set_center(const double *xi)
  {
    if(xi_==NULL){
      xi_ = new double[n_];     
    }
    memcpy(xi_,xi, n_*sizeof(double));
  }

  //
  // this are pure virtual in hiop::hiopInterfaceMDS and need to be implemented
  //
  bool get_prob_sizes(long long& n, long long& m)
  {
    n = n_;
    m = n_; 
    //assert(false && "not implemented");
    return true; 
  }

  bool get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type)
  {
    // y_1 bounded
    xlow[0] = 0.;
    xupp[0] = 1e20;
    for(int i=1; i<n; ++i) xlow[i] = -1e+20;
    for(int i=1; i<n; ++i) xupp[i] = +1e+20;
    for(int i=0; i<n; ++i) type[i]=hiopNonlinear;
    //assert(false && "not implemented");
    return true;
  }

  bool get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type)
  {
    assert(m == n_);
    for(int i=0;i<n_-1;i++)
    {
      clow[i] = 0.;
      cupp[i] = 1e20;
    }
    clow[n_-1] = 1.;
    cupp[n_-1] = 1e20;
    return true;
  }

  bool get_sparse_dense_blocks_info(int& nx_sparse, int& nx_dense,
				    int& nnz_sparse_Jace, int& nnz_sparse_Jaci,
				    int& nnz_sparse_Hess_Lagr_SS, int& nnz_sparse_Hess_Lagr_SD)
  {
    nx_sparse = nsparse_;  
    nx_dense = n_-nsparse_;
    assert(nx_sparse>0);

    nnz_sparse_Jace = 0;
    nnz_sparse_Jaci = nsparse_+(nsparse_-1)*2+1;
    nnz_sparse_Hess_Lagr_SS = nsparse_;  //Lagrangian?
    nnz_sparse_Hess_Lagr_SD = 0;
    return true;
  }

  bool eval_f(const long long& n, const double* x, bool new_x, double& obj_value)
  {
    assert(n_==n);
    obj_value = 0.;
    for(int i=0;i<n;i++){
      obj_value += (x[i]-x_[i])*(x[i]-x_[i]);
    }
    obj_value *= 0.5/nS_;

    return true;
  }
 
  virtual bool eval_cons(const long long& n, const long long& m, 
			 const long long& num_cons, const long long* idx_cons,  
			 const double* x, bool new_x, double* cons)
  {
    
    assert(n==n_); assert(m==n_);
    //printf("number of constraints %d\n",num_cons); 
    //printf("number of constraints should be %d)\n",n_);
    assert(num_cons==n_||num_cons==0);

    if(num_cons==0)
    {
      return true;
    }

    for(auto j=0;j<m; j++) cons[j]=0.;

    for(int irow=0; irow<num_cons; irow++){
      const int con_idx = (int) idx_cons[irow];
      if(con_idx<m-1){
        cons[con_idx] = x[con_idx+1]-x[con_idx];
      }else{
        assert(con_idx==m-1);
        cons[m-1] = (1-x[0]+xi_[0])*(1-x[0]+xi_[0]);
        for(int i=1;i<nS_;i++)
        {
          cons[m-1] += (x[i] + xi_[i])*(x[i] + xi_[i]);
        }
        for(int i=nS_;i<n_;i++)
        {
          cons[m-1] += x[i]*x[i];
        }
      } 
    }

    return true; 
  }
  
  //  r_i(x;\xi^i) = 1/S *  min_y 0.5 || y - x ||^2 such that 
  bool eval_grad_f(const long long& n, const double* x, bool new_x, double* gradf)
  {
    assert(n_==n);    
    for(int i=0;i<n_;i++) gradf[i] = (x[i]-x_[i])/nS_;
    return true;
  }
 
  virtual bool eval_Jac_cons(const long long& n, const long long& m, 
		             const long long& num_cons, const long long* idx_cons,
		             const double* x, bool new_x,
		             const long long& nsparse, const long long& ndense, 
		             const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS, 
		             double* JacD)
  {

    assert(num_cons==n_||num_cons==0);
    //indexes for sparse part
    if(num_cons==0)
    {
      return true;
    }

    if(iJacS!=NULL && jJacS!=NULL) {
      int nnzit=0;
      for(int itrow=0; itrow<num_cons; itrow++) {
	const int con_idx = (int) idx_cons[itrow];
	if(con_idx<nsparse_-1) {
	  //sparse Jacobian eq w.r.t. x and s
	  //yk+1
	  iJacS[nnzit] = con_idx;
	  jJacS[nnzit] = con_idx+1; //1
	  nnzit++;

	  //yk
	  iJacS[nnzit] = con_idx;
	  jJacS[nnzit] = con_idx; //-1
	  nnzit++;

	}else if(con_idx==nsparse_-1){
	  iJacS[nnzit] = con_idx;
	  jJacS[nnzit] = con_idx; //-1
	  nnzit++;
	}else if(con_idx==m-1)
	{ 
	  iJacS[nnzit] = m-1;
	  jJacS[nnzit] = 0;
	  nnzit++;
          //cons[m-1] = (1-x[0]+xi_[0]);
	  if(nsparse_<=nS_){
            for(int i=1;i<nsparse_;i++)
            {
	      iJacS[nnzit] = m-1;
	      jJacS[nnzit] = i;
	      nnzit++;
              //cons[m-1] += (x[i] + xi_[i])*(x[i] + xi_[i]);
            }
	  }else{
	    for(int i=1;i<nS_;i++)
            {
	      iJacS[nnzit] = m-1;
	      jJacS[nnzit] = i;
	      nnzit++;
              //cons[m-1] += (x[i] + xi_[i])*(x[i] + xi_[i]);
            }
	    for(int i=nS_;i<nsparse_;i++)
            {
	      iJacS[nnzit] = m-1;
	      jJacS[nnzit] = i;
	      nnzit++;
              //cons[m-1] += x[i]*x[i];
            }
	  }
	  //sparse Jacobian ineq w.r.t x and s
	}
      }
      assert(nnzit==nnzJacS);
    }
    //values for sparse Jacobian if requested by the solver
    if(MJacS!=NULL) {
      int nnzit=0;
      for(int itrow=0; itrow<num_cons; itrow++) {
	const int con_idx = (int) idx_cons[itrow];
	if(con_idx<nsparse_-1) 
	{
	  //sparse Jacobian eq w.r.t. x and s
	  //yk+1
	  MJacS[nnzit] = 1.;
	  nnzit++;

	  //yk
	  MJacS[nnzit] = -1.;
	  nnzit++;

	}else if(con_idx==nsparse_-1)
	{
	  MJacS[nnzit] = -1.;
	  nnzit++;
	}else if(con_idx==m-1)
	{
        
	  MJacS[nnzit] = -2*(1-x[0]+xi_[0]);
	  nnzit++;
          //cons[m-1] = (1-x[0]+xi_[0])^2;
	  if(nsparse_<=nS_){
            for(int i=1;i<nsparse_;i++)
            {
	      MJacS[nnzit] = 2*(x[i]+xi_[i]);
	      nnzit++;
              //cons[m-1] += (x[i] + xi_[i])*(x[i] + xi_[i]);
            }
	  }else{
	    for(int i=1;i<nS_;i++)
            {
	      MJacS[nnzit] = 2*(x[i]+xi_[i]);
	      nnzit++;
              //cons[m-1] += (x[i] + xi_[i])*(x[i] + xi_[i]);
            }
	    for(int i=nS_;i<nsparse_;i++)
            {
	      MJacS[nnzit] = 2*x[i];
	      nnzit++;
              //cons[m-1] += x[i]*x[i];
            }
	  }
	  //sparse Jacobian ineq w.r.t x and s
	}
      }
      assert(nnzit==nnzJacS);
    }
    
    //dense Jacobian w.r.t ydense
    //it has row number of m
    if(JacD!=NULL) {
      for(int itrow=0; itrow<num_cons; itrow++) {
	const int con_idx = (int) idx_cons[itrow];
	if(con_idx==nsparse_-1) 
	{
          JacD[(n_-nsparse_)*con_idx+(con_idx-nsparse_+1)] = 1.0;
	}else if(con_idx>nsparse_-1 && con_idx!=m-1)
	{
          JacD[(n_-nsparse_)*con_idx+(con_idx-nsparse_)] = -1.0;
          JacD[(n_-nsparse_)*con_idx+(con_idx-nsparse_)+1] = 1.0;
	}else if(con_idx==m-1)
	{
          if(nsparse_<=nS_){
              //cons[m-1] += (x[i] + xi_[i])*(x[i] + xi_[i]);
            for(int i=nsparse_; i<nS_;i++){
              JacD[(n_-nsparse_)*con_idx+i-nsparse_] = 2*(x[i] + xi_[i]);	
	    }
            for(int i=nS_; i<m;i++){
              JacD[(n_-nsparse_)*con_idx+i-nsparse_] = 2*x[i] ;	
	    }
          }else{
            for(int i=nsparse_; i<m;i++){
              JacD[(n_-nsparse_)*con_idx+i-nsparse_] = 2*x[i] ;	
	    }
	  }
	}
      }
    }
    //assert("for debugging" && false); //for debugging purpose
    return true;
  }
  //to do
  bool eval_Hess_Lagr(const long long& n, const long long& m, 
                      const double* x, bool new_x, const double& obj_factor,
                      const double* lambda, bool new_lambda,
                      const long long& nsparse, const long long& ndense, 
                      const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS, 
                      double* HDD,
                      int& nnzHSD, int* iHSD, int* jHSD, double* MHSD)
  {
    assert(nnzHSS==nsparse_);
    assert(nnzHSD==0);
    //    r_i(x;\xi^i) = 1/S *  min_y 0.5 || y - x ||^2 such that 
    if(iHSS!=NULL && jHSS!=NULL) {
      for(int i=0;i<nsparse_;i++) iHSS[i] = jHSS[i] = i;     
    }
    // need lambda
    if(MHSS!=NULL) {
      for(int i=0;i<nsparse_;i++) MHSS[i] =  obj_factor/nS_; //what is this?     
      MHSS[0] += -2*lambda[m-1];
      for(int i=1;i<nsparse_;i++){
        MHSS[i] += -lambda[m-1]* 2.; //what is this?     
      } 
    }
    //assert(false && "not implemented");

    if(HDD!=NULL){
      //HDD size: ndense_*ndense_
      for(int i=0; i<ndense;i++){
        HDD[ndense*i+i] = obj_factor/nS_;
	HDD[ndense*i+i] += -2*lambda[m-1];
      }
    }
    return true;
  }

  /* Implementation of the primal starting point specification */
  bool get_starting_point(const long long& global_n, double* x0)
  {    
    assert(global_n==n_);
    for(int i=0; i<global_n; i++) x0[i]=1.;
    return true;
  }

  //todo: what should I do here?
  bool get_starting_point(const long long& n, const long long& m,
				  double* x0,
				  bool& duals_avail,
				  double* z_bndL0, double* z_bndU0,
				  double* lambda0)
  {
    //assert(false && "not implemented");
    return false;
  }

  // computing the derivative of the recourse function with respect to x in the problem description
  // which is the x_ in the protected variable, while x in the function implementation
  // represents y in the problem description
  bool compute_gradx(const int n, const double* y, double*  gradx)
  {
    assert(n_==n);
    for(int i=0;i<n_;i++) gradx[i] = 2*(x_[i]-y[i]);
    return true;
  };

  /**
   * Returns COMM_SELF communicator since this example is only intended to run 
   * on one MPI process 
   */
  bool get_MPI_comm(MPI_Comm& comm_out) {
    comm_out=MPI_COMM_SELF;
    return true;
  }

protected:
  double* x_;
  double* xi_;
  int n_;
  int nS_;
  double sparse_ratio = 0.3;
  int nsparse_;
};

#endif