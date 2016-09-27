#include "hiopInterface.hpp"
#include "hiopNlpFormulation.hpp"
#include "hiopAlgFilterIPM.hpp"

#include <cmath>
#include <cassert>
#include <cstring> //for memcpy

#ifdef WITH_MPI
#include "mpi.h"
#else
#define MPI_COMM_WORLD 0
#define MPI_Comm int
#endif

/* Test with bounds and constraints of all types. 
 *  min   sum { (x_{i}-1)^4 : i=1,...,n} / 4
 *  s.t.  sum x_i = n+1
 *        5<= 2*x_1 + sum {x_i : i=2,...,n} 
 *        1<= 2*x_1 + 0.5*x_2 + sum{x_i : i=3,...,n} <= 2*n
 *            4*x_1 + 2  *x_2 + 2*x_3 + sum{x_i : i=4,...,n} <=4*n
 *        x_1 free
 *        0.0 <= x_2 
 *        1.5 <= x_3 <= 10
 *        x_i >=0.5, i=4,...,n
 */
class Ex2 : public hiopInterfaceDenseConstraints
{
public: 
  Ex2(int n)
    : n_vars(n), n_cons(4), comm(MPI_COMM_WORLD)
  {
    comm_size=1; my_rank=0; 
#ifdef WITH_MPI
    int ierr = MPI_Comm_size(comm, &comm_size); assert(MPI_SUCCESS==ierr);
    ierr = MPI_Comm_rank(comm, &my_rank); assert(MPI_SUCCESS==ierr);
#endif

    // set up vector distribution for primal variables - easier to store it as a member in this simple example
    col_partition = new long long[comm_size];
    long long quotient=n_vars/comm_size, remainder=n_vars-comm_size*quotient;
    if(my_rank==0) printf("reminder=%d quotient=%d\n", remainder, quotient);
    int i=0; col_partition[i]=0; i++;
    while(i<=remainder) { col_partition[i] = col_partition[i-1]+quotient+1; i++; }
    while(i<=comm_size) { col_partition[i] = col_partition[i-1]+quotient;   i++; }

    if(my_rank==0) {
      for(int i=0;i<=comm_size;i++) 
        printf("%3d ", col_partition[i]);
      printf("\n");
    }
  }
  virtual ~Ex2()
  {
    delete[] col_partition;
  }


  bool get_prob_sizes(long long& n, long long& m)
  { n=n_vars; m=n_cons; return true; }

  bool get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type)
  {
    int i_local;
    for(int i=col_partition[my_rank]; i<col_partition[my_rank+1]; i++) {
      i_local=idx_global2local(n,i);
      if(i==0) { xlow[i_local]=-1e20; xupp[i_local]=1e20;type[i_local]=hiopNonlinear; continue; }
      if(i==1) { xlow[i_local]= 0.0;  xupp[i_local]=1e20;type[i_local]=hiopNonlinear; continue; }
      if(i==2) { xlow[i_local]= 1.5;  xupp[i_local]=10.0;type[i_local]=hiopNonlinear; continue; }
      //this is for x_4, x_5, ... , x_n (i>=3), which are bounded only from below
      xlow[i_local]= 0.5; xupp[i_local]=1e20;type[i_local]=hiopNonlinear;
    }
    return true;
  }
  bool get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type)
  {
    assert(m==n_cons);
    clow[0]= 5.0;  cupp[0]=  5.0;     type[0]=hiopInterfaceBase::hiopLinear;
    clow[1]= 0.0;  cupp[1]= 1e20;     type[1]=hiopInterfaceBase::hiopLinear;
    clow[2]= 1.0;  cupp[2]= 2*n_vars; type[2]=hiopInterfaceBase::hiopLinear;
    clow[3]=-1e20; cupp[3]= 4*n_vars; type[3]=hiopInterfaceBase::hiopLinear;
    return true;
  }
  bool eval_f(const long long& n, const double* x, bool new_x, double& obj_value)
  {
    int n_local=col_partition[my_rank+1]-col_partition[my_rank];
    obj_value=0.; 
    for(int i=0;i<n_local;i++) obj_value += 0.25*pow(x[i]-1., 4);
#ifdef WITH_MPI
    double obj_global;
    int ierr=MPI_Allreduce(&obj_value, &obj_global, 1, MPI_DOUBLE, MPI_SUM, comm); assert(ierr==MPI_SUCCESS);
    obj_value=obj_global;
#endif
    return true;
  }
  bool eval_grad_f(const long long& n, const double* x, bool new_x, double* gradf)
  {
    int n_local=col_partition[my_rank+1]-col_partition[my_rank];
    for(int i=0;i<n_local;i++) gradf[i] = pow(x[i]-1.,3);
    return true;
  }

  /* Four constraints no matter how large n is */
  bool eval_cons(const long long& n, const long long& m, 
		 const long long& num_cons, const long long* idx_cons,  
		 const double* x, bool new_x, double* cons)
  {
    assert(n==n_vars); assert(m==n_cons); assert(n_cons==4);
    assert(num_cons<=m); assert(num_cons>=0);
    //local contributions to the constraints in cons are reset
    for(int j=0;j<num_cons; j++) cons[j]=0.;


    //compute the constraint one by one.
    for(int itcon=0; itcon<num_cons; itcon++) {
      
      // --- constraint 1 body ---> sum x_i = n+1
      if(idx_cons[itcon]==0) {
	int n_local=col_partition[my_rank+1]-col_partition[my_rank];
	//loop over x in local indexes and add its entries to the result
	for(int i=0;i<n_local;i++) cons[itcon] += x[i];
	continue; //done with this constraint
      }
    
      // --- constraint 2 body ---> 2*x_1 + sum {x_i : i=2,...,n} 
      if(idx_cons[itcon]==1) {
	int i_local;
	//loop over x in global indexes 
	for(int i_global=col_partition[my_rank]; i_global<col_partition[my_rank+1]; i_global++) {
	  i_local=idx_global2local(n,i_global);
	  //x_1 has a different contribution to constraint 2 than the rest
	  if(i_global==0) cons[itcon] += 2*x[i_local]; 
	  else            cons[itcon] +=   x[i_local];
	}
	continue;
      }
      // --- constraint 3 body ---> 2*x_1 + 0.5*x_2 + sum{x_i : i=3,...,n}
      if(idx_cons[itcon]==2) {
	int i_local;
	//loop over x in global indexes 
	for(int i_global=col_partition[my_rank]; i_global<col_partition[my_rank+1]; i_global++) {
	  i_local=idx_global2local(n,i_global);
	  //x_1 and x_2 have a different contributions to constraint 3 than the rest
	  if(i_global==0)   cons[itcon] += 2.0*x[i_local]; 
	  else 
	    if(i_global==1) cons[itcon] += 0.5*x[i_local];
	    else            cons[itcon] +=     x[i_local];
	}
	continue;	
      }
      // --- constraint 4 body ---> 4*x_1 + 2*x_2 + 2*x_3 + sum{x_i : i=4,...,n}
      if(idx_cons[itcon]==3) {
	int i_local;
	//loop over x in global indexes 
	for(int i_global=col_partition[my_rank]; i_global<col_partition[my_rank+1]; i_global++) {
	  i_local=idx_global2local(n,i_global);
	  //x_1, x_2, and x_3 have a different contributions to constraint 3 than the rest
	  if(i_global==0)                  cons[itcon] += 4*x[i_local]; 
	  else 
	    if(i_global==1 || i_global==2) cons[itcon] += 2*x[i_local];
	    else                           cons[itcon] +=   x[i_local];
	}
	continue;	
      }
    } //end for loop over constraints


#ifdef WITH_MPI
    double* cons_global=new double[num_cons];
    int ierr=MPI_Allreduce(cons, cons_global, num_cons, MPI_DOUBLE, MPI_SUM, comm); assert(ierr==MPI_SUCCESS);
    memcpy(cons, cons_global, num_cons*sizeof(double));
    delete[] cons_global;
#endif

    return true;
  }
  bool eval_Jac_cons(const long long& n, const long long& m,
		     const long long& num_cons, const long long* idx_cons,  
                     const double* x, bool new_x, double** Jac) 
  {
    assert(n==n_vars); assert(m==n_cons); 
    int n_local=col_partition[my_rank+1]-col_partition[my_rank];
    int i;
    //here we will iterate over the local indexes, however we still need to work with the
    //global indexes to correctly determine the entries in the Jacobian corresponding
    //to the 'rebels' variables x_1, x_2, x_3 


    for(int itcon=0; itcon<num_cons; itcon++) {
      //Jacobian of constraint 1 is all ones
      if(idx_cons[itcon]==0) {
	for(i=0; i<n_local; i++) Jac[itcon][i]=1.0;
	continue;
      }

      //Jacobian of constraint 2 is all ones except the first entry, which is 2
      if(idx_cons[itcon]==1) {
	for(i=1; i<n_local; i++) Jac[itcon][i]=1.0;
	//this is an overkill, but finding it useful for educational purposes
	//is local index 0 the global index 0 (x_1)? If yes the Jac should be 2.0
	Jac[itcon][0] = idx_local2global(n,0)==0?2.:1.;
	continue;
      }

      //Jacobian of constraint 3
      if(idx_cons[itcon]==2) {
	for(i=2; i<n_local; i++) Jac[itcon][i]=1.0;
	Jac[itcon][0] = idx_local2global(n,0)==0?2.:1.;
	Jac[itcon][1] = idx_local2global(n,1)==1?0.5:1.;
	continue;
      }
      
      //Jacobian of constraint  4
      if(idx_cons[itcon]==3) {
	for(i=2; i<n_local; i++) Jac[itcon][i]=1.0;
	Jac[itcon][0] = idx_local2global(n,0)==0?4.:1.; 
	Jac[itcon][1] = idx_local2global(n,1)==1?2.:1.;
	Jac[itcon][2] = idx_local2global(n,2)==2?2.:1.;  
      }
    }
    return true;
  }

  bool get_vecdistrib_info(long long global_n, long long* cols)
  {
    if(global_n==n_vars)
      for(int i=0; i<=comm_size; i++) cols[i]=col_partition[i];
    else 
      assert(false && "You shouldn't need distrib info for this size.");
    return true;
  }
private:
  int n_vars, n_cons;
  MPI_Comm comm;
  int my_rank, comm_size;
  long long* col_partition;
public:
  inline int idx_local2global(long long global_n, int idx_local) 
  { 
    assert(idx_local + col_partition[my_rank]<col_partition[my_rank+1]);
    if(global_n==n_vars)
      return idx_local + col_partition[my_rank]; 
    assert(false && "you shouldn't need global index for a vector of this size.");
  }
  inline int idx_global2local(long long global_n, long long idx_global)
  {
    assert(idx_global>=col_partition[my_rank]   && "global index does not belong to this rank");
    assert(idx_global< col_partition[my_rank+1] && "global index does not belong to this rank");
    assert(global_n==n_vars && "your global_n does not match the number of variables?");
    return idx_global-col_partition[my_rank];
  }
};

int main(int argc, char **argv)
{
  int rank=0;
#ifdef WITH_MPI
  MPI_Init(&argc, &argv);
  assert(MPI_SUCCESS==MPI_Comm_rank(MPI_COMM_WORLD,&rank));
  //assert(MPI_SUCCESS==MPI_Comm_size(MPI_COMM_WORLD,&numRanks));
  if(0==rank) printf("Support for MPI is enabled\n");
#endif

  Ex2 nlp_interface(7);
  if(rank==0) printf("interface created\n");
  hiopNlpDenseConstraints nlp(nlp_interface);
  if(rank==0) printf("nlp formulation created\n");

  hiopAlgFilterIPM solver(&nlp);
  solver.run();
#ifdef WITH_MPI
  MPI_Finalize();
#endif
  
 

  return 0;
}
