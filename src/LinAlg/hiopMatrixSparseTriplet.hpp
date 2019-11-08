#ifndef HIOP_SPARSE_MATRIX_TRIPLET
#define HIOP_SPARSE_MATRIX_TRIPLET

#include "hiopVector.hpp"
#include "hiopMatrix.hpp"

namespace hiop
{

/** Sparse matrix in triplet format - it is not distributed
 */
class hiopMatrixSparseTriplet : public hiopMatrix
{
public:
  hiopMatrixSparseTriplet(int rows, int cols, int nnz);
  virtual ~hiopMatrixSparseTriplet(); 

  virtual void setToZero();
  virtual void setToConstant(double c);
  virtual void copyFrom(const hiopMatrixSparseTriplet& dm);

  virtual void timesVec(double beta,  hiopVector& y,
			double alpha, const hiopVector& x) const;
  virtual void timesVec(double beta,  double* y,
			double alpha, const double* x) const;

  virtual void transTimesVec(double beta,   hiopVector& y,
			     double alpha, const hiopVector& x) const;
  virtual void transTimesVec(double beta,   double* y,
			     double alpha, const double* x) const;

  virtual void timesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;

  virtual void transTimesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;

  virtual void timesMatTrans(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;

  virtual void addDiagonal(const hiopVector& d_);
  virtual void addDiagonal(const double& value);
  virtual void addSubDiagonal(long long start, const hiopVector& d_);

  virtual void addMatrix(double alpah, const hiopMatrix& X);
  virtual double max_abs_value();

  virtual bool isfinite() const;
  
  //virtual void print(int maxRows=-1, int maxCols=-1, int rank=-1) const;
  virtual void print(FILE* f=NULL, const char* msg=NULL, int maxRows=-1, int maxCols=-1, int rank=-1) const;

  virtual hiopMatrix* alloc_clone() const;
  virtual hiopMatrix* new_copy() const;

  virtual long long m() const {return nrows;}
  virtual long long n() const {return ncols;}
  virtual long long nnz() const {return nonzeroes;}

  inline int* i_row() { return iRow; }
  inline int* j_col() { return jCol; }
  inline double* M() { return values; }
#ifdef HIOP_DEEPCHECKS
  virtual bool assertSymmetry(double tol=1e-16) const;
#endif
protected:
  int nrows; ///< number of rows
  int ncols; ///< number of columns
  int nonzeroes;  ///< number of nonzero entries
   
  int* iRow; ///< row indices of the nonzero entries
  int* jCol; ///< column indices of the nonzero entries
  double* values; ///< values of the nonzero entries
private:
  hiopMatrixSparseTriplet() : nrows(0), ncols(0), nonzeroes(0), iRow(NULL), jCol(NULL), values(NULL)
  {
  };
  hiopMatrixSparseTriplet(const hiopMatrixSparseTriplet&) {};
};

/** Sparse symmetric matrix in triplet format. Only the lower triangle is stored */
class hiopMatrixSymSparseTriplet : public hiopMatrixSparseTriplet 
{
public: 
  hiopMatrixSymSparseTriplet(int n, int nnz)
    : hiopMatrixSparseTriplet(n, n, nnz)
  {
  }
  virtual ~hiopMatrixSymSparseTriplet(); 

  /** y = beta * y + alpha * this * x */
  virtual void timesVec(double beta,  hiopVector& y,
			double alpha, const hiopVector& x) const;
  virtual void timesVec(double beta,  double* y,
			double alpha, const double* x) const;

  virtual void transTimesVec(double beta,   hiopVector& y,
			     double alpha, const hiopVector& x) const
  {
    return timesVec(beta, y, alpha, x);
  }
  virtual void transTimesVec(double beta,   double* y,
			     double alpha, const double* x) const
  {
    return timesVec(beta, y, alpha, x);
  }

  virtual hiopMatrix* alloc_clone() const;
  virtual hiopMatrix* new_copy() const;
};

} //end of namespace

#endif
