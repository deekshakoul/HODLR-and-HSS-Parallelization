#ifndef __HODLR_Matrix__
#define __HODLR_Matrix__
#include <Eigen/Dense>
#include <set>
#include <vector>
#include<omp.h>
#include<Eigen/StdVector>
class HODLR_Matrix {
	friend class HODLR_Tree;
	private:
	int N;
	public:
	HODLR_Matrix(int N);

	virtual double get_Matrix_Entry(int j, int k) {
		return 0.0;
	}
	virtual Eigen::VectorXd get_array(int start , int size){

		return  Eigen::VectorXd::Zero(3);
	}
	virtual	double get_array_element(int index){
		return 0.0;
	}
	Eigen::VectorXd get_Row(int j, int nColStart, int nColEnd);
	Eigen::VectorXd get_Col(int k, int nRowStart, int nRowEnd);
	Eigen::MatrixXd get_Matrix(int j, int k, int nRows, int nCols);
	Eigen::VectorXd get_Row_p(int j, int nColStart, int nColEnd);
	Eigen::VectorXd get_Col_p(int k, int nRowStart, int nRowEnd);
	Eigen::MatrixXd get_Matrix_p(int j, int k, int nRows, int nCols);
	void rook_Piv_p(int nRowStart, int nColStart, int nRows, int nCols, double tolerance, Eigen::MatrixXd& L, Eigen::MatrixXd& R, int& computedRank);	
	int max_Abs_Vector(const Eigen::VectorXd& v, const std::set<int>& allowed_Indices, double& max);
	void rook_Piv(int nRowStart, int nColStart, int nRows, int nCols, double tolerance, Eigen::MatrixXd& L, Eigen::MatrixXd& R, int& computedRank);

	~HODLR_Matrix();
};

class myHODLR_Matrix : public HODLR_Matrix {
private:
	Eigen::VectorXd x;
public:
	//Eigen::VectorXd x;
	myHODLR_Matrix(int N) : HODLR_Matrix(N) {
		x	=	Eigen::VectorXd::Random(N);
		std::sort(x.data(),x.data()+x.size());
	};
	double get_Matrix_Entry(int j, int k) {
		if(j==k) {
			return 10;
		}
		else {
			//return 1.0/(1.0+((x(j)-x(k))*(x(j)-x(k))));
			//return sqrt(1.0 + ((x(j)-x(k))*(x(j)-x(k))));
			// return exp(-fabs(x(j)-x(k)));
			
			return exp((-(x(j)-x(k))*(x(j)-x(k))));		//kernel function
						
			//return exp2((-(x(j)-x(k))*(x(j)-x(k)))*3.322);		
		}
	}

	Eigen::VectorXd get_array(int start , int size){
	return x.segment(start,size);	
	}
	
	double get_array_element(int index){
		return x(index);
	}
	~myHODLR_Matrix() {}; //Destructor
};

/********************************************************/
/*	PURPOSE OF EXISTENCE:	Constructor for the class	*/
/********************************************************/

/************/
/*	INPUTS	*/
/************/

///	N	-	Number of rows/columns of the matrix

HODLR_Matrix::HODLR_Matrix(int N) {
	this->N	=	N;
}

/********************************************************/
/*	PURPOSE OF EXISTENCE:	Destructor for the class	*/
/********************************************************/

HODLR_Matrix::~HODLR_Matrix() {};

/********************************************************/
/*	PURPOSE OF EXISTENCE:	Obtains a row of the matrix	*/
/********************************************************/

/************/
/*	INPUTS	*/
/************/

///	j			-	row index
///	nColStart	-	Starting column
///	nCols		-	Number of columns

/************/
/*	OUTPUT	*/
/************/

///	row			-	row of the matrix



Eigen::VectorXd HODLR_Matrix::get_Row(const int j, const int nColStart, const int nCols) {

    Eigen::VectorXd row(nCols);

  
	row =  (-(get_array(nColStart,nCols).array() - get_array_element(j)).square()).exp();

    return row;
}

Eigen::VectorXd HODLR_Matrix::get_Row_p(const int j, const int nColStart, const int nCols) {   //given nCols set each thread to get two(or any number) elements - launch specific number of threads
	Eigen::VectorXd row(nCols);	
	#pragma omp parallel
        {
                int num_threads = omp_get_num_threads();
                int tid = omp_get_thread_num();
                int n_per_thread = nCols / num_threads;
                if ((n_per_thread * num_threads < nCols)) n_per_thread++;
                int start = tid * n_per_thread;
                int len = n_per_thread;
                if (tid + 1 == num_threads) len = nCols - start;

                if(start < nCols)
                        row.segment(start, len) = (-(get_array(nColStart+start,len).array() - get_array_element(j)).square()).exp().matrix();

        }

	return row;
}

/************************************************************/
/*	PURPOSE OF EXISTENCE:	Obtains a column of the matrix	*/
/************************************************************/

/************/
/*	INPUTS	*/
/************/

///	k			-	column index
///	nRowStart	-	Starting row
///	nRows		-	Number of rows

/************/
/*	OUTPUT	*/
/************/

///	col			-	column of the matrix



Eigen::VectorXd HODLR_Matrix::get_Col(const int j, const int nColStart, const int nCols) {

	Eigen::VectorXd row(nCols);
	row =  (-(get_array(nColStart,nCols).array() - get_array_element(j)).square()).exp();//.matrix();		

    return row;
}


Eigen::VectorXd HODLR_Matrix::get_Col_p(const int j, const int nColStart, const int nCols) {

	Eigen::VectorXd col(nCols);
 	omp_set_num_threads(50);
	#pragma omp parallel
        {	
                int num_threads = omp_get_num_threads();
                int tid = omp_get_thread_num();
                int n_per_thread = nCols / num_threads;
                if ((n_per_thread * num_threads < nCols)) n_per_thread++;
                int start = tid * n_per_thread;
                int len = n_per_thread;
                if (tid + 1 == num_threads) len = nCols - start;

                if(start < nCols)
                        col.segment(start, len) = (-(get_array(nColStart+start,len).array() - get_array_element(j)).square()).exp().matrix();

        }
       return col;
}
/****************************************************************/
/*	PURPOSE OF EXISTENCE:	Obtains a sub-matrix of the matrix	*/
/****************************************************************/

/************/
/*	INPUTS	*/
/************/

///	nRowStart	-	starting row
///	nColStart	-	Starting column
///	nRows		-	Number of rows
/// nCols		-	Number of columns

/************/
/*	OUTPUT	*/
/************/

///	mat			-	sub-matrix of the matrix

Eigen::MatrixXd HODLR_Matrix::get_Matrix(const int nRowStart, const int nColStart, const int nRows, const int nCols) {    	
	Eigen::MatrixXd mat(nRows, nCols);
	// std::cout<<"Get MAtrix  "<<nRows<<" x " <<nCols<<"\n";
	for (int j=0; j<nRows; ++j) {
		for (int k=0; k<nCols; ++k) {
			mat(j,k)	=	get_Matrix_Entry(j+nRowStart, k+nColStart);
		}
	}
       
	return mat;
}

Eigen::MatrixXd HODLR_Matrix::get_Matrix_p(const int nRowStart, const int nColStart, const int nRows, const int nCols) {    	
	Eigen::MatrixXd mat(nRows, nCols);
	 
//#pragma omp parallel for
//#pragma omp parallel for collapse(2)
	for (int j=0; j<nRows; ++j) {
//#pragma omp parallel for
		for (int k=0; k<nCols; ++k) {
			mat(j,k)	=	get_Matrix_Entry(j+nRowStart, k+nColStart);
		}
	}
       
	return mat;
}
/********************************************************************************************/
/*	PURPOSE OF EXISTENCE:	Obtains the index and value of the maximum entry of a vector 	*/
/********************************************************************************************/

/************/
/*	INPUTS	*/
/************/

///	v				-	vector
///	allowed_Indices	-	indices that need to be searched

/************/
/*	OUTPUT	*/
/************/

///	max		-	Value of the maximum
/// index	-	Index of the maximum
int HODLR_Matrix::max_Abs_Vector(const Eigen::VectorXd& v, const std::set<int>& allowed_Indices, double& max) {
	std::set<int>::iterator it	=	allowed_Indices.begin();
	int index	=	*allowed_Indices.begin();
	max		=	v(index);
	
	for (it=allowed_Indices.begin(); it!=allowed_Indices.end(); ++it) {
		if(fabs(v(*it))>fabs(max)) {
			index	=	*it;
			max		=	v(index);
		}
		
	}
	return index;
}
 

/****************************************************************************************************/
/*	PURPOSE OF EXISTENCE:	Obtains the low-rank decomposition of the matrix to a desired tolerance	*/
/* 	using rook pivoting, i.e., given a sub-matrix 'A' and tolerance 'epsilon', computes matrices	*/
/*	'L' and 'R' such that ||A-LR'||_F < epsilon. The norm is Frobenius norm.						*/
/****************************************************************************************************/

void HODLR_Matrix::rook_Piv_p(int nRowStart, int nColStart, int nRows, int nCols, double tolerance, Eigen::MatrixXd& L, Eigen::MatrixXd& R, int& computedRank) {
	std::vector<int> rowIndex;		///	This stores the row indices, which have already been used.
	std::vector<int> colIndex;		///	This stores the column indices, which have already been used.
	std::set<int> remainingRowIndex;/// Remaining row indicies
	std::set<int> remainingColIndex;/// Remaining row indicies
	std::vector<Eigen::VectorXd> u;	///	Stores the column basis.
	std::vector<Eigen::VectorXd> v;	///	Stores the row basis.

/*	for (int k=0; k<nRows; ++k) {
		remainingRowIndex.insert(k);
	}
*/	//double st = omp_get_wtime();
int k;
	for (k = 0; k < nRows; ++k) 
    		remainingRowIndex.insert(remainingRowIndex.end(), k);
	for (k = 0; k < nCols; ++k) 
    		remainingColIndex.insert(remainingColIndex.end(), k);
	
/*	for (int k=0; k<nCols; ++k) {
		remainingColIndex.insert(k);
	}
*/
	srand (time(NULL));
	double max, Gamma, unused_max;

	/*  INITIALIZATION  */

	/// Initialize the matrix norm and the the first row index
	double matrix_Norm  =   0;
	rowIndex.push_back(0);
	remainingRowIndex.erase(0);

	int pivot;

	computedRank   =   0;

	Eigen::VectorXd a, row, col;

	double row_Squared_Norm, row_Norm, col_Squared_Norm, col_Norm;

	int max_tries  =   10;

	int count;
	
	/// Repeat till the desired tolerance is obtained
	do {
		/// Generation of the row
		/// Row of the residuum and the pivot column
		// row =   A.row(rowIndex.back());
		//std::cout<<"heck";
		row	=	get_Row_p(nRowStart+rowIndex.back(), nColStart, nCols);
		
		for (int l=0; l<computedRank; ++l) {		//comprank = 0 ;
			row =   row-u[l](rowIndex.back())*v[l];
		}

		pivot   =   max_Abs_Vector(row, remainingColIndex, max);


		count	=   0;

		/// This randomization is needed if in the middle of the algorithm the row happens to be exactly the linear combination of the previous rows upto some tolerance.
		while (fabs(max)<tolerance && count < max_tries && remainingColIndex.size() >0 && remainingRowIndex.size() >0) {
			rowIndex.pop_back();
			int new_rowIndex	=	*remainingRowIndex.begin();
			rowIndex.push_back(new_rowIndex);
			remainingRowIndex.erase(new_rowIndex);

			/// Generation of the row
			// a	=	A.row(new_rowIndex);
			a	=	get_Row_p(nRowStart+new_rowIndex, nColStart, nCols); //moving to next row of the block
			/// Row of the residuum and the pivot column
			row =   a;

			for (int l=0; l<computedRank; ++l) {
				row =   row-u[l](rowIndex.back())*v[l];
			}
			pivot   =   max_Abs_Vector(row, remainingColIndex, max);
			++count;
		}

		if (count == max_tries || remainingColIndex.size() == 0 || remainingRowIndex.size() == 0) break;

		count = 0;

		colIndex.push_back(pivot);
		remainingColIndex.erase(pivot);

		/// Normalizing constant
		Gamma   =   1.0/max;

		/// Generation of the column
		// a	=	A.col(colIndex.back());
		a	=	get_Col_p(nColStart+colIndex.back(), nRowStart, nRows);
		/// Column of the residuum and the pivot row
		col =   a;
		for (int l=0; l<computedRank; ++l) {
			col =   col-v[l](colIndex.back())*u[l];
		}
		pivot   =   max_Abs_Vector(col, remainingRowIndex, unused_max);

		/// This randomization is needed if in the middle of the algorithm the columns happens to be exactly the linear combination of the previous columns.
		while (fabs(max)<tolerance && count < max_tries && remainingColIndex.size() >0 && remainingRowIndex.size() >0) {
			colIndex.pop_back();
			int new_colIndex	=	*remainingColIndex.begin();
			colIndex.push_back(new_colIndex);
			remainingColIndex.erase(new_colIndex);

			/// Generation of the column
			// a	=	A.col(new_colIndex);
			a	=	get_Col_p(nColStart+new_colIndex, nRowStart, nRows);

			/// Column of the residuum and the pivot row
			col =   a;
			for (int l=0; l<computedRank; ++l) {
				col =   col-u[l](colIndex.back())*v[l];
			}
			pivot   =   max_Abs_Vector(col, remainingRowIndex, unused_max);
			++count;
			//std::cout << count << "\n";
		}

		if (count == max_tries || remainingColIndex.size() == 0 || remainingRowIndex.size() == 0) break;

		count = 0;

		rowIndex.push_back(pivot);
		remainingRowIndex.erase(pivot);

		/// New vectors
		u.push_back(Gamma*col);
		v.push_back(row);

		/// New approximation of matrix norm
		row_Squared_Norm    =   row.squaredNorm();
		row_Norm            =   sqrt(row_Squared_Norm);

		col_Squared_Norm    =   col.squaredNorm();
		col_Norm            =   sqrt(col_Squared_Norm);

		matrix_Norm         =   matrix_Norm +   Gamma*Gamma*row_Squared_Norm*col_Squared_Norm;

		for (int j=0; j<computedRank; ++j) {
			matrix_Norm     =   matrix_Norm +   2.0*(u[j].dot(u.back()))*(v[j].dot(v.back()));
		}
		++computedRank;
	} while (computedRank*(nRows+nCols)*row_Norm*col_Norm > fabs(max)*tolerance*matrix_Norm && computedRank < fmin(nRows, nCols));

	/// If the computedRank is close to full-rank then return the trivial full-rank decomposition

	if (computedRank>=fmin(nRows, nCols)-1) {
		if (nRows < nCols) {
			L   =   Eigen::MatrixXd::Identity(nRows,nRows);
			R	=	get_Matrix_p(nRowStart, nColStart, nRows, nCols).transpose();
			// V	=	A.transpose();
			computedRank   =   nRows;
		}
		else {
			// U	=	A;
			L	=	get_Matrix_p(nRowStart, nColStart, nRows, nCols);
			R   =   Eigen::MatrixXd::Identity(nCols,nCols);
			computedRank   =   nCols;
		}
	}
	else {
		L   =   Eigen::MatrixXd(nRows,computedRank);
		R   =   Eigen::MatrixXd(nCols,computedRank);
		for (int j=0; j<computedRank; ++j) {
			L.col(j)    =   u[j];
			R.col(j)    =   v[j];
		}
	}
	//double e = omp_get_wtime();
	//std::cout<<"time for rook piv p"<<e-st<<" for  N is  "<<nRows<<" "<<nCols<<"\n";
	// std::cout << "Size of row index: " << rowIndex.size() << "\n";
	// std::cout << "Size of remaining row index: " << remainingRowIndex.size() << "\n";
		//std::cout<<"rows: "<<cut<<" "<<cop<<"\n";
	return;
}



void HODLR_Matrix::rook_Piv(int nRowStart, int nColStart, int nRows, int nCols, double tolerance, Eigen::MatrixXd& L, Eigen::MatrixXd& R, int& computedRank) { // 4 loops are vectorized
	

	//std::cout<<"non Parallel rook - piv "<<nRows<<"\n"; 
	std::vector<int> rowIndex;		///	This stores the row indices, which have already been used.
	std::vector<int> colIndex;		///	This stores the column indices, which have already been used.
	std::set<int> remainingRowIndex;/// Remaining row indicies
	std::set<int> remainingColIndex;/// Remaining row indicies
	std::vector<Eigen::VectorXd> u;	///	Stores the column basis.
	std::vector<Eigen::VectorXd> v;	///	Stores the row basis.

/*	for (int k=0; k<nRows; ++k) {
		remainingRowIndex.insert(k);
	}
*/	//double st = omp_get_wtime();
int k;
	for (k = 0; k < nRows; ++k) 
    		remainingRowIndex.insert(remainingRowIndex.end(), k);
	for (k = 0; k < nCols; ++k) 
    		remainingColIndex.insert(remainingColIndex.end(), k);
	
/*	for (int k=0; k<nCols; ++k) {
		remainingColIndex.insert(k);
	}
*/
	srand (time(NULL));
	double max, Gamma, unused_max;

	/*  INITIALIZATION  */

	/// Initialize the matrix norm and the the first row index
	double matrix_Norm  =   0;
	rowIndex.push_back(0);
	remainingRowIndex.erase(0);

	int pivot;

	computedRank   =   0;

	Eigen::VectorXd a, row, col;

	double row_Squared_Norm, row_Norm, col_Squared_Norm, col_Norm;

	int max_tries  =   10;

	int count;

	/// Repeat till the desired tolerance is obtained
	do {
		/// Generation of the row
		/// Row of the residuum and the pivot column
		// row =   A.row(rowIndex.back());
		row	=	get_Row(nRowStart+rowIndex.back(), nColStart, nCols);
		for (int l=0; l<computedRank; ++l) {		//comprank = 0 ;
			row =   row-u[l](rowIndex.back())*v[l];
		}

		pivot   =   max_Abs_Vector(row, remainingColIndex, max);


		count	=   0;

		/// This randomization is needed if in the middle of the algorithm the row happens to be exactly the linear combination of the previous rows upto some tolerance.
		while (fabs(max)<tolerance && count < max_tries && remainingColIndex.size() >0 && remainingRowIndex.size() >0) {
			rowIndex.pop_back();
			int new_rowIndex	=	*remainingRowIndex.begin();
			rowIndex.push_back(new_rowIndex);
			remainingRowIndex.erase(new_rowIndex);

			/// Generation of the row
			// a	=	A.row(new_rowIndex);
			a	=	get_Row(nRowStart+new_rowIndex, nColStart, nCols); //moving to next row of the block
			/// Row of the residuum and the pivot column
			row =   a;

			for (int l=0; l<computedRank; ++l) {
				row =   row-u[l](rowIndex.back())*v[l];
			}
			pivot   =   max_Abs_Vector(row, remainingColIndex, max);
			++count;
		}

		if (count == max_tries || remainingColIndex.size() == 0 || remainingRowIndex.size() == 0) break;

		count = 0;

		colIndex.push_back(pivot);
		remainingColIndex.erase(pivot);

		/// Normalizing constant
		Gamma   =   1.0/max;

		/// Generation of the column
		// a	=	A.col(colIndex.back());
		a	=	get_Col(nColStart+colIndex.back(), nRowStart, nRows);
		/// Column of the residuum and the pivot row
		col =   a;
		for (int l=0; l<computedRank; ++l) {
			col =   col-v[l](colIndex.back())*u[l];
		}
		pivot   =   max_Abs_Vector(col, remainingRowIndex, unused_max);

		/// This randomization is needed if in the middle of the algorithm the columns happens to be exactly the linear combination of the previous columns.
		while (fabs(max)<tolerance && count < max_tries && remainingColIndex.size() >0 && remainingRowIndex.size() >0) {
			colIndex.pop_back();
			int new_colIndex	=	*remainingColIndex.begin();
			colIndex.push_back(new_colIndex);
			remainingColIndex.erase(new_colIndex);

			/// Generation of the column
			// a	=	A.col(new_colIndex);
			a	=	get_Col(nColStart+new_colIndex, nRowStart, nRows);

			/// Column of the residuum and the pivot row
			col =   a;
			for (int l=0; l<computedRank; ++l) {
				col =   col-u[l](colIndex.back())*v[l];
			}
			pivot   =   max_Abs_Vector(col, remainingRowIndex, unused_max);
			++count;
			//std::cout << count << "\n";
		}

		if (count == max_tries || remainingColIndex.size() == 0 || remainingRowIndex.size() == 0) break;

		count = 0;

		rowIndex.push_back(pivot);
		remainingRowIndex.erase(pivot);

		/// New vectors
		u.push_back(Gamma*col);
		v.push_back(row);

		/// New approximation of matrix norm
		row_Squared_Norm    =   row.squaredNorm();
		row_Norm            =   sqrt(row_Squared_Norm);

		col_Squared_Norm    =   col.squaredNorm();
		col_Norm            =   sqrt(col_Squared_Norm);

		matrix_Norm         =   matrix_Norm +   Gamma*Gamma*row_Squared_Norm*col_Squared_Norm;

		for (int j=0; j<computedRank; ++j) {
			matrix_Norm     =   matrix_Norm +   2.0*(u[j].dot(u.back()))*(v[j].dot(v.back()));
		}
		++computedRank;
	} while (computedRank*(nRows+nCols)*row_Norm*col_Norm > fabs(max)*tolerance*matrix_Norm && computedRank < fmin(nRows, nCols));

	/// If the computedRank is close to full-rank then return the trivial full-rank decomposition

	if (computedRank>=fmin(nRows, nCols)-1) {
		if (nRows < nCols) {
			L   =   Eigen::MatrixXd::Identity(nRows,nRows);
			R	=	get_Matrix(nRowStart, nColStart, nRows, nCols).transpose();
			// V	=	A.transpose();
			computedRank   =   nRows;
		}
		else {
			// U	=	A;
			L	=	get_Matrix(nRowStart, nColStart, nRows, nCols);
			R   =   Eigen::MatrixXd::Identity(nCols,nCols);
			computedRank   =   nCols;
		}
	}
	else {
		L   =   Eigen::MatrixXd(nRows,computedRank);
		R   =   Eigen::MatrixXd(nCols,computedRank);
		for (int j=0; j<computedRank; ++j) {
			L.col(j)    =   u[j];
			R.col(j)    =   v[j];
		}
	}
	//double e = omp_get_wtime();
	//std::cout<<"time for rook piv "<<e-st<<" for  N is  "<<nRows<<" "<<nCols<<"\n";
	// std::cout << "Size of row index: " << rowIndex.size() << "\n";
	// std::cout << "Size of remaining row index: " << remainingRowIndex.size() << "\n";
	return;
}


#endif /*__HODLR_Matrix__*/
