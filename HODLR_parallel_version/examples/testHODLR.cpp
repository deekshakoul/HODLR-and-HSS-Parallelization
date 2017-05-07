#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include "HODLR_Tree.hpp"
#include "HODLR_Matrix.hpp"




int main(int argc, char* argv[]) {
	double start_time = omp_get_wtime();	
	
	int N	= atoi(argv[1]);
	myHODLR_Matrix* A	=	new myHODLR_Matrix(N);
	int M	=	atoi(argv[2]);
	int  level	=	N /M ; 
	const int nLevels	=	log2(level);
	//std::cout<<"first argument"<<level;
	//std::cout<<"second argument"<<;
	//std::cout<<"third argument"<<N;

	std::cout<<"nlevels are: "<<nLevels<<"\n";
	
	const double tolerance	=	1e-13;
	double start, end;
	double CPS	=	1.0;
	 
	//std::cout << "\nFast method...\n";
	
	start	=	omp_get_wtime();
	HODLR_Tree* myMatrix	=	new HODLR_Tree(nLevels, tolerance, A);
	Eigen::MatrixXd x	=	Eigen::MatrixXd::Random(N,1);
	Eigen::MatrixXd bFast;
	myMatrix->assemble_Tree();
	end		=	omp_get_wtime();
        std::cout << "Time taken for assembling the matrix in HODLR form is: " << (end-start)/CPS << "\n";
	

	start	=	omp_get_wtime();
	myMatrix->matmat_Product(x, bFast);
	end	=	omp_get_wtime();
	std::cout << "Time for fast matrix-vector product is: " << (end-start)/CPS << "\n"; //0.0002

	Eigen::MatrixXd xFast;
	start	=	omp_get_wtime();
	myMatrix->factorize();
	end		=	omp_get_wtime();
	std::cout << "Time taken to factorize is: " << (end-start)/CPS << "\n";
//std::cout << (end-start)/CPS << "\n";

	start	=	omp_get_wtime();
	xFast	=	myMatrix->solve(bFast);
	end	=	omp_get_wtime();
	std::cout << "Time taken to solve is: " << (end-start)/CPS << "\n";
	std::cout << "Error in the solution is: " << (xFast-x).norm()/*/(1.0+x.norm())*/ << "\n";
	
	double end_time	=	omp_get_wtime();
	double time_taken =	end_time - start_time;
	std::cout<<"Net Time Taken: "<< time_taken << "\n\n" ;
	//std::cout<<time_taken<<"\n";
	//std::cout<<(xFast-x).norm()<<"\n";
return 0 ; 
}
