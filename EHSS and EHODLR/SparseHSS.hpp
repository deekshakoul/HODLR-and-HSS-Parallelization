#ifndef __SparseHSS__
#define __SparseHSS__
#include<stdio.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include<Eigen/StdVector>
#include <vector>
#include <algorithm> 
#include "chebNodes.hpp"
#include <math.h> 
#include<Eigen/SparseLU>
#include <Eigen/OrderingMethods>
#include<iomanip>
#include<omp.h>

class SparseHSS{
	private:
		int N,nlevels,nleaves,leafsize,Asize;
		double xradius;
		chebNodes* myChebNodes;	
		typedef Eigen::Triplet<double> T; 
		std::vector<T>tripleList; 
		void assembly_leaf_level(int k,double xcenter);
	
	public:	
		SparseHSS( int N , int r, Eigen::VectorXd x, chebNodes* myChebNodes);
		~SparseHSS();
		int r;
		Eigen::VectorXd x;
		void solution();
};

SparseHSS::SparseHSS(int N, int r, Eigen::VectorXd x, chebNodes* myChebNodes){
 	this->N = N;
	this->r = r;
	this->x = x;
	this->myChebNodes = myChebNodes;
	nlevels = floor(log2(N/r));
	nleaves =   exp2(nlevels); 
	leafsize =   N/nleaves;
	xradius = 1.0/nleaves;	
	Asize = 3*N-4*r;
}



void SparseHSS::solution(){
		//int N =exp2(r);
	std::cout<<"System Size,N: "<<N<<" "<<" rank,r: "<<r<<"\n";
	std::cout<<"levels: "<<nlevels<<"\n";
//		int leafsize =   N/nleaves;	//this must be integer implying N must be in form of multiple of power of 2^(nlevels).
	Eigen::VectorXd xcenter(nleaves);			
	for(int i=1;i<=nleaves;i++){
		xcenter(i-1) = 2*i-1;
	}
	xcenter = xcenter*xradius; 
	xcenter = xcenter.array() - 1;
	Eigen::SparseMatrix<double> A(Asize,Asize);	
	for(int k=0 ; k<nleaves;k++){		
		assembly_leaf_level(k,xcenter(k));
	}
	
	Eigen::VectorXd radius(nlevels);	
	for(int i =0;i<nlevels;i++)
		radius(i) = 1.0/exp2(i+1);               

	Eigen::MatrixXd xleftnode = Eigen::MatrixXd::Zero(r,nlevels); 		
	Eigen::MatrixXd xrightnode = Eigen::MatrixXd::Zero(r,nlevels);
	std::map<int,Eigen::MatrixXd> mapp;
	for(int n = 0; n<nlevels;n++){
        	xleftnode.col(n)  =  myChebNodes->getscaledchebnodes(-radius(n),radius(n));
		xrightnode.col(n) = -xleftnode.col(n);
		std::sort(xrightnode.col(n).data(), xrightnode.col(n).data() + xrightnode.rows());
        	mapp[n] =  myChebNodes->Kernel(xleftnode.block(0,n,r,1),xrightnode.block(0,n,r,1));
    	}
	
	Eigen::VectorXd levelindex(nlevels); //levelindex is the size of each big block corresponding to a level
        for(int n=0;n<nlevels;n++)
		levelindex(n) = 5*exp2(n)*r;
		
	levelindex(0) = levelindex(0) - r;
	Eigen::VectorXd levelstart = Eigen::VectorXd::Zero(nlevels);
	Eigen::VectorXd levelend = Eigen::VectorXd::Zero(nlevels);	
	levelstart(nlevels-1) = N+1; 
	levelend(nlevels-1) = N+levelstart(nlevels-1);
	for (int n = nlevels-2;n>=0;n--){
		levelstart(n) = levelstart(n+1) + 2*exp2(n+2)*r;
		levelend(n) =   levelstart(n) - 1 + levelindex(n);
	}

	Eigen::MatrixXd transfer = myChebNodes->childparenttransfer();
	Eigen::MatrixXd transfer_transpose = transfer.transpose();

	//Assembly at higher levels
	for(int n=nlevels;n>=1;n--){
		int size = exp2(n)*r;	
		Eigen::VectorXd rowindex(size),colindex(size);
		for(int i=0;i<size;i++){
			rowindex(i) = i;
			colindex(i) = i+size;			
		}
		rowindex = rowindex.array() + levelstart(n-1) - 1;
		colindex = colindex.array() + levelstart(n-1) - 1; 
		//Assembling the identity matrix
		Eigen::MatrixXd identity_mat = Eigen::MatrixXd::Identity(size,size); 
		identity_mat.diagonal() = -identity_mat.diagonal();
 
		int rows = A.rows();
		int cols = A.cols();
			
		if(rowindex(size-1)>A.rows()){
			int diff = rowindex(size-1) - A.rows();
			rows = rows+diff;
			cols = cols+diff;
			A.conservativeResize(rows,cols);
		}			
			
		for(int i=rowindex(0);i<=rowindex(size-1);i++)
			for(int j=colindex(0);j<=colindex(size-1);j++)
				tripleList.push_back(T(i,j,identity_mat(i-rowindex(0),j-colindex(0))));

		if(colindex(size-1)>A.rows()){
			int diff = colindex(size-1) - A.rows();
			rows = rows+diff+1;
			cols = cols+diff+1;
			A.conservativeResize(colindex(size-1)+1,colindex(size-1)+1);
		}
	
		for(int i=colindex(0);i<=colindex(size-1);i++)
			for(int j=rowindex(0);j<=rowindex(size-1);j++)
				tripleList.push_back(T(i,j,identity_mat(i-colindex(0),j-rowindex(0))));		

		rowindex.resize(r);
		colindex.resize(r);
		for(int i = 0;i<r;i++)
			rowindex(i) = i;
		rowindex = rowindex.array() + levelstart(n-1) + size;
		colindex = rowindex; 
		Eigen::MatrixXd mat = mapp[n-1];
		Eigen::MatrixXd mat_trans = mat.transpose();
		//Assembling the interaction				
		for(int k=0;k<=exp2(n-1)-1;k++){
			rowindex = rowindex.array() + 2*k*r -1;
			colindex = colindex.array() + (2*k+1)*r -1; 
			
			if(rowindex(rowindex.size()-1)>A.rows()){
				int diff = rowindex(rowindex.size()-1) - A.rows();
				rows = rows+diff;
				cols = cols+diff;
				A.conservativeResize(rowindex(size-1)+1,rowindex(size-1)+1);
			}
			for(int i=rowindex(0);i<=rowindex(rowindex.size()-1);i++)
				for(int j=colindex(0);j<=colindex(colindex.size()-1);j++)
					tripleList.push_back(T(i,j,mat(i-rowindex(0),j-colindex(0))));
		
			if(colindex(colindex.size()-1)>A.rows()){
				int diff = colindex(colindex.size()-1) - A.rows();
				rows = rows+diff;
				cols = cols+diff;
				A.conservativeResize(colindex(size-1)+1,colindex(size-1)+1);
			}
			for(int i=colindex(0);i<=colindex(colindex.size()-1);i++)
				for(int j=rowindex(0);j<=rowindex(rowindex.size()-1);j++)
					tripleList.push_back(T(i,j,mat_trans(i-colindex(0),j-rowindex(0))));
		
			rowindex = rowindex.array() - 2*k*r +1;
			colindex = colindex.array() - (2*k+1)*r +1;  
		}

		if(n==1)
			break;
		
		int size1 = r*2;
		rowindex.resize(size1);
		for(int i=0;i<r;i++){
			rowindex(i) = i;
			colindex(i) = i;			
		}
		//Assembling the transfer
		for(int i=r;i<size1;i++)
			rowindex(i) = i;
			rowindex = rowindex.array() + levelstart(n-1) + size;
			colindex = colindex.array() + levelstart(n-1) + size*2;
			for(int k=0;k<exp2(n-1);k++){
				rowindex = rowindex.array() + 2*k*r -1;
				colindex = colindex.array() + k*r -1;
				int row_diff = rowindex(rowindex.size()-1)-A.rows(); 
				int col_diff = colindex(colindex.size()-1)-A.cols();
				if(row_diff>0){
				A.conservativeResize(A.rows()+row_diff+1,A.rows()+row_diff+1);
				}

				if(col_diff>0){
				A.conservativeResize(A.cols()+col_diff+1,A.cols()+col_diff+1);
				}
				for(int i=rowindex(0);i<=rowindex(rowindex.size()-1);i++)
					for(int j=colindex(0);j<=colindex(colindex.size()-1);j++)
						tripleList.push_back(T(i,j,transfer(i-rowindex(0),j-colindex(0)))); 

				row_diff = colindex(colindex.size()-1)-A.rows(); 
				col_diff = rowindex(rowindex.size()-1)-A.cols();
				
				if(row_diff>0){
				A.conservativeResize(A.cols()+row_diff+1,A.rows()+row_diff+1);
				}

				if(col_diff>0){
				A.conservativeResize(A.cols()+col_diff+1,A.cols()+col_diff+1);
				}
				for(int i=colindex(0);i<=colindex(colindex.size()-1);i++)
					for(int j=rowindex(0);j<=rowindex(rowindex.size()-1);j++)
						tripleList.push_back(T(i,j,transfer_transpose(i-colindex(0),j-rowindex(0)))); 
				rowindex = rowindex.array() - 2*k*r +1;
				colindex = colindex.array() - k*r +1;

			}
		}	

	A.setFromTriplets(tripleList.begin(),tripleList.end());

//std::cout<<"the number of nonzeros with  corrected  comparison: "<< (Eigen::Map<Eigen::VectorXd> (A.valuePtr(), A.nonZeros()).array() != 0.0).count()<<"\n";

	Eigen::MatrixXd K = myChebNodes->Kernel(x,x);
	Eigen::VectorXd solexact = Eigen::VectorXd::Random(N,1); 	
	solexact = (solexact.array()+1.0)/2.0 ;
	Eigen::VectorXd b = myChebNodes->calc_old_error(x,solexact,N); //checkc
	Eigen::VectorXd b_cat = Eigen::MatrixXd::Zero(A.rows()-N,1);
	Eigen::VectorXd bnew(b.rows()+b_cat.rows()); //Asixe x 1 
	bnew<<b,b_cat;
	Eigen::SparseLU<Eigen::SparseMatrix<double> > solverA;
	A.makeCompressed();
	double in1 = omp_get_wtime();
	solverA.analyzePattern(A);
	solverA.factorize(A);
	Eigen::VectorXd solnew = solverA.solve(bnew);
	double in2 = omp_get_wtime();
	std::cout<<"total time taken for sparse, AX=B  : "<<in2-in1<<"\n";	
	Eigen::VectorXd num_norm = solnew.segment(0,N).array() - solexact.array() ; 		
	double err = (solnew.segment(0,N)-solexact).norm()/solexact.norm(); 	
	std::cout<<"New Error in the solution is: "<<err<<"\n";
}

void SparseHSS::assembly_leaf_level(int k,double xcenter){
	Eigen::MatrixXd mat,mat_kernel;
	Eigen::VectorXd index = myChebNodes->getindices(k+1,leafsize);
	Eigen::VectorXd indexr(r);
	int num = N+k*r;
	for(int i = num ; i<N+(k+1)*r; i++)
		indexr(i-num) = i;
	Eigen::VectorXd xvalues = myChebNodes->get_xvalues(index);
	mat_kernel = myChebNodes->Kernel(xvalues,xvalues);
	int num1 = index(0);
	for(int i=num1;i<=index(leafsize-1);i++)
		for(int j=num1;j<=index(leafsize-1);j++)
			tripleList.push_back(T(i,j,mat_kernel(i-num1,j-num1)));
	mat = myChebNodes->Chebyshevinterpolants(xvalues,xcenter,xradius);
	for(int i=num1;i<=index(leafsize-1);i++)
		for(int j=num;j<=indexr(r-1);j++)
			tripleList.push_back(T(i,j,mat(i-num1,j-num)));
	mat = mat.transpose().eval();
	for(int j=num;j<=indexr(r-1);j++)
		for(int i=num1;i<=index(leafsize-1);i++)
			tripleList.push_back(T(j,i,mat(j-num,i-num1)));
}
#endif
