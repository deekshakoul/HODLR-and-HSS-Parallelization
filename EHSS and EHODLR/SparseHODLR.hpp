#ifndef __SparseHODLR__
#define __SparseHODLR__
#include<stdio.h>
#include <iostream>
#include <cmath>
//#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include<Eigen/StdVector>
#include <vector>
#include <algorithm> 
#include "chebNodes.hpp"
#include <math.h> 
#include<Eigen/SparseLU>
//#include <Eigen/OrderingMethods>
#include<iomanip>
#include<omp.h>

class SparseHODLR{
	private:
		int N,nlevels,nleaves,leafsize,Asize;
		double inv_leaf;
		chebNodes* myChebNodes;	
		typedef Eigen::Triplet<double> T; 
		std::vector<T>tripleList; 
		void assembly_leaf_level(int k,double xcenter);
		void assembly_clusters(int k,int clustersize, int N1,double xcenter_cluster,double inv_cluster);
		void assembly_identity(int NN, int count);
	public:	
		SparseHODLR( int N , int r, Eigen::VectorXd x, chebNodes* myChebNodes);
		~SparseHODLR();
		int r;
		Eigen::VectorXd x;
		void solution();

};

SparseHODLR::SparseHODLR(int N, int r, Eigen::VectorXd x, chebNodes* myChebNodes){
	this->N = N;
	this->r = r;
	this->x = x;
	this->myChebNodes = myChebNodes;
	nlevels = floor(log2(N/r)-1);
	nleaves =   exp2(nlevels); 
	leafsize =   N/nleaves;
	inv_leaf = 1.0/nleaves;	
	Asize = N+2*r*(2*nleaves-2);
}
void SparseHODLR::solution(){
	std::cout<<"\nSystem Size,N: "<<N<<" "<<" rank,r: "<<r<<"\n";
	std::cout<<"nlevels: "<<nlevels<<"\n";
	Eigen::VectorXd xcenter(nleaves),xradius(nlevels+1);   				
	Eigen::VectorXd nend = Eigen::VectorXd::Zero(nlevels) ;
	//otherNodes* myNodes = new otherNodes(); 
	   
	for(int i =0;i<=nlevels;i++){
		xradius(i) = 1.0/exp2(i);               
	} 

	for(int i=1;i<=nleaves;i++){
		xcenter(i-1) = 2*i-1;
	} 

	xcenter = xcenter/nleaves; 			
	xcenter = xcenter.array() - 1;
	nend(nlevels-1)       =   N+2*r*nleaves;
	std::map<int,Eigen::MatrixXd> mapp; 
	Eigen::SparseMatrix<double> A(Asize,Asize);	

	Eigen::MatrixXd xleftnode(r,nlevels),xrightnode(r,nlevels);   
	for(int n = 0; n<nlevels;n++){
       		xleftnode.col(n)  =  myChebNodes->getscaledchebnodes(-xradius(n+1),xradius(n+1));
		xrightnode.col(n) = -xleftnode.col(n);
		std::sort(xrightnode.col(n).data(), xrightnode.col(n).data() + xrightnode.rows());
       		mapp[n] =  myChebNodes->Kernel(xleftnode.block(0,n,r,1),xrightnode.block(0,n,r,1));
	}

	for(int k=0;k<nleaves;k++){
		assembly_leaf_level(k,xcenter(k));			
	}

	assembly_identity(N,nleaves);	
			
	Eigen::MatrixXd mat = myChebNodes->Kernel(xleftnode.block(0,nlevels-1,r,1),xrightnode.block(0,nlevels-1,r,1));		
	for(int k=0;k<nleaves;k=k+2){
		int num1 = 	N+r*nleaves+k*r;
		int num2 = N+r*nleaves+(k+1)*r;
		for(int i = N+r*nleaves+k*r;i<N+r*nleaves+(k+1)*r;i++)
			for(int j = N+r*nleaves+(k+1)*r; j<N+r*nleaves+(k+2)*r; j++ )	
				tripleList.push_back(T(i,j,mat(i-num1,j-num2)));
		Eigen::MatrixXd mat_trans = mat.adjoint();
		for(int j = N+r*nleaves+(k+1)*r; j<N+r*nleaves+(k+2)*r; j++ ) 	
			for(int i = N+r*nleaves+k*r;i<N+r*nleaves+(k+1)*r;i++)					
				tripleList.push_back(T(j,i,mat_trans(j-num2,i-num1)));
	} 	


	for(int n = nlevels-1;n>=1;n-- ){
		int nclusters = exp2(n);
		int clustersize = N/nclusters;
		double inv_cluster = 1.0/nclusters; 
		nend(n-1) = nend(n) + 2*r*nclusters; 
		int N1 =  nend(n);
		Eigen::VectorXd xcenter_cluster(nclusters);
		
		for(int i=1;i<=nclusters;i++)
			xcenter_cluster(i-1) = 2*i-1;
		xcenter_cluster = xcenter_cluster/nclusters;
		xcenter_cluster = xcenter_cluster.array() - 1;
		Eigen::MatrixXd index_cluster(nclusters,clustersize);
		for(int k=0;k<nclusters;k++){
			assembly_clusters(k,clustersize, N1 ,xcenter_cluster(k), inv_cluster);
		}

		assembly_identity(N1,nclusters);
		Eigen::MatrixXd mat = mapp[n-1];
		Eigen::MatrixXd mat_trans = mat.adjoint();
		for(int k=0;k<nclusters;k=k+2){
			int num1 = 	N1+r*nclusters+k*r;
			int num2 = N1+r*nclusters+(k+1)*r;
			for(int i = N1+r*nclusters+k*r;i<N1+r*nclusters+(k+1)*r;i++)
				for(int j = N1+r*nclusters+(k+1)*r; j<N1+r*nclusters+(k+2)*r; j++ )	
					tripleList.push_back(T(i,j,mat(i-num1,j-num2)));
			for(int j = N1+r*nclusters+(k+1)*r; j<N1+r*nclusters+(k+2)*r; j++ ) 	
				for(int i = N1+r*nclusters+k*r;i<N1+r*nclusters+(k+1)*r;i++)					
					tripleList.push_back(T(j,i,mat_trans(j-num2,i-num1)));
		} 
	}


		A.setFromTriplets(tripleList.begin(),tripleList.end());


		//Printing Column wise Sparse Matrix
		/*for (int k=0; k < A.outerSize(); ++k)
		{
		    for (Eigen::SparseMatrix<double>::InnerIterator it(A,k); it; ++it)
		    {
		 	if(it.value()== 0){}

			else	{
			std::cout << it.row()+1<<" "<<it.col()+1<<" "<<it.value()<<"\n";	
			}
		    }
		}
		*/

//std::cout<<"Frobenius Norm of sparse matrix A after  all changes are completed: "<<A.norm()<<"\n";
//std::cout<<"the number of nonzeros with  corrected  comparison: "<< (Eigen::Map<Eigen::VectorXd> (A.valuePtr(), A.nonZeros()).array() != 0.0).count()<<"\n";

	Eigen::VectorXd solexact = Eigen::VectorXd::Random(N,1);
	solexact = ((solexact.array()+1.0)/2.0);
	Eigen::VectorXd b = myChebNodes->calc_old_error(x,solexact,N);
	Eigen::VectorXd b_cat = Eigen::MatrixXd::Zero(Asize-N,1);
	Eigen::VectorXd bnew(b.rows()+b_cat.rows());  
	bnew<<b,b_cat;

	A.makeCompressed();
	Eigen::SparseLU <Eigen::SparseMatrix<double> > solverA;
	double in1 = omp_get_wtime();
	solverA.analyzePattern(A);
	solverA.factorize(A);
	//solverA.info()!=Eigen::Success can be used to check working of compute and solve method.
	Eigen::VectorXd solnew = solverA.solve(bnew);
	double in2 = omp_get_wtime();
	std::cout<<"total time taken for sparse, AX=B  : "<<in2-in1<<"\n";		
	Eigen::VectorXd num_norm = solnew.segment(0,N)-solexact; 
	double err = (solnew.segment(0,N)-solexact).norm()/solexact.norm(); 	
	std::cout<<"New Error in the solution is: "<<err<<"\n";

}
void SparseHODLR::assembly_leaf_level(int k,double xcenter){ 
	Eigen::VectorXd index = myChebNodes->getindices(k+1,leafsize);
	Eigen::VectorXd xvalues = myChebNodes->get_xvalues(index);
	Eigen::MatrixXd mat,mat_cheby; 
	mat =  myChebNodes->Kernel(xvalues,xvalues);
	int num1 = index(0);
	for(int i=num1;i<=index(leafsize-1);i++)
		for(int j=num1;j<=index(leafsize-1);j++)
			tripleList.push_back(T(i,j,mat(i-num1,j-num1)));

	mat_cheby = myChebNodes->Chebyshevinterpolants(xvalues,xcenter,inv_leaf);		
	int num2 = N+k*r;
	for(int i = num1; i<=index(leafsize-1);i++)
		for(int j = num2; j<N+(k+1)*r;j++)	
			tripleList.push_back(T(i,j,mat_cheby(i-num1,j-num2)));					
	
	mat_cheby = mat_cheby.adjoint().eval();  
	for(int i = N+k*r; i<N+(k+1)*r;i++)
		for(int j = num1; j<=index(leafsize-1);j++) 
			tripleList.push_back(T(i,j,mat_cheby(i-num2,j-num1)));	
}

void SparseHODLR::assembly_clusters(int k,int clustersize, int N1,double xcenter_cluster,double inv_cluster){
	Eigen::VectorXd index_cluster = myChebNodes->getindices(k+1,clustersize); 
	Eigen::VectorXd xvalues = myChebNodes->get_xvalues(index_cluster);
	Eigen::MatrixXd mat = myChebNodes->Chebyshevinterpolants(xvalues,xcenter_cluster,inv_cluster);				
	int num1 = index_cluster(0),num2 = N1+k*r;
	for(int i = index_cluster(0); i<=index_cluster(clustersize-1);i++)
		for(int j =N1+k*r; j<N1+(k+1)*r;j++)	
			tripleList.push_back(T(i,j,mat(i-num1,j-num2)));					
	
	mat = mat.adjoint().eval();
	for(int i = N1+k*r; i<N1+(k+1)*r;i++)
		for(int j = num1; j<=index_cluster(clustersize-1);j++) 
			tripleList.push_back(T(i,j,mat(i-num2,j-num1)));
}
void SparseHODLR::assembly_identity(int NN, int count){
	Eigen::MatrixXd identity_mat = Eigen::MatrixXd::Identity(r*count,r*count); 
	int num1=NN,num2=NN+r*count;
	for(int i = num1;i<num2;i++)						
		for(int j=num2;j<(NN+2*r*count);j++)
			tripleList.push_back(T(i,j,identity_mat(i-num1,j-num2)));
	for(int i=num2;i<=(NN+2*r*count - 1);i++)
		for(int j = num1;j<=(num2 - 1);j++)
			tripleList.push_back(T(i,j,identity_mat(i-num2,j-num1)));
}
#endif
