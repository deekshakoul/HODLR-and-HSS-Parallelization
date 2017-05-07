#ifndef __chebNodes__
#define __chebNodes__
#include <Eigen/Dense>
#include<Eigen/StdVector>
#include <vector>
#include <iostream>
#define pi 3.141592653589793
Eigen::VectorXd standardchebnodes(12);
class chebNodes{
	private:
		int r;
		Eigen::VectorXd x;
		//Eigen::VectorXd getstandardchebnodes();


	public:
		chebNodes( int r , Eigen::VectorXd x);
		~chebNodes();
		Eigen::VectorXd getscaledchebnodes(double center, double radius);
		Eigen::VectorXd	getindices(int k , int leaf);	
		Eigen::VectorXd	get_xvalues(Eigen::VectorXd indices);
		Eigen::MatrixXd Kernel(Eigen::VectorXd x, Eigen::VectorXd y);
		Eigen::MatrixXd Chebyshevpolynomials(Eigen::VectorXd xx);
		Eigen::MatrixXd Chebyshevinterpolants(Eigen::VectorXd xx, double xcenter, double xradius);
		Eigen::MatrixXd getinteraction(double xcenter,double xradius,double ycenter,double yradius);		
		Eigen::MatrixXd childparenttransfer();
		Eigen::VectorXd calc_old_error(Eigen::VectorXd x, Eigen::VectorXd solexact, int N);		
		
};

chebNodes::chebNodes( int r,Eigen::VectorXd x){
	this->r = r;
	this->x = x; 
	standardchebnodes.resize(r);
	for(int i=1 ; i<r+1 ; i++){
		standardchebnodes(i-1)   =   -cos((2*i-1)*pi/2.0/r);  //total number of itr = r : returns same values for a similar r 
	}

}

	
Eigen::VectorXd	chebNodes::getscaledchebnodes(double center, double radius){
		return center + radius*standardchebnodes.array();
}



Eigen::VectorXd	chebNodes::getindices(int k , int leaf){
	Eigen::VectorXd req_column(leaf);	
	req_column(0) = (k-1)*leaf ;
	for(int i=1 ;i<leaf;i++)
		req_column(i) = req_column(i-1)+ 1;
	
	return req_column;
}



Eigen::VectorXd	chebNodes::get_xvalues(Eigen::VectorXd indices){
	Eigen::VectorXd x_values(indices.size());
	for(int i =0;i<indices.size();i++)
	x_values(i) = x(indices(i));
	
	return x_values;
}


Eigen::MatrixXd chebNodes::Kernel(Eigen::VectorXd xx, Eigen::VectorXd y){
	int m = xx.size();
	int n = y.size();
	Eigen::MatrixXd z(m,n),K(m,n);   //vector is already 32 x 1 

	z = (xx*Eigen::MatrixXd::Ones(1,n)) - (Eigen::MatrixXd::Ones(m,1)*(y.transpose()));
	z = z.array().abs(); 
	K = (-(z.array().square())).exp();

//find indices of z where z=0 then make k=0 at that index

	for(int i=0;i<m;i++)
		for(int j=0;j<n;j++){
			if(z(i,j)==0)
				K(i,j)=0;
		}

	return K;	
}


Eigen::MatrixXd chebNodes::Chebyshevpolynomials(Eigen::VectorXd xx){
	int n = xx.size(); Eigen::VectorXd x1(n);
	Eigen::MatrixXd T = Eigen::MatrixXd::Ones(n,r);
	if(r>=2)
		T.col(1) = xx;   //dont change the rank 
	if(r>=3)
		for(int k=1;k<=r-2;k++){
			//x1 = xx*2;	
			T.col(k+1) = (2*xx.array()*T.col(k).array()).matrix() - T.col(k-1);  //ELEMENT WISE Multilply  T.col(k+1) = (2*xx.array()*T.col(k).array())- T.col(k-1);
//P.cwiseProduct(Q) = P.*Q
			}
	return T;	

}

Eigen::MatrixXd chebNodes::Chebyshevinterpolants(Eigen::VectorXd xx, double xcenter, double xradius){   //correct 
	int n = xx.size();
	Eigen::MatrixXd S(n,r),Tx,Tcheb;
	Eigen::VectorXd xstandard   =   xx.array()-xcenter;
	xstandard = xstandard/xradius; 
	Tx          =   Chebyshevpolynomials(xstandard);
	Tcheb       =   Chebyshevpolynomials(standardchebnodes);
	S           =   ((2*(Tx*Tcheb.transpose())).array()-1)/r;
	return S;
}

Eigen::MatrixXd chebNodes::getinteraction(double xcenter,double xradius,double ycenter,double yradius){
	Eigen::VectorXd xscaledchebnodes = getscaledchebnodes(xcenter,xradius);
	Eigen::VectorXd yscaledchebnodes = getscaledchebnodes(ycenter,yradius);
	return Kernel(xscaledchebnodes,yscaledchebnodes);
}


Eigen::MatrixXd chebNodes::childparenttransfer(){
	Eigen::VectorXd chebnode = standardchebnodes;
	chebnode = 0.5*chebnode;
	Eigen::VectorXd leftchebnode    =   chebnode.array() - 0.5;
	Eigen::VectorXd rightchebnode   =   chebnode.array() + 0.5;
	std::sort(leftchebnode.data(), leftchebnode.data() + r);
	std::sort(rightchebnode.data(), rightchebnode.data() + r);
	Eigen::VectorXd vec_joined(chebnode.size()*2);
	vec_joined<<leftchebnode,rightchebnode;
	return  Chebyshevinterpolants(vec_joined, 0, 1);	

}
Eigen::VectorXd chebNodes::calc_old_error(Eigen::VectorXd x, Eigen::VectorXd solexact, int N){
	Eigen::MatrixXd K = Kernel(x,x);	
	Eigen::VectorXd b = K*solexact;
	Eigen::PartialPivLU<Eigen::MatrixXd> solverK ;
	solverK.compute(K);
	double in2 = omp_get_wtime(); 
	Eigen::VectorXd sol = solverK.solve(b);
	double in1 = omp_get_wtime();
	std::cout<<"total time taken for KX=B: "<<in1-in2<<"\n";		
	double err = (sol.segment(0,N)-solexact).norm()/solexact.norm(); 	
	std::cout<<"old Error in the solution is: "<<err<<"\n";
	return b;
}

#endif
