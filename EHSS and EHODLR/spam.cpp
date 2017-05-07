#include<stdio.h>
#include<string.h>
#include "SparseHODLR.hpp"
#include "SparseHSS.hpp"
 
//extern "C" int main(int argc, char *argv[]){
extern "C" void main_a( char *arg1,char *arg2,char *arg3){
	int N = atoi(arg1);	
	int r = atoi(arg2);
	int option;
	std::string matrix_type = arg3;
	
	int xx = !matrix_type.compare("HODLR") ^ !matrix_type.compare("hodlr");
	int yy = !matrix_type.compare("HSS") ^ !matrix_type.compare("hss");

	Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(N,-1,1);	
	chebNodes* myChebNodes = new chebNodes(r,x);
 
	if(xx==1){
		SparseHODLR* hodlr = new SparseHODLR(N,r,x,myChebNodes);
		hodlr->solution();
	}

	if(yy==1){
		SparseHSS* hss = new SparseHSS(N,r,x,myChebNodes);
		hss->solution();
	} 

}




