#ifndef __HODLR_Tree__
#define __HODLR_Tree__
#define total_threads 128
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include "HODLR_Matrix.hpp"
#include "HODLR_Node.hpp"
#include<omp.h>
#include<algorithm>
class HODLR_Tree {
	private:
		int N;
		int nLevels;
		double tolerance;
		std::vector<int> nodesInLevel;
		std::vector<std::pair<int, int> >myPairs;
		HODLR_Matrix* A;
		std::vector< std::vector<HODLR_Node*> > tree;
		void createTree();
		void createRoot();
		void createChildren(int j, int k);


		//  Variables and methods needed for HODLR solver
		void factorize_Leaf(int k);
		void factorize_Non_Leaf(int j, int k);
		void factorize_Non_Leaf_p(int j, int k);
		Eigen::MatrixXd solve_Leaf(int k, Eigen::MatrixXd b);
		Eigen::MatrixXd solve_Non_Leaf(int j, int k, Eigen::MatrixXd b);


	public:
		HODLR_Tree(int nLevels, double tolerance, HODLR_Matrix* A);
		~HODLR_Tree();
		virtual double get_Matrix_Element(int j, int k) {
			return 0.0;
		}

		//  Methods for HODLR solver
		void assemble_Tree();
		void factorize();
		double determinant();
		void matmat_Product(Eigen::MatrixXd x, Eigen::MatrixXd& b);
		Eigen::MatrixXd solve(Eigen::MatrixXd b);

		};

/*******************************************************/
/*	PURPOSE OF EXISTENCE: Constructor for the class    */
/*******************************************************/

/************/
/*	INPUTS	*/
/************/

/// nLevels      -   Number of levels in the HODLR Tree
/// tolerance    -   Relative error for approximating the off-diagonal blocks
/// A            -   HODLR matrix for the kernel matrix

HODLR_Tree::HODLR_Tree(const int nLevels, const double tolerance, HODLR_Matrix* A) {
	// // std::cout << "\nStart HODLR_Tree\n";
	this->nLevels		=	nLevels;
	this->tolerance		=	tolerance;
	this->A			=	A;
	this->N			=	A->N;
	nodesInLevel.push_back(1);
	for (int j=1; j<=nLevels; ++j) {
		nodesInLevel.push_back(2*nodesInLevel.back());
	}

	for(int j=0;j<nLevels;j++){
		for(int i=0;i<nodesInLevel[j];i++){
			myPairs.push_back(std::make_pair(j,i));
		}
	}	

      //std::cout << "\nDone HODLR_Tree\n";
	createTree();
}

/*******************************************************/
/*	PURPOSE OF EXISTENCE: Builds the root of HODLR Tree*/
/*******************************************************/

void HODLR_Tree::createRoot() {
	// // std::cout << "\nStart createRoot\n";
	HODLR_Node* root	=	new HODLR_Node(0, 0, 0, 0, N, tolerance);
	std::vector<HODLR_Node*> level;
	level.push_back(root);
	tree.push_back(level);
	 //std::cout << "\nDone createRoot\n";
}

/******************************************************************************/
/*	PURPOSE OF EXISTENCE: Builds HODLR child nodes for HODLR nodes in the tree*/
/******************************************************************************/

/************/
/*	INPUTS	*/
/************/

/// j           -   Level number
/// k           -   Node number

void HODLR_Tree::createChildren(int j, int k) {
	// // std::cout << "\nStart createChildren\n";
	//	Adding left child
	HODLR_Node* left	=	new HODLR_Node(2*j, k+1, 0, tree[j][k]->cStart[0], tree[j][k]->cSize[0], tolerance);
	tree[j+1].push_back(left);

	//	Adding right child
	HODLR_Node* right	=	new HODLR_Node(2*j+1, k+1, 1, tree[j][k]->cStart[1], tree[j][k]->cSize[1], tolerance);
	tree[j+1].push_back(right);
	//std::cout << "\nDone createChildren\n";
}

/****************************************************************/
/*	PURPOSE OF EXISTENCE: Builds HODLR nodes for HODLR Matrix A */
/****************************************************************/

void HODLR_Tree::createTree() {
	//std::cout << "\nStart createTree\n";
	createRoot();
	
	for (int j=0; j<nLevels; ++j) { 
		std::vector<HODLR_Node*> level; //otherwise segmentation fault
		tree.push_back(level);
	}
std::vector<std::pair<int, int> >::iterator it;

#pragma omp parllel
{
       
	#pragma omp for private(it) schedule(static) 	
	for(it = myPairs.begin(); it < myPairs.end(); ++it) {
		createChildren(it->first,it->second);
	
	}

}


	//std::cout<<"End creating tree"<<"\n";


}

/************************************************************************************************************************************************/
/*	PURPOSE OF EXISTENCE: Obtains a factorization of the leaf nodes and computes the low rank approximations of the off-diagonal blocks, Z=UV'. */
/************************************************************************************************************************************************/
void HODLR_Tree::assemble_Tree() {
	//std::cout << "\nStart assemble_Tree\n";
	int threshold=5;
	int first_levels =nLevels-threshold;

	omp_set_dynamic(0);
	omp_set_nested(1);
	
	int threads_first_levels = 4; //no of threads 
 	int left_threads = (total_threads - threads_first_levels*first_levels)/threshold;

	#pragma omp parallel num_threads(nLevels) 
	{ 	

		if (omp_get_thread_num() <first_levels)
				omp_set_num_threads(threads_first_levels);
		else
				omp_set_num_threads(left_threads);

		int j = omp_get_thread_num();
		int itr = nodesInLevel[j];
	

	     #pragma omp parallel ///launches threads
		{	 
				#pragma omp for schedule(dynamic) nowait
				for (int i=0; i<itr; ++i) {
					int start =tree[j][i]->cStart[0];
					int end  = tree[j][i]->cStart[1];
					int size0   =tree[j][i]->cSize[0];
					int size1	=tree[j][i]->cSize[1];
				A->rook_Piv(start,end,size0,size1, tolerance, tree[j][i]->U[0], tree[j][i]->V[1],tree[j][i]->rank[0]);	
			 	} 
		

			       #pragma omp for schedule(guided) 
				for (int i=0; i<itr; ++i) {
					int start =tree[j][i]->cStart[0];
					int end  = tree[j][i]->cStart[1];
					int size0   =tree[j][i]->cSize[0];
					int size1	=tree[j][i]->cSize[1];
				A->rook_Piv(end,start,size1,size0, tolerance,tree[j][i]->U[1], tree[j][i]->V[0], tree[j][i]->rank[1]);
			 	}
		
		}

	}




const int itre = nodesInLevel[nLevels];
	#pragma omp parallel 
	{
		#pragma omp for
			for (int k=0; k<itre; ++k) {
				tree[nLevels][k]->assemble_Leaf_Node(A);
			}
	}




}


/*******************************************************/
/*	PURPOSE OF EXISTENCE:	 Matrix-matrix product     */
/*******************************************************/

/************/
/*	INPUTS	*/
/************/

/// x   -   Matrix to be multiplied on the right of the HODLR matrix

/************/
/*	OUTPUTS	*/
/************/

/// b   -   Matrix matrix product

void HODLR_Tree::matmat_Product(const Eigen::MatrixXd x, Eigen::MatrixXd& b) {
	// // std::cout << "\nStart matmat_Product\n";
	int r	=	x.cols();
	b 	=	Eigen::MatrixXd::Zero(N,r);
	for(int j=0;j<nLevels;j++){
		int itr = nodesInLevel[j];
		#pragma omp parallel 
		{	
			#pragma omp for schedule(dynamic) nowait
				for(int i=0 ; i<itr;i++){
		                        int start =tree[j][i]->cStart[0];
					int end  = tree[j][i]->cStart[1];
					int size0   =tree[j][i]->cSize[0];
					int size1	=tree[j][i]->cSize[1];
					b.block(start,0,size0,x.cols())+=(tree[j][i]->U[0]*(tree[j][i]->V[1].transpose()*x.block(end,0,size1,x.cols())));
				}

 	                #pragma omp for schedule(guided) 	
				for(int i=0 ; i<itr;i++){
		                        int start =tree[j][i]->cStart[0];
					int end  = tree[j][i]->cStart[1];
					int size0   =tree[j][i]->cSize[0];
					int size1	=tree[j][i]->cSize[1];
	    				b.block(end,0,size1,x.cols())+=(tree[j][i]->U[1]*(tree[j][i]->V[0].transpose()*x.block(start,0,size0,x.cols())));
				}				
	
 		}

	}  
	
	const int itre = nodesInLevel[nLevels]; 
	//int val = itre/total_threads;
	#pragma omp parallel 
	{
		#pragma omp for
			for (int k=0; k<itre; ++k) {
				int start =tree[nLevels][k]->nStart;	
				int size =tree[nLevels][k]->nSize;
			        b.block(start,0,size,x.cols())+=tree[nLevels][k]->K*x.block(start,0,size,x.cols());
			}
	}

}


/****************************************************************************************/
/*	PURPOSE OF EXISTENCE:	Routine is used for obtaining factorisations of leaf nodes	*/
/****************************************************************************************/

/************/
/*	INPUTS	*/
/************/

/// k           -   Leaf Node number



/************************************************************************************/
/*	PURPOSE OF EXISTENCE:	Routine for solving Kx = b, where K is the leaf node k. */
/************************************************************************************/

/************/
/*	INPUTS	*/
/************/

/// k           -   Node number
///	b			-	Input matrix

/************/
/*	OUTPUT	*/
/************/

///	x			-	Inverse of K multiplied with b


Eigen::MatrixXd HODLR_Tree::solve_Leaf(int k, Eigen::MatrixXd b) {
	// std::cout << "\nStart solve Leaf: " << k << "\n";
	Eigen::MatrixXd x	=	tree[nLevels][k]->Kfactor.solve(b);
	// std::cout << "\nDone solve Leaf: " << k << "\n";
	return x;
}

/********************************************************************************************/
/*	PURPOSE OF EXISTENCE:	Routine is used for obtaining factorisations of non-leaf nodes	*/
/********************************************************************************************/

/************/
/*	INPUTS	*/
/************/

/// j           -   Level number
/// k           -   Node number

void HODLR_Tree::factorize_Non_Leaf(int j, int k) {
	//std::cout << "\nStart factorize; Level: " << j << "Node: " << k << "\n";
	int r0	=	tree[j][k]->rank[0];
	int r1	=	tree[j][k]->rank[1];
	tree[j][k]->K	=	Eigen::MatrixXd::Identity(r0+r1, r0+r1);
	
	tree[j][k]->K.block(0, r0, r0, r1)	=	tree[j][k]->Vfactor[1].transpose()*tree[j][k]->Ufactor[1];
	tree[j][k]->K.block(r0, 0, r1, r0)	=	tree[j][k]->Vfactor[0].transpose()*tree[j][k]->Ufactor[0];
	tree[j][k]->Kfactor.compute(tree[j][k]->K);
	int parent	=	k;
	int child	=	k;
	int size	=	tree[j][k]->nSize;
	int tstart, r;
	
	for (int l=j-1; l>=0; --l) {
		child	=	parent%2;
		parent	=	parent/2;
		tstart	=	tree[j][k]->nStart-tree[l][parent]->cStart[child];
		r		=	tree[l][parent]->rank[child];
		tree[l][parent]->Ufactor[child].block(tstart,0,size,r)	=	solve_Non_Leaf(j, k, tree[l][parent]->Ufactor[child].block(tstart,0,size,r));
	}
	
}

void HODLR_Tree::factorize_Non_Leaf_p(int j, int k) {
	//std::cout << "\nStart factorize; Level: " << j << "\n";
	
	int r0	=	tree[j][k]->rank[0];
	int r1	=	tree[j][k]->rank[1];
	tree[j][k]->K	=	Eigen::MatrixXd::Identity(r0+r1, r0+r1);
	
	
	tree[j][k]->K.block(0, r0, r0, r1)	=	tree[j][k]->Vfactor[1].transpose()*tree[j][k]->Ufactor[1];
	tree[j][k]->K.block(r0, 0, r1, r0)	=	tree[j][k]->Vfactor[0].transpose()*tree[j][k]->Ufactor[0];
	tree[j][k]->Kfactor.compute(tree[j][k]->K);
	int size	=	tree[j][k]->nSize;

	int child[j+1];
	int parent[j+1];

	parent[j] =k;	
	for (int l=j-1; l>=0; --l) {
	child[l] = parent[l+1]%2;
	parent[l] = parent[l+1]/2;
	}
	#pragma omp parallel for num_threads(2)
	for (int l=j-1; l>=0; --l) {
		int tstart	=	tree[j][k]->nStart-tree[l][parent[l]]->cStart[child[l]];
		int r		=	tree[l][parent[l]]->rank[child[l]];
		tree[l][parent[l]]->Ufactor[child[l]].block(tstart,0,size,r)	=	solve_Non_Leaf(j, k, tree[l][parent[l]]->Ufactor[child[l]].block(tstart,0,size,r));
	}
}

/********************************************************************************************************************************/
/*	PURPOSE OF EXISTENCE:	Routine for solving (I+UKV')x = b. The method uses Sherman-Morrison-Woodsbury formula to obtain x.	*/
/********************************************************************************************************************************/

/************/
/*	INPUTS	*/
/************/

/// j           -   Level number
/// k           -   Node number
///	b			-	Input matrix

/************/
/*	OUTPUT	*/
/************/

///	matrix			-	(I-U(inverse of (I+K))KV') multiplied by b

Eigen::MatrixXd HODLR_Tree::solve_Non_Leaf(int j, int k, Eigen::MatrixXd b) {
	int r0	=	tree[j][k]->rank[0];
	int r1	=	tree[j][k]->rank[1];
	int n0	=	tree[j][k]->cSize[0];
	int n1	=	tree[j][k]->cSize[1];
	int r	=	b.cols();
	Eigen::MatrixXd temp(r0+r1, r);
	temp << tree[j][k]->Vfactor[1].transpose()*b.block(n0,0,n1,r),
	     tree[j][k]->Vfactor[0].transpose()*b.block(0,0,n0,r);
	temp	=	tree[j][k]->Kfactor.solve(temp);
	Eigen::MatrixXd y(n0+n1, r);
	y << tree[j][k]->Ufactor[0]*temp.block(0,0,r0,r), tree[j][k]->Ufactor[1]*temp.block(r0,0,r1,r);
	return (b-y);
}

/*********************************************************/
/*	PURPOSE OF EXISTENCE: Factorises the kernel matrix A.*/
/*********************************************************/
void HODLR_Tree::factorize_Leaf(int k) {  
	// std::cout << "\nStart factorize Leaf: " << k << "\n";
	tree[nLevels][k]->Kfactor.compute(tree[nLevels][k]->K);
	int parent	=	k;
	int child	=	k;
	int size	=	tree[nLevels][k]->nSize;
	int tstart, r;
	
	for (int l=nLevels-1; l>=0; --l) {	
		child	=	parent%2;
		parent	=	parent/2;
		tstart	=	tree[nLevels][k]->nStart-tree[l][parent]->cStart[child];
		r	=	tree[l][parent]->rank[child];
		tree[l][parent]->Ufactor[child].block(tstart,0,size,r)	=	solve_Leaf(k, tree[l][parent]->Ufactor[child].block(tstart,0,size,r));
	}
}

void HODLR_Tree::factorize() {
	// std::cout << "\nStart factorize...\n";
	#pragma omp parallel 
	{
		#pragma omp for schedule(static)
		 for(std::vector<std::pair<int, int> >::iterator it = myPairs.begin(); it < myPairs.end(); ++it){
			for (int l=0; l<2; ++l) {
					tree[it->first][it->second]->Ufactor[l]	=	tree[it->first][it->second]->U[l];
					tree[it->first][it->second]->Vfactor[l]	=	tree[it->first][it->second]->V[l];
				}
		}
	
		#pragma omp for schedule(dynamic) 		
		for (int k=0; k<nodesInLevel[nLevels]; ++k) {
			factorize_Leaf(k);
		}
	}

	omp_set_dynamic(0);
	omp_set_nested(1);
	int th =6;
	if(nLevels-1<th)
		th=-1;

	for (int j=nLevels-1; j>th; --j) {
		#pragma omp parallel for schedule(dynamic)//num_threads(64)
			for (int k=0; k<nodesInLevel[j]; ++k) {
				factorize_Non_Leaf(j, k);
			}
	} 


	for (int j=th; j>=0; --j) {  
		int itr = nodesInLevel[j];
		int out_th = itr;
		int in_th = total_threads/out_th;
		#pragma omp parallel num_threads(out_th )
		{ 		
			int k = omp_get_thread_num();
			int r0	=	tree[j][k]->rank[0];
	 		int r1	=	tree[j][k]->rank[1];
			tree[j][k]->K	=	Eigen::MatrixXd::Identity(r0+r1, r0+r1);
			tree[j][k]->K.block(0, r0, r0, r1)	=	tree[j][k]->Vfactor[1].transpose()*tree[j][k]->Ufactor[1];
			tree[j][k]->K.block(r0, 0, r1, r0)	=	tree[j][k]->Vfactor[0].transpose()*tree[j][k]->Ufactor[0];
			tree[j][k]->Kfactor.compute(tree[j][k]->K);
			int child[j+1];	
			int parent[j+1];
			int size	=	tree[j][k]->nSize;
			parent[j] =k;	
			for (int l=j-1; l>=0; --l) {
				child[l] = parent[l+1]%2;
				parent[l] = parent[l+1]/2;
			}
			if(omp_get_thread_num()<out_th )
				omp_set_num_threads(in_th);
 
			#pragma omp parallel for 
			for (int l=j-1; l>=0; --l) {
				int tstart	=	tree[j][k]->nStart-tree[l][parent[l]]->cStart[child[l]];
				int r		=	tree[l][parent[l]]->rank[child[l]];
				tree[l][parent[l]]->Ufactor[child[l]].block(tstart,0,size,r)	=	solve_Non_Leaf(j, k, tree[l][parent[l]]->Ufactor[child[l]].block(tstart,0,size,r));
			}	

		}

	} 
}

/**********************************************************************************************/
/*	PURPOSE OF EXISTENCE:	Solves for the 	linear system Ax=b, where A is the kernel matrix. */
/**********************************************************************************************/

/************/
/*	INPUTS	*/
/************/

///	b			-	Input matrix

/************/
/*	OUTPUT	*/
/************/

///	x			-	Inverse of kernel matrix multiplied by input matrix

Eigen::MatrixXd HODLR_Tree::solve(Eigen::MatrixXd b) {
	// std::cout << "\nStart solve...\n";
	int start, size;
	Eigen::MatrixXd x	=	Eigen::MatrixXd::Zero(b.rows(),b.cols());
	int r	=	b.cols();
	int k;

	int threads_required = (exp2(nLevels)) -1 ;
	if(threads_required < total_threads){
		omp_set_num_threads(threads_required+1);
	}
	#pragma omp parallel for private(k,start,size) schedule(static) 
	for (k=0; k<nodesInLevel[nLevels]; ++k) {	
		start	=	tree[nLevels][k]->nStart;
		size	=	tree[nLevels][k]->nSize;
		x.block(start, 0, size, r)	=	solve_Leaf(k, b.block(start, 0, size, r));  
	}
	
	b=x;

	for (int j=nLevels-1; j>=0; --j) {
	int k; //omp_set_num_threads(nodesInLevel[j]);
		#pragma omp parallel for private(k,start,size) schedule(static)
		for (k=0; k<nodesInLevel[j]; ++k) {
			start	=	tree[j][k]->nStart;
			size	=	tree[j][k]->nSize;
			x.block(start, 0, size, r)	=	solve_Non_Leaf(j, k, b.block(start, 0, size, r));
		}
		b=x;
	}
	// std::cout << "\nEnd solve...\n";


	return x;
}



#endif /*__HODLR_Tree__*/

