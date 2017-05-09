Parallelization achieved using OpenMP for "Factorization based Solver of HODLR Matrices" 
==============================================================================================

Information about the solver
------------------------------------
    1.The parallel solver will be most effective when levels >=7 as the levels below that take very less amount of time both serially and parallely, hence 
    no speedup can be seen, therefore levels >=7 are better inputs.   

    2.In the **rook_piv** method mentioned in HODLR_Matrix.hpp, I have introduced new methods like "get_array_element", "get_array_element" that exploit the vectorization feature of Eigen Library and 
    help reducing time to complete assembly phase. Some more methods like "get_col_p", "get_row_p" were introduced but they did not affect the time taken by assembly.    

    3.As N increases, the speedup(serial time/parallel time) measured tends to increase.

    4.In 'HODLR_Tree' number of threads have been mentioned as per the system details.

 
Working
------------

	1.Run the script s.sh
	:: 
	    sh s.sh
	2. Edit the system parameters in the script itself. 

	3. The script runs makefile.mk and save the executables in the exec folder. 
	  
Input
------------
	System Size, N:
	
	Size of the smallest possible block, M:

	number of simulations, sim:
	

Output
------------
	Error and time taken by each phase in the new solver.


System Specifications
----------------------------
	g++ 5.4.0 

	Intel Xeon E7-8860 v3 machine

	Intel i7 processor
	

