Python bindings to the solver using foreign function 'ctypes'
=============================================================

This is a python wrapper around the extended-sparsification solver for HODLR and HSS matrices.

Working
------------

	1.Install python along with numpy and ctypes modules. 
	
	2.Compile the main file(spam.cpp) as shared library 
	::
		g++ -fopenmp -shared -o spam.so -fPIC spam.cpp -I header/
	
	3. 'spamM.py' is the python wrapper, execute it using 
	::
		python spamM.py
	4. If any changes are made to the header files, need to compile it again.
	  
Input
------------
	System Size, N: 
	
	Rank, r: 
	
	type of matrix, HODLR/HSS:

Output
------------
	Error and time taken by the new solver

System 
------------
	Python 2.7.12 
	
	g++ 5.4.0 
	

