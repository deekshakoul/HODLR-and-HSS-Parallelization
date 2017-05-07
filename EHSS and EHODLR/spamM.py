#g++ -fopenmp -shared -o spam.so -fPIC spam.cpp -I header/ 
import numpy.ctypeslib as ctl
import ctypes

libname = 'spam.so'
libdir = './'
lib=ctl.load_library(libname, libdir)
py_main_a = lib.main_a
py_main_a.argtypes = [ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p]

q = raw_input('To quit,press Q: ')
while(q!='Q'):
	N = raw_input('Enter data size, N: ')
	r = raw_input('Enter the rank,r: ')
	s = raw_input('Extended Sparsification: choose HODLR/HSS: ') 
	nn =ctypes.c_char_p(N)
	rr=ctypes.c_char_p(r)
	ss=ctypes.c_char_p(s)
	cc = [nn ,rr,ss]
	results = py_main_a(nn,rr,ss)
	q = raw_input('To quit,press Q: ')
print('Exiting the system')
