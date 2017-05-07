#!/bin/bash  
 
make -f makefile.mk clean
make -f makefile.mk
cd exec
#rm ans.txt

sim=1
M=200
T=12
for N in 51200 #12800 25600 51200 102400 204800 409600 819200 1633600 1638400 3276800 6533600 13107200 26214400 
	do	
	echo $N
		c=0 	
		while [ $c -lt $sim ]
		do
		./HODLR_Test $N $M $T
		c=`expr $c + 1`
		done
	done
