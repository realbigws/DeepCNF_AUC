SHELL :=/bin/bash
CXX = g++
#-> OpenMP
MFLAG = -O3 -fopenmp
#-> normal
FLAG = -O3

# compile directory
all:
	$(CXX) DeepCNF_Pred.cpp DeepCNF_Misc.cpp DeepCNF.cpp Chebyshev.cpp getopt.cpp $(FLAG) -o DeepCNF_Pred

mpi:
	$(CXX) DeepCNF_Pred.cpp DeepCNF_Misc.cpp DeepCNF.cpp Chebyshev.cpp getopt.cpp $(MFLAG) -o DeepCNF_Pred
	
# clean all the temp files
clean:

