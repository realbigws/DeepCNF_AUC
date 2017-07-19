
SHELL :=/bin/bash

#-> MPI 
CMM = mpiCC
MFLAG = -O3 -D_FEAT_FLOAT -D_MPI
#-> normal
CXX = g++
FLAG = -O3 -D_FEAT_FLOAT 

# compile directory
all:
	$(CXX) DeepCNF_Train.cpp DeepCNF_Misc.cpp DeepCNF.cpp Chebyshev.cpp getopt.cpp $(FLAG) -o DeepCNF_Train

mpi:
	$(CMM) DeepCNF_Train.cpp DeepCNF_Misc.cpp DeepCNF.cpp Chebyshev.cpp getopt.cpp $(MFLAG) -o DeepCNF_Train

# clean all the temp files
clean:

