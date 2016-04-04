

# make the subdir first
CXX = g++
FLAG = -O3
SHELL :=/bin/bash

DeepCNF_Train_Dir = ./DeepCNF_Train_src
DeepCNF_Pred_Dir = ./DeepCNF_Pred_src
DeepCNF_Train_EXE = /DeepCNF_Train
DeepCNF_Pred_EXE = /DeepCNF_Pred

all:
	cd $(DeepCNF_Train_Dir)&&make
	mv ${DeepCNF_Train_Dir}$(DeepCNF_Train_EXE) ../
	cd $(DeepCNF_Pred_Dir)&&make
	mv $(DeepCNF_Pred_Dir)$(DeepCNF_Pred_EXE) ../

# clean all the temp files
clean:
