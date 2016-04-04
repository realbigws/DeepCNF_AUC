//#define _MPI

#ifdef _MPI
#include <mpi.h>  //-> include MPI here !!
#endif

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <map>
#include <string>
#include <string.h>
#include "getopt.h"
#include "DeepCNF.h"
#include "LBFGS.h"
#include "Fast_Sort.h"
#include "DeepCNF_Misc.h"
using namespace std;


// MPI
int proc_id;
int num_procs;
// end

//-> for training purpose
vector<DeepCNF_Seq*> testData,trainData;
DeepCNF_Model *m_pModel;
int GATE_FUNCTION=1;                //-> default fate function is tanh
string model_outdir_ = "MODELS";    //-> default should be "MODELS/";
double regularizer_ = 0.5;          //-> default should be 0.5 -> for each layer
int METHOD=0;       //[0] for maximal probability, [1] for maximal accuracy
vector <double> state_auc;

//-> datastructure for maximize AUC
//1. datastructure for sending
double **prob_out_auc_send_test;
int **label_out_auc_send_test;
double **prob_out_auc_send_train;
int **label_out_auc_send_train;
//2. datastructure for gathering
U_INT gather_totnum_test;
U_INT *gather_size_test;
U_INT *gather_offset_test;
double **prob_out_auc_gather_test;
int **label_out_auc_gather_test;
U_INT *auc_gather_index_test;
U_INT gather_totnum_train;
U_INT *gather_size_train;
U_INT *gather_offset_train;
double **prob_out_auc_gather_train;
int **label_out_auc_gather_train;
U_INT *auc_gather_index_train;
//2.X temp dataset
double *prob_out_auc_gather_test_tmp;
int *label_out_auc_gather_test_tmp;
double *prob_out_auc_gather_train_tmp;
int *label_out_auc_gather_train_tmp;


//============= Max AUC related =============//__2015_0520__//
//--- label should be 0 or 1
double Calc_AUC_Value(double *value,int *label,U_INT *index,U_INT totnum)
{
	//Fast_Sort
	Fast_Sort <double> fast_sort_d;
	fast_sort_d.fast_sort_1(value,index,totnum);
	//get zero number
	U_INT zero_num=0;
	for(U_INT i=0;i<totnum;i++)
	{
		if(label[i]==0)zero_num++;
	}
	U_INT one_num=totnum-zero_num;
	//calculate
	double passed_zero=0;
	double auc_totnum=0;
	for(U_INT i=0;i<totnum;i++)
	{
		U_INT idx=index[i];
		if(label[idx]==1)
		{
			auc_totnum+=(zero_num-passed_zero);
		}
		else
		{
			passed_zero++;
		}
	}
	//final value
	return 1.0*auc_totnum/zero_num/one_num;
}


//=============== patameter initialization ==============//
//-> we use a "normalized initialization" method as shown in the following paper:
/*
Understanding the difficulty of training deep feedforward neural networks
	Xavier Glorot  &  Yoshua Bengio
	AISTATS 2010


[remarks]:
The normalization factor may therefore be important when
initializing deep networks because of the multiplicative effect
through layers, and we suggest the following initialization
procedure to approximately satisfy our objectives of
maintaining activation variances and back-propagated gradients
variance as one moves up or down the network. We
call it the normalized initialization:

W ~ U[ - sqrt( 6.0/(n_j + n_j+1) ) , + sqrt( 6.0/(n_j + n_j+1) ) ],

where U[-a, +a] is the uniform distribution in the interval
(-a, +a) and n_k is the size of the k-th layer.
*/
void Parameter_Initialization(int tot_layer,U_INT* layer_count,double ** layer_weight)
{
	//--- init random generator ---//
	srand ( unsigned ( time(0) ) );

	//--- init layer-wise parameter ---//
	for(int i=0;i<tot_layer;i++)
	{

		//get upper/lower bound
		double value=0.05;
/*
		if(i<tot_layer-2)          //-> neuron layer
		{
			//--> get current/next number
			double lower=layer_count[i];
			double upper=layer_count[i+1];
			value=sqrt(6.0/(lower+upper));
		}
		else if(i==tot_layer-2)    //-> neuron to state
		{
			//--> get current number
			double curr=layer_count[i];
			value=sqrt(1.0/curr);
		}
		else                       //-> state to state
		{
			//--> just random
			value=0.05;
		}
*/

		//init
		for(U_INT j=0;j<layer_count[i];j++)
		{
			//generate random number
			double rand_zerone=((double)rand()/(double)RAND_MAX);
			double rand_val=2.0*rand_zerone-1.0; //-> between (-1,+1)
			layer_weight[i][j]=rand_val*value;
		}
	}
}


//==================== init data =====================//

//--- initialize data ----//
void InitData(
	vector <vector <string> > & feat_in, vector <vector <int> > & label_in,
	DeepCNF_Model *pModel, vector <DeepCNF_Seq*> &out_data)
{
	//read in total data
	out_data.clear();
	for(U_INT i=0;i<feat_in.size();i++)
	{
		//-> get length
		int length_s=(int)feat_in[i].size();
		//-> construct a new sequence
		DeepCNF_Seq *seq = new DeepCNF_Seq(length_s, pModel);
		seq->Read_Init_Features(feat_in[i]);
		for(int j=0;j<length_s;j++)seq->observed_label[j]=label_in[i][j];
		//-> add to data
		out_data.push_back(seq);
	}
}
void InitData(
	vector <vector <vector <FeatScore> > > & feat_in, vector <vector <int> > & label_in,
	DeepCNF_Model *pModel, vector <DeepCNF_Seq*> &out_data)
{
	//read in total data
	out_data.clear();
	for(U_INT i=0;i<feat_in.size();i++)
	{
		//-> get length
		int length_s=(int)feat_in[i].size();
		//-> construct a new sequence
		DeepCNF_Seq *seq = new DeepCNF_Seq(length_s, pModel);
		seq->Read_Init_Features(feat_in[i]);
		for(int j=0;j<length_s;j++)seq->observed_label[j]=label_in[i][j];
		//-> add to data
		out_data.push_back(seq);
	}
}


//=================================== LBFGS class related =========================//
//----- LBFGS class ------//
class _LBFGS: public Minimizer 
{
public:
	void Report(const string &s);
	void Report(const vector<double> &theta, int iteration, double objective,double step_length);
	void ComputeGradient(vector<double> &g, const vector<double> &x);
	double ComputeFunction(const vector<double> &x);
public:
	_LBFGS();
	string model_outdir;    //-> default should be "MODELS/";
	double regularizer;     //-> default should be 0.5
};
_LBFGS::_LBFGS() :
	Minimizer(false) {
}

void _LBFGS::Report(const string &s) {
}//if(!proc_id) cerr << s << endl;}


//------ report function --------//
void _LBFGS::Report(const vector<double> &theta, int iteration, double objective, double step_length)
{
	//if(iteration%5) return;
	if(METHOD==2)ComputeFunction(theta);

	//-> set model parameter
	double *xx = new double[theta.size()];
	for(U_INT i=0;i<theta.size();i++)xx[i] = theta[i];
#ifdef _MPI
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(xx, theta.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	m_pModel->cnf_model_load(xx,theta.size());

	//-> 1. calculate the norm of the parameters
	double norm_w = 0;
	for(U_INT i=0;i<theta.size();i++)norm_w +=theta[i]*theta[i];


	//==================== 2. output training results ===================//
	//-> calculate testing accuracy
	double test_totalPos = 0;
	double test_totalCorrect = 0;
	U_INT test_curi=0;
	for(U_INT i=0;i<testData.size();i++)
	{
		if(METHOD!=2)
		{
			testData[i]->MAP_Assign();
			testData[i]->ComputeTestAccuracy_Weight(test_totalPos,test_totalCorrect);
		}
		else
		{
			vector <vector <double> > prob_output;
			testData[i]->MAP_Probability(prob_output);
			for(int t=0;t<m_pModel->dim_states;t++)
			{
				for(int k=0;k<testData[i]->sequence_length;k++)
				{
					//determine label
					int label=0;
					if(testData[i]->observed_label[k]==t)label=1;
					//output probability
					prob_out_auc_send_test[t][test_curi+k]=prob_output[k][t];
					label_out_auc_send_test[t][test_curi+k]=label;
				}
			}
			test_curi+=testData[i]->sequence_length;
		}
	}
#ifdef _MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	//-> for max AUC
	double test_auc=0;
	vector <double> test_lab_auc;
	//-> for max log_prob and accuracy
	double test_tp_sum = 0;
	double test_tc_sum = 0;
	//----- max AUC -----//
	if(METHOD==2)
	{
		//-> gather data for AUC 
		for(int t=0;t<m_pModel->dim_states;t++)
		{
			if(proc_id==0)
			{
				for(U_INT i=0;i<test_curi;i++)
				{
					prob_out_auc_gather_test[t][i]=prob_out_auc_send_test[t][i];
					label_out_auc_gather_test[t][i]=label_out_auc_send_test[t][i];
				}
			}
#ifdef _MPI
			//-> gather prob_out_auc_gather_test
			MPI_Barrier(MPI_COMM_WORLD);
			if(proc_id!=0)
			{
				MPI_Send(prob_out_auc_send_test[t], test_curi, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			}
			else
			{
				for(int i=1;i<num_procs;i++) 
				{
					MPI_Recv(prob_out_auc_gather_test_tmp,gather_size_test[i],MPI_DOUBLE,i,0,
						MPI_COMM_WORLD,MPI_STATUS_IGNORE);
					for(U_INT j=0;j<gather_size_test[i];j++)
					{
						prob_out_auc_gather_test[t][gather_offset_test[i]+j]=prob_out_auc_gather_test_tmp[j];
					}
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);
			//-> gather label_out_auc_gather_test
			if(proc_id!=0)
			{
				MPI_Send(label_out_auc_send_test[t], test_curi, MPI_INT, 0, 0, MPI_COMM_WORLD);
			}
			else
			{
				for(int i=1;i<num_procs;i++)
				{
					MPI_Recv(label_out_auc_gather_test_tmp,gather_size_test[i],MPI_INT,i,0,
						MPI_COMM_WORLD,MPI_STATUS_IGNORE);
					for(U_INT j=0;j<gather_size_test[i];j++)
					{
						label_out_auc_gather_test[t][gather_offset_test[i]+j]=label_out_auc_gather_test_tmp[j];
					}
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);
#endif
		}
		//-> calculate for AUC 
		if(proc_id==0)
		{
			test_lab_auc.clear();
			for(int t=0;t<m_pModel->dim_states;t++)
			{
				double cur_auc=Calc_AUC_Value(prob_out_auc_gather_test[t],label_out_auc_gather_test[t],
					auc_gather_index_test,gather_totnum_test);
				test_lab_auc.push_back(cur_auc);
				test_auc+=cur_auc;
			}
			test_auc/=m_pModel->dim_states;
		}
	}
	//----- max probability -----//
	else
	{
#ifdef _MPI
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(&test_totalPos, &test_tp_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&test_totalCorrect, &test_tc_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
#else
		test_tp_sum = test_totalPos;
		test_tc_sum = test_totalCorrect;
#endif
	}

	//==================== 3. output testing results ===================//
	//-> calculate training accuracy
	double train_totalPos = 0;
	double train_totalCorrect = 0;
	U_INT train_curi=0;
	for(U_INT i=0;i<trainData.size();i++)
	{
		if(METHOD!=2)
		{
			trainData[i]->MAP_Assign();
			trainData[i]->ComputeTestAccuracy_Weight(train_totalPos,train_totalCorrect);
		}
		else
		{
			vector <vector <double> > prob_output;
			trainData[i]->MAP_Probability(prob_output);
			for(int t=0;t<m_pModel->dim_states;t++)
			{
				for(int k=0;k<trainData[i]->sequence_length;k++)
				{
					//determine label
					int label=0;
					if(trainData[i]->observed_label[k]==t)label=1;
					//output probability
					prob_out_auc_send_train[t][train_curi+k]=prob_output[k][t];
					label_out_auc_send_train[t][train_curi+k]=label;
				}
			}
			train_curi+=trainData[i]->sequence_length;
		}
	}
#ifdef _MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	//-> for max AUC
	double train_auc=0;
	vector <double> train_lab_auc;
	//-> for max log_prob and accuracy
	double train_tp_sum = 0;
	double train_tc_sum = 0;
	//----- max AUC -----//
	if(METHOD==2)
	{
		//-> gather data for AUC 
		for(int t=0;t<m_pModel->dim_states;t++)
		{
			if(proc_id==0)
			{
				for(U_INT i=0;i<train_curi;i++)
				{
					prob_out_auc_gather_train[t][i]=prob_out_auc_send_train[t][i];
					label_out_auc_gather_train[t][i]=label_out_auc_send_train[t][i];
				}
			}
#ifdef _MPI
			//-> gather prob_out_auc_gather_test
			MPI_Barrier(MPI_COMM_WORLD);
			if(proc_id!=0)
			{
				MPI_Send(prob_out_auc_send_train[t], train_curi, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			}
			else
			{
				for(int i=1;i<num_procs;i++)
				{
					MPI_Recv(prob_out_auc_gather_train_tmp,gather_size_train[i],MPI_DOUBLE,i,0,
						MPI_COMM_WORLD,MPI_STATUS_IGNORE);
					for(U_INT j=0;j<gather_size_train[i];j++)
					{
						prob_out_auc_gather_train[t][gather_offset_train[i]+j]=prob_out_auc_gather_train_tmp[j];
					}
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);
			//-> gather label_out_auc_gather_test
			if(proc_id!=0)
			{
				MPI_Send(label_out_auc_send_train[t], train_curi, MPI_INT, 0, 0, MPI_COMM_WORLD);
			}
			else
			{
				for(int i=1;i<num_procs;i++)
				{
					MPI_Recv(label_out_auc_gather_train_tmp,gather_size_train[i],MPI_INT,i,0,
						MPI_COMM_WORLD,MPI_STATUS_IGNORE);
					for(U_INT j=0;j<gather_size_train[i];j++)
					{
						label_out_auc_gather_train[t][gather_offset_train[i]+j]=label_out_auc_gather_train_tmp[j];
					}
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);
#endif
		}
		//-> calculate for AUC 
		if(proc_id==0)
		{
			train_lab_auc.clear();
			for(int t=0;t<m_pModel->dim_states;t++)
			{
				double cur_auc=Calc_AUC_Value(prob_out_auc_gather_train[t],label_out_auc_gather_train[t],
					auc_gather_index_train,gather_totnum_train);
				train_lab_auc.push_back(cur_auc);
				train_auc+=cur_auc;
			}
			train_auc/=m_pModel->dim_states;
		}
	}
	//----- max probability -----//
	else
	{
#ifdef _MPI
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(&train_totalPos, &train_tp_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&train_totalCorrect, &train_tc_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
#else
		train_tp_sum = train_totalPos;
		train_tc_sum = train_totalCorrect;
#endif
	}

	//----- output to file ------//
	if(proc_id==0)
	{
		//-> output results to screen
		cout << endl << "Iteration:  " << iteration << endl;
		cout << " Objective: " << objective << endl;
		cout << " Weight Norm: " << sqrt(norm_w) << endl;
		if(METHOD==2)
		{
			cout << " obj AUC:   ";
			double averate_auc=0;
			for(int i=0;i<(int)state_auc.size();i++)
			{
				cout << state_auc[i] << " ";
				averate_auc+=state_auc[i];
			}
			averate_auc/=state_auc.size();
			cout << " | average: " << (double)averate_auc<<endl;
		}
		if(METHOD!=2)
		{
			cout << " test ACC(MAP) : " << (double) test_tc_sum/test_tp_sum <<"   " << (int)test_tc_sum << "/" << (int)test_tp_sum << endl;
			cout << " train ACC(MAP): " << (double) train_tc_sum/train_tp_sum << "   " << (int)train_tc_sum << "/" << (int)train_tp_sum << endl;
		}
		else
		{
			cout << " test AUC:  ";
			for(int i=0;i<(int)test_lab_auc.size();i++)cout << test_lab_auc[i] << " ";
			cout << " | average: " << (double)test_auc<<endl;
			cout << " train AUC: ";
			for(int i=0;i<(int)train_lab_auc.size();i++)cout << train_lab_auc[i] << " ";
			cout << " | average: " << (double)train_auc<<endl;
		}
		//-> output model to file
		char command[10000];
		sprintf(command,"%s/model.%d",model_outdir.c_str(),iteration);
		ofstream fout(command);
		for(U_INT i=0;i<theta.size();i++)fout << theta[i] << " ";
		fout << endl;
		fout.close();
	}

	//--- delete xx ----//
	delete [] xx;

#ifdef _MPI
MPI_Barrier(MPI_COMM_WORLD);
#endif
}


//------ ComputeGradient function --------//
void _LBFGS::ComputeGradient(vector<double> &g, const vector<double> &x)
{
	if(METHOD==2)ComputeFunction(x);

	//-> set model parameter
	double *xx = new double[x.size()];
	for(U_INT i=0;i<x.size();i++)xx[i] = x[i];
#ifdef _MPI
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(xx, x.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	m_pModel->cnf_model_load(xx,x.size());

	//-> calculate gradient
	for(U_INT i=0;i<x.size();i++)m_pModel->grad_sum[i]=0;
	double total_count=0;
	for(U_INT k=0;k<trainData.size();k++)
	{
		//-> calculate forward_backward
		trainData[k]->Calc_Forward_Backward();
		//-> calculate gradient
		if(METHOD==0) trainData[k]->Grad_Calc();             //-> maximize log_prob
		else if(METHOD==1) trainData[k]->Grad_Calc2();      //-> maximize accuracy
		else trainData[k]->Grad_Calc3();                    //-> maximize AUC
		//-> collect gradient
		m_pModel->Collect_GradSum();
		total_count+=trainData[k]->sequence_length;
	}

	//-> collect gradient
	double final_total_count=0;
#ifdef _MPI
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(m_pModel->grad_sum, m_pModel->grad, x.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Bcast(m_pModel->grad, x.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&total_count,&final_total_count,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	MPI_Bcast(&final_total_count,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
#else
	for(U_INT i=0;i<x.size();i++)m_pModel->grad[i]=m_pModel->grad_sum[i];
	final_total_count=total_count;
#endif

	//-> final process  with regularizer
	if(METHOD==2) for(U_INT i=0;i<x.size();i++) g[i] = -1.0*final_total_count*m_pModel->grad[i];
	else for(U_INT i=0;i<x.size();i++) g[i] = -1.0*m_pModel->grad[i];
	for(U_INT i=0;i<x.size();i++) g[i] += x[i]*2*regularizer;

	//--- delete xx ----//
	delete [] xx;

#ifdef _MPI
MPI_Barrier(MPI_COMM_WORLD);
#endif
}

//------ ComputeFunction function --------//
double _LBFGS::ComputeFunction(const vector<double> &x)
{
	//-> set model parameter
	double *xx = new double[x.size()];
	for(U_INT i=0;i<x.size();i++)xx[i] = x[i];
#ifdef _MPI
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(xx, x.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	m_pModel->cnf_model_load(xx,x.size());

	//-> calculate objective function
	double obj = 0;
	double total_count=0;
	if(METHOD==2) m_pModel->Initialize_SVvalue();   //-> only for maximize AUC
	for(U_INT k=0;k<trainData.size();k++)
	{
		//-> calculate forward_backward
		trainData[k]->Calc_Forward_Backward();
		//-> calculate objective function
		double obj_cur=0;
		if(METHOD==0)obj_cur=trainData[k]->Obj_Calc();         //-> maximize log_prob
		else if(METHOD==1) obj_cur=trainData[k]->Obj_Calc2();  //-> maximize accuracy
		else trainData[k]->Obj_Calc3();                        //-> maximize AUC
		//-> collect objective function
		obj+=obj_cur;
		if(METHOD==2) m_pModel->SumOver_SVvalue();   //-> only for maximize AUC
		//-> collect total
		total_count+=trainData[k]->sequence_length;
	}

	//-> collect objective function
	double obj_sum = 0;
	double final_total_count=0;
#ifdef _MPI
	if(METHOD!=2)
	{
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(&obj, &obj_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Bcast(&obj_sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
	}
	else
	{
		U_INT total_num=(m_pModel->max_degree + 2)*(m_pModel->max_degree + 1)/2;
		U_INT totnum=total_num*m_pModel->dim_states;
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(m_pModel->s_value_sum, m_pModel->s_value_out, totnum, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Bcast(m_pModel->s_value_out, totnum, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(m_pModel->v_value_sum, m_pModel->v_value_out, totnum, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Bcast(m_pModel->v_value_out, totnum, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(m_pModel->s_value_sum_n, m_pModel->s_value_out_n, totnum, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Bcast(m_pModel->s_value_out_n, totnum, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(m_pModel->v_value_sum_n, m_pModel->v_value_out_n, totnum, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Bcast(m_pModel->v_value_out_n, totnum, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		//-> add pseudocount to v_value_out_n and s_value_out_n
		for(U_INT i=0;i<totnum;i++)m_pModel->s_value_out_n[i]++;
		for(U_INT i=0;i<totnum;i++)m_pModel->v_value_out_n[i]++;
		//-> calculate objective function
		obj_sum=m_pModel->Calculate_AUC_Value(); //-> only for maximize AUC
		//-> get total number
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(&total_count,&final_total_count,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Bcast(&final_total_count,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
	}
#else
	if(METHOD!=2) obj_sum = obj;
	else
	{
		U_INT total_num=(m_pModel->max_degree + 2)*(m_pModel->max_degree + 1)/2;
		U_INT totnum=total_num*m_pModel->dim_states;
		for(U_INT i=0;i<totnum;i++)m_pModel->s_value_out[i]=m_pModel->s_value_sum[i];
		for(U_INT i=0;i<totnum;i++)m_pModel->v_value_out[i]=m_pModel->v_value_sum[i];
		for(U_INT i=0;i<totnum;i++)m_pModel->s_value_out_n[i]=m_pModel->s_value_sum_n[i];
		for(U_INT i=0;i<totnum;i++)m_pModel->v_value_out_n[i]=m_pModel->v_value_sum_n[i];
		//-> add pseudocount to v_value_out_n and s_value_out_n
		for(U_INT i=0;i<totnum;i++)m_pModel->s_value_out_n[i]++;
		for(U_INT i=0;i<totnum;i++)m_pModel->v_value_out_n[i]++;
		//-> calculate objective function
		obj_sum=m_pModel->Calculate_AUC_Value(); //-> only for maximize AUC
		//-> get total number
		final_total_count=total_count;
	}
#endif

	//-> for AUC calculation
	if(METHOD==2)
	{
		state_auc.clear();
		for(int t=0;t<m_pModel->dim_states;t++)state_auc.push_back(m_pModel->auc_for_label[t]);
	}

	//-> final process with regularizer
	if(METHOD==2)obj_sum = -1.0*final_total_count*obj_sum;
	else obj_sum = -obj_sum;
	for(U_INT i=0;i<x.size();i++) obj_sum += x[i]*x[i]*regularizer;

	//--- delete xx ----//
	delete [] xx;

#ifdef _MPI
MPI_Barrier(MPI_COMM_WORLD);
#endif
	//--- return ----//
	return obj_sum;
}

//------ Init MaxAUC Data_Structure --------//
void Max_AUC_DataStructure_Init(void)
{
	//-> Part I. datastructure for sending 
	//--> test
	U_INT total_length_test=0;
	for(U_INT i=0;i<testData.size();i++)total_length_test+=testData[i]->sequence_length;
	U_INT current_size_test = total_length_test;
	prob_out_auc_send_test=new double*[m_pModel->dim_states];
	label_out_auc_send_test=new int*[m_pModel->dim_states];
	for(int t=0;t<m_pModel->dim_states;t++)
	{
		prob_out_auc_send_test[t]=new double[current_size_test];
		label_out_auc_send_test[t]=new int[current_size_test];
	}
	//--> train
	U_INT total_length_train=0;
	for(U_INT i=0;i<trainData.size();i++)total_length_train+=trainData[i]->sequence_length;
	U_INT current_size_train = total_length_train;
	prob_out_auc_send_train=new double*[m_pModel->dim_states];
	label_out_auc_send_train=new int*[m_pModel->dim_states];
	for(int t=0;t<m_pModel->dim_states;t++)
	{
		prob_out_auc_send_train[t]=new double[current_size_train];
		label_out_auc_send_train[t]=new int[current_size_train];
	}

	//-> Part II. datastructure for gathering
	if(proc_id==0)
	{
		gather_size_test=new U_INT[num_procs];
		gather_offset_test=new U_INT[num_procs];
		gather_size_train=new U_INT[num_procs];
		gather_offset_train=new U_INT[num_procs];
	}
	if(proc_id==0)
	{
		gather_size_test[0]=current_size_test;
		gather_size_train[0]=current_size_train;
	}
#ifdef _MPI
	//-> gathre for current_size_test
	//->    https://msdn.microsoft.com/en-us/library/Dn473377(v=VS.85).aspx
	MPI_Barrier(MPI_COMM_WORLD);
	if(proc_id!=0)
	{
		MPI_Send(&current_size_test, 1, MPI_LONG_LONG_INT, 0, 0, MPI_COMM_WORLD);
	}
	else
	{
		for(int i=1;i<num_procs;i++) 
		{
			U_INT tmp_rec;
			MPI_Recv(&tmp_rec, 1, MPI_LONG_LONG_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			gather_size_test[i]=tmp_rec;
		}
	}
	//-> gather for current_size_train
	MPI_Barrier(MPI_COMM_WORLD);
	if(proc_id!=0)
	{
		MPI_Send(&current_size_train, 1, MPI_LONG_LONG_INT, 0, 0, MPI_COMM_WORLD);
	}
	else
	{
		for(int i=1;i<num_procs;i++) 
		{
			U_INT tmp_rec;
			MPI_Recv(&tmp_rec, 1, MPI_LONG_LONG_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			gather_size_train[i]=tmp_rec;
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	if(proc_id==0)
	{
		//-> 1. get gathered data size
		//--> test
		U_INT offset_test=0;
		for(int i=0;i<num_procs;i++)
		{
			gather_offset_test[i]=offset_test;
			offset_test+=gather_size_test[i];
		}
		gather_totnum_test=offset_test;
		//--> train
		U_INT offset_train=0;
		for(int i=0;i<num_procs;i++)
		{
			gather_offset_train[i]=offset_train;
			offset_train+=gather_size_train[i];
		}
		gather_totnum_train=offset_train;
		//-> 2. create data structure
		//--> test
		prob_out_auc_gather_test=new double*[m_pModel->dim_states];
		label_out_auc_gather_test=new int*[m_pModel->dim_states];
		for(int t=0;t<m_pModel->dim_states;t++)
		{
			prob_out_auc_gather_test[t]=new double[gather_totnum_test];
			label_out_auc_gather_test[t]=new int[gather_totnum_test];
		}
		auc_gather_index_test=new U_INT[gather_totnum_test];
		//--> train
		prob_out_auc_gather_train=new double*[m_pModel->dim_states];
		label_out_auc_gather_train=new int*[m_pModel->dim_states];
		for(int t=0;t<m_pModel->dim_states;t++)
		{
			prob_out_auc_gather_train[t]=new double[gather_totnum_train];
			label_out_auc_gather_train[t]=new int[gather_totnum_train];
		}
		auc_gather_index_train=new U_INT[gather_totnum_train];
		//-> 3. create tmp structure
		//--> test
		prob_out_auc_gather_test_tmp=new double[gather_totnum_test];
		label_out_auc_gather_test_tmp=new int[gather_totnum_test];
		prob_out_auc_gather_train_tmp=new double[gather_totnum_train];
		label_out_auc_gather_train_tmp=new int[gather_totnum_train];
	}
}

//------ Dele MaxAUC Data_Structure --------//
void Max_AUC_DataStructure_Dele(void)
{
	//-> datastructure for sending 
	for(int t=0;t<m_pModel->dim_states;t++)
	{
		delete [] prob_out_auc_send_test[t];
		delete [] label_out_auc_send_test[t];
		delete [] prob_out_auc_send_train[t];
		delete [] label_out_auc_send_train[t];
	}
	delete [] prob_out_auc_send_test;
	delete [] label_out_auc_send_test;
	delete [] prob_out_auc_send_train;
	delete [] label_out_auc_send_train;

	//-> datastructure for gathering
	if(proc_id==0)
	{
		delete [] gather_size_test;
		delete [] gather_offset_test;
		delete [] auc_gather_index_test;
		delete [] gather_size_train;
		delete [] gather_offset_train;
		delete [] auc_gather_index_train;
		for(int t=0;t<m_pModel->dim_states;t++)
		{
			delete [] prob_out_auc_gather_test[t];
			delete [] label_out_auc_gather_test[t];
			delete [] prob_out_auc_gather_train[t];
			delete [] label_out_auc_gather_train[t];
		}
		delete [] prob_out_auc_gather_test;
		delete [] label_out_auc_gather_test;
		delete [] prob_out_auc_gather_train;
		delete [] label_out_auc_gather_train;
	}
}

//=================================== LBFGS class related =========================//over



//----------- Layer-wise Training Procedure ----------------//__150121__//
//-> this procedure is only for initializing the parameter
void LayerWise_Training_Initialize(
	vector <int> &range_out, U_INT local_num_ori,                            //-> for feature range process
	string &train_file, string &test_file, int num_procs,int proc_id,        //-> input for data
	string & window_str, string & node_str, int state_num, U_INT local_num_, //-> input for parameter
	string & iterative_number_str, string & regularizer_str,                 //-> input for layer-wise
	vector <double> &in_param, vector <double> &out_param)                   //-> output for initialized weight
{
	//data structure 
	vector <vector <vector <FeatScore> > > feat_in_train;
	vector <vector <int> > label_in_train;
	vector <vector <vector <FeatScore> > > feat_in_test;
	vector <vector <int> > label_in_test;
	//load data
	LoadData(train_file, num_procs, proc_id, local_num_ori,range_out,feat_in_train, label_in_train);
	LoadData(test_file, num_procs, proc_id, local_num_ori,range_out,feat_in_test, label_in_test);
	//parse string
	vector <int> window_rec;
	vector <int> node_rec;
	vector <int> iterative_number_rec;
	vector <double> regularizer_rec;
	char separator=',';
	int ws1=Parse_Str(window_str,window_rec,separator);
	int ws2=Parse_Str(node_str,node_rec,separator);
	int ws3=Parse_Str(iterative_number_str,iterative_number_rec,separator);
	int ws4=Parse_Str_Double(regularizer_str,regularizer_rec,separator);
	//check 
	if(ws1!=ws2 || ws1!=ws3 || ws1!=ws4)
	{
		fprintf(stderr,"dimension not equal !! window_str %d, node_str %d, iterative_number_str %d, regularizer_rec %d \n",ws1,ws2,ws3,ws4);
		exit(-1);
	}

	//pre-training
	U_INT current_pos=0;
	U_INT init_num=local_num_;
	out_param.clear();
	for(int l=0;l<ws3;l++)
	{
		//get current window and node
		vector <double> window_in,node_in;
		window_in.push_back(window_rec[l]);
		node_in.push_back(node_rec[l]);
		string window_str_in,node_str_in;
		char separator=' ';
		Parse_Double(window_in, window_str_in,separator);
		Parse_Double(node_in, node_str_in,separator);

		//init model
		U_INT local_num=init_num;
		m_pModel = new DeepCNF_Model(1,window_str_in,node_str_in,state_num,local_num,1);
		m_pModel->Gate_Function=GATE_FUNCTION;

		//init weight
		U_INT WD=m_pModel->total_param_count;
		vector<double> w0 (WD,0);
		for (U_INT i = 0; i < m_pModel->layer_count[0]; i++)
		{
			w0[i]=in_param[current_pos];
			current_pos++;
		}

		//input feature
		InitData(feat_in_train,label_in_train,m_pModel,trainData);
		InitData(feat_in_test,label_in_test,m_pModel,testData);

		//start training
		//============================== MAIN TRAINING PROCEDURE HERE ================//
		//----- run LBFGS -----//
		_LBFGS *lbfgs = new _LBFGS;
		//--> set model_outdir and regularizer
		char command[30000];
		sprintf(command,"mkdir -p %s_%d/",model_outdir_.c_str(),l+1);
		int retv=system(command);
		sprintf(command,"%s_%d/",model_outdir_.c_str(),l+1);
		lbfgs->model_outdir=command;
		lbfgs->regularizer=regularizer_rec[l];
		//--> training
		lbfgs->LBFGS(w0,iterative_number_rec[l]);
		delete lbfgs;
		//----- run LBFGS -----//over
		//============================== MAIN TRAINING PROCEDURE HERE ================//over

		//assign trained weights
		if(l<ws3-1)
		{
			for (U_INT i = 0; i < m_pModel->layer_count[0]; i++)out_param.push_back(m_pModel->layer_weight[0][i]);
		}
		else
		{
			for (U_INT i = 0; i < m_pModel->total_param_count; i++)out_param.push_back(m_pModel->weights[i]);
		}

		//update features of the current layer
		init_num=node_rec[l];
		for (U_INT i = 0; i < feat_in_train.size(); i++)
		{
			vector <vector <FeatScore> > output;
			trainData[i]->Gate_Output(feat_in_train[i],output);
			feat_in_train[i]=output;
		}
		for (U_INT i = 0; i < feat_in_test.size(); i++)
		{
			vector <vector <FeatScore> > output;
			testData[i]->Gate_Output(feat_in_test[i],output);
			feat_in_test[i]=output;
		}

		//free all the alignments
		for (U_INT i = 0; i < trainData.size(); i++) delete trainData[i];
		for (U_INT i = 0; i < testData.size(); i++) delete testData[i];
		delete m_pModel;
	}
}


//------------ usage -------------//
void Usage() {
	cerr << "DeepCNF_Train v1.4 (mpi version) [2015_09_10] \n\n";
	cerr << "Usage : \n\n";
	cerr << "mpirun -np NP ./DeepCNF_Train -r train_file -t test_file \n";
	cerr << "             -w window_str -d node_str -s state_num -l feat_num [-S feat_select] \n";
	cerr << "            [-n finetune_num] [-f finetune_reg] [-m init_model] [-o out_root] \n";
	cerr << "            [-G gate_function] [-W label_weight] [-M method] [-D AUC_degree] \n";
//	cerr << "            [-M method] [-W label_weight] [-D AUC_degree] [-B AUC_beta] \n";
	cerr << "Options:\n\n";
	cerr << "-np NP :             number of processors. \n\n";
	//-> required parameter
	cerr << "-r train_file :      training file. \n\n";
	cerr << "-t test_file :       testing file. \n\n";
	cerr << "-w window_str :      window string for DeepCNF. e.g., '5,5' \n\n";
	cerr << "-d node_str :        node string for DeepCNF. e.g., '40,20' \n\n";
	cerr << "-s state_num :       state number. \n\n";
	cerr << "-l feat_num :        feature number at each position. \n\n";
	cerr << "-S feat_select :     feature selection. e.g., '1-7,9' [default uses all features] \n\n";
	//-> optional parameter
//	cerr << "-u layer_num :       layerwise pre-training number. e.g.,'100,50' [default is -1]\n\n";
//	cerr << "-v layer_reg :       layerwise pre-training regularizer. e.g., '0.5,0.2' [default is -1]\n\n";
	cerr << "-n finetune_num :    fine-tune iteration number. [default is 200] \n\n";
	cerr << "-f finetune_reg :    fine-tune regularizer. [default is 0.5] \n\n";
	cerr << "-m init_model :      file for initial model parameters. [optional, and default is NULL] \n\n";
	cerr << "-o out_root :        output directory for trained models. [optional, and default is 'MODELS/'] \n\n";
	//-> method related parameter
	cerr << "-G gate_function :   gate function type: sigmoid (1), tanh (2), and relu (0). [default is 1] \n\n";
	cerr << "-M method :          maximize log_prob (0), posterior_prob (1), or AUC (2). [default is 0] \n\n";
	cerr << "-W label_weight :    label weight. e.g., '0.1,0.9'. [default is 1 for each label] \n\n";
	//-> max AUC related
	cerr << "-D AUC_degree :      degree (1-30) of polynomials for AUC [default is 3]. \n\n";
//	cerr << "-B AUC_beta :        sigmoid parameter beta (1-100) for AUC [default is -1]. \n\n";
}

//------------ main -------------//
int main(int argc, char** argv)
{
METHOD=0;

//mpi init
proc_id=0;
num_procs=1;

	//----- MPI init -----//
#ifdef _MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
#endif
	//----- MPI init -----//over

	//-- help --//
	if (argc < 2)
	{
		if(proc_id==0) Usage();
#ifdef _MPI
		MPI_Finalize();
#endif
		exit(0);
	}

	//---- init parameter ----//
	string iterative_number_str="-1";
	string regularizer_str="-1";
	int fine_tune_number = 200;
	//-> basic inputs
	string train_list_file = "";
	string test_list_file = "";
	//-> required parameters
	string window_str = "";    //-> window string
	string node_str = "";      //-> node string
	int state_num = -1;        //-> state number
	int local_num_ori = -1;    //-> original feature number
	int local_num = -1;        //-> processed feature number
	string feat_range_str="";  //-> feature range string
	//-> advanced parameters
	bool use_trained_model = false;
	string model_file = "";
	int gate_function_=1;      //-> gate function [0 for relu, 1 for tanh, 2 for sigmoid ]
	int method_=0;             //-> training methodology [0 for max_logprob, 1 for max_accuracy, 2 for max_AUC]
	string label_weight_str="";
	int max_auc_degree=3;      //-> default use degree=3
	double max_auc_beta=-1;    //-> default use real_AUC


	//command-line arguments process
	extern char* optarg;
	char c = 0;
	while ((c = getopt(argc, argv, "r:t:w:d:s:l:S:u:v:n:f:m:o:G:M:W:D:B:")) != EOF) {
		switch (c) {
		//-> training/testing file
		case 'r':
			train_list_file = optarg;
			break;
		case 't':
			test_list_file = optarg;
			break;
		//-> window/node string
		case 'w':
			window_str = optarg;
			break;
		case 'd':
			node_str = optarg;
			break;
		//-> state/local number
		case 's':
			state_num = atoi(optarg);
			break;
		case 'l':
			local_num_ori = atoi(optarg);
			break;
		case 'S':
			feat_range_str = optarg;
			break;
		//-> layer-wise parameters for training
		case 'u':
			iterative_number_str = optarg;
			break;
		case 'v':
			regularizer_str = optarg;
			break;
		//-> fine-tune parameters for training
		case 'n':
			fine_tune_number = atoi(optarg);
			break;
		case 'f':
			regularizer_ = atof(optarg);
			break;
		//-> model related
		case 'm':
			model_file = optarg;
			use_trained_model = true;
			break;
		case 'o':
			model_outdir_ = optarg;
			break;
		//-> gate function (relu, sigmoid, or tanh)
		case 'G':
			gate_function_ = atoi(optarg);
			break;
		//-> method (max_logprob, max_accuracy, or max_AUC)
		case 'M':
			method_ = atoi(optarg);
			break;
		//-> label weight
		case 'W':
			label_weight_str = optarg;
			break;
		//-> max AUC degree
		case 'D':
			max_auc_degree = atoi(optarg);
			break;
		case 'B':
			max_auc_beta = atof(optarg);
			break;
		default:
			Usage();
			exit(-1);
		}
	}

	//----- check parameter -----//
	//-> check input file
	if(train_list_file=="" || test_list_file=="")
	{
		if(proc_id==0) fprintf(stderr,"train_list_file %s or test_list_file %s is NULL\n",train_list_file.c_str(),test_list_file.c_str());
#ifdef _MPI
		MPI_Finalize();
#endif
		exit(-1);
	}

	//-> check window/node string
	if(window_str=="" || node_str=="")
	{
		if(proc_id==0) fprintf(stderr,"window_str %s or node_str %s is NULL\n",window_str.c_str(),node_str.c_str());
#ifdef _MPI
		MPI_Finalize();
#endif
		exit(-1);
	}

	//-> check dimension
	int ws1_;
	{
		vector <int> wstmp;
		char separator=',';
		int ws1=Parse_Str(window_str,wstmp,separator);
		int ws2=Parse_Str(node_str,wstmp,separator);
		ws1_=ws1;
		//--> check window_str and node_str
		if(ws1!=ws2)
		{
			if(proc_id==0) fprintf(stderr,"window_str %s dimension not equal to node_str %s \n",window_str.c_str(),node_str.c_str());
#ifdef _MPI
			MPI_Finalize();
#endif
			exit(-1);
		}
	}

	//-> check pre_training iterative_num
	if(iterative_number_str!="-1")
	{
		vector <int> wstmp;
		char separator=',';
		int ws3=Parse_Str(iterative_number_str,wstmp,separator);
		//--> check iterative_number_str
		if(ws1_!=ws3 && ws3>=1)
		{
			if(proc_id==0) fprintf(stderr,"window_str %s dimension not equal to iterative_number_str %s \n",window_str.c_str(),iterative_number_str.c_str());
#ifdef _MPI
			MPI_Finalize();
#endif
			exit(-1);
		}
		//--> assign iterative_number
		if(ws3==0)
		{
			iterative_number_str="-1";
		}
		else if(ws3==1)
		{
			vector <double> iterative_number;
			for(int i=0;i<ws1_;i++)iterative_number.push_back(wstmp[0]);
			char separator=',';
			Parse_Double(iterative_number, iterative_number_str,separator);
		}
	}

	//-> check pre_training regularizer
	if(regularizer_str!="-1")
	{
		vector <int> wstmp;
		char separator=',';
		int ws3=Parse_Str(regularizer_str,wstmp,separator);
		//--> check iterative_number_str
		if(ws1_!=ws3 && ws3>=1)
		{
			if(proc_id==0) fprintf(stderr,"window_str %s dimension not equal to regularizer_str %s \n",window_str.c_str(),regularizer_str.c_str());
#ifdef _MPI
			MPI_Finalize();
#endif
			exit(-1);
		}
		//--> assign iterative_number
		if(ws3==0)
		{
			regularizer_str="-1";
		}
		else if(ws3==1)
		{
			vector <double> iterative_reg;
			for(int i=0;i<ws1_;i++)iterative_reg.push_back(wstmp[0]);
			char separator=',';
			Parse_Double(iterative_reg, regularizer_str,separator);
		}
	}

	//-> check state/local number
	if(state_num==-1 || local_num_ori==-1)
	{
		if(proc_id==0) fprintf(stderr,"state_num %d and local_num_ori %d must be assigned\n",state_num,local_num_ori);
#ifdef _MPI
		MPI_Finalize();
#endif
		exit(-1);
	}
	local_num=local_num_ori;
	//-> process range
	vector <int> range_out(local_num_ori,1);
	if(feat_range_str!="")
	{
		local_num=Parse_Feature_Range(feat_range_str,range_out);
	}

	//-> check fine-tune number
	if(fine_tune_number<0 )
	{
		if(proc_id==0) fprintf(stderr,"finetune_num %d should be positive \n",fine_tune_number);
#ifdef _MPI
		MPI_Finalize();
#endif
		exit(-1);
	}

	//-> check regularizer
	if(regularizer_<0 )
	{
		if(proc_id==0) fprintf(stderr,"finetune_reg %f should be positive \n",regularizer_);
#ifdef _MPI
		MPI_Finalize();
#endif
		exit(-1);
	}

	//-> check method
	if(method_ <0 || method_ >2)
	{
		if(proc_id==0) fprintf(stderr,"method %d should be 0, 1 or 2 \n",method_);
#ifdef _MPI
		MPI_Finalize();
#endif
		exit(-1);
	}
	if(method_==2)
	{
		if(max_auc_degree<1 || max_auc_degree>30)
		{
			if(proc_id==0) fprintf(stderr,"max_auc_degree %d should be 1 to 30 \n",method_);
#ifdef _MPI
			MPI_Finalize();
#endif
			exit(-1);
		}
	}

	//-> check gate_function
	if(gate_function_ < 0 || gate_function_ > 2)
	{
		if(proc_id==0) fprintf(stderr,"gate_function %d should be 0, 1 or 2 \n",gate_function_);
#ifdef _MPI
		MPI_Finalize();
#endif
		exit(-1);
	}
	GATE_FUNCTION=gate_function_;

	//-> check label_weight
	vector <double> label_weight;
	if(label_weight_str!="")
	{
		char separator=',';
		int label_weight_num=Parse_Str_Double(label_weight_str,label_weight,separator);
		if(label_weight_num!=state_num)
		{
			if(proc_id==0) fprintf(stderr,"label_weight_str %s dimension not equal to state_num %d \n",label_weight_str.c_str(),state_num);
#ifdef _MPI
			MPI_Finalize();
#endif
			exit(-1);
		}
	}
	else
	{
		label_weight.resize(state_num);
		for(int i=0;i<state_num;i++)label_weight[i]=1;
	}



//---- output command ----//
if(proc_id==0)
{
	cout << "#------------------------------------------#"<<endl;
	cout << "$ " << "mpirun -np "<<num_procs<<" ";
	for(int i=0;i<argc;i++)cout<<argv[i]<<" ";
	cout<<endl;
	cout << "#------------------------------------------#"<<endl<<endl;
}
//---- output parameter ---//
if(proc_id==0) cout << "model options: window_str "<< window_str << ", node_str " << node_str << ", state_num " << state_num << ", feat_num "<< local_num << ", feat_range_str " << feat_range_str << ", label_weight_str " << label_weight_str << endl;

	//===================== initilize weights ================//start
	//----------- init model to get the dimension of weights ------------//
	m_pModel = new DeepCNF_Model(ws1_,window_str,node_str,state_num,local_num,0);
	m_pModel->Gate_Function=GATE_FUNCTION;
	U_INT WD=m_pModel->total_param_count;
if(proc_id==0) cout << "parameter number: "<< WD << " --------------------------------------------------"<<endl;
	for(int i=0;i<state_num;i++)m_pModel->label_weight[i]=label_weight[i];

	//------ load weights from model_file or randomize weights -----//
	double *w00 = new double[WD];
	if(proc_id==0)
	{
		//-> random init parameter
		Parameter_Initialization(m_pModel->tot_layer,m_pModel->layer_count,m_pModel->layer_weight);
		m_pModel->MultiDim_To_OneDim(m_pModel->tot_layer,m_pModel->layer_count,m_pModel->layer_weight,w00);
		//-> use trained model or not
		if(use_trained_model)
		{
if(proc_id==0) cout << "to use the input model from file: " << model_file <<endl;
	
			ifstream ifmodel(model_file.c_str());
			for(U_INT i=0;i<WD;i++) if( ! ( ifmodel>>w00[i] ) ) break;
		}
	}

	//---- MPI BARRIER ---//
#ifdef _MPI
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(w00, WD, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	//---- MPI BARRIER ---//over

	vector<double> w0 (WD,0);
	for (U_INT i = 0; i < WD; i++) w0[i]=w00[i];
	delete [] w00;
	delete m_pModel;
	//===================== initilize weights ================//over




	//========= pre-training or not ==========//start
	if(iterative_number_str!="-1" || regularizer_str!="-1" )
	{
if(proc_id==0) cout << "start pre-training: " << iterative_number_str << " , " << regularizer_str << endl;

		//-> check null status
		if(iterative_number_str=="-1")
		{
			vector <double> iterative_num;
			for(int i=0;i<ws1_;i++)iterative_num.push_back(fine_tune_number);
			char separator=',';
			Parse_Double(iterative_num, iterative_number_str,separator);
		}
		if(regularizer_str=="-1")
		{
			vector <double> iterative_reg;
			for(int i=0;i<ws1_;i++)iterative_reg.push_back(regularizer_);
			char separator=',';
			Parse_Double(iterative_reg, regularizer_str,separator);
		}

		//-> start pre-training
		vector<double> w0_ (WD,0);
		LayerWise_Training_Initialize(
			range_out,local_num_ori,
			train_list_file,test_list_file,num_procs,proc_id,
			window_str,node_str,state_num,local_num,
			iterative_number_str,regularizer_str,w0,w0_);
		w0=w0_;

	}
	//========= pre-training or not ==========//over


	//============================== MAIN TRAINING PROCEDURE HERE ================//start
	METHOD=method_;
	{
		//--------- init model and load data-----//
		if(METHOD!=2)m_pModel = new DeepCNF_Model(ws1_,window_str,node_str,state_num,local_num,1);
		else m_pModel = new DeepCNF_Model(ws1_,window_str,node_str,state_num,local_num,1,max_auc_degree,max_auc_beta);
		for(int i=0;i<state_num;i++)m_pModel->label_weight[i]=label_weight[i];
		m_pModel->Gate_Function=GATE_FUNCTION;
		//data structure
		vector <vector <string> > feat_in_train;
		vector <vector <int> > label_in_train;
		vector <vector <string> > feat_in_test;
		vector <vector <int> > label_in_test;
		//load data
		LoadData(train_list_file, num_procs, proc_id, local_num_ori,range_out,feat_in_train, label_in_train);
		LoadData(test_list_file, num_procs, proc_id, local_num_ori,range_out,feat_in_test, label_in_test);
		//init data
		InitData(feat_in_train,label_in_train,m_pModel,trainData);
		InitData(feat_in_test,label_in_test,m_pModel,testData);

if(proc_id==0) cout << "start: " << fine_tune_number << " , " << regularizer_ <<endl;

		//----- run LBFGS -----//
		_LBFGS *lbfgs = new _LBFGS;
//		if(fine_tune_number==0) //-> 0 iteration
		{
			if(METHOD==2)Max_AUC_DataStructure_Init();
			lbfgs->regularizer=regularizer_;
			double objective=lbfgs->ComputeFunction(w0);
			lbfgs->Report(w0, 0, objective, 0);
		}
		if(fine_tune_number>0)  //-> more iterations
		{
			//--> set model_outdir and regularizer
			char command[30000];
			sprintf(command,"mkdir -p %s/",model_outdir_.c_str());
			int retv=system(command);
			lbfgs->model_outdir=model_outdir_;
			lbfgs->regularizer=regularizer_;
			//--> training
			lbfgs->LBFGS(w0,fine_tune_number);
		}
		delete lbfgs;

		//------ free all data and model ----//
		if(METHOD==2)Max_AUC_DataStructure_Dele();
		for (U_INT i = 0; i < trainData.size(); i++) delete trainData[i];
		for (U_INT i = 0; i < testData.size(); i++) delete testData[i];
		delete m_pModel;
	}
	//============================== MAIN TRAINING PROCEDURE HERE ================//over


	//----- MPI finalize -----//
#ifdef _MPI
	MPI_Barrier( MPI_COMM_WORLD);
	MPI_Finalize();
#endif
	exit(0);
}

