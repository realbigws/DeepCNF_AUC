#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <vector> 
#include <iostream>
#include <fstream>
#include <ostream>
#include <sstream>
#include <cmath>
using namespace std;


//-------- utility ------//
void getBaseName(string &in,string &out,char slash,char dot)
{
	int i,j;
	int len=(int)in.length();
	for(i=len-1;i>=0;i--)
	{
		if(in[i]==slash)break;
	}
	i++;
	for(j=len-1;j>=0;j--)
	{
		if(in[j]==dot)break;
	}
	if(j==-1)j=len;
	out=in.substr(i,j-i);
}
void getRootName(string &in,string &out,char slash)
{
	int i;
	int len=(int)in.length();
	for(i=len-1;i>=0;i--)
	{
		if(in[i]==slash)break;
	}
	if(i<=0)out=".";
	else out=in.substr(0,i);
}

//---------- input a string, output a vector -----//
int String_To_Vector(string &input,vector <double> &output)
{
	istringstream www(input);
	output.clear();
	int count=0;
	double value;
	for(;;)
	{
		if(! (www>>value) )break;
		output.push_back(value);
		count++;
	}
	return count;
}


//------ load matrix file ------//
int Load_Matrix(string &input_file, vector < vector < double > > &output_mat)
{
	//start
	ifstream fin;
	string buf,temp;
	//read
	fin.open(input_file.c_str(), ios::in);
	if(fin.fail()!=0)
	{
		fprintf(stderr,"input_file %s not found!\n",input_file.c_str());
		return -1;
	}
	//load
	int count=0;
	int colnum;
	int colcur;
	int first=1;
	output_mat.clear();
	vector <double> tmp_rec;
	for(;;)
	{
		if(!getline(fin,buf,'\n'))break;
		colcur=String_To_Vector(buf,tmp_rec);
		if(first==1)
		{
			first=0;
			colnum=colcur;
		}
		else
		{
			if(colcur!=colnum)
			{
				fprintf(stderr,"current column number %d not equal to the first column number %d \n",
					colcur,colnum);
				return -1;
			}
		}
		output_mat.push_back(tmp_rec);
		count++;
	}
	//return
	return count;
}


//------ load label file -------//
//-> example
/*
>1a0sP
0000000001111111111111111111
*/
int Load_LAB_File(string &lab_file,vector <int> &lab_number)
{
	//start
	ifstream fin;
	string buf,temp;
	//read
	fin.open(lab_file.c_str(), ios::in);
	if(fin.fail()!=0)return -1;
	//skip
	for(int i=0;i<2;i++)
	{
		if(!getline(fin,buf,'\n'))
		{
			fprintf(stderr,"file %s format bad!\n",lab_file.c_str());
			exit(-1);
		}
	}
	//load
	lab_number.clear();
	int count=0;
	for(int i=0;i<(int)buf.length();i++)
	{
		char c=buf[i];
		int lab=c-'0';
		lab_number.push_back(lab);
	}
	//return
	return (int)buf.length();
}


//-------------- for prediction -------------//
//given label and matrix, generate feat for DeepCNF
void Feature_Make(string &matrix_file,string &label_file)
{
	//-- load mat file --//
	vector < vector < double > > output_mat;
	int mat_len=Load_Matrix(matrix_file, output_mat);
	if(mat_len<=0)exit(-1);
	//-- load lab_file --//
	vector <int> lab_number;
	int lab_len=Load_LAB_File(label_file,lab_number);
	//-- check --//
	if(mat_len!=lab_len)
	{
		lab_number.clear();
		for(int i=0;i<mat_len;i++)lab_number.push_back(-1);
	}
	int length=mat_len;
	int featdim=output_mat[0].size();

	//==== generate feature =====//
	vector <string> output;
	output.clear();
	for(int k=0;k<length;k++)
	{
		//------ output ------//
		stringstream oss;
		for(int i=0;i<featdim;i++)
		{
			int wsiii=(int)output_mat[k][i];
			if(wsiii!=output_mat[k][i])oss << output_mat[k][i] << " ";
			else oss << wsiii << " ";
		}
		string wsbuf=oss.str();
		output.push_back(wsbuf);
	}
	//-> printf
	printf("%d\n",length);
	for(int k=0;k<length;k++)printf("%s\n",output[k].c_str());
	for(int k=0;k<length;k++)printf("%d\n",lab_number[k]);
}


//----------- main -------------//
int main(int argc,char **argv)
{
	//------- DeepCNF_FeatMake --------// 
	{
		if(argc<3)
		{
			printf("DeepCNF_FeatMake <matrix_file> <label_file> \n");
			printf("[note]: the label_file should be two lines !!! \n");
			exit(-1);
		}
		string matrix_file=argv[1];
		string label_file=argv[2];
		//process
		Feature_Make(matrix_file,label_file);
		exit(0);
	}
}

