#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <sys/time.h>

int form_matrix(char input[],double mat[][500],int *i,int *j,bool test)
{
	int l=0,k=0;
	char temp[10000];
	int number_of_cols=0;
	while(input[k]!='\n')
    {  		
    	if(input[k]!=' ')
	    {
	   		temp[l]=input[k];
	   		l++;
	   		k++;
	   	}
    	else if (input[k]==' ') 
    	{
    		for (int z=l;z<10000;z++)
    			temp[z]=0;
    		mat[*i][*j]=atof(temp);
    		strcpy(temp," ");
    		l=0;
    		k++;
    		(*j)++;
    		number_of_cols++;    		
    	}
    }
   	if (input[k]=='\n' && test!=true)
   	{
   		for (int z=l;z<10000;z++)
    			temp[z]=0;
   		mat[*i][*j]=atof(temp);
    	strcpy(temp," ");
    	l=0;
    	k=0;
    	(*i)++;
    	*j=0;
    	number_of_cols++;
   	}
   	input[0]='\0';
   	return number_of_cols;
}
__global__ void
convolution(double *a, double *h, double *c,int c_rows, int c_cols,int z,int i,int j,int k)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x; //idx
	int m = blockIdx.y * blockDim.y + threadIdx.y; //idy
  //printf("%d %d\n",m,n);
  if (m<c_rows && n<c_cols)
  {
    for(int p=0;p<=(j-1);p++)
   	{
   		for(int q=0;q<=(k-1);q++)
   		{
      if(!((m-p)<0 || (n-q)<0 || (m-p)>=z || (n-q)>=i))
      {
        c[(m*c_cols)+n]+=h[(p*k)+q]*a[((m-p)*i)+(n-q)];			
        __syncthreads();
      }
   		}
   	}
  }
}
int main(int argc, char **argv)
{
	FILE *read_file;
	char input[10000];
	int e=0,d=0,m=0,k=0,j=0,n=0,select=1,u=0,v=0;
	double a[500][500],h[500][500];
  cudaError_t err = cudaSuccess;
	int flag1=0,flag2=0;
  char *input_file;
  input_file=argv[1];
	read_file=fopen(input_file,"r");
	if (read_file==NULL)
	{
		printf("Error opening file\n");
		exit(1);
	}
	while(fgets(input,10000,read_file)) 
    {
    	bool test=false;

    	if (strcmp(input,"\n")==0)
    	{
    		select=2;
    		test=true;
    	}
    	if (select==1)
    	{
    		(m)++;
        if (test!=true && flag1==0)
        {
    		  n=form_matrix(input,a,&e,&d,test);
          flag1=1;
        }else
          form_matrix(input,a,&e,&d,test);
    	}
    	else if (select==2)
    	{
    		(j)++;
    		if (test!=true && flag2==0)
    		{
    			k=form_matrix(input,h,&u,&v,test);
    			flag2=1;
    		}
    		else 
    			form_matrix(input,h,&u,&v,test);
    	}
    	input[0]='\0';
    }
    --j;
    double *h_a=NULL,*h_h=NULL,*h_c=NULL;
   
    int c_rows=(m+j-1);
    int c_cols=(n+k-1);
    size_t size_a=(m*n)*sizeof(double);

    err=cudaMallocManaged((void**)&h_a, size_a);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix a (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    for (int i=0;i<m;i++)
    {
      for(int j=0;j<n;j++)
      {
        h_a[(i*n)+j]=a[i][j];
      }
    }

    size_t size_h=(j*k)*sizeof(double);

    err=cudaMallocManaged((void**)&h_h, size_h);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix h (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    for (int p=0;p<j;p++)
    {
      for(int q=0;q<k;q++)
      {
        h_h[(p*k)+q]=h[p][q];
      }
    }

    size_t size_c=(c_rows*c_cols)*sizeof(double);
    err=cudaMallocManaged((void**)&h_c, size_c);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix c (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    dim3 threadsPerBlock(32,32);
    dim3 numBlocks(c_rows/threadsPerBlock.x+1,c_cols/threadsPerBlock.y+1);
    //struct timeval begin, end;
    //gettimeofday(&begin, NULL);
    convolution<<<numBlocks, threadsPerBlock>>>(h_a,h_h, h_c,c_rows, c_cols, m, n, j, k);
    //gettimeofday(&end, NULL);
    err = cudaGetLastError();

    //int time_in_us = 1e6*(end.tv_sec-begin.tv_sec) + (end.tv_usec-begin.tv_usec);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
   err = cudaDeviceSynchronize();
  
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize the device (error code: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
   	
   	for (int i=0;i<(m+j-1);i++)
    {
      for(int z=0;z<(n+k-1);z++)
      {
        printf("%0.3lf ",h_c[(i*c_cols)+z]);
      }
      printf("\n");
    }
    //printf("Time for V2 Kernel = %d us\n", time_in_us);
    err = cudaFree(h_a);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device matrix A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
  
    err = cudaFree(h_h);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device matrix h (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(h_c);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device matrix c (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    fclose(read_file);
	return 0;
}