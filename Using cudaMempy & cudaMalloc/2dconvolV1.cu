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
        //printf("%s\n",temp);
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
      //printf("%s\n",temp);
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
convolution(const double *a, const double *h, double *c,int c_rows, int c_cols,int z,int i,int j,int k)
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
        }
        else
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
    /*printf("Size of matrix 1:%d * %d\n",m,n);
    printf("Size of matrix 2:%d * %d\n",j,k);
   	for (int i=0;i<5;i++)
   	{
   		for(int j=0;j<5;j++)
   		{
   			printf("%f ",a[i][j]);
   		}
   		printf("\n");
   	}
   	for (int i=0;i<j;i++)
   	{
   		for(int z=0;z<k;z++)
   		{
   			printf("%f ",h[i][z]);
   		}
   		printf("\n");
   	}*/
    size_t size_a=(m*n)*sizeof(double);
    double *h_a=(double*)malloc(size_a);
    for (int i=0;i<m;i++)
    {
      for(int j=0;j<n;j++)
      {
        h_a[(i*n)+j]=a[i][j];
      }
    }
   	double *d_a=NULL;
   	err = cudaMalloc((void **)&d_a, size_a);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix a (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    size_t size_h=(j*k)*sizeof(double);
    double *h_h=(double*)malloc(size_h);
    for (int p=0;p<j;p++)
    {
      for(int q=0;q<k;q++)
      {
        h_h[(p*k)+q]=h[p][q];
      }
    }
    double *d_h=NULL;
   	err = cudaMalloc((void **)&d_h, size_h);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix h (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    int c_rows=(m+j-1);
    int c_cols=(n+k-1);
	  double *d_c=NULL;
   	size_t size_c=(c_rows*c_cols)*sizeof(double);
    double *c=(double*)malloc(size_c);
   	err = cudaMalloc((void **)&d_c, size_c);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix c (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix a from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_h, h_h, size_h, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix h from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    dim3 threadsPerBlock(32,32);
    dim3 numBlocks(c_rows/threadsPerBlock.x+1,c_cols/threadsPerBlock.y+1);
    //struct timeval begin, end;
    //gettimeofday(&begin, NULL);
    convolution<<<numBlocks, threadsPerBlock>>>(d_a, d_h, d_c,c_rows, c_cols, m, n, j, k);
    //gettimeofday(&end, NULL);
  
    //int time_in_us = 1e6*(end.tv_sec-begin.tv_sec) + (end.tv_usec-begin.tv_usec);
    err = cudaGetLastError();

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

    err = cudaMemcpy(c, d_c, size_c, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix c from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
   	
   	for (int i=0;i<(m+j-1);i++)
   	{
   		for(int z=0;z<(n+k-1);z++)
   		{
   			printf("%0.3lf ",c[(i*c_cols)+z]);
   		}
   		printf("\n");
   	}
    //printf("Time for V1 Kernel = %d us\n", time_in_us);
   	err = cudaFree(d_a);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device matrix a (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_h);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device matrix h (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_c);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device matrix c (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    free(h_a);
    free(h_h);
    free(c);
    fclose(read_file);
	return 0;
}