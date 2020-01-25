#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define  MASTER		0

float smooth_F[] = {1,1,1,1,1,1,1,1,1};
float blur_F[] = {1,2,1,2,4,2,1,2,1};
float sharpen_F[] = {0,-2,0,-2,11,-2,0,-2,0};
float mean_F[] = {-1,-1,-1,-1,9,-1,-1,-1,-1};
float emboss_F[] = {0,-1,0,0,0,0,0,1,0};

char img_type;
int width,height,nrch;
unsigned char max_val;
unsigned char* image;
int size;

int base_size;
int num_add,st,dr;

int numtasks, rank, len;

unsigned char* my_part;

void recv_image();
 
void init_data(char* name)
{
    int i,j,k;
	
	for(i=0;i<9;i++)
	{
		smooth_F[i]/=9;
		blur_F[i]/=16;
		sharpen_F[i]/=3;
	}
	char linieee; //ignorare linie in plus
	FILE* f;

	if (rank ==0)
	{
		f = fopen(name, "r"); //deschidere poza si citire tip poza
		fscanf(f, "%c", &img_type); fscanf(f, "%c", &img_type);

		
		fscanf(f, "%c", &linieee);
		linieee = 1;
		while (linieee != '\n')
			fscanf(f, "%c", &linieee);

		img_type -= '0';
		fscanf(f, "%d%d", &width, &height);
		fscanf(f, "%hhu", &max_val);
		if (img_type == 5) //determinare numar canale
			nrch = 1;
		else
			nrch = 3;
		MPI_Bcast(&max_val, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
		MPI_Bcast(&img_type, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
		MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&nrch, 1, MPI_INT, 0, MPI_COMM_WORLD);
	}
	else
	{
		MPI_Bcast(&max_val, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
		MPI_Bcast(&img_type, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
		MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&nrch, 1, MPI_INT, 0, MPI_COMM_WORLD);
	}

	size = width * height;
	image = (unsigned char*)calloc((width + 2) * (height + 2) * nrch, sizeof(unsigned char));

	for (i = 0; i < height + 2; i++) //bordare cu 0
		for (k = 0; k < nrch; k++)
			image[i * (width + 2) * nrch + k] = image[i * (width + 2) * nrch + (width + 1) * nrch + k] = 0;
	for (i = 0; i < width + 2; i++)
		for (k = 0; k < nrch; k++)
			image[i * nrch + k] = 0;
	int val = (width + 2) * (height + 2);
	for (i = (width + 2) * (height + 1); i < val; i++)
		for (k = 0; k < nrch; k++)
			image[i * nrch + k] = 0;

	if (rank == 0)
	{
		fscanf(f, "%c", &linieee); //citire /n in plus
		unsigned char* aux = image + (width + 3) * nrch; //pointerul catre pixelul actual
		for (i = 0; i < height; i++) //citire imagine
		{
			for (j = 0; j < width; j++)
				for (k = 0; k < nrch; k++)
				{
					fscanf(f, "%c", aux);
					aux++;
				}
			aux += 2 * nrch; //sarire peste bordura
		}
	}

	recv_image();
}

void write_image_ascii(char* name)
{
    int i,j,k;
    FILE* f=fopen(name,"w");
    if(img_type==5)
       fprintf(f,"P2\n");
    else
       fprintf(f,"P3\n");
    fprintf(f,"%d %d\n%d\n",width,height,max_val);
    unsigned char* aux=image+(width+3)*nrch;
    for(i=0;i<height;i++)
    {
        for(j=0;j<width;j++)
            for(k=0;k<nrch;k++)
            {
                fprintf(f,"%hhu ",(int)*aux);
                aux++;
            }
        aux+=2*nrch;
        fprintf(f,"\n");
    }
}

void write_image_binary(char* name)
{
    int i,j,k;
    FILE* f=fopen(name,"wb");
    fprintf(f,"P%c\n",img_type+'0');
    fprintf(f,"%d %d\n%d\n",width,height,max_val);
    unsigned char* aux=image+(width+3)*nrch;
    for(i=0;i<height;i++)
    {
        for(j=0;j<width;j++)
            for(k=0;k<nrch;k++)
            {
                fprintf(f,"%c",(unsigned char)(*aux));
                aux++;
            }
        aux+=2*nrch;
    }
}

int ok=0;


unsigned char filter_Pixel(unsigned char* img,float* kernel)
{
	ok++;
	float rez = 0;
	img -= (width + 2 + 1) * nrch; rez += kernel[0] * (*img); 
    img+=nrch; rez+=kernel[1]*(*img); 
    img+=nrch; rez+=kernel[2]*(*img); 
    img+=(width+2)*nrch; img-=2*nrch;	rez+=kernel[3]*(*img); 
    img+=nrch; rez+=kernel[4]*(*img); 
    img+=nrch; rez+=kernel[5]*(*img); 
    img+=(width+2)*nrch; img-=2*nrch; rez+=kernel[6]*(*img); 
    img+=nrch; rez+=kernel[7]*(*img); 
    img+=nrch; rez+=kernel[8]*(*img); 
    if(rez<0) rez=0;
    if(rez>max_val) rez=max_val;
    return (unsigned char)rez;
}

	

void apply_filter(char* name,int st,int dr)
{
    float* filter;
    if(!strcmp(name,"smooth"))
    {
        filter=smooth_F;
    }
    if(!strcmp(name,"blur"))
    {
        filter=blur_F;
    }
    if(!strcmp(name,"sharpen"))
    {
        filter=sharpen_F;
    }
    if(!strcmp(name,"mean"))
    {
        filter=mean_F;
    }
    if(!strcmp(name,"emboss"))
    {
        filter=emboss_F;
    }
	
	unsigned char* aux;
	unsigned char* img_idx;
	
	int num;
	int num_rows,i,j,k;
	for (k = 0; k < 1; k++)
	{
		num = st;
		aux = my_part;
		num_rows = st / (width + 2);
		img_idx = image + (width + 3 + st + 2 * (st /width)) * nrch;
	//	printf("%d %d %d\n", st, dr, (width + 3 + st + 2 * (st / width)) * nrch);
		while (num <= dr)
		{
			for (i = 0; i < nrch; i++)
			{
				*aux = filter_Pixel(img_idx, filter);
				aux++;
				img_idx++;
			}
		//	printf("%d ", num);
			num++;
			if (num % width == 0)
			{
				img_idx += 2 * nrch;
			//	printf("\n");
			}
		}
		//printf("\n");
	}
}

int min(int a,int b)
{
    if(a<b) return a;
    return b;
}

int max(int a,int b)
{
    if(a>b) return a;
    return b;
}

void send_image()
{
	int i,j,k;
	
	unsigned char* aux;
	unsigned char* img_idx;
	int num;
	int st1,dr1;
	st1=st; dr1=dr;
	
	if(rank==0)
    {
        i=0;
        while(1)
        {
			aux=my_part;
            num=st1;
            img_idx=image+(width+3+ st1+ 2*(st1/width))*nrch;
			while(num<=dr1)
            {
                for(j=0;j<nrch;j++)
                {
                    *img_idx=*aux;
                    aux++;
                    img_idx++;
                }
                num++;
                if(num%width==0)
                    img_idx+=2*nrch;
            }
            st1=(base_size+1)*min(i+1,num_add) + (base_size)*max(0,i+1-num_add);
            dr1 = st1 + base_size + (i+1<num_add)-1;
            i++;
            if(i>=numtasks)
                break;
            MPI_Recv(my_part,(dr1-st1+1)*nrch,MPI_CHAR,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
    }
	else
	{
		MPI_Send(my_part, (dr1 - st1 + 1) * nrch, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
	}
}

void recv_image()
{
	int i,sz = (width+2)*(height+2)*nrch;
	if(rank==0)
	{
		for(i=1;i<numtasks;i++)
			MPI_Send(image,sz,MPI_CHAR,i,0,MPI_COMM_WORLD);
	}
	else
		MPI_Recv(image,sz,MPI_CHAR,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
}

int main (int argc, char *argv[])
{
   
    char hostname[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Get_processor_name(hostname, &len);
    init_data(argv[1]);

	MPI_Barrier(MPI_COMM_WORLD);

	base_size=size/numtasks;
	num_add=size%numtasks;
    int size_part=0;

    st=(base_size+1)*min(rank,num_add) + (base_size)*max(0,rank-num_add);
    dr = st + base_size + (rank<num_add)-1;
	my_part=(unsigned char*)calloc((dr-st+1)*nrch,sizeof(unsigned char));
	
	int i;
    for(i=3;i<argc;i++)
    {
		apply_filter(argv[i],st,dr);
		send_image();
		if(i!=argc-1)
			recv_image();
    }

    if(rank==0)
        write_image_binary(argv[2]);
    MPI_Finalize();
    return 0;
}

