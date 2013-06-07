#include <stdio.h>
#include "mpi.h"
#include <math.h>
#include <stdlib.h>
#include <strings.h>

typedef struct {
    int       nb_proc;    // Number of processus in our program 
    MPI_Comm  grid_comm;  // Communicator for the entire grid
    MPI_Comm  row_comm;   // Communicator for the specific row of our processus in the grid 
    MPI_Comm  col_comm;   // Same thing for column
    int       order;      // Order of our matrix to compute
    int       pos_row;    // Coordinates of our processus in the row_communicator
    int       pos_col;    // Same thing for column
    int       grid_rank;  // Global rank of our processus
} GRID_INFO_T;


// Initialization of the processus grid and all the communicators.
void init_grid(GRID_INFO_T* /*out*/);
// Read matrix from a file defined by a string. Result in filling a 2D matrix onto a continuous array of float
void read_matrix(GRID_INFO_T* /*in*/, char* /*in*/, float* /*out*/, int /*in*/);
// Display global matrix contained in second parameter and any blocks of that matrix
void display_matrix(GRID_INFO_T* /*in*/, float* /*in*/, float* /*in*/);
// Display global matrix
void display_matrix2(GRID_INFO_T* /*in*/, float* /*in*/);
// Multiply the first and second matrix (array of float) into the third one
void multiply_matrix(GRID_INFO_T* /*in*/, float* /*in*/, float* /*in*/, float* /*out*/);
// Variant of the above function
void multiply_matrix2(int, float*, float*, float*);
// Fox's algorithm. Multiply two first matrix into the third one. Avoid O(n^3)
void fox(GRID_INFO_T* /*in*/, float* /*in*/, float* /*in*/, float* /*out*/);
// Cannon's algorithm. Same as above
void cannon(GRID_INFO_T* /*in*/, float* /*in*/, float* /*in*/, float* /*out*/);


int main(int argc, char** argv)
{

	GRID_INFO_T grid;

	// Declaration of block and matrix that we will compute
	float* block_A; 
	float* block_B;
	float* block_C;
    	float* mat_A;
    	float* mat_B; 
    	float* mat_C;
	
	double time1, time2, time3, time4, time5; 


	time1 = MPI_Wtime();
	
    	// Initialization MPI and our GRID_INFO_T struct
	MPI_Init(&argc, &argv);
	init_grid(&grid);

	if(grid.grid_rank == 0)
	{
		mat_A = (float*) malloc(grid.order*grid.order * sizeof(float));
		mat_B = (float*) malloc(grid.order*grid.order * sizeof(float));
		mat_C = (float*) malloc(grid.order*grid.order * sizeof(float));
		read_matrix(&grid, "mat21", mat_A, grid.order);
		read_matrix(&grid, "mat22", mat_B, grid.order);		
	
		int mat_col;
		int mat_row;

		// We make sure that mat_C is initialized correctly. No need for mat_A and mat_B that are filled from files 
		for(mat_row = 0; mat_row < grid.order; ++mat_row)
		{
			for(mat_col = 0; mat_col < grid.order; ++mat_col)
			{
				
				mat_C[mat_row*grid.order+mat_col] = 0.0; 
			}
		}
	}


	 if(argc > 0)
        {
                if(strcasecmp(argv[1], "linear") == 0)
                {
                        if(grid.grid_rank==0)
			{
				printf("Linear computation\n \n");        
                        	multiply_matrix2(grid.nb_proc, mat_A, mat_B, mat_C);
                		display_matrix2(&grid, mat_C);
			}
			MPI_Finalize();
			return 0;  
		}
        }

	// In the next block of code, we are creating a new Datatype that will define data layout in our matrix
	// The array_size var indicates that our are of size nb_proc*nb_proc
	// Subarray_size indicates that our future blocks from matrix will be sqrt(nb_proc)*sqrt(nb_proc) blocks
	// Then we call MPI_Type_create_subarray that will put in blocktype the exact layout of our blocks
	// We then resize it due to MPI_FLOAT type being four time bigger than C float
	// In order to use the resized type we have to commit it

	MPI_Datatype blocktype, type;	

	int array_size[2] = {grid.nb_proc, grid.nb_proc};
	int subarray_sizes[2] = {(int)sqrt(grid.nb_proc), (int) sqrt(grid.nb_proc)};
	int array_start[2] = {0,0};
	
	MPI_Type_create_subarray(2, array_size, subarray_sizes, array_start, MPI_ORDER_C, MPI_FLOAT, &blocktype); 
	MPI_Type_create_resized(blocktype, 0, (int)sqrt(grid.nb_proc)*sizeof(float), &type);
	MPI_Type_commit(&type);
	
	int i, j;
	int displs[grid.nb_proc];
	int send_counts[grid.nb_proc];

	// For each processus, we are allocating a continuous array of float. 
	// They are our blocks used in Fox's and Cannon's algorithms. 

	block_A = (float*) malloc(grid.nb_proc*sizeof(float));
	block_B = (float*) malloc(grid.nb_proc*sizeof(float));
	block_C = (float*) malloc(grid.nb_proc*sizeof(float));

	for(i=0; i< grid.nb_proc; ++i)
		block_C[i] = 0.0; 


	// Setting the offset that we'll use in MPI_Scatterv. 
	// It indicates how much you have to shift in order to distribute the correct blocks to all our processus

	if (grid.grid_rank == 0) 
	{
		for(i=0; i<grid.nb_proc; i++) 
		{
			send_counts[i] = 1;
		}

		int disp = 0;
		for (i=0; i<(int)sqrt(grid.nb_proc); i++) {
			for (j=0; j<(int)sqrt(grid.nb_proc); j++) {
				displs[i*(int)sqrt(grid.nb_proc)+j] = disp;
				disp += 1;
			}
			disp += ((grid.nb_proc/(int)sqrt(grid.nb_proc)-1))*(int)sqrt(grid.nb_proc);
		}
	}

	// MPI_Scatterv takes the global matrix mat_A and mat_B then "subdivides" them into blocks that are sent to each
	// processus of our communicator MPI_COMM_WORLD
	// Note that we are using the Datatype we defined at #87 -> #89

	time2 = MPI_Wtime();

	MPI_Scatterv(mat_A, send_counts, displs, type, block_A,grid.nb_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);	
	MPI_Scatterv(mat_B, send_counts, displs, type, block_B,grid.nb_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);                                              


	int algorithm = 0;

	if(argc <= 0)
	{
		if(grid.grid_rank == 0)
			printf("Fox's algorithm\n \n");
	}
	else if(strcasecmp(argv[1], "cannon") == 0)
	{
		if(grid.grid_rank == 0)
			printf("Cannon's algorithm\n \n");
		algorithm = 1;
	}
	else if(strcasecmp(argv[1], "fox") == 0)
	{        
       		if(grid.grid_rank == 0)
			printf("Fox's algorithm\n \n");
	}


	// We are calling our algorithms based on user input
	// Note that we are only sending blocks from our matrix and not entire matrix

	time3 = MPI_Wtime();

	if(algorithm == 1)
		cannon(&grid, block_A, block_B, block_C);
	else if(algorithm == 0)
		fox(&grid, block_A, block_B, block_C);

	time4 = MPI_Wtime();

	// MPI_GATHERV is the inverse operation to MPI_SCATTERV
	// We are building the matrix back from the blocks

	MPI_Gatherv(block_C, grid.nb_proc,  MPI_FLOAT,
                 mat_C, send_counts, displs, type,
                 0, MPI_COMM_WORLD);

	time5 = MPI_Wtime();

	//display_matrix2(&grid, mat_A);
	//display_matrix2(&grid, mat_B);
	//display_matrix2(&grid, mat_C);

	if(grid.grid_rank == 0)
		printf("%f, %f, %f, %f, %f \n", time5-time4, time4-time3, time3-time2, time2-time1, time5-time1 );

	/*free(block_A);
        free(block_B);
        free(block_C);
        free(mat_A);
        free(mat_B);
        free(mat_C);*/
	
	MPI_Finalize();
	return 0;
}


void fox(GRID_INFO_T* grid, float* block_A, float* block_B, float* block_C)
{
	int src;		
	int dst;		
	int root = 0;
	int ON_DIAGONAL = 0;
	MPI_Status status;

	// Defines the size of a block. So each block is sqroot*sqroot

	int sqroot = (int) sqrt(grid->nb_proc);

	// We compute the source and destination for each processus
	// We use it when we have to shift B blocks 

	src = (grid->pos_row + 1) % sqroot;
	dst = (grid->pos_row - 1 + sqroot ) % sqroot;
	
	// We have to use those var to avoid losing our data contained in block_A and block_B

	float* temp_A = malloc(grid->nb_proc * sizeof(float));
	float* temp_B = malloc(grid->nb_proc * sizeof(float));
	
	int i;
	for(i=0; i < grid->nb_proc; ++i)
	{
		temp_A[i] = 0.;
		temp_B[i] = 0.;
	}

	// Waiting for every process to complete before starting the core of the algorithm

	MPI_Barrier(grid->row_comm);		
	
	// For each iterations we 
	// + find the blocks that are forming a diagonal
	// + diffuse that block on the row it belongs to
	// + multiply the updated block_A (or temp_A) with block_B onto block_C
	// + shift the B blocks upward

	for(i = 0; i < sqroot; i++ )
	{
		root = (grid->pos_row + i)%sqroot;
		if( root == grid->pos_col ){
			MPI_Bcast( block_A, grid->nb_proc, MPI_FLOAT, root, grid->row_comm);
			ON_DIAGONAL = 1;	
		}
		else{
			MPI_Bcast(temp_A, grid->nb_proc, MPI_FLOAT, root, grid->row_comm );
			ON_DIAGONAL = 0;
		}
		
		( ON_DIAGONAL )? multiply_matrix(grid, block_A, block_B,block_C ) : 
				multiply_matrix(grid, temp_A,  block_B, block_C ); 

	
		MPI_Send( block_B, grid->nb_proc, MPI_FLOAT, dst, 333, grid->col_comm);
		MPI_Recv( temp_B, grid->nb_proc, MPI_FLOAT, src, 333,grid->col_comm, &status);
		int k;
		for(k = 0; k < grid->nb_proc; ++k)
			block_B[k] = temp_B[k]; 
	}

}


void cannon(GRID_INFO_T* grid, float* block_A, float* block_B, float* block_C)
{
	int sqroot = sqrt(grid->nb_proc);
  	int shift_source, shift_dest;
	MPI_Status status;
	int up_rank, down_rank, left_rank, right_rank;
	int i;

	// Pre-skewing

	MPI_Cart_shift(grid->grid_comm, 1, -1, &right_rank, &left_rank); 
	MPI_Cart_shift(grid->grid_comm, 0, -1, &down_rank, &up_rank); 
	MPI_Cart_shift(grid->grid_comm, 1, -grid->pos_row, &shift_source, &shift_dest); 

	// Execute a blocking send and receive. 
	// The same buffer is used both for the send and for the receive
	// The sent data is replaced by received data

	MPI_Sendrecv_replace(block_A, sqroot*sqroot, MPI_FLOAT, shift_dest, 1, shift_source, 1, grid->grid_comm, &status); 

	MPI_Cart_shift(grid->grid_comm, 0, -grid->pos_col, &shift_source, &shift_dest); 
	MPI_Sendrecv_replace(block_B, sqroot*sqroot, MPI_FLOAT, shift_dest, 1, shift_source, 1, grid->grid_comm, &status); 
   
    for (i=0; i<sqroot; i++) 
    { 
        multiply_matrix(grid, block_A, block_B, block_C); 
        MPI_Sendrecv_replace(block_A, grid->nb_proc, MPI_FLOAT, left_rank, 1, right_rank, 1, grid->grid_comm, &status); 
        MPI_Sendrecv_replace(block_B, grid->nb_proc, MPI_FLOAT, up_rank, 1, down_rank, 1, grid->grid_comm, &status); 
    } 
   
   	// Post-skewing

    MPI_Cart_shift(grid->grid_comm, 1, +grid->pos_row, &shift_source, &shift_dest); 
    MPI_Sendrecv_replace(block_B, grid->nb_proc, MPI_FLOAT, shift_dest, 1, shift_source, 1, grid->grid_comm, &status); 

    MPI_Cart_shift(grid->grid_comm, 0, +grid->pos_col, &shift_source, &shift_dest); 
    MPI_Sendrecv_replace(block_B, grid->nb_proc, MPI_FLOAT, shift_dest, 1, shift_source, 1, grid->grid_comm, &status); 	
}


void multiply_matrix(GRID_INFO_T* grid, float* block_A, float* block_B, float* block_C)
{
	// Simple matrix multiplication

	int i, j, k;
        int sqroot = (int)sqrt(grid->nb_proc);

    	for (i = 0; i < sqroot; i++)
        	for (j = 0; j < sqroot; j++)
            		for (k = 0; k < sqroot; k++)
                		block_C[i*sqroot+j] += block_A[i*sqroot+k]*block_B[k*sqroot+j];
}

void multiply_matrix2(int size, float* block_A, float* block_B, float* block_C)
{
	int i, j, k;

	for (i = 0; i < size; i++)
                for (j = 0; j < size; j++)
                        for (k = 0; k < size; k++)
                                block_C[i*size+j] += block_A[i*size+k]*block_B[k*size+j];
}

void display_matrix2(GRID_INFO_T* grid, float* mat)
{
	int i;
	if (grid->grid_rank == 0) {
                printf("Global matrix: \n");
                int ii, jj;
                for (ii=0; ii< grid->order ; ii++) {
                    for (jj=0; jj< grid->order; jj++) {
                        if(log10((int)mat[ii*grid->order+jj])+1 <= 2)
							printf("%2d ",(int)mat[ii*grid->order+jj]);
						else if(log10((int)mat[ii*grid->order+jj])+1 > 2)
                                printf("%4d ",(int)mat[ii*grid->order+jj]);
                    }
                    printf("\n");                                                                                                                       
                }
		printf("\n\n");                                                                                                                                       
            }       
}


void display_matrix(GRID_INFO_T* grid, float* mat, float* block)
{
	int i;
	for (i = 0; i < grid->nb_proc; i++) {                                                                                                                
        if (i == grid->grid_rank) {                                                                                                                      
            printf("Rank = %d\n", grid->grid_rank);                                                                                                      
            if (grid->grid_rank == 0) {                                                                                                                  
                printf("Global matrix: \n");                                                                                                            
                int ii, jj;                                                                                                                             
                for (ii=0; ii< grid->order ; ii++) {                                                                                                     
                    for (jj=0; jj< grid->order; jj++) {                                                                                                  
                        printf("%3d ",(int)mat[ii*grid->order+jj]);                                                                                    
                    }                                                                                                                                   
                    printf("\n");                                                                                                                       
                }                                                                                                                                       
            }                                                                                                                                           
            printf("Local Matrix:\n");                                                                                                                  
            int ii, jj;                                                                                                                                 
            for (ii=0; ii<(int)sqrt(grid->order); ii++) {                                                                                                
                for (jj=0; jj<(int)sqrt(grid->order); jj++) {                                                                                            
                    printf("%3d ",(int)block[ii*(int)sqrt(grid->order)+jj]);                                                                            
                }                                                                                                                                       
                printf("\n");                                                                                                                           
            }                                                                                                                                           
            printf("\n");                                                                                                                               
        }
        MPI_Barrier(MPI_COMM_WORLD);                                                                                                                    
    }

}


void init_grid(GRID_INFO_T*  grid_info  /* out */) {
    int rank;
    int dims[2];
    int period[2];
    int coords[2];
    int free_coords[2];

    MPI_Comm_size(MPI_COMM_WORLD, &(grid_info->nb_proc));
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    grid_info->order = grid_info->nb_proc;
    dims[0] = dims[1] = (int) sqrt(grid_info->order);
    period[0] = period[1] = 1;

    // Create a grid of processus 
    // Store global rank in grid
    // Find the communicators and the coordinates for each processus

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, 1, &(grid_info->grid_comm));
    MPI_Comm_rank(grid_info->grid_comm, &(grid_info->grid_rank));
    MPI_Cart_coords(grid_info->grid_comm, grid_info->grid_rank, 2, coords);
    grid_info->pos_row = coords[0];
    grid_info->pos_col = coords[1];

    free_coords[0] = 0; 
    free_coords[1] = 1;
    MPI_Cart_sub(grid_info->grid_comm, free_coords, &(grid_info->row_comm));

    free_coords[0] = 1; 
    free_coords[1] = 0;
    MPI_Cart_sub(grid_info->grid_comm, free_coords, &(grid_info->col_comm));
} 


void read_matrix(GRID_INFO_T* info, char* namefile, float* mat, int order)
{
	// Open the file named namefile

	FILE* file = fopen(namefile, "r"); 
	if(file == NULL)
	{
		perror("invalid read");
		exit(1);  
	}

	// For each row and column we read a float and store it in the array of float mat.

	int col; 
	int row;
	for(row = 0; row < order; ++row)
	{
		for(col = 0; col < order; ++col)
		{
			fscanf(file, "%f", &mat[row*order+col]);
			// To skip whitespaces
			char c = fgetc(file);
		}
	}

	fclose(file);
}

