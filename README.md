#Fox & Cannon implementation, MPI
###*Florian Gasc, Silouane Gerin*

##Compilation

* mpicc main.c -o gemm

##Execution

* mpirun -n 16 gemm fox
* mpirun -n 64 gemm cannon
* mpirun -n 100 gemm linear

