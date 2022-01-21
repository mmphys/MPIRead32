# MPIRead32

Reproduce MPI error when reading > 2GB to a single rank

Required argument 1: Filename used for destructive test
Optional argument 2: Size of header in bytes. Default 0
Optional argument 3:    MPI dimensions. Default: world_size,1,1,1
Optional argument 4: Global dimensions. Default: 48,48,48,96
NB: arguments 3 and four can be n-dimensional, but must match

##Fails:

    mpirun --mca io romio321 -np 2 MPIRead32 a.out 0 2.1 2304.4608
    mpirun --mca io romio321 -np 2 MPIRead32 a.out 0 2.1 4608.2304

##Succeeds:

    mpirun --mca io romio321 -np 2 MPIRead32 a.out 0 1.2 2304.4608
    mpirun --mca io romio321 -np 2 MPIRead32 a.out 0 1.2 4608.2304
    mpirun --mca io ompio    -np 2 MPIRead32 a.out 0 2.1 2304.4608
    mpirun --mca io ompio    -np 2 MPIRead32 a.out 0 2.1 4608.2304
    mpirun --mca io ompio    -np 2 MPIRead32 a.out 0 1.2 2304.4608
    mpirun --mca io ompio    -np 2 MPIRead32 a.out 0 1.2 4608.2304

##Hint:

Replacing both occurrences of MPI_ORDER_FORTRAN with MPI_ORDER_C
causes the success / fail MPI ordering with romio321 to reverse
