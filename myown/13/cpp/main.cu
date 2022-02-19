#include "common.cuh"
#include "memory.cuh"
#include "initialize.cuh"
#include "neighbor.cuh"
#include "integrate.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int main(int argc, char **argv)
{
    int nx = 5;
    int Ne = 20000;
    int Np = 20000;
    cudaSetDevice(0);

    if (argc != 3) 
    { 
        printf("Usage: %s nx Ne\n", argv[0]);
        exit(1);
    }
    else
    {
        nx = atoi(argv[1]);
        Ne = atoi(argv[2]);
        Np = Ne;
    }

    int N = 4 * nx * nx * nx;
    int Ns = 100;
    int MN = 200; // max number of neighbors
    real T_0 = 60.0;
    real ax = 5.385;
    real time_step = 5.0 / TIME_UNIT_CONVERSION;
    Atom atom;
    allocate_memory(N, MN, &atom);
    for (int n = 0; n < N; ++n) { atom.m[n] = 40.0; }
    initialize_position(nx, ax, &atom);
    initialize_velocity(N, T_0, &atom);
    find_neighbor(N, MN, &atom);
    clock_t startTime = clock();
    CHECK(cudaDeviceSynchronize());
    equilibration(Ne, N, MN, T_0, time_step, &atom);
    printf("%g\n", float(clock() - startTime) / CLOCKS_PER_SEC);
    production(Np, Ns, N, MN, T_0, time_step, &atom);
    printf("%g\n", float(clock() - startTime) / CLOCKS_PER_SEC);
    deallocate_memory(&atom);
    return 0;
}

