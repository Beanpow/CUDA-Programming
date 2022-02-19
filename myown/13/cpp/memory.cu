#include "memory.cuh"
#include <stdlib.h>
#include "error.cuh"

void allocate_memory(int N, int MN, Atom *atom)
{
    atom->NN = (int*) malloc(N * sizeof(int));
    atom->NL = (int*) malloc(N * MN * sizeof(int));
    atom->m  = (real*) malloc(N * sizeof(real));
    atom->x  = (real*) malloc(N * sizeof(real));
    atom->y  = (real*) malloc(N * sizeof(real));
    atom->z  = (real*) malloc(N * sizeof(real));
    atom->vx = (real*) malloc(N * sizeof(real));
    atom->vy = (real*) malloc(N * sizeof(real));
    atom->vz = (real*) malloc(N * sizeof(real));
    atom->fx = (real*) malloc(N * sizeof(real));
    atom->fy = (real*) malloc(N * sizeof(real));
    atom->fz = (real*) malloc(N * sizeof(real));
    atom->pe = (real*) malloc(N * sizeof(real));
    atom->ke = (real*) malloc(N * sizeof(real));
    atom->box = (real*) malloc(6 * sizeof(real));


    CHECK(cudaMalloc((void**)&atom->g_NN, N * sizeof(int)));
    CHECK(cudaMalloc((void**)&atom->g_NL, N * MN * sizeof(int)));
    CHECK(cudaMalloc((void**)&atom->g_x, N * sizeof(real)));
    CHECK(cudaMalloc((void**)&atom->g_y, N * sizeof(real)));
    CHECK(cudaMalloc((void**)&atom->g_z, N * sizeof(real)));
    CHECK(cudaMalloc((void**)&atom->g_fx, N * sizeof(real)));
    CHECK(cudaMalloc((void**)&atom->g_fy, N * sizeof(real)));
    CHECK(cudaMalloc((void**)&atom->g_fz, N * sizeof(real)));
    CHECK(cudaMalloc((void**)&atom->g_pe, N * sizeof(real)));
}

void deallocate_memory(Atom *atom)
{
    free(atom->NN);
    free(atom->NL);
    free(atom->m);
    free(atom->x);
    free(atom->y);
    free(atom->z);
    free(atom->vx);
    free(atom->vy);
    free(atom->vz);
    free(atom->fx);
    free(atom->fy);
    free(atom->fz);
    free(atom->pe);
    free(atom->ke);
    free(atom->box);

    CHECK(cudaFree(atom->g_NN));
    CHECK(cudaFree(atom->g_NL));
    CHECK(cudaFree(atom->g_x));
    CHECK(cudaFree(atom->g_y));
    CHECK(cudaFree(atom->g_z));
    CHECK(cudaFree(atom->g_fx));
    CHECK(cudaFree(atom->g_fy));
    CHECK(cudaFree(atom->g_fz));
    CHECK(cudaFree(atom->g_pe));
}

