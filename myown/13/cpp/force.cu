#include "force.cuh"
#include "mic.cuh"
#include "error.cuh"

struct Constants
{
    real cutoff_square;
    real e24s6;
    real e48s12;
    real e4s6;
    real e4s12;
    int MaxN;
};

void __global__ gpu_find_force(
    Constants constants, int N, int *g_NN, int *g_NL, real *box,
    real *g_x, real *g_y, real *g_z, real *g_pe, real *g_fx, real *g_fy, real *g_fz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    for (int k = 0; k < g_NN[i]; k++)
    {
        // int j = g_NL[i * constants.MaxN + k];
        continue;
        // if (j < i)
        //     continue;

        // real dx = g_x[i] - g_x[j];
        // real dy = g_y[i] - g_y[j];
        // real dz = g_z[i] - g_z[j];
        // apply_mic(box, &dx, &dy, &dz);

        // real r2 = dx * dx + dy * dy + dz * dz;
        // if (r2 > constants.cutoff_square)
        //     continue;

        // real r2inv = 1.0 / r2;
        // real r4inv = r2inv * r2inv;
        // real r6inv = r2inv * r4inv;
        // real r8inv = r4inv * r4inv;
        // real r12inv = r4inv * r8inv;
        // real r14inv = r6inv * r8inv;

        // real f = constants.e24s6 * r8inv - constants.e48s12 * r14inv;
        // g_pe[i] += constants.e4s12 * r12inv - constants.e4s12 * r14inv;
        // g_fx[i] += f * dx;
        // g_fx[j] -= f * dx;
        // g_fy[i] += f * dy;
        // g_fy[j] -= f * dy;
        // g_fz[i] += f * dz;
        // g_fz[j] -= f * dz;
    }
}

void find_force(int N, int MN, Atom *atom)
{
    int *NN = atom->NN;
    int *NL = atom->NL;
    real *x = atom->x;
    real *y = atom->y;
    real *z = atom->z;
    real *fx = atom->fx;
    real *fy = atom->fy;
    real *fz = atom->fz;
    real *pe = atom->pe;
    real *box = atom->box;
    const real epsilon = 1.032e-2;
    const real sigma = 3.405;
    const real cutoff = 10.0;
    const real cutoff_square = cutoff * cutoff;
    const real sigma_3 = sigma * sigma * sigma;
    const real sigma_6 = sigma_3 * sigma_3;
    const real sigma_12 = sigma_6 * sigma_6;
    const real e24s6 = 24.0 * epsilon * sigma_6;
    const real e48s12 = 48.0 * epsilon * sigma_12;
    const real e4s6 = 4.0 * epsilon * sigma_6;
    const real e4s12 = 4.0 * epsilon * sigma_12;

    Constants constants;
    constants.cutoff_square = cutoff_square;
    constants.e24s6 = e24s6;
    constants.e48s12 = e48s12;
    constants.e4s12 = e4s12;
    constants.e4s6 = e4s6;
    constants.MaxN = MN;

    int blockSize = 128;
    int gridSize = (N - 1) / blockSize + 1;
    int m = sizeof(real) * N;

    for (int n = 0; n < N; ++n)
    {
        fx[n] = fy[n] = fz[n] = pe[n] = 0.0;
    }

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(atom->g_NL, atom->NL, N * MN * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(atom->g_NN, NN, N * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(atom->g_x, atom->x, m, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(atom->g_y, atom->y, m, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(atom->g_z, atom->z, m, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(atom->g_pe, atom->pe, m, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(atom->g_fx, atom->fx, m, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(atom->g_fy, atom->fy, m, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(atom->g_fz, atom->fz, m, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());


    gpu_find_force<<<gridSize, blockSize>>>(
        constants, N, atom->g_NN, atom->g_NL, box, atom->g_x, atom->g_y, atom->g_z,
        atom->pe, atom->fx, atom->fy, atom->fz);
}
