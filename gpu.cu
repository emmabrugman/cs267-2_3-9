#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "common.h"
#include <cuda.h>
#include <stdio.h>

#define NUM_THREADS 256
#define BIN_SIZE 0.01 

int blks;

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;

    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__global__ void compute_forces_gpu(particle_t* particles, int num_parts, int* bins, int num_bins_per_row, int num_bins) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    int bin_id = bins[tid];
    int bx = bin_id % num_bins_per_row;
    int by = bin_id / num_bins_per_row;

    particles[tid].ax = 0;
    particles[tid].ay = 0;

    __shared__ particle_t shared_particles[NUM_THREADS];

    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            int nbx = bx + dx;
            int nby = by + dy;

            if (nbx >= 0 && nbx < num_bins_per_row && nby >= 0 && nby < num_bins_per_row) {
                int neighbor_bin = nbx + nby * num_bins_per_row;
                int bin_start = bins[neighbor_bin];
                int bin_end = (neighbor_bin + 1 < num_bins) ? bins[neighbor_bin + 1] : num_parts;

                for (int j = bin_start + threadIdx.x; j < bin_end; j += blockDim.x) {
                    shared_particles[threadIdx.x] = particles[j];
                    __syncthreads();

                    for (int k = 0; k < blockDim.x && (bin_start + k) < bin_end; k++) {
                        apply_force_gpu(particles[tid], shared_particles[k]);
                    }
                    __syncthreads();
                }
            }
        }
    }
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];

    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
}

__global__ void computeBinIDs(int *bin_ids, particle_t *parts, int num_parts, double size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < num_parts) {
        int num_bins_per_row = (int)(size / BIN_SIZE);
        int bx = (int)(parts[i].x / BIN_SIZE);
        int by = (int)(parts[i].y / BIN_SIZE);
        bin_ids[i] = bx + by * num_bins_per_row;
    }
}
__global__ void countParticlesPerBin(int *bin_counts, int *bin_ids, int num_parts) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < num_parts) {
        atomicAdd(&bin_counts[bin_ids[i]], 1);
    }
}

__global__ void assignParticlesToBins(particle_t *sorted_parts, int *bins, particle_t *parts, int *bin_ids, int num_parts) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < num_parts) {
        int bin_id = bin_ids[i];
        int insert_idx = atomicAdd(&bins[bin_id], 1);
        sorted_parts[insert_idx] = parts[i];
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    int num_bins_per_row = (int)ceil(size / BIN_SIZE);
    int num_bins = num_bins_per_row * num_bins_per_row;

    particle_t *gpu_parts, *gpu_sorted_parts;
    int *bin_ids, *bin_counts, *bins;

    cudaMalloc(&gpu_parts, num_parts * sizeof(particle_t));
    cudaMalloc(&gpu_sorted_parts, num_parts * sizeof(particle_t));
    cudaMalloc(&bin_ids, num_parts * sizeof(int));
    cudaMalloc(&bin_counts, num_bins * sizeof(int));
    cudaMalloc(&bins, (num_bins + 1) * sizeof(int));

    cudaMemset(bin_counts, 0, num_bins * sizeof(int));

    cudaMemcpy(gpu_parts, parts, num_parts * sizeof(particle_t), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (num_parts + threadsPerBlock - 1) / threadsPerBlock;

    computeBinIDs<<<numBlocks, threadsPerBlock>>>(bin_ids, gpu_parts, num_parts, size);
    cudaDeviceSynchronize();

    countParticlesPerBin<<<numBlocks, threadsPerBlock>>>(bin_counts, bin_ids, num_parts);
    cudaDeviceSynchronize();

    thrust::inclusive_scan(thrust::device, bin_counts, bin_counts + num_bins, bins);
    cudaDeviceSynchronize();

    cudaMemcpy(bins + 1, bins, (num_bins - 1) * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemset(bins, 0, sizeof(int));

    assignParticlesToBins<<<numBlocks, threadsPerBlock>>>(gpu_sorted_parts, bins, gpu_parts, bin_ids, num_parts);
    cudaDeviceSynchronize();

    compute_forces_gpu<<<blks, NUM_THREADS>>>(gpu_sorted_parts, num_parts, bins, num_bins_per_row, num_bins);
    cudaDeviceSynchronize();

    move_gpu<<<blks, NUM_THREADS>>>(gpu_sorted_parts, num_parts, size);
    cudaDeviceSynchronize();

    cudaMemcpy(parts, gpu_sorted_parts, num_parts * sizeof(particle_t), cudaMemcpyDeviceToHost);

    cudaFree(gpu_parts);
    cudaFree(gpu_sorted_parts);
    cudaFree(bin_ids);
    cudaFree(bin_counts);
    cudaFree(bins);
}

//hw2-correctness/correctness-check.py gpu2.out correct.out