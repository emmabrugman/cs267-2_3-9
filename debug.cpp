#include "common.h"
#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;
int num_bins;

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

__global__ void bin_particles(particle_t* particles, int* bin_counts, int num_parts, double size, int num_bins) {
    /*
    particles (array): original array of particles shared across all blocks
    bin_counts (array) : stores the number of particles in each bin
    num_parts (int) : total number of particles
    size : size of the simulation space
    */
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;


    int bin_x = (int)(particles[tid].x / size * num_bins);
    int bin_y = (int)(particles[tid].y / size * num_bins);
    int bin_idx = bin_y * num_bins + bin_x;

    // increment the count of the correspond bin 
    // use atomic operations because there will be multiple
    // particles with the same bin id
   if (bin_idx >= 0 && bin_idx < (num_bins * num_bins)) {
        atomicAdd(&bin_counts[bin_idx], 1);
   }

}

__global__ void reorder_particles(particle_t* particles, particle_t* sorted_particles, int* bin_offsets, int num_parts, double size, int num_bins) {
    /*
        particles (array): original array of particles shared across all blocks
        sorted_partciles (array) : contain all particles in orignal array, sorted by bin ID
        bin_offsets (array) : prefix sum array of bin counts-- each values corresponds to the starting index of each bin
        num_parts (int) : total number of particles
        size : size of simulation space
        Kernel reorders particles into new array based on their bin indices. Ensures particles fromm the same bin
        are stored contiguously in memory
    */
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    int bin_x = (int)(particles[tid].x / size * num_bins);
    int bin_y = (int)(particles[tid].y / size * num_bins);
    int bin_idx = bin_y * num_bins + bin_x;

    // atomic operation prevents mutliple threads from accessing shared array
    // gets the current offset of the bin
       if (bin_idx >= 0 && bin_idx < num_bins * num_bins) {
        // incremeber the approproate bin_offset value, and records old one in idx
        int idx = atomicAdd(&bin_offsets[bin_idx], 1);
        if (idx >= 0 && idx < num_parts) {
            sorted_particles[idx] = particles[tid];
        }
    }

}

__global__ void compute_forces_gpu(particle_t* particles, int num_parts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particles[tid].ax = particles[tid].ay = 0;
    for (int j = 0; j < num_parts; j++)
        apply_force_gpu(particles[tid], particles[j]);
    
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
    double bin_size = 2.0 * cutoff;
    int num_bins_dimension = (int)(size / bin_size);
    num_bins = num_bins_dimension * num_bins_dimension;
}


void simulate_one_step(particle_t* parts, int num_parts, double size) {
    int* bin_counts;
    int* bin_offsets;
    size_t bin_counts_size = num_bins * num_bins * sizeof(int);
    size_t bin_offsets_size = num_bins * num_bins * sizeof(int);

    // allocates memory for bin_counts and bin_offsets GLOBALLY
    cudaMalloc((void**)&bin_counts, bin_counts_size);
    cudaMalloc((void**)&bin_offsets, bin_offsets_size);

    
    cudaMemset(bin_counts, 0, bin_counts_size);

    // allocate particles to bins
    bin_particles<<<blks, NUM_THREADS>>>(parts, bin_counts, num_parts, size, num_bins);

    // wrap pointers to use thrust 
    thrust::device_ptr<int> thrust_bin_counts(bin_counts);
    thrust::device_ptr<int> thrust_bin_offsets(bin_offsets);

    // determine prefix sum of bin_counts 
    thrust::inclusive_scan(thrust_bin_counts, thrust_bin_counts + (num_bins * num_bins), thrust_bin_offsets);

    // allocates memory for sorted_particles and bin_offsets GLOBALLY
    particle_t* sorted_particles;
    cudaMalloc((void**)&sorted_particles, num_parts * sizeof(particle_t));

    reorder_particles<<<blks, NUM_THREADS>>>(parts, sorted_particles, bin_offsets, num_parts, size, num_bins);

    compute_forces_gpu<<<blks, NUM_THREADS>>>(sorted_particles, num_parts);
    move_gpu<<<blks, NUM_THREADS>>>(sorted_particles, num_parts, size);
    
    // copy sorted particles back to parts
    cudaMemcpy(parts, sorted_particles, num_parts * sizeof(particle_t), cudaMemcpyDeviceToDevice);
    

    cudaFree(bin_counts);
    cudaFree(bin_offsets);
    cudaFree(sorted_particles);
}