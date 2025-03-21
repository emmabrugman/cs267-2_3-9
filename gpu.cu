#include "common.h"
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <iostream>

#define NUM_THREADS 256

int blks;
int num_bins_x;
int total_bins;

struct BinningData {
    thrust::device_vector<int> d_particle_bins;
    thrust::device_vector<int> d_bin_counts;
    thrust::device_vector<int> d_bin_start_indices;
    thrust::device_vector<int> d_particle_ids;
    thrust::device_vector<int> d_bin_offsets;

    void clear() {
        d_particle_bins.clear();
        d_bin_counts.clear();
        d_bin_start_indices.clear();
        d_particle_ids.clear();
        d_bin_offsets.clear();

        d_particle_bins.shrink_to_fit();
        d_bin_counts.shrink_to_fit();
        d_bin_start_indices.shrink_to_fit();
        d_particle_ids.shrink_to_fit();
        d_bin_offsets.shrink_to_fit();
    }
};

BinningData* g_bin_data = nullptr;

struct ComputeBinId {
    double bin_size;
    int num_bins_x;

    __host__ __device__ ComputeBinId(double size, int num_bins_x)
        : bin_size(size / max(num_bins_x, 1)), num_bins_x(num_bins_x) {}

    __host__ __device__ int operator()(const particle_t& p) const {
        int bin_x = static_cast<int>(p.x / bin_size);
        int bin_y = static_cast<int>(p.y / bin_size);
        bin_x = max(0, min(bin_x, num_bins_x - 1));
        bin_y = max(0, min(bin_y, num_bins_x - 1));
        return bin_x + bin_y * num_bins_x;
    }
};

__global__ void assign_bins(int* particle_bins, particle_t* parts, int num_parts, double bin_size, int num_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_parts) {
        int bin_x = static_cast<int>(parts[idx].x / bin_size);
        int bin_y = static_cast<int>(parts[idx].y / bin_size);
        bin_x = max(0, min(bin_x, num_bins - 1));
        bin_y = max(0, min(bin_y, num_bins - 1));
        int bin_id = bin_y * num_bins + bin_x;
        particle_bins[idx] = bin_id;
    }
}

__global__ void count_particles_per_bin(int* bin_counts, int* particle_bins, int num_parts, int num_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_parts) {
        int bin_id = particle_bins[idx];
        if (bin_id >= 0 && bin_id < num_bins * num_bins) {
            atomicAdd(&bin_counts[bin_id], 1);
        }
    }
}

__global__ void place_particles_in_bins(int* particle_ids, int* bin_start_indices,
                                        int* particle_bins, int num_particles, int* bin_offsets, int num_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles) {
        int bin_id = particle_bins[idx];
        if (bin_id < num_bins * num_bins) {
            int insert_idx = atomicAdd(&bin_offsets[bin_id], 1);
            particle_ids[bin_start_indices[bin_id] + insert_idx] = idx;
        }
    }
}

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff) return;
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__global__ void compute_forces_gpu(particle_t* particles, int num_parts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;
    particles[tid].ax = particles[tid].ay = 0;
    for (int j = 0; j < num_parts; j++)
        apply_force_gpu(particles[tid], particles[j]);
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;

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

void cleanup_simulation() {
    std::cout << "[DEBUG] Begin cleanup_simulation()\n";

    if (g_bin_data) {
        g_bin_data->clear();
        delete g_bin_data;
        g_bin_data = nullptr;
        std::cout << "[DEBUG] Deleted g_bin_data\n";
    }

    std::cout << "[DEBUG] Calling cudaDeviceReset()\n";
    cudaDeviceReset();
    std::cout << "[DEBUG] Finished cleanup_simulation()\n";
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    g_bin_data = new BinningData();

    num_bins_x = min(static_cast<int>(size / cutoff), num_parts);
    total_bins = num_bins_x * num_bins_x;

    g_bin_data->d_particle_bins.resize(num_parts);
    g_bin_data->d_bin_counts.resize(total_bins, 0);
    g_bin_data->d_bin_start_indices.resize(total_bins, 0);
    g_bin_data->d_particle_ids.resize(num_parts);
    g_bin_data->d_bin_offsets.resize(total_bins, 0);

    thrust::transform(thrust::device, parts, parts + num_parts, g_bin_data->d_particle_bins.begin(),
                      ComputeBinId(size, num_bins_x));
    thrust::fill(thrust::device, g_bin_data->d_bin_counts.begin(), g_bin_data->d_bin_counts.end(), 0);

    int grid_size = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    count_particles_per_bin<<<grid_size, NUM_THREADS>>>(
        thrust::raw_pointer_cast(g_bin_data->d_bin_counts.data()),
        thrust::raw_pointer_cast(g_bin_data->d_particle_bins.data()),
        num_parts, total_bins);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(g_bin_data->d_bin_counts.begin(), g_bin_data->d_bin_counts.end(), g_bin_data->d_bin_start_indices.begin());
    thrust::fill(thrust::device, g_bin_data->d_bin_offsets.begin(), g_bin_data->d_bin_offsets.end(), 0);

    place_particles_in_bins<<<grid_size, NUM_THREADS>>>(
        thrust::raw_pointer_cast(g_bin_data->d_particle_ids.data()),
        thrust::raw_pointer_cast(g_bin_data->d_bin_start_indices.data()),
        thrust::raw_pointer_cast(g_bin_data->d_particle_bins.data()),
        num_parts,
        thrust::raw_pointer_cast(g_bin_data->d_bin_offsets.data()), total_bins);
    cudaDeviceSynchronize();
}

__global__ void compute_forces_bin_based(
    int* particle_ids, int* bin_start_indices, int* bin_counts,
    int* particle_bins, particle_t* parts, int num_parts, int num_bins_x, int total_bins, double size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts) return;

    int bin_id = particle_bins[tid];
    int bin_x = bin_id % num_bins_x;
    int bin_y = bin_id / num_bins_x;

    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            int neighbor_x = bin_x + dx;
            int neighbor_y = bin_y + dy;

            if (neighbor_x >= 0 && neighbor_x < num_bins_x &&
                neighbor_y >= 0 && neighbor_y < num_bins_x)
            {
                int neighbor_bin = neighbor_x + neighbor_y * num_bins_x;
                if (neighbor_bin < 0 || neighbor_bin >= total_bins) continue;

                int start = bin_start_indices[neighbor_bin];
                int end = start + bin_counts[neighbor_bin];

                for (int i = start; i < end; i++) {
                    apply_force_gpu(parts[tid], parts[particle_ids[i]]);
                }
            }
        }
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    num_bins_x = min(static_cast<int>(size / cutoff), num_parts);
    total_bins = num_bins_x * num_bins_x;

    int blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts);
    cudaDeviceSynchronize();

    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
    cudaDeviceSynchronize();

    ComputeBinId bin_id_op(size, num_bins_x);
    thrust::transform(thrust::device, parts, parts + num_parts, g_bin_data->d_particle_bins.begin(), bin_id_op);
    cudaDeviceSynchronize();

    thrust::fill(thrust::device, g_bin_data->d_bin_counts.begin(), g_bin_data->d_bin_counts.end(), 0);
    cudaDeviceSynchronize();

    count_particles_per_bin<<<(num_parts + 255) / 256, 256>>>(
        thrust::raw_pointer_cast(g_bin_data->d_bin_counts.data()),
        thrust::raw_pointer_cast(g_bin_data->d_particle_bins.data()),
        num_parts, total_bins);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(g_bin_data->d_bin_counts.begin(), g_bin_data->d_bin_counts.end(), g_bin_data->d_bin_start_indices.begin());
    thrust::fill(thrust::device, g_bin_data->d_bin_offsets.begin(), g_bin_data->d_bin_offsets.end(), 0);

    place_particles_in_bins<<<(num_parts + 255) / 256, 256>>>(
        thrust::raw_pointer_cast(g_bin_data->d_particle_ids.data()),
        thrust::raw_pointer_cast(g_bin_data->d_bin_start_indices.data()),
        thrust::raw_pointer_cast(g_bin_data->d_particle_bins.data()),
        num_parts,
        thrust::raw_pointer_cast(g_bin_data->d_bin_offsets.data()), total_bins);
    cudaDeviceSynchronize();

    compute_forces_bin_based<<<blks, NUM_THREADS>>>(
        thrust::raw_pointer_cast(g_bin_data->d_particle_ids.data()),
        thrust::raw_pointer_cast(g_bin_data->d_bin_start_indices.data()),
        thrust::raw_pointer_cast(g_bin_data->d_bin_counts.data()),
        thrust::raw_pointer_cast(g_bin_data->d_particle_bins.data()),
        parts, num_parts, num_bins_x, total_bins, size);

    std::vector<int> h_bin_counts(total_bins);
    cudaMemcpy(h_bin_counts.data(), thrust::raw_pointer_cast(g_bin_data->d_bin_counts.data()),
               total_bins * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}


//hw2-correctness/correctness-check.py gpu2.out correct.out
/*
Traceback (most recent call last):
  File "/global/homes/s/sabrinat/hw2-correctness/correctness-check.py", line 84, in <module>
    check_conditions( avg_dists )
  File "/global/homes/s/sabrinat/hw2-correctness/correctness-check.py", line 68, in check_conditions
    assert( np.mean( avg_dists[:50] ) < 3e-7 )
AssertionError
*/