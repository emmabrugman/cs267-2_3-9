
    #include "common.h"
    #include <cuda.h>
    #include <thrust/device_vector.h>
    #include <thrust/sort.h>
    #include <thrust/scan.h>
    #include <thrust/execution_policy.h>
    #include <thrust/transform.h>
    
    #define NUM_THREADS 256  // Define number of threads per block for CUDA kernels

/*
    //number of threads X number of blocks - lecture 9 
    //number of blocks consistent and we control number of threads( they give us) - 


    number of bins per row x number of bins per row to round up == TRY with the row? 

    inclusive scan -- cumulative sum of all the indices = 
        = inclusive each element at i includes the value at i in the sum 
    
    
    exclusive scan = i-1 but not i itself 

    
*/



    // Number of blocks used for GPU execution
    int blks;
    
    // Global device vectors to store particle bin information
    thrust::device_vector<int> d_particle_bins;      // Stores bin index for each particle
    thrust::device_vector<int> d_bin_counts;        // Stores number of particles per bin
    thrust::device_vector<int> d_bin_start_indices; // Stores start index of each bin in sorted array
    thrust::device_vector<int> d_particle_ids;      // Stores particle indices after sorting into bins
    thrust::device_vector<int> d_bin_offsets;       // Tracks bin offset while placing particles
    
    // Global variables for bin dimensions
    int num_bins_x;  // Number of bins along X-axis
    int total_bins;  // Total number of bins in the simulation space
    

// Computes bin ID for a given particle based on its position

//needed to use thrust transform (error)
struct ComputeBinId {
    double bin_size; // Size of a single bin
    int num_bins_x;  // Number of bins along X-axis

    // Constructor to initialize bin_size and number of bins
    __host__ __device__ ComputeBinId(double size, int num_bins_x)
    : bin_size(size / max(num_bins_x, 1)), num_bins_x(num_bins_x) {}

    // Functor to compute bin ID based on particle position  
    __host__ __device__ int operator()(const particle_t& p) const {
        int bin_x = static_cast<int>(p.x / bin_size);
        int bin_y = static_cast<int>(p.y / bin_size);

        // bin indices within valid range
        bin_x = max(0, min(bin_x, num_bins_x - 1));
        bin_y = max(0, min(bin_y, num_bins_x - 1));

        // Convert 2D bin index to 1D array index
        return bin_x + bin_y * num_bins_x;
    }
};




__global__ void assign_bins(int* particle_bins, particle_t* parts, int num_parts, double bin_size, int num_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_parts) {
        // Correctly compute the bin indices
        int bin_x = static_cast<int>(parts[idx].x / bin_size);
        int bin_y = static_cast<int>(parts[idx].y / bin_size);

        // Ensure bin indices are within bounds
        bin_x = max(0, min(bin_x, num_bins - 1));
        bin_y = max(0, min(bin_y, num_bins - 1));

        // Correct 2D to 1D bin indexing
        int bin_id = bin_y * num_bins + bin_x;


        if (idx < 10 || idx % 100 == 0) {  // Print first 10 and every 100th particle
            printf("[DEBUG] Particle %d -> Position (%.3f, %.3f) -> Bin (%d, %d) -> Bin ID %d\n",
                   idx, parts[idx].x, parts[idx].y, bin_x, bin_y, bin_id);
        }
        

        // Debugging print statement for first few particles


        // Sanity check
        if (bin_id < 0 || bin_id >= num_bins * num_bins) {
            printf("[ERROR] Particle %d assigned to INVALID bin %d (x=%.3f, y=%.3f)\n", 
                   idx, bin_id, parts[idx].x, parts[idx].y);
        }

        // Assign bin to particle
        particle_bins[idx] = bin_id;
    }
}


// Count Particles Per Bin
__global__ void count_particles_per_bin(int* bin_counts, int* particle_bins, int num_parts, int num_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_parts) {
        int bin_id = particle_bins[idx];
        if (bin_id >= 0 && bin_id < num_bins * num_bins) {
            int old_value = atomicAdd(&bin_counts[bin_id], 1);
            
            // ✅ Print BEFORE and AFTER atomicAdd()
            if (idx < 10) {
                printf("[GPU] Particle %d: Bin %d (before=%d, after=%d)\n",
                       idx, bin_id, old_value, old_value + 1);
            }
        } else {
            printf("[GPU] ERROR: Particle %d invalid bin ID %d\n", idx, bin_id);
        }
    }
}



// Place Particles in Their Bins
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
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
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

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
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

    cudaDeviceReset();  // Ensures proper cleanup when the program exits
}



void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

        // Determine number of bins based on simulation space and cutoff distance
num_bins_x = min(static_cast<int>(size / cutoff), num_parts);  // Limit bins to at most num_parts
total_bins = num_bins_x * num_bins_x;

std::cout << "[DEBUG] num_parts: " << num_parts 
<< ", num_bins_x: " << num_bins_x 
<< ", total_bins: " << total_bins 
<< std::endl;


    // Resize global Thrust vectors for binning
    d_particle_bins.resize(num_parts);
    d_bin_counts.resize(total_bins, 0);
    d_bin_start_indices.resize(total_bins, 0);
    d_particle_ids.resize(num_parts);
    d_bin_offsets.resize(total_bins, 0);

    // Assign particles to bins using Thrust transform -- applies computebinid functor to compute bin ID for all particles in parallel ******
    thrust::transform(thrust::device, parts, parts + num_parts, d_particle_bins.begin(),
                      ComputeBinId(size, num_bins_x));

    // Reset bin counts before counting - d_bin_counts to zero before countint particles
    //"0.0"
    thrust::fill(thrust::device, d_bin_counts.begin(), d_bin_counts.end(), 0);

    int grid_size = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    // Count particles per bin --Cuda kernel to count particles per bin
    count_particles_per_bin<<<grid_size, NUM_THREADS>>>(
        thrust::raw_pointer_cast(d_bin_counts.data()), 
        thrust::raw_pointer_cast(d_particle_bins.data()), 
        num_parts, total_bins);
    cudaDeviceSynchronize();

    //Debugging: Print bin counts after counting particles
    std::vector<int> h_bin_counts(total_bins);
    cudaMemcpy(h_bin_counts.data(), thrust::raw_pointer_cast(d_bin_counts.data()), 
               total_bins * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::cout << "[DEBUG] Initial Non-Empty Bins:\n";
    for (int i = 0; i < total_bins; i++) {
        if (h_bin_counts[i] > 0) {
            std::cout << "Bin " << i << " has " << h_bin_counts[i] << " particles\n";
        }
    }
    

    // Compute prefix sum for bin start indices - determin the bin start indices *** 
    thrust::exclusive_scan(d_bin_counts.begin(), d_bin_counts.end(), d_bin_start_indices.begin());
    cudaDeviceSynchronize();

    // Reset bin offsets before placing particles in bins
    thrust::fill(thrust::device, d_bin_offsets.begin(), d_bin_offsets.end(), 0);

    // Place particles in bins -based on the computed bin indices = reseting the bin_offsets to 0 before updating in place particles
    place_particles_in_bins<<<grid_size, NUM_THREADS>>>(
        thrust::raw_pointer_cast(d_particle_ids.data()),
        thrust::raw_pointer_cast(d_bin_start_indices.data()),
        thrust::raw_pointer_cast(d_particle_bins.data()),
        num_parts,
        thrust::raw_pointer_cast(d_bin_offsets.data()), total_bins);
    cudaDeviceSynchronize();
}

    
    // // Cleanup
    // cudaFree(particle_bins_gpu);
    // cudaFree(bin_counts_gpu);
    // cudaFree(bin_start_indices_gpu);
    // cudaFree(bin_offsets_gpu);


    __global__ void compute_forces_bin_based(
        int* particle_ids, int* bin_start_indices, int* bin_counts, 
        int* particle_bins, particle_t* parts, int num_parts, int num_bins_x, int total_bins, double size) 
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_parts) return;
    
        // Find the bin this particle belongs to
        int bin_id = particle_bins[tid];  // Get the precomputed bin ID
        int bin_x = bin_id % num_bins_x;
        int bin_y = bin_id / num_bins_x;
    
        // Iterate over the bin and its 8 neighboring bins
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int neighbor_x = bin_x + dx;
                int neighbor_y = bin_y + dy;
    
                if (neighbor_x >= 0 && neighbor_x < num_bins_x &&
                    neighbor_y >= 0 && neighbor_y < num_bins_x) 
                {
                    int neighbor_bin = neighbor_x + neighbor_y * num_bins_x;
                    if (neighbor_bin < 0 || neighbor_bin >= total_bins) continue;  // to stay within the total bin bounds
    
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
        num_bins_x = min(static_cast<int>(size / cutoff), num_parts);  // Limit bins to at most num_parts
        total_bins = num_bins_x * num_bins_x;
        
        int blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    
        // Compute Forces (should use bins, but currently applies all-to-all)
        compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts);
        cudaDeviceSynchronize();
    
        // Move Particles
        move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
        cudaDeviceSynchronize();
    
        //  Reassign Bins After Movement
        ComputeBinId bin_id_op(size, num_bins_x);
        thrust::transform(thrust::device, parts, parts + num_parts, d_particle_bins.begin(), bin_id_op);
        cudaDeviceSynchronize();
    
        // Reset Bin Counts Before Recomputing
        thrust::fill(thrust::device, d_bin_counts.begin(), d_bin_counts.end(), 0);
        cudaDeviceSynchronize();
    
        //Count Particles Per Bin
        count_particles_per_bin<<<(num_parts + 255) / 256, 256>>>(
            thrust::raw_pointer_cast(d_bin_counts.data()), 
            thrust::raw_pointer_cast(d_particle_bins.data()), 
            num_parts, total_bins);
        cudaDeviceSynchronize();  
    
        //Compute Prefix Sum for Bin Start Indices
        thrust::exclusive_scan(d_bin_counts.begin(), d_bin_counts.end(), d_bin_start_indices.begin());
        cudaDeviceSynchronize();
    
        //Reset Bin Offsets Before Reassigning Particles
        thrust::fill(thrust::device, d_bin_offsets.begin(), d_bin_offsets.end(), 0);
    
        //Reassign Particles to Bins
        place_particles_in_bins<<<(num_parts + 255) / 256, 256>>>(
            thrust::raw_pointer_cast(d_particle_ids.data()),
            thrust::raw_pointer_cast(d_bin_start_indices.data()),
            thrust::raw_pointer_cast(d_particle_bins.data()),
            num_parts,
            thrust::raw_pointer_cast(d_bin_offsets.data()), total_bins);
        cudaDeviceSynchronize();
    
        // Compute Forces Efficiently Using Neighbor Bins (TODO: Implement this)
        compute_forces_bin_based<<<blks, NUM_THREADS>>>(
            thrust::raw_pointer_cast(d_particle_ids.data()),  
            thrust::raw_pointer_cast(d_bin_start_indices.data()),  
            thrust::raw_pointer_cast(d_bin_counts.data()),  
            thrust::raw_pointer_cast(d_particle_bins.data()),  // ✅ Pass this explicitly
            parts, num_parts, num_bins_x, total_bins, size);
        
    
        //Debugging Bin Assignments
        std::vector<int> h_bin_counts(total_bins);
        cudaMemcpy(h_bin_counts.data(), thrust::raw_pointer_cast(d_bin_counts.data()), 
                   total_bins * sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    
        std::cout << "[DEBUG] Total Particles Counted in Bins: " 
                  << thrust::reduce(d_bin_counts.begin(), d_bin_counts.end(), 0, thrust::plus<int>())
                  << " / " << num_parts << std::endl;
    
                  std::cout << "[DEBUG] After move Non-Empty Bins:\n";
                  for (int i = 0; i < total_bins; i++) {
                      if (h_bin_counts[i] > 0) {
                          std::cout << "Bin " << i << " has " << h_bin_counts[i] << " particles\n";
                      }
                  }
                  
    }
    


    

