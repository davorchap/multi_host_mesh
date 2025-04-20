#pragma once
#include <mpi.h>
#include <cstdint>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <string>
#include <sstream>
#include <iomanip>
#include <limits> // Required for numeric_limits
#include <string>   // Required for std::stoi

namespace mesh {

// Debug Print Configuration Namespace
namespace Debug {
    enum class Mode { NONE, ALL, SPECIFIC_RANK };
    static Mode current_mode = Mode::NONE;
    static int target_rank = -1; // Target rank if mode is SPECIFIC_RANK

    // Configure debug printing. Should be called after MPI_Init potentially.
    inline void configure(Mode mode, int rank_id = -1) {
        current_mode = mode;
        target_rank = rank_id;
        // Optional: Add barrier if config needs to be globally consistent immediately
        // MPI_Barrier(MPI_COMM_WORLD); 
    }

    // Check if the current rank should print debug messages
    inline bool should_print(int current_process_rank) {
        switch (current_mode) {
            case Mode::NONE:
                return false;
            case Mode::ALL:
                return true;
            case Mode::SPECIFIC_RANK:
                return current_process_rank == target_rank;
        }
        return false; // Should not happen
    }
} // namespace Debug

struct Shape { 
    uint32_t x{}, y{}; 
    Shape(uint32_t x_val = 0, uint32_t y_val = 0) : x(x_val), y(y_val) {}
};
struct Range { 
    uint32_t start{}, end{}; 
    Range(uint32_t s = 0, uint32_t e = 0) : start(s), end(e) {}
};

// Moved DeviceCQ and Device definitions after Shape/Range
struct DeviceCQ {
    std::vector<uint64_t> cmds_;
};

class Device {
public:
    Shape global_coords; // Global coordinates of this device
    Shape local_coords;  // Local coordinates within the host submesh
    DeviceCQ cq_;        // Command queue for this specific device

    explicit Device(Shape global_c, Shape local_c) 
        : global_coords(global_c), local_coords(local_c) {}

    void print_creation_info(int rank) const {
        if (Debug::should_print(rank)) {
            std::cout << "[rank " << rank << "] Initialized Device @ global (" 
                      << global_coords.x << "," << global_coords.y << ") / local (" 
                      << local_coords.x << "," << local_coords.y << ")\n";
        }
    }
};

inline std::string to_string(const Shape& s) {
    return std::to_string(s.x) + "x" + std::to_string(s.y);
}

inline std::string to_string(const Range& r) {
    return "[" + std::to_string(r.start) + ".." + std::to_string(r.end) + ")";
}

inline bool is_power_of_2(uint32_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

inline Shape validate_mesh_shape(Shape shape) {
    if (!is_power_of_2(shape.x) || !is_power_of_2(shape.y)) {
        std::cerr << "Error: mesh dimensions must be powers of 2\n";
        std::exit(1);
    }
    return shape;
}

inline Shape validate_host_submesh_shape(Shape mesh_shape, Shape host_submesh_shape) {
    if (!is_power_of_2(host_submesh_shape.x) || !is_power_of_2(host_submesh_shape.y)) {
        std::cerr << "Error: host submesh dimensions must be powers of 2\n";
        std::exit(1);
    }
    if (mesh_shape.x % host_submesh_shape.x != 0 || mesh_shape.y % host_submesh_shape.y != 0) {
        std::cerr << "Error: host submesh must evenly divide mesh dimensions\n";
        std::exit(1);
    }
    return host_submesh_shape;
}

struct Validation {
    static void enabled(bool on) { instance().on_ = on; }
    static bool on()             { return instance().on_; }
private:
    bool on_ = true;
    static Validation& instance() { static Validation v; return v; }
};

class HostBuffer {
public:
    void*  ptr()  { return data_; }
    size_t size() const { return bytes_; }

private:
    friend class MeshBuffer;
    explicit HostBuffer(size_t sz) : bytes_(sz) { data_ = std::malloc(sz); }
    void*  data_;
    size_t bytes_;
};

class MeshBuffer {
public:
    size_t   bytes() const { return shape_.x * shape_.y; }
    HostBuffer host_view() const;

private:
    friend class MeshDevice;
    MeshBuffer(uint64_t b, Shape shape, Shape owning_mesh_shape) : base_(b), shape_(shape), owning_mesh_shape_(owning_mesh_shape) {}
    uint64_t  base_;
    Shape     shape_;
    Shape     owning_mesh_shape_;
};

class MeshWorkload {
public:
    explicit MeshWorkload(std::vector<uint64_t>&& words, Shape target_mesh_shape)
        : cmds_(std::move(words)), target_mesh_shape_(target_mesh_shape) 
    {
        // Print informational message if debug enabled for this rank
        int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (Debug::should_print(rank)) {
            std::cout << "[rank " << rank << "] Creating MeshWorkload for target mesh " 
                      << to_string(target_mesh_shape_) << "...\n";
        }

        if (!Validation::on()) return;
        /* hash for lock‑step test */
        uint64_t h = 0;
        for (auto w : cmds_) h ^= w * 0x9ddfea08eb382d69ULL;
        uint64_t x;
        MPI_Allreduce(&h, &x, 1, MPI_UINT64_T, MPI_BXOR, MPI_COMM_WORLD);
        assert(x == 0 && "ranks diverged while building workload");
        // Print success message if debug enabled for this rank
        if (Debug::should_print(rank)) {
            std::cout << "[rank " << rank << "] Validation: MeshWorkload constructor for target mesh " 
                      << to_string(target_mesh_shape_) << " OK\n"; 
        }
    }
    const std::vector<uint64_t>& words() const { return cmds_; }

private:
    std::vector<uint64_t> cmds_;
    Shape target_mesh_shape_;
};

class MeshDevice; // Forward declaration

class MeshCQ {
public:
    // Constructor takes owning device
    explicit MeshCQ(MeshDevice& dev) : dev_(dev) {}

    // Push dispatches workload to local device CQs
    void push(const MeshWorkload& wl);
    
    // const std::vector<uint64_t>& cmds() const { return cmds_; } // REMOVE - No longer stores commands
private:
    // friend class MeshDevice; // Keep friendship if needed for cq_ init
    // std::vector<uint64_t> cmds_; // REMOVE
    MeshDevice& dev_; // Reference to owning device
};

struct HostSubmesh {
    Range x_range;
    Range y_range;
    Shape shape;

    std::string to_string() const {
        return "x" + mesh::to_string(x_range) + " y" + mesh::to_string(y_range) + 
               " shape=" + mesh::to_string(shape);
    }
};

class MeshDevice {
public:
    friend class MeshCQ; // Grant MeshCQ access to private members

    // Update signature to include config args
    static MeshDevice& open(Shape mesh_shape, Shape host_submesh_shape, 
                           bool enable_validation, Debug::Mode debug_mode, int debug_rank) 
    {
        // Configure validation and debugging *early* so constructor messages are gated
        // Note: MPI is guaranteed to be initialized within the constructor called below
        Validation::enabled(enable_validation);
        Debug::configure(debug_mode, debug_rank);
        
        // Static local guarantees construction only happens once
        static MeshDevice dev(mesh_shape, host_submesh_shape); 
        return dev;
    }
    static void close() { get().teardown(); }

    MeshBuffer allocate(Shape shape);
    // Add overload for overriding owning mesh shape
    MeshBuffer allocate(Shape shape, Shape owning_mesh_shape_override);
    MeshCQ&    cq() { return cq_; }

    void dispatch_pending();   /* encode only rank‑local cmds (stub) */
    void wait();               /* poll + final barrier          (stub) */

    int rank()  const { return rank_;  }
    int world() const { return world_; }
    Shape host_submesh_shape() const { return host_submesh_shape_; }
    Shape mesh_shape() const { return mesh_shape_; }
    const HostSubmesh& host_submesh() const { return host_submesh_; }

private:
    // Private helper for allocation logic
    MeshBuffer allocate_impl(Shape buffer_shape, Shape owning_mesh_shape);

    explicit MeshDevice(Shape mesh_shape, Shape host_submesh_shape);
    void teardown();
    static MeshDevice& get() {
        // If get() is called, constructor must have run, so first_call_done is true.
        assert(first_call_done && "MeshDevice not initialized. Call open() first.");
        // Update call to open with default config values
        static MeshDevice& instance_ref = open({1,1}, {1,1}, 
                                                true, Debug::Mode::NONE, -1); // Validation ON, Debug NONE by default for internal get()
        return instance_ref;
    }
    void print_host_submesh_layout();

    void print_system_config() const {
        if (rank_ == 0) {
            std::cout << "\nSystem Configuration:\n"
                      << "  MeshDevice Shape: " << mesh_shape_.x << "x" << mesh_shape_.y << "\n"
                      << "  World Size: " << world_ << " ranks\n"
                      << "  Host SubMesh: " << host_submesh_shape_.x << "x" << host_submesh_shape_.y << "\n"
                      << "  Host Mesh: " << (mesh_shape_.x / host_submesh_shape_.x) << "x" 
                                       << (mesh_shape_.y / host_submesh_shape_.y) << "\n\n";
        }
    }

    int     rank_, world_;
    Shape   mesh_shape_;
    Shape   host_submesh_shape_;
    HostSubmesh host_submesh_;
    MeshCQ  cq_;
    static bool first_call_done;
    // Store local devices, not just their CQs
    std::vector<Device> local_devices_; // Devices locally owned by this host
};

inline MeshDevice::MeshDevice(Shape mesh_shape, Shape host_submesh_shape)
    : mesh_shape_(validate_mesh_shape(mesh_shape))
    , host_submesh_shape_(validate_host_submesh_shape(mesh_shape, host_submesh_shape))
    , cq_(*this)
{
    // Check if constructor is being entered a second time (problematic with static local in open)
    // This check needs to be robust across MPI processes.
    // We use a simple flag, assuming MPI processes execute this constructor roughly simultaneously
    // after the static variable initialization barrier.
    if (first_call_done) {
        // This scenario shouldn't happen with the static local in open(),
        // but we keep the check as a safeguard.
        std::cerr << "Error: MeshDevice constructor called more than once.\n";
        std::exit(1); // Or MPI_Abort
    }

    int init = 0;
    MPI_Initialized(&init);
    if (!init) MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &world_);

    // Configuration calls moved to open(), called before this constructor runs

    // Verify MPI world size matches host submesh partitioning
    uint32_t expected_hosts = (mesh_shape_.x / host_submesh_shape_.x) * 
                             (mesh_shape_.y / host_submesh_shape_.y);
    if (world_ != expected_hosts) {
        if (rank_ == 0) { // Print error only from rank 0 to avoid flood
            std::cerr << "Error: MPI world size " << world_ 
                      << " does not match expected host count " << expected_hosts << "\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1); // Use MPI_Abort for cleaner exit across ranks
        // std::exit(1); // Replaced with MPI_Abort
    } else {
        // Print success message if debug enabled for this rank
        if (Debug::should_print(rank_)) {
            std::cout << "[rank " << rank_ << "] Validation: MPI world size (" << world_ << ") matches expected host count (" << expected_hosts << ") OK\n";
        }
    }

    // Calculate this host's submesh range
    uint32_t hosts_x = mesh_shape_.x / host_submesh_shape_.x;
    uint32_t hosts_y = mesh_shape_.y / host_submesh_shape_.y;
    
    uint32_t host_x = rank_ % hosts_x;
    uint32_t host_y = rank_ / hosts_x;

    host_submesh_.x_range = {
        host_x * host_submesh_shape_.x,
        (host_x + 1) * host_submesh_shape_.x
    };
    host_submesh_.y_range = {
        host_y * host_submesh_shape_.y,
        (host_y + 1) * host_submesh_shape_.y
    };
    host_submesh_.shape = host_submesh_shape_;

    // Initialize the local devices vector
    uint32_t submesh_width = host_submesh_shape_.x;
    uint32_t submesh_height = host_submesh_shape_.y;
    size_t local_device_count = static_cast<size_t>(submesh_width) * submesh_height;
    local_devices_.reserve(local_device_count); // Reserve space

    uint32_t global_start_x = host_submesh_.x_range.start;
    uint32_t global_start_y = host_submesh_.y_range.start;

    if (Debug::should_print(rank_)) {
         std::cout << "[rank " << rank_ << "] Initializing " << local_device_count << " local devices...\n";
    }
    for (uint32_t ly = 0; ly < submesh_height; ++ly) {
        for (uint32_t lx = 0; lx < submesh_width; ++lx) {
            uint32_t gx = global_start_x + lx;
            uint32_t gy = global_start_y + ly;
            // Pass both global and local coordinates to Device constructor
            local_devices_.emplace_back(Shape(gx, gy), Shape(lx, ly)); 
            local_devices_.back().print_creation_info(rank_); 
        }
    }

    // Determine if this rank should print the global config/layout
    bool should_print_global = false;
    if (Debug::current_mode == Debug::Mode::ALL && rank_ == 0) {
        // Mode 'all': only rank 0 prints global info
        should_print_global = true;
    } else if (Debug::current_mode == Debug::Mode::SPECIFIC_RANK && rank_ == Debug::target_rank) {
        // Mode 'specific': only the target rank prints global info
        should_print_global = true;
    }
    // Mode 'none': should_print_global remains false

    // Print global info if this rank is designated
    if (should_print_global) {
        print_system_config(); 
        print_host_submesh_layout();
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    // Gate the rank-specific ownership message with general debug settings
    if (Debug::should_print(rank_)) {
        std::cout << "[rank " << rank_ << "] owns " << host_submesh_.to_string() << " region.\n";
    }

    // Set the flag only *after* all initialization and validation is complete
    first_call_done = true;
}

inline void MeshDevice::print_host_submesh_layout() {
    uint32_t hosts_x = mesh_shape_.x / host_submesh_shape_.x;
    uint32_t hosts_y = mesh_shape_.y / host_submesh_shape_.y;
    
    std::cout << "\nHost Submesh Layout (" << hosts_x << "x" << hosts_y << " hosts):\n";
    
    // Calculate cell width based on the longest possible content
    uint32_t cell_width = 16; // Increased from 14 to 16 for better alignment
    
    // Print top border
    std::cout << std::string(hosts_x * (cell_width + 1) - 5, '-') << "\n";
    
    for (uint32_t y = 0; y < hosts_y; ++y) {
        // First line: rank numbers
        for (uint32_t x = 0; x < hosts_x; ++x) {
            uint32_t rank = y * hosts_x + x;
            std::cout << "|Rank " << std::setw(2) << rank << "      ";
        }
        std::cout << "|\n";
        
        // Second line: x ranges
        for (uint32_t x = 0; x < hosts_x; ++x) {
            uint32_t start_x = x * host_submesh_shape_.x;
            uint32_t end_x = (x + 1) * host_submesh_shape_.x;
            std::cout << "|x[" << std::setw(2) << start_x << ".." << std::setw(2) << end_x << ")    ";
        }
        std::cout << "|\n";
        
        // Third line: y ranges
        for (uint32_t x = 0; x < hosts_x; ++x) {
            uint32_t start_y = y * host_submesh_shape_.y;
            uint32_t end_y = (y + 1) * host_submesh_shape_.y;
            std::cout << "|y[" << std::setw(2) << start_y << ".." << std::setw(2) << end_y << ")    ";
        }
        std::cout << "|\n";
        
        // Bottom border for this row
        std::cout << std::string(hosts_x * (cell_width + 1) - 5, '-') << "\n";
    }
    std::cout << "\n";
}

inline void MeshDevice::teardown() {
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all ranks reach teardown
    static bool once = false;
    if (!once) { MPI_Finalize(); once = true; }
}

// Original allocate method - now delegates to impl
inline MeshBuffer MeshDevice::allocate(Shape shape) {
    return allocate_impl(shape, mesh_shape_); 
}

// Overload for allocating with an overridden owning mesh shape - now delegates to impl
inline MeshBuffer MeshDevice::allocate(Shape buffer_shape, Shape owning_mesh_shape_override) {
    return allocate_impl(buffer_shape, owning_mesh_shape_override);
}

// Implementation of the private helper
inline MeshBuffer MeshDevice::allocate_impl(Shape buffer_shape, Shape owning_mesh_shape) {
    // Use a single static epoch counter for both allocation paths
    static uint64_t epoch = 0; 
    uint64_t base = (0x9e3779b97f4a7c15ULL * ++epoch) & 0x7fffffffffffULL;

    // Print allocation message if debug enabled for this rank, using the provided owning shape
    if (Debug::should_print(rank_)) {
        // Distinguish if the owning shape is the device's default or an override implicitly
        bool is_override = !(owning_mesh_shape.x == mesh_shape_.x && owning_mesh_shape.y == mesh_shape_.y);
        std::cout << "[rank " << rank_ << "] Allocating MeshBuffer shape=" << to_string(buffer_shape) 
                  << (is_override ? " with OVERRIDDEN Owning MeshDevice shape=" : " for MeshDevice shape=")
                  << to_string(owning_mesh_shape) << "\n";
    }

    if (Validation::on()) {
        // Validation still checks consistency of buffer_shape and base across ranks
        uint64_t crc = base ^ buffer_shape.x ^ buffer_shape.y;
        uint64_t x;
        MPI_Allreduce(&crc, &x, 1, MPI_UINT64_T, MPI_BXOR, MPI_COMM_WORLD);
        assert(x == 0 && "ranks diverged during allocation");
        // Print validation success message if debug enabled for this rank
        if (Debug::should_print(rank_)) { 
            std::cout << "[rank " << rank_ << "] Validation: MeshBuffer allocation OK\n"; 
        }
    }
    // Pass the effective owning shape to the constructor
    return MeshBuffer(base, buffer_shape, owning_mesh_shape); 
}

inline HostBuffer MeshBuffer::host_view() const {
    /* equally divide bytes across ranks */
    int world;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    return HostBuffer(bytes() / world);
}

inline void MeshCQ::push(const MeshWorkload& wl) {
    const auto& words = wl.words();
    if (words.empty()) return;

    // Dispatch commands to all local Devices owned by the associated MeshDevice.
    size_t local_device_count = dev_.local_devices_.size(); // Access via friend
    if (Debug::should_print(dev_.rank())) {
        std::cout << "[rank " << dev_.rank() << "] MeshCQ::push: Dispatching " << words.size() 
                  << " command(s) to " << local_device_count << " local Devices\n";
    }

    for (size_t i = 0; i < local_device_count; ++i) {
        // Access the cq_ member of the Device via friend access
        dev_.local_devices_[i].cq_.cmds_.insert(
            dev_.local_devices_[i].cq_.cmds_.end(),
            words.begin(),
            words.end()
        );
    }
}

inline void MeshDevice::dispatch_pending() {
    // Iterate through local Devices and process/print their commands
    if (Debug::should_print(rank_)) {
        std::cout << "[rank " << rank_ << "] dispatch_pending: Processing local Devices...\n";
    }
    size_t local_device_count = local_devices_.size();

    for (size_t i = 0; i < local_device_count; ++i) {
        Device& device = local_devices_[i]; 
        DeviceCQ& d_cq = device.cq_;      

        if (!d_cq.cmds_.empty()) {
            uint32_t global_x = device.global_coords.x;
            uint32_t global_y = device.global_coords.y;
            uint32_t local_x = device.local_coords.x;
            uint32_t local_y = device.local_coords.y;
            
            if (Debug::should_print(rank_)) {
                std::cout << "[rank " << rank_ 
                          << "]   Dispatching for Device @ global (" << global_x << "," << global_y 
                          << ") / local (" << local_x << "," << local_y
                          << "): " << d_cq.cmds_.size() << " command(s)\n";
            }
            // In a real implementation: Send commands from d_cq.cmds_ to the specific hardware device
            
            // Clear the queue after dispatching
            d_cq.cmds_.clear();
        }
    }
    if (Debug::should_print(rank_)) {
        std::cout << "[rank " << rank_ << "] dispatch_pending: Finished processing local Devices.\n";
    }
}

inline void MeshDevice::wait() {
    // Print message before barrier if debug enabled for this rank
    if (Debug::should_print(rank_)) {
        std::cout << "[rank " << rank_ << "] Entering wait (MPI_Barrier)\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // Print message after barrier if debug enabled for this rank
    if (Debug::should_print(rank_)) {
         std::cout << "[rank " << rank_ << "] Exiting wait (MPI_Barrier complete)\n"; // Changed message
    }
}

// Define the static member
bool MeshDevice::first_call_done = false;

}  // namespace mesh
