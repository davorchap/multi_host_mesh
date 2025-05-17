#include "multi_host_mesh_runtime.hpp"
#include <cstdlib>
#include <stdexcept>
using namespace mesh;

void usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <mesh_x> <mesh_y> <host_submesh_x> <host_submesh_y>"
              << " [--validate on|off] [--debug <mode>]\n"
              << "  mesh_x, mesh_y: overall mesh dimensions (must be powers of 2)\n"
              << "  host_x, host_y: host submesh dimensions (must be powers of 2)\n"
              << "                  must evenly divide mesh dimensions\n"
              << "  --validate on|off: Enable or disable runtime validation checks (default: on)\n"
              << "  --debug <mode>: Set debug print mode (default: none)\n"
              << "                  mode can be 'none', 'all', or a specific integer rank ID\n";
    std::exit(1);
}

// Struct to hold parsed arguments
struct ProgramArgs {
    Shape mesh_shape;
    Shape host_submesh_shape;
    bool validation_enabled;
    mesh::Debug::Mode debug_mode = mesh::Debug::Mode::NONE; // Default debug mode
    int debug_rank = -1;
};

// Function to parse command line arguments
ProgramArgs parse_args(int argc, char** argv) {
    ProgramArgs args;
    args.validation_enabled = true; // Default validation
    // Default debug handled by struct initializer

    int current_arg = 1;
    if (current_arg + 3 >= argc) usage(argv[0]); // Need at least 4 shape args

    args.mesh_shape = {
        static_cast<uint32_t>(std::atoi(argv[current_arg++])),
        static_cast<uint32_t>(std::atoi(argv[current_arg++]))
    };
    args.host_submesh_shape = {
        static_cast<uint32_t>(std::atoi(argv[current_arg++])),
        static_cast<uint32_t>(std::atoi(argv[current_arg++]))
    };

    // Parse optional arguments
    while (current_arg < argc) {
        std::string flag = argv[current_arg++];
        if (current_arg >= argc) { // Each flag needs a value
            std::cerr << "Error: Flag '" << flag << "' requires an argument.\n";
            usage(argv[0]);
        }
        std::string value = argv[current_arg++];

        if (flag == "--validate") {
            if (value == "on") args.validation_enabled = true;
            else if (value == "off") args.validation_enabled = false;
            else {
                std::cerr << "Error: Invalid value for --validate flag. Use 'on' or 'off'.\n";
                usage(argv[0]);
            }
        } else if (flag == "--debug") {
            if (value == "none") {
                args.debug_mode = mesh::Debug::Mode::NONE;
                args.debug_rank = -1;
            } else if (value == "all") {
                args.debug_mode = mesh::Debug::Mode::ALL;
                args.debug_rank = -1;
            } else {
                // Try to parse as rank ID
                try {
                    int rank_id = std::stoi(value);
                    if (rank_id < 0) {
                         std::cerr << "Error: Invalid rank ID for --debug flag. Must be non-negative.\n";
                         usage(argv[0]);
                    }
                    args.debug_mode = mesh::Debug::Mode::SPECIFIC_RANK;
                    args.debug_rank = rank_id;
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Error: Invalid value for --debug flag. Use 'none', 'all', or a rank ID.\n";
                    usage(argv[0]);
                } catch (const std::out_of_range& e) {
                    std::cerr << "Error: Rank ID for --debug flag is out of range.\n";
                    usage(argv[0]);
                }
            }
        } else {
             std::cerr << "Error: Unknown optional argument '" << flag << "'\n";
             usage(argv[0]);
        }
    }

    return args;
}

// User-defined test function moved from runtime
// Update signature to accept target mesh shape
inline MeshWorkload fabric_multicast_test(const MeshBuffer& in_buf, const MeshBuffer& out_buf, Shape target_mesh_shape) {
    // Mock‑up of a TT‑fabric stress test
    // This simulates a multicast operation with a fixed test pattern
    // All ranks must create identical workloads
    uint64_t test_pattern = 0xCAFEBABEULL;
    // Example: Incorporate both buffer sizes into the command (customize as needed)
    uint64_t cmd = (test_pattern << 32) | ((in_buf.bytes() + out_buf.bytes()) & 0xFFFFFFFFULL);
    
    // Pass target mesh shape to MeshWorkload constructor
    return MeshWorkload({cmd}, target_mesh_shape);
}

int main(int argc, char** argv) {
    ProgramArgs args = parse_args(argc, argv);

    // Pass config args directly to open
    auto& dev = MeshDevice::open(args.mesh_shape, args.host_submesh_shape, 
                               args.validation_enabled, args.debug_mode, args.debug_rank);
    
    auto& cq  = dev.cq();

    // Get mesh shape from device
    Shape mesh_shape = dev.mesh_shape();

    // Create a test buffer
    Shape test_shape = {1024, 1024};  // 1MB buffer
    MeshBuffer test_buf = dev.allocate(test_shape);
    MeshBuffer output_buf = dev.allocate(test_shape, mesh_shape);

    // WIP: HostBuffer design
    //  HostBuffer host_buf = dev.allocate_host_buffer<uint64_t>(test_shape, mesh_shape, submesh_shape);

    // Create and push the multicast test workload
    // All ranks create identical workloads
    // Pass mesh shape to the test function
    MeshWorkload multicast_test = fabric_multicast_test(test_buf, output_buf, mesh_shape); 
    cq.push(multicast_test);

    dev.dispatch_pending();
    dev.wait();

    MeshDevice::close();
    return 0;
} 