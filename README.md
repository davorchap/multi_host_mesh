# Multi-Host Mesh Runtime

**Disclaimer:** This is a simplified proof-of-concept and mock-up runtime system. It demonstrates core concepts like SPMD execution, global workload definition vs. local dispatch, and host-based mesh partitioning. It is **not** intended for production use and lacks many features of a real hardware runtime (e.g., actual device communication, complex memory management, robust error handling).

**Goal:** The primary motivation is to explore how concepts similar to those in TT-Metalium (like `MeshDevice`, `MeshWorkload`, `MeshBuffer`), which are natively multi-device aware, can be extended to become natively **multi-host** aware. This involves managing a single logical mesh distributed across multiple hosts communicating via MPI.

## Design Philosophy & Rationale

![image](https://github.com/user-attachments/assets/40aba6fa-b170-49ef-91a4-6cff1e4a81ea)


This is a simple MPI-based mesh runtime system designed for SPMD (Single Program, Multiple Data) execution across multiple hosts (MPI ranks) cooperatively managing **a single, large logical mesh**. It does **not** currently model multi-mesh scenarios.

The core design principle is to **separate the global, host-symmetric definition of work from the local, host-specific execution.**

*   **Global View (All Hosts):**
    *   Users interact with a virtual `MeshDevice` representing the entire logical mesh.
    *   `MeshBuffer` allocation and `MeshWorkload` creation are performed identically and deterministically across all participating hosts.
    *   This ensures every host has a consistent, global view of the resources and the intended computation, simplifying workload design and debugging.
*   **Local Ownership (Per Host):**
    *   Each host (rank) physically manages a specific sub-mesh region of the overall logical mesh.
    *   Internally, the `MeshDevice` on a given host owns and manages a collection of `Device` objects representing the mesh nodes within its assigned sub-mesh.
    *   Each local `Device` has its own `DeviceCQ` (Device Command Queue).
*   **SPMD & Lockstep Behavior:** The requirement for identical, deterministic creation of the global view (`MeshBuffer`, `MeshWorkload`) across hosts enforces an SPMD model. All hosts proceed in logical lockstep during the definition phase.
*   **Optional Validation:** To aid debugging and explicitly verify this lockstep behavior, the runtime includes a validation mechanism (`--validate on`, default). This uses MPI collectives to assert that key operations are indeed identical across all ranks. Disable (`--validate off`) for performance.
*   **Local Dispatch Abstraction:** The host-local `MeshCQ` serves as the interface for submitting the globally defined `MeshWorkload`. Internally (`MeshCQ::push`), it dispatches the relevant commands from the global workload to the appropriate local `DeviceCQ`s managed by that host. The subsequent `MeshDevice::dispatch_pending` call processes these local `DeviceCQ`s, notionally interacting with the hardware resources associated with those local mesh devices.

**Crucially, Determinism is Required:** Both the runtime itself *and* the user's application code (including workload generation functions) **must be deterministic**. This means using deterministic algorithms and data structures (e.g., avoiding hash maps with non-deterministic iteration order if the order affects workload generation). If any host diverges due to non-determinism, the system's behavior becomes undefined. The `--validate` option can help *detect* divergence after it occurs, but it cannot prevent it; determinism is the fundamental requirement for correctness and run-to-run reproducibility.

**Layered Architecture per Host:** The separation between global definition and local execution can be visualized as a layered stack on each host:

```
+----------------------------------------------------+
|          Global View (Identical on all Hosts)      |
|----------------------------------------------------|
| - MeshDevice (Full Logical Mesh)                   |
| - MeshBuffer (Global Resource Spec)                |
| - MeshWorkload (Global Computation Spec)           |
+----------------------- | --------------------------+
                         |
                         | Submission Interface
                         V
+----------------------- - --------------------------+
|                  MeshCQ (Global Input)             |
|----------------------------------------------------|
|          Local Dispatch Logic (Host-Specific)      |
|            (Maps Global Work to Local CQs)         |
|----------------------- | --------------------------+
                         |
                         V Local CQs
+----------------------- - --------------------------+
|          Local View (Host-Specific Subset)         |
|----------------------------------------------------|
| - Set of local Device objects                      |
| - Associated local DeviceCQ per Device             |
| - MeshDevice::dispatch_pending() processes these   |
+----------------------------------------------------+
```

This architecture ensures that all hosts agree on the *what* (the global `MeshWorkload` submitted to `MeshCQ`) before diverging into the *how* (the host-specific dispatch to local `DeviceCQ`s).

**Single-Rank Debugging:** A significant advantage of this design is that because the user's application code interacts primarily with the global view objects (`MeshDevice`, `MeshBuffer`, `MeshWorkload`) and this view is identical across all ranks up to the `MeshCQ` submission point, much of the application logic can often be debugged effectively by running with a single rank (`mpirun -np 1 ...`). This drastically simplifies the debugging process, especially for systems with a large number of hosts.

In essence, the user defines *what* computation should happen on the *entire mesh* (global view), and the runtime handles *how* to distribute and execute that computation on the *local devices* managed by each host.

### Visualization (4-Host Example: 16x8 Mesh / 8x4 Sub-Meshes)

Imagine a 16x8 logical mesh divided among 4 hosts (ranks 0-3), where each host manages an 8x4 sub-mesh:

```
+---------------------------------------------------+
|            Logical Mesh (16x8 Global)              |
|                                                    |
|       +---------------------+----------------------+       |
|       | Host 0 (Rank 0)     | Host 1 (Rank 1)      |       |    Global View (All Hosts):
|       | Owns 8x4 Sub-Mesh   | Owns 8x4 Sub-Mesh    |       |    - MeshDevice (16x8)
|       | Global Coords:      | Global Coords:       |       |    - MeshBuffer (Global)
|       | x=[0..8), y=[0..4)  | x=[8..16), y=[0..4)  |       |    - MeshWorkload (Global)
|       | (Local Devs+CQs)    | (Local Devs+CQs)     |       |    - MeshCQ (Submit Global)
|       +---------------------+----------------------+       |
|       | Host 2 (Rank 2)     | Host 3 (Rank 3)      |       |
|       | Owns 8x4 Sub-Mesh   | Owns 8x4 Sub-Mesh    |       |    Local Ownership (per Host):
|       | Global Coords:      | Global Coords:       |       |    - 8x4 set of Devices
|       | x=[0..8), y=[4..8)  | x=[8..16), y=[4..8)  |       |    - 32 DeviceCQs
|       | (Local Devs+CQs)    | (Local Devs+CQs)     |       |    - dispatch_pending
|       +---------------------+----------------------+       |
|                                                    |
+----------------------------------------------------+

  ^                           ^
  |                           |
All Hosts (Ranks 0-3) create the same GLOBAL specifications
for Buffers & Workloads relative to the 16x8 Logical Mesh.

Each Host only DISPATCHES commands to the 32 LOCAL Devices
(and their DeviceCQs) within its owned 8x4 Sub-Mesh region.

```

## Architecture TODO

*   Define and implement `HostBuffer` across a mesh of hosts (e.g., spanning a 2x2 host configuration).
*   Explore how host-side functions (like tilize, padding, data transformations) operating on `HostBuffer`s fit into the programming model, potentially integrating with or mirroring concepts from TT-Metalium's host API layer.

## Dependencies

*   **MPI Implementation:** An MPI library is required (e.g., Open MPI, MPICH, Intel MPI). The `mpic++` compiler wrapper and `mpirun` command must be available in your PATH.
    *   **On macOS with Homebrew:** You can install Open MPI using:
        ```bash
        brew install open-mpi
        ```
        This typically makes `mpic++` and `mpirun` available in your default PATH.
*   **C++ Compiler:** A C++11 compliant compiler (tested on Mac Pro with clang 17)

## Components

*   `multi_host_mesh_runtime.hpp`: Header-only library providing:
    *   `MeshDevice`: Represents the virtual view of the entire logical mesh, but internally manages locally owned `Device`s.
    *   `Device`: Represents a single device in the mesh, storing its global/local coordinates.
    *   `DeviceCQ`: Command Queue specific to a single local `Device`.
    *   `MeshBuffer`: Specification of a global buffer resource.
    *   `MeshWorkload`: Specification of a global workload.
    *   `MeshCQ`: Interface for submitting global workloads, handles internal dispatch to local `DeviceCQ`s.
    *   Validation & Debugging logic.
*   `multi_host_mesh_example.cpp`: Example program demonstrating how to use the runtime, including argument parsing and a sample workload (`fabric_multicast_test`).

## Compile

```bash
mpic++ multi_host_mesh_example.cpp -o multi_host_mesh_example -std=c++11
```
(Requires C++11).

## Run

Example with 4 ranks, 16x8 MeshDevice, and 8x4 Host SubMeshes:

**1. Default (Validation On, Debug None):**
```bash
mpirun -np 4 ./multi_host_mesh_example 16 8 8 4
```

**2. Validation Off:**
```bash
mpirun -np 4 ./multi_host_mesh_example 16 8 8 4 --validate off
```

**3. Debug All Ranks (Validation On by default):**
```bash
mpirun -np 4 ./multi_host_mesh_example 16 8 8 4 --debug all
```

**4. Debug Rank 0 Only (Validation Off):**
```bash
mpirun -np 4 ./multi_host_mesh_example 16 8 8 4 --validate off --debug 0
```

This creates:
- A 16x8 virtual `MeshDevice` representing the logical mesh.
- 4 ranks (hosts) arranged according to the mesh division (e.g., a 2x2 arrangement of hosts for this example).
- Each host owns an 8x4 sub-mesh region and manages the `Device` objects and associated `DeviceCQ`s within that region.

### Command-Line Arguments

```
Usage: ./multi_host_mesh_example <mesh_x> <mesh_y> <host_submesh_x> <host_submesh_y> \
                              [--validate on|off] [--debug <mode>]
  mesh_x, mesh_y: overall mesh dimensions (must be powers of 2)
  host_x, host_y: host submesh dimensions (must be powers of 2)
                  must evenly divide mesh dimensions
  --validate on|off: Enable or disable runtime validation checks (default: on)
  --debug <mode>: Set debug print mode (default: none)
                  mode can be 'none', 'all', or a specific integer rank ID
```

*   Mesh dimensions and host submesh dimensions must be powers of 2.
*   Host submesh dimensions must evenly divide the mesh dimensions.
*   The number of MPI ranks (`mpirun -np N`) must equal `(mesh_x / host_submesh_x) * (mesh_y / host_submesh_y)`.

### Validation

To enforce and verify the required symmetric, lockstep behavior during workload/buffer creation, the runtime includes optional validation checks (`--validate on`). These use MPI collectives to assert consistency across ranks. Disable with `--validate off` for performance.

### Debug Printing

The runtime includes internal print statements for various operations. The verbosity is controlled by the `--debug` flag:

*   `--debug none` (default): Minimal output.
*   `--debug all`: All ranks print debug messages, useful for tracing execution flow across the system.
*   `--debug <rank_id>`: Only the specified rank prints debug messages.
