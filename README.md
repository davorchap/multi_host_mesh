# Multi-Host Mesh Runtime

**Disclaimer:** This is a simplified proof-of-concept and mock-up runtime system. It demonstrates core concepts like SPMD execution, global workload definition vs. local dispatch, and host-based mesh partitioning. It is **not** intended for production use and lacks many features of a real hardware runtime (e.g., actual device communication, complex memory management, robust error handling).

**Goal:** The primary motivation is to explore how concepts similar to those in TT-Metalium (like `MeshDevice`, `MeshWorkload`, `MeshBuffer`), which are natively multi-device aware, can be extended to become natively **multi-host** aware. This involves managing a single logical mesh distributed across multiple hosts using MPI (or a similar mechanism) primarily for inter-host coordination. Implicit in this "single logical mesh" concept is the assumption of a unified underlying fabric that spans all devices within the mesh, allowing direct device-to-device communication.

## Table of Contents

- [Design Philosophy & Rationale](#design-philosophy--rationale)
- [Comparison with Other Architectures](#comparison-with-other-architectures)
  - [1. Single-Host, Single-Device](#1-single-host-single-device)
  - [2. Single-Host, Multi-Device (e.g., Galaxy)](#2-single-host-multi-device-eg-galaxy)
  - [3. Multi-Host, Multi-Device (Single Controller, Multiple Executors)](#3-multi-host-multi-device-single-controller-multiple-executors)
  - [Visualization (4-Host Example: 16x8 Mesh / 8x4 Sub-Meshes)](#visualization-4-host-example-16x8-mesh--8x4-sub-meshes)
- [Host Coordination Dependency](#host-coordination-dependency)
- [Architecture TODO](#architecture-todo)
- [Dependencies](#dependencies)
- [Components](#components)
- [Compile](#compile)
- [Run](#run)
  - [Command-Line Arguments](#command-line-arguments)
  - [Validation](#validation)
  - [Debug Printing](#debug-printing)

## Design Philosophy & Rationale

![image](https://github.com/user-attachments/assets/40aba6fa-b170-49ef-91a4-6cff1e4a81ea)


This is a simple MPI-based mesh runtime system designed for SPMD (Single Program, Multiple Data) execution. Crucially, this means the user's application itself runs as multiple processes (one MPI rank per host), a shift from typical single-process host applications. MPI (or a similar mechanism) is used here primarily for host-to-host coordination to maintain the SPMD lockstep behavior during workload definition and for optional validation checks. It is **not** intended for bulk data movement between host processes or between hosts and devices. All substantial data movement, including device-to-device transfers and collective communications across the logical mesh, is assumed to occur over the underlying TT-fabric. These processes cooperatively manage **a single, large logical mesh** distributed across the hosts. It does **not** currently model multi-mesh scenarios (Scenarios involving multiple distinct logical meshes, even if potentially residing on the same underlying fabric, are outside the current scope).

The core design principle is to **separate the global, host-symmetric definition of work from the local, host-specific execution.**

*   **Global View (All Hosts):**
    *   Users interact with a virtual `MeshDevice` representing the entire logical mesh.
    *   `MeshBuffer` allocation and `MeshWorkload` creation are performed identically and deterministically across all participating hosts/processes.
    *   This ensures every host process has a consistent, global view of the resources and the intended computation, simplifying workload design and debugging.
*   **Local Ownership (Per Host):**
    *   Each host process (rank) physically manages a specific sub-mesh region of the overall logical mesh.
    *   Internally, the `MeshDevice` on a given host owns and manages a collection of `Device` objects representing the mesh nodes within its assigned sub-mesh.
    *   Each local `Device` has its own `DeviceCQ` (Device Command Queue).
*   **Underlying Fabric Assumption:** This design assumes that the devices comprising the single logical mesh are connected via a single, unified fabric. Direct device-to-device communication within the mesh occurs over this fabric without host intervention.
*   **SPMD & Lockstep Behavior:** The requirement for identical, deterministic creation of the global view (`MeshBuffer`, `MeshWorkload`) across hosts/processes enforces an SPMD model. All host processes proceed in logical lockstep during the definition phase.
*   **Optional Validation:** To aid debugging and explicitly verify this lockstep behavior, the runtime includes a validation mechanism (`--validate on`, default). This uses MPI collectives to assert that key operations are indeed identical across all ranks. Disable (`--validate off`) for performance.
*   **Local Dispatch Abstraction:** The host-local `MeshCQ` serves as the interface for submitting the globally defined `MeshWorkload`. Internally (`MeshCQ::push`), it dispatches the relevant commands from the global workload to the appropriate local `DeviceCQ`s managed by that host process. The subsequent `MeshDevice::dispatch_pending` call processes these local `DeviceCQ`s, notionally interacting with the hardware resources associated with those local mesh devices.
*   **Control Plane (Conceptual):** While not fully implemented here, the conceptual model aligns with a control plane (likely running alongside the user process on a designated host, e.g., rank 0) having a global view of the entire mesh topology (potentially provided by the user via a mesh description). This central control plane would generate necessary configurations (like routing tables), but delegate the task of applying these configurations to the specific host processes directly managing the target devices.

**Crucially, Determinism is Required:** Both the runtime itself *and* the user's application code (including workload generation functions) **must be deterministic**. This means using deterministic algorithms and data structures (e.g., avoiding hash maps with non-deterministic iteration order if the order affects workload generation). If any host process diverges due to non-determinism, the system's behavior becomes undefined. The `--validate` option can help detect divergence after it occurs, but it cannot prevent it; determinism is the fundamental requirement for correctness and run-to-run reproducibility.

### Determinism Considerations for Python Bindings

When using this runtime (especially in the SPMD model) via Python bindings (e.g., pybind11), additional care must be taken regarding determinism:

1.  **Python Garbage Collection (GC) and Resource Deallocation:**
    *   **Problem:** Standard Python GC timing is non-deterministic across different processes (ranks). If C++ resource deallocation (like freeing a `MeshBuffer`) directly modifies allocator state (which is necessary for correct subsequent allocations) and is tied *only* to the Python object's destruction (e.g., via `__del__`), this critical state change will occur at different times on different ranks, leading to divergence.
    *   **Solution:** The C++ resource lifetime management **must be decoupled** from Python's GC timing. The Python bindings **must provide explicit, deterministic mechanisms** for resource deallocation.
    *   **Recommendation:** Users **must** use these explicit mechanisms. The strongly recommended approach is to use **context managers (`with` statement)** for resources like `MeshBuffer`. This ensures deallocation happens deterministically upon exiting the `with` block scope, synchronized across all ranks.
        ```python
        # Example of deterministic deallocation with context manager
        with device.allocate_buffer(...) as buf:
            # ... use buf ...
        # <-- buf is deterministically deallocated here on all ranks
        ```
    *   An alternative might be an explicit `buffer.free()` method, but it must be called at the exact same logical point in the code by all ranks.

2.  **Python Hash Randomization:**
    *   **Problem:** Since Python 3.3, dictionary and set iteration order is randomized by default across different process invocations. If user logic iterates over these collections and the order affects workload generation (e.g., determines the order of operations), the generated `MeshWorkload` will differ between ranks, breaking the SPMD model.
    *   **Recommendation:** User code **must not** rely on unordered collection iteration if it impacts workload definition. Either use ordered collections (e.g., `collections.OrderedDict`, lists) or ensure the logic is insensitive to iteration order. Alternatively, setting the `PYTHONHASHSEED=0` environment variable for all ranks can enforce deterministic iteration, but makes the application reliant on this external setting.

3.  **Other Sources:** Standard pitfalls like rank-dependent branching logic that affects workload definition, uncoordinated I/O, or multi-threading races within a rank's definition phase must also be avoided.

The runtime's validation layer (`--validate on`) can help *detect* divergence caused by these issues but cannot prevent them. Careful deterministic programming is essential when using Python with this SPMD runtime.

**Layered Architecture per Host:** The separation between global definition and local execution can be visualized as a layered stack on each host process:

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

This architecture ensures that all host processes agree on the *what* (the global `MeshWorkload` submitted to `MeshCQ`) before diverging into the *how* (the host-specific dispatch to local `DeviceCQ`s).

**Single-Rank Debugging:** A significant advantage of this design is that because the user's application code interacts primarily with the global view objects (`MeshDevice`, `MeshBuffer`, `MeshWorkload`) and this view is identical across all ranks/processes up to the `MeshCQ` submission point, much of the application logic can often be debugged effectively by running with a single rank/process (`mpirun -np 1 ...`). This drastically simplifies the debugging process, especially for systems with a large number of hosts.

In essence, the user defines *what* computation should happen on the *entire mesh* (global view), and the runtime handles *how* to distribute and execute that computation on the *local devices* managed by each host process.

## Comparison with Other Architectures

To better understand the proposed design (SPMD / Multiple Lockstep Controllers), it's helpful to contrast it with other common system architectures:

### 1. Single-Host, Single-Device

*   **Execution:** User code runs as a single process.
*   **Scope:** `MeshDevice` represents a 1x1 mesh. The `MeshCQ` maps directly (1:1) to the single underlying `DeviceCQ`.
*   **Relevance:** Simplest baseline, not applicable to multi-device or multi-host scenarios.

```ascii
+-------------+
| Host        |
| Process (1) |
|-------------|
| User Code   |
| MeshDevice  |
|  (1x1)      |
|   |         |
| MeshCQ      |
|   | (1:1)   |
|   V         |
| DeviceCQ    |
+-------------+
```

### 2. Single-Host, Multi-Device (e.g., Galaxy)

*   **Execution:** User code runs as a single process.
*   **Scope:** `MeshDevice` represents the multi-device mesh (e.g., 8x4). `MeshBuffer` and `MeshWorkload` are defined globally for this mesh.
*   **Dispatch:** The single `MeshCQ` receives the global workload and internally dispatches relevant commands to the multiple `DeviceCQ`s corresponding to the devices managed by the host. This dispatch might be multi-threaded for efficiency.
*   **Relevance:** Represents the current standard for multi-device systems *within* a single host. The proposed multi-host design leverages similar concepts for the *local* dispatch part on each host.

```ascii
+--------------------+
| Host               |
| Process (1)        |
|--------------------|
| User Code          |
| MeshDevice (NxM)   |
|      |             |
|    MeshCQ          |
|      | (Dispatch)  |
|  +---V---+---V---+ |
|  | DevCQ | DevCQ | |
|  +-------+-------+ |
|    ... (NxM) ...   |
+--------------------+
```

### 3. Multi-Host, Multi-Device (Single Controller, Multiple Executors)

This is a common alternative approach to managing multi-host systems:

*   **Execution:** User code runs as a **single process** on a designated controller host (or potentially a separate dedicated host).
*   **Scope:** The Controller process holds the global view (`MeshDevice`, `MeshWorkload`).
*   **Dispatch:** The Controller process serializes commands derived from the `MeshWorkload` and sends them over the network (e.g., using RPC or a message queue) to Executor processes running on the other hosts. Each Executor process manages its local sub-mesh devices and has a form of "Remote Mesh CQ" that receives and executes commands from the Controller.
*   **Key Contrast:** Unlike the proposed SPMD model where *every* host process runs the *same* user code up to the `MeshCQ` submission, here only the Controller runs the main user logic. Executors are passive receivers of commands.

```ascii
    +----------------------------+
    | Controller Host            |
    | Process (1)                |
    |----------------------------|
    | User Code                  |
    | MeshDevice (Global)        |
    | MeshWorkload               |
    | (Serialize Cmds)           |
    |                            |
    | Network Dispatch           |
    +----|-----------------|-----+
         |                 |
         |     (Cmds)      |
         |                 |
         V                 V
+------------------+  +------------------+
| Executor Host 1  |  | Executor Host N  |
| Process (1)      |  | Process (1)      |
|------------------|  |------------------|
| Executor Logic   |  | Executor Logic   |
| RemoteMeshCQ     |  | RemoteMeshCQ     |
|   | (Dispatch)   |  |   | (Dispatch)   |
|   V   Local      |  |   V   Local      |
| +-------+        |  | +-------+        |
| | DevCQ | ...    |  | | DevCQ | ...    |
| +-------+        |  | +-------+        |
+------------------+  +------------------+
```

*   **Potential Pros:**
    *   **Avoids User Code Divergence:** A major advantage. Since user application logic runs only on the Controller, the risk of divergence between hosts due to non-deterministic user code (e.g., unordered map iteration impacting command generation) or bugs in rank-specific logic is eliminated. This contrasts with the SPMD model's strict requirement for deterministic user code on all ranks.
    *   **Familiar User Model:** User code remains single-process, which might be simpler for developers accustomed to single-host programming paradigms.
    *   **Centralized Host State:** If complex host-side global state needs to be managed or calculated *during* workload generation, doing so in a single Controller process can be simpler than coordinating it across multiple SPMD processes.
*   **Potential Cons:**
    *   **Serialization Overhead:** Requires defining, maintaining, and executing serialization/deserialization protocols for commands sent between Controller and Executors, adding complexity and potential performance cost.
    *   **Controller Bottleneck:** The single Controller's computation and network dispatch capabilities can become a performance bottleneck, limiting throughput and scalability, especially with many hosts or high-frequency command submission.
    *   **Executor Complexity:** Requires implementing non-trivial Executor processes capable of receiving, deserializing, and executing commands, plus managing local resources.
    *   **Debugging Distributed System:** Debugging involves understanding and potentially coordinating logs/state across the Controller, the network communication layer, and multiple Executor processes, presenting different challenges than debugging SPMD divergence.

In summary, the Single Controller model offers robustness against user-code-induced host divergence and presents a familiar single-process programming model, but potentially at the cost of serialization overhead, controller bottlenecks, and the complexity of building and debugging the distributed Controller/Executor system. The proposed SPMD / "Multiple Lockstep Controllers" design trades the determinism burden on user code for potentially better scalability (avoiding the controller bottleneck) and reduced serialization overhead by replicating the workload definition phase across hosts.

#### **This Proposal:** Multi-Host, Multi-Device (SPMD / Multiple Lockstep Controllers)

```ascii
      +---------------------------------------------------+
      |                 MPI Coordination                  |
      |               (All-to-All/Global)                 |
      +---|-------------------|---------------------|-----+
          ^                   ^                     ^
          |                   |                     |
          V                   V                     V
+-------------------+  +-------------------+  +-------------------+
| Host 1 (Rank 0)   |  | Host 2 (Rank 1)   |  | Host N (Rank N-1) |
| Process (SPMD)    |  | Process (SPMD)    |  | Process (SPMD)    |
|-------------------|  |-------------------|  |-------------------|
| User Code (Same)  |  | User Code (Same)  |  | User Code (Same)  |
| MeshDevice(Global)|  | MeshDevice(Global)|  | MeshDevice(Global)|
| MeshWorkload(Same)|  | MeshWorkload(Same)|  | MeshWorkload(Same)|
|     |             |  |     |             |  |     |             |
|   MeshCQ          |  |   MeshCQ          |  |   MeshCQ          |
|     | (Dispatch)  |  |     | (Dispatch)  |  |     | (Dispatch)  |
|     V   Local     |  |     V   Local     |  |     V   Local     |
|   +-------+       |  |   +-------+       |  |   +-------+       |
|   | DevCQ | ...   |  |   | DevCQ | ...   |  |   | DevCQ | ...   |
|   +-------+       |  |   +-------+       |  |   +-------+       |
+-------------------+  +-------------------+  +-------------------+
```

### Visualization (4-Host Example: 16x8 Mesh / 8x4 Sub-Meshes)

Imagine a 16x8 logical mesh divided among 4 hosts (ranks 0-3), where each host manages an 8x4 sub-mesh:

```ascii
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

## Host Coordination Dependency

Any multi-host system requires a mechanism for communication and coordination between the host processes. In the SPMD model proposed here, this is needed for optional validation checks and potentially for future extensions involving host-level barriers or collective operations. In the Single Controller model, it's needed for the Controller to dispatch commands to Executors (e.g., via RPC, ZeroMQ, etc.).

This proof-of-concept currently uses **MPI** for this coordination layer, primarily due to its ubiquity in HPC environments and its straightforward primitives for collective operations (like the ones used in the `--validate` checks).

However, relying strictly on MPI introduces a potentially heavy dependency that might not be desirable or necessary for all users, especially those not running in traditional HPC environments or those only targeting single-host configurations.

Future evolution of this multi-host runtime should consider making the coordination layer more flexible:

*   **Optional MPI:** The requirement for MPI could be gated by a build flag, removing the dependency entirely for users only building/running single-host configurations.
*   **Abstract Interface:** Define an abstract C++ interface (e.g., `IHostCoordinator`) specifying the necessary coordination primitives (like `barrier`, `allreduce`, `bcast`, potentially basic point-to-point messaging if needed for other models).
*   **Pluggable Implementations:** The core runtime would use the `IHostCoordinator` interface. We could then provide:
    *   A default implementation based on MPI (shipped with the package for ease of use).
    *   Potentially other implementations (e.g., using ZeroMQ, gRPC, or other messaging libraries).
    *   Allow users (consumers) to provide their own custom implementation if they have specific infrastructure or performance requirements.

This approach allows users to choose the coordination mechanism that best fits their environment while keeping the core runtime logic agnostic to the specific underlying library.

**Note:** For this specific example code (`multi_host_mesh_example.cpp`), MPI is currently required for compilation and execution in multi-host mode (`mpirun -np > 1`).

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
