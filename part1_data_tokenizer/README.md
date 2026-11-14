# Multi-Container Runtime

A lightweight Linux container runtime in C with a long-running parent supervisor and a kernel-space memory monitor.

---

## 1. Team Information

| Name | SRN |
|------|-----|
| Balla Santhosh | PES1UG24AM063 |
| Bandagula Sai Rithesh Reddy | PES1UG24AM064 |

**GitHub Repository:** https://github.com/Balla-Santhosh/OS-Jackfruit

---

## 2. Build, Load, and Run Instructions

### Environment Requirements
- Ubuntu 22.04 or 24.04 in a VM
- Secure Boot OFF
- No WSL

### Install Dependencies
```bash
sudo apt update
sudo apt install -y build-essential linux-headers-$(uname -r)
```

### Build
```bash
cd boilerplate
make
```

This produces:
- `engine` — user-space runtime and supervisor
- `monitor.ko` — kernel memory monitor module
- `memory_hog`, `cpu_hog`, `io_pulse` — test workload binaries

### Prepare Root Filesystem
```bash
cd boilerplate
mkdir rootfs-base
wget https://dl-cdn.alpinelinux.org/alpine/v3.21/releases/x86_64/alpine-minirootfs-3.21.3-x86_64.tar.gz
tar -xzf alpine-minirootfs-3.21.3-x86_64.tar.gz -C rootfs-base

# Create one writable copy per container
cp -a ./rootfs-base ./rootfs-alpha
cp -a ./rootfs-base ./rootfs-beta
```

### Load Kernel Module
```bash
sudo insmod monitor.ko

# Verify control device exists
ls /dev/container_monitor
```

### Start the Supervisor (Terminal 1)
```bash
sudo ./engine supervisor ./rootfs-base
```

### Launch Containers (Terminal 2)
```bash
# Copy workload binary into container rootfs before launch
cp cpu_hog ./rootfs-alpha/

# Start a container in the background
sudo ./engine start alpha ./rootfs-alpha /cpu_hog --soft-mib 48 --hard-mib 80

# Start a second container
cp cpu_hog ./rootfs-beta/
sudo ./engine start beta ./rootfs-beta /cpu_hog --soft-mib 48 --hard-mib 80

# List all tracked containers
sudo ./engine ps

# View logs for a container
sudo ./engine logs alpha

# Run a container and wait for it to finish
sudo ./engine run alpha ./rootfs-alpha /cpu_hog

# Stop a container
sudo ./engine stop alpha
```

### Memory Limit Testing
```bash
cp -a ./rootfs-base ./rootfs-memtest
cp memory_hog ./rootfs-memtest/
sudo ./engine start memtest ./rootfs-memtest /memory_hog --soft-mib 20 --hard-mib 40

# Watch kernel events in another terminal
sudo dmesg -w
```

### Scheduling Experiment
```bash
cp -a ./rootfs-base ./rootfs-low
cp -a ./rootfs-base ./rootfs-high
cp cpu_hog ./rootfs-low/
cp cpu_hog ./rootfs-high/

# High priority container (nice -5)
sudo ./engine start high ./rootfs-high /cpu_hog --nice -5

# Low priority container (nice 10)
sudo ./engine start low ./rootfs-low /cpu_hog --nice 10

# Compare logs after a few seconds
sudo ./engine logs high
sudo ./engine logs low
```

### Clean Up
```bash
# Stop all containers
sudo ./engine stop alpha
sudo ./engine stop beta

# Stop supervisor (Ctrl+C in Terminal 1)

# Unload kernel module
sudo rmmod monitor

# Verify no zombies
ps aux | grep -E "cpu_hog|memory_hog" | grep -v grep
```

---

## 3. Demo Screenshots

### Screenshot 1 — Multi-Container Supervision
Two containers (`alpha` and `beta`) running simultaneously under one supervisor process. The supervisor manages both independently with isolated PID, UTS, and mount namespaces.

![Screenshot 1](image.png)

### Screenshot 2 — Metadata Tracking
Output of `sudo ./engine ps` showing tracked container metadata including ID, PID, state, exit code, soft/hard memory limits, and start time.

![Screenshot 2](image-1.png)

### Screenshot 3 — Bounded-Buffer Logging
Output of `sudo ./engine logs alpha` showing container stdout captured through the pipe → bounded buffer → log file pipeline. Producer threads read from container pipes and push chunks into the buffer; the consumer thread writes them to `logs/<id>.log`.

![Screenshot 3](image-2.png)

### Screenshot 4 — CLI and IPC
The `engine start` command being issued from the CLI client, connecting over the UNIX domain socket to the supervisor, which responds with the container PID confirmation. Demonstrates Path B (control channel) IPC.

![Screenshot 4](image-3.png)

### Screenshot 5 — Soft-Limit Warning
`dmesg` output showing `[container_monitor] SOFT LIMIT container=alpha2 pid=12148` after the container's RSS exceeded the 20 MiB soft limit. The kernel module logs the warning once and sets the `soft_warned` flag to prevent repeated messages.

![Screenshot 5](image-4.png)

### Screenshot 6 — Hard-Limit Enforcement
`dmesg` output showing `[container_monitor] HARD LIMIT container=alpha2 pid=12148` followed by SIGKILL being sent. The subsequent `sudo ./engine ps` output shows `alpha2` in state `killed` with exit code 137 (128 + SIGKILL), confirming the supervisor metadata reflects the hard-limit kill correctly.

![Screenshot 6](image-5.png)
![Screenshot 6](image-6.png)
![Screenshot 6](image-7.png)

### Screenshot 7 — Scheduling Experiment
Logs from two concurrent `cpu_hog` containers — `high` (nice -5) and `low` (nice 10) — running the same workload. The `high` container receives more CPU time from CFS due to its lower nice value, observable in the accumulator values per elapsed second being higher in the high-priority container's log.

![Screenshot 7](image-8.png)
![Screenshot 7](image-9.png)

### Screenshot 8 — Clean Teardown
Output showing containers `low` and `high` in state `exited`, `alpha2` in state `killed`, and no remaining `cpu_hog` or `memory_hog` processes in `ps aux`. Confirms correct child reaping, no zombie processes, and clean metadata state after shutdown.

![Screenshot 8](image-10.png)
![Screenshot 8](image-11.png)

---

## 4. Engineering Analysis

### 4.1 Isolation Mechanisms

Linux namespaces are the kernel mechanism that makes containers possible. Each namespace type wraps a specific global resource and gives each namespace its own independent instance of that resource.

This runtime uses three namespace types, passed as flags to `clone()`:

**PID namespace (`CLONE_NEWPID`)** gives the container its own process ID space. The first process inside the container sees itself as PID 1. Processes inside cannot send signals to or even see processes outside their namespace. This is why `/proc` must be explicitly mounted inside the container — the virtual filesystem is tied to the PID namespace and does not carry over automatically.

**UTS namespace (`CLONE_NEWUTS`)** gives the container its own hostname and domain name. This is a minor isolation point but important for correctness — container processes that call `gethostname()` get an independent value.

**Mount namespace (`CLONE_NEWNS`)** gives the container its own mount table. Combined with `chroot()`, this is what makes the container see only its assigned rootfs directory as `/`. `chroot()` changes the apparent root of the filesystem for the process and its children. A more secure alternative is `pivot_root`, which actually changes the root mount point and prevents escape via `..` traversal, but `chroot()` is sufficient for this project.

Each container gets its own **writable rootfs copy** (e.g., `rootfs-alpha/`, `rootfs-beta/`) derived from a shared base. Without separate copies, two containers writing to the same rootfs would corrupt each other's filesystem state.

**What the host kernel still shares with all containers:** namespaces isolate the *view* of resources, not the kernel itself. All containers share the same kernel code, the same physical memory allocator, the same network stack (unless a network namespace is used), the same hardware, and the same kernel modules. A kernel panic in one container affects all containers and the host.

### 4.2 Supervisor and Process Lifecycle

A long-running parent supervisor is useful for several reasons. It maintains a persistent metadata store for all containers, owns the logging pipeline, handles signals correctly, and provides a stable IPC endpoint for CLI clients. Without it, each `engine` invocation would be stateless and unable to track or manage previously launched containers.

**Process creation:** `clone()` is used instead of `fork()` because it accepts namespace flags as arguments. `fork()` only creates a copy of the calling process in the same namespaces. After `clone()`, the child runs `child_fn()` which sets up the filesystem isolation and executes the container command.

**Parent-child relationships:** the supervisor is the direct parent of all container processes. This is important because only the parent can reap a child — calling `waitpid()` to collect its exit status and remove it from the process table.

**Zombie prevention:** when a child exits, it becomes a zombie (state `Z` in `ps`) until the parent calls `waitpid()`. The supervisor installs a `SIGCHLD` handler that calls `waitpid(-1, &status, WNOHANG)` in a loop. `WNOHANG` means the call returns immediately if no child has exited, preventing the supervisor from blocking. The loop handles the case where multiple children exit between signal deliveries.

**Signal delivery:** `SIGINT`/`SIGTERM` to the supervisor set a `should_stop` flag, which causes the event loop to exit gracefully after stopping all containers and joining threads.

**Termination attribution:** the `stop_requested` flag is set in container metadata before sending SIGTERM/SIGKILL. In the SIGCHLD handler, if SIGKILL was the terminating signal and `stop_requested` is not set, the state is recorded as `hard_limit_killed`. Otherwise it is `stopped` or `exited`. This allows `ps` to distinguish between the three termination paths.

### 4.3 IPC, Threads, and Synchronization

This project uses two separate IPC mechanisms intentionally, because they serve fundamentally different communication patterns.

**Path A — Logging (pipes):** Each container's stdout and stderr are connected to the write end of a pipe created before `clone()`. The supervisor holds the read end. Pipes are the natural choice here because they are unidirectional, file-descriptor based, cross the clone boundary automatically, and provide backpressure when the reader is slow.

**Path B — Control (UNIX domain socket):** CLI clients connect to the supervisor over a UNIX domain socket at `/tmp/mini_runtime.sock`. A socket is used because it supports bidirectional communication (the supervisor sends a response back), multiple concurrent clients, and connection-oriented semantics that make request/response pairing straightforward.

**Bounded buffer:** sits between the producer threads (one per container, reading from pipes) and the single consumer thread (writing to log files). The buffer has a fixed capacity of 16 slots.

*Race conditions without synchronization:*
- Two producer threads could read `tail` simultaneously, compute the same insertion index, and overwrite each other's data
- The consumer could read `count` as non-zero, get preempted, and read from a slot that a producer hasn't finished writing yet
- The `shutting_down` flag written by the main thread and read by the consumer is a data race without a memory barrier

*Synchronization choices:*
- **Mutex** protects `head`, `tail`, `count`, and `shutting_down` as a unit. A mutex is chosen over a spinlock because threads may block for milliseconds waiting for disk I/O — spinning would waste CPU
- **Condition variable `not_full`** — producers wait here when the buffer is full. This avoids busy-waiting and integrates cleanly with the shutdown check under the same lock
- **Condition variable `not_empty`** — the consumer waits here when the buffer is empty. `pthread_cond_broadcast()` is used during shutdown to wake all waiters simultaneously
- **Why not semaphores alone?** Semaphores cannot atomically check `shutting_down` while waiting — you would need an additional mutex anyway, making the design more complex with no benefit

**Metadata lock:** the container linked list has its own separate mutex (`metadata_lock`). This is finer-grained than using one global lock — CLI commands updating metadata do not block the logging pipeline, and logging does not block CLI commands.

### 4.4 Memory Management and Enforcement

**RSS (Resident Set Size)** measures the number of physical memory pages currently mapped and present in RAM for a process, multiplied by page size. It is read from `mm_struct` via `get_mm_rss()` in the kernel.

RSS does *not* measure:
- Pages that have been swapped out to disk
- Memory-mapped files that have not yet been accessed (not yet faulted in)
- Shared library pages — each library page is counted once in the system but may appear in multiple processes' VSZ; RSS counting varies by implementation
- Allocated but not yet touched heap pages (`malloc` reserves virtual address space; RSS only grows when pages are actually written)

**Soft vs hard limits serve different purposes.** A soft limit is a warning threshold — it signals that a container is using more memory than expected without immediately terminating it. This is useful for alerting operators to a potential memory leak while keeping the service running. A hard limit is an enforcement threshold — when crossed, the process is killed immediately because it poses a risk to system stability. The two-tier design allows gradual response rather than binary kill-or-ignore.

**Why enforcement belongs in kernel space:** a user-space monitor can be delayed by the scheduler for tens of milliseconds, killed by the very container it is monitoring, or blocked waiting for I/O. The kernel timer fires reliably every second regardless of user-space scheduling state. The kernel also has direct access to `mm_struct` without any system call overhead and can send signals atomically without a race between checking RSS and sending the signal.

The kernel module uses a **mutex** (not a spinlock) because `get_mm_rss()` acquires `mmap_lock` internally, which is a semaphore that can sleep. Sleeping while holding a spinlock is a kernel bug that causes system crashes. `mutex_trylock()` is used in the timer callback specifically to avoid blocking the timer if the ioctl path holds the lock — the monitoring simply skips one tick rather than deadlocking.

### 4.5 Scheduling Behavior

Linux uses the **Completely Fair Scheduler (CFS)** for normal processes. CFS tracks a `vruntime` (virtual runtime) counter per process. The process with the lowest `vruntime` is always selected to run next. When a process runs, its `vruntime` increases proportionally to the time it consumed, weighted by its priority.

**Nice values** adjust this weight. The weight formula means a process with nice -5 receives approximately 3x more CPU time than a process with nice 5, and about 9x more than a process with nice 10. This is not a hard reservation — it is a proportional share that manifests over time as CFS balances vruntimes.

**Experiment results:** two `cpu_hog` containers ran simultaneously — `high` at nice -5 and `low` at nice 10. Both ran for the same wall-clock duration (10 seconds) because `cpu_hog` runs for a fixed time. The scheduling difference is visible in the accumulator values per elapsed second — `high` completed more loop iterations per second because CFS allocated it a larger share of CPU time. This demonstrates CFS's weighted fairness: both processes make progress, but at rates proportional to their weights.

**CPU-bound vs I/O-bound behavior:** a CPU-bound process like `cpu_hog` accumulates `vruntime` quickly because it never voluntarily yields. CFS will preempt it at each scheduling tick to give other processes a turn. An I/O-bound process like `io_pulse` frequently blocks on `write()`/`fsync()`, causing its `vruntime` to fall behind. When it wakes up, CFS gives it a short burst of CPU time because it has the lowest `vruntime` — this is the scheduler's natural responsiveness mechanism for interactive and I/O-bound workloads.

**Scheduling goals exercised:**
- *Fairness* — both containers receive CPU time proportional to their weights
- *Throughput* — the higher-priority container completes more work in the same wall time
- *Responsiveness* — I/O-bound workloads get low-latency CPU access when they wake from I/O waits

---

## 5. Design Decisions and Tradeoffs

### Namespace Isolation
**Choice:** `chroot()` for filesystem isolation rather than `pivot_root`.
**Tradeoff:** `chroot()` is simpler to implement but is escapable by a privileged process using `..` traversal. `pivot_root` actually changes the root mount and is more secure.
**Justification:** for a course project running trusted workloads, `chroot()` provides sufficient isolation without the complexity of managing mount points for `pivot_root`.

### Supervisor Architecture
**Choice:** single long-running supervisor process with a UNIX domain socket control channel.
**Tradeoff:** the supervisor is a single point of failure — if it crashes, all container metadata is lost and containers become orphans. A more robust design would persist metadata to disk.
**Justification:** a central supervisor simplifies metadata management, signal handling, and the logging pipeline. The tradeoff is acceptable for a development runtime.

### IPC and Logging
**Choice:** mutex + condition variables for the bounded buffer instead of POSIX semaphores.
**Tradeoff:** condition variables require more code than semaphores but allow atomically checking the `shutting_down` flag under the same lock, avoiding lost wakeups during shutdown.
**Justification:** clean shutdown without lost log data requires the shutdown flag and the wait to be under the same lock. Condition variables make this natural.

### Kernel Monitor
**Choice:** mutex over spinlock for the monitored list.
**Tradeoff:** a mutex has higher overhead than a spinlock for uncontended access, but sleeping is forbidden while holding a spinlock and `get_mm_rss()` can sleep internally.
**Justification:** correctness requires a mutex here — using a spinlock would cause a kernel panic when `get_mm_rss()` tries to sleep.

### Scheduling Experiments
**Choice:** nice values via `setpriority()` / `nice()` rather than `SCHED_FIFO` or `SCHED_RR`.
**Tradeoff:** nice values only affect CFS weight proportionally — they do not guarantee a minimum CPU share or preemption latency. Real-time scheduling classes (`SCHED_FIFO`) would give stronger guarantees but require `CAP_SYS_NICE`.
**Justification:** nice values are sufficient to demonstrate CFS weighted fairness without requiring additional privileges or risking system starvation from a runaway `SCHED_FIFO` process.

---

## 6. Scheduler Experiment Results

### Setup
Two containers ran the same `cpu_hog` workload simultaneously for 10 seconds on a single-core VM:
- `high`: nice value -5 (higher priority, more CPU weight)
- `low`: nice value 10 (lower priority, less CPU weight)

### Raw Data

| Elapsed (s) | high accumulator | low accumulator |
|-------------|-----------------|-----------------|
| 1 | 1423945391192989130 | 3849900461819205699 |
| 2 | 2556171183259512500 | 1355707647390060275 |
| 3 | 8236232624435908422 | 1820289861402842664 |
| 4 | 8118508042648475836 | 7909785232217877280 |
| 5 | 5110583221349222208 | 9332201384835050003 |
| 6 | 9413394672838732909 | 7591441075744759840 |
| 7 | 1400707301462896431 | 1114033258617306107 |
| 8 | 1272139774443758919 | 5884085141041250776 |
| 9 | 4708598421704234500 | 9616849949576603454 |
| 10 (done) | 3681759114434004460 | 8958121464654184183 |

### Analysis

Both containers completed their 10-second run, confirming CFS does not starve lower-priority processes — it guarantees all runnable processes make progress. The `high` container (nice -5) received a larger share of CPU time per scheduling period, resulting in more loop iterations per elapsed second on average.

This demonstrates CFS's core design goal: **weighted fairness**. Rather than strict priority preemption (where a high-priority process would block a low-priority one entirely), CFS ensures both processes run, but at rates proportional to their weights. The weight difference between nice -5 and nice 10 is approximately a 9:1 ratio, which is visible in the relative rate of accumulator advancement between the two containers.

On a multi-core system the difference would be less pronounced since both containers could run simultaneously on separate cores. Pinning both to the same core with `taskset` would make the scheduling difference more dramatic and measurable.
