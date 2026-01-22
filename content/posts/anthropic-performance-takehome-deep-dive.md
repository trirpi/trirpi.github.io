+++
author = "Tristan Trouwen"
title = "Deep Dive: Anthropic's Performance Take-Home (The One Claude Beat Humans At)"
date = "2026-01-21"
description = "A detailed breakdown of Anthropic's original performance take-home test - the one Claude Opus 4.5 beat most humans at. We'll explore the custom VLIW SIMD architecture simulator, the tree traversal problem, and why this challenge is so fascinating."
tags = [
    "performance",
    "compilers",
    "SIMD",
    "VLIW",
    "optimization",
    "AI",
]
categories = [
    "tutorials",
]
+++

Today, Anthropic [open-sourced their original performance engineering take-home](https://github.com/anthropics/original_performance_takehome) - the one they retired after Claude Opus 4.5 started outperforming most humans given only 2 hours. As someone who works on AI kernels, I couldn't resist diving in.

What makes this challenge particularly interesting is that it's not your typical "optimize this Python function" interview problem. Instead, candidates are given a complete **custom VLIW SIMD architecture simulator** and asked to optimize a kernel running on it. It's essentially a puzzle that tests your understanding of computer architecture, instruction-level parallelism, and low-level optimization - all areas where humans have traditionally had an edge over AI systems.

In this post, I'll break down exactly how this whole system works, from the machine architecture to the reference kernel, and explain why it's such a clever test of performance engineering skills.

## The Big Picture

Before we dive into code, let me paint a picture of what's happening here:

1. **A custom processor architecture** is simulated in Python
2. **A specific computational problem** (tree traversal with hashing) needs to be solved on this processor
3. **Your job** is to optimize the kernel to minimize clock cycles

The baseline implementation takes a whopping **147,734 cycles**. Claude Opus 4.5's best result after 11.5 hours of test-time compute? **1,487 cycles**. That's a **99x speedup**. The best human performance is apparently even better, but Anthropic isn't saying by how much.

Let's understand what we're working with.

---

## Part 1: The VLIW SIMD Architecture

The heart of this challenge is a simulated [VLIW](https://en.wikipedia.org/wiki/Very_long_instruction_word) (Very Large Instruction Word) [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) (Single Instruction, Multiple Data) processor. If you've never heard these terms before, here's the quick version:

- **VLIW**: Multiple operations can execute in a single cycle, as long as they use different "engines"
- **SIMD**: Vector operations that process multiple data elements at once

### The Machine Class

The entire simulator lives in the [`Machine` class](https://github.com/anthropics/original_performance_takehome/blob/main/problem.py#L64-L95):

```python
class Machine:
    """
    Simulator for a custom VLIW SIMD architecture.

    VLIW (Very Large Instruction Word): Cores are composed of different
    "engines" each of which can execute multiple "slots" per cycle in parallel.
    How many slots each engine can execute per cycle is limited by SLOT_LIMITS.
    Effects of instructions don't take effect until the end of cycle. Each
    cycle, all engines execute all of their filled slots for that instruction.
    Effects like writes to memory take place after all the inputs are read.

    SIMD: There are instructions for acting on vectors of VLEN elements in a
    single slot. You can use vload and vstore to load multiple contiguous
    elements but not non-contiguous elements. Use vbroadcast to broadcast a
    scalar to a vector and then operate on vectors with valu instructions.
    """
```

### Execution Engines and Slot Limits

This is where things get interesting. The processor has multiple **engines**, each capable of executing multiple **slots** per cycle. Here are the [limits defined in the code](https://github.com/anthropics/original_performance_takehome/blob/main/problem.py#L48-L55):

```python
SLOT_LIMITS = {
    "alu": 12,      # 12 scalar ALU operations per cycle
    "valu": 6,      # 6 vector ALU operations per cycle
    "load": 2,      # 2 load operations per cycle
    "store": 2,     # 2 store operations per cycle
    "flow": 1,      # 1 flow control operation per cycle
    "debug": 64,    # Debug operations (not counted)
}
```

This is where the "VLIW" magic happens. In a single cycle, you could theoretically execute:
- 12 scalar math operations
- 6 vector math operations (each operating on 8 elements!)
- 2 memory loads
- 2 memory stores
- 1 control flow operation

That's a lot of parallelism per cycle. The challenge is structuring your code to actually *use* all of it.

### What an Instruction Looks Like

An instruction is a dictionary mapping engine names to lists of operations (slots). Here's an [example from the docstring](https://github.com/anthropics/original_performance_takehome/blob/main/problem.py#L86-L87):

```python
{"valu": [("*", 4, 0, 0), ("+", 8, 4, 0)], "load": [("load", 16, 17)]}
```

This single instruction does **three things in one cycle**:
1. Vector multiply: `scratch[4:12] = scratch[0:8] * scratch[0:8]`
2. Vector add: `scratch[8:16] = scratch[4:12] + scratch[0:8]`
3. Scalar load: `scratch[16] = memory[scratch[17]]`

All in one clock cycle. Beautiful, right?

### Memory Model: Scratch Space and Main Memory

The machine has two memory spaces:

1. **Main Memory** (`self.mem`): This is where the problem input lives and where outputs are written
2. **Scratch Space** (`core.scratch`): Think of this as registers + constant memory + a manually managed cache

From the [code constants](https://github.com/anthropics/original_performance_takehome/blob/main/problem.py#L57-L60):

```python
VLEN = 8          # Vector length: 8 elements
N_CORES = 1       # Single core (older versions had multiple)
SCRATCH_SIZE = 1536  # 1536 words of scratch space
```

The scratch space is crucial. Every ALU operation reads and writes to scratch addresses. Constants need to be loaded into scratch before use. It's like programming a [GPU with shared memory](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/), but more explicit.

### The ALU Instruction Set

Let's look at what [scalar operations are available](https://github.com/anthropics/original_performance_takehome/blob/main/problem.py#L219-L252):

```python
def alu(self, core, op, dest, a1, a2):
    a1 = core.scratch[a1]
    a2 = core.scratch[a2]
    match op:
        case "+":
            res = a1 + a2
        case "-":
            res = a1 - a2
        case "*":
            res = a1 * a2
        case "//":
            res = a1 // a2
        case "cdiv":
            res = cdiv(a1, a2)  # Ceiling division
        case "^":
            res = a1 ^ a2       # XOR
        case "&":
            res = a1 & a2       # AND
        case "|":
            res = a1 | a2       # OR
        case "<<":
            res = a1 << a2      # Left shift
        case ">>":
            res = a1 >> a2      # Right shift
        case "%":
            res = a1 % a2       # Modulo
        case "<":
            res = int(a1 < a2)  # Less than (returns 0 or 1)
        case "==":
            res = int(a1 == a2) # Equality (returns 0 or 1)
    res = res % (2**32)  # 32-bit wrap-around
    self.scratch_write[dest] = res
```

Pretty standard stuff, but note that all arithmetic is **32-bit unsigned with wrap-around**. This is important for the hash function.

### Vector ALU Operations

The vector unit is where the SIMD magic happens. Here's the [valu implementation](https://github.com/anthropics/original_performance_takehome/blob/main/problem.py#L254-L267):

```python
def valu(self, core, *slot):
    match slot:
        case ("vbroadcast", dest, src):
            # Broadcast scalar to vector
            for i in range(VLEN):
                self.scratch_write[dest + i] = core.scratch[src]
        case ("multiply_add", dest, a, b, c):
            # Fused multiply-add: dest = a * b + c
            for i in range(VLEN):
                mul = (core.scratch[a + i] * core.scratch[b + i]) % (2**32)
                self.scratch_write[dest + i] = (mul + core.scratch[c + i]) % (2**32)
        case (op, dest, a1, a2):
            # Apply scalar op element-wise
            for i in range(VLEN):
                self.alu(core, op, dest + i, a1 + i, a2 + i)
```

Three key operations:
1. **vbroadcast**: Copy a scalar to all 8 positions of a vector
2. **multiply_add**: Fused multiply-add (very common in DSP and ML)
3. **Element-wise ops**: Any scalar ALU op can be applied to vectors

### Load and Store

[Memory operations](https://github.com/anthropics/original_performance_takehome/blob/main/problem.py#L269-L298) are limited but essential:

```python
def load(self, core, *slot):
    match slot:
        case ("load", dest, addr):
            # Scalar load: scratch[dest] = memory[scratch[addr]]
            self.scratch_write[dest] = self.mem[core.scratch[addr]]
        case ("vload", dest, addr):
            # Vector load: 8 consecutive elements
            addr = core.scratch[addr]
            for vi in range(VLEN):
                self.scratch_write[dest + vi] = self.mem[addr + vi]
        case ("const", dest, val):
            # Load immediate constant
            self.scratch_write[dest] = (val) % (2**32)
```

Key insight: **only 2 loads and 2 stores per cycle**. This is often the bottleneck. Vector loads help because `vload` gets you 8 elements in one slot!

### Flow Control

[Flow control operations](https://github.com/anthropics/original_performance_takehome/blob/main/problem.py#L300-L335) include jumps, selects, and more:

```python
def flow(self, core, *slot):
    match slot:
        case ("select", dest, cond, a, b):
            # Conditional select: dest = cond ? a : b
            self.scratch_write[dest] = (
                core.scratch[a] if core.scratch[cond] != 0 else core.scratch[b]
            )
        case ("vselect", dest, cond, a, b):
            # Vector conditional select
            for vi in range(VLEN):
                self.scratch_write[dest + vi] = (
                    core.scratch[a + vi]
                    if core.scratch[cond + vi] != 0
                    else core.scratch[b + vi]
                )
        case ("cond_jump", cond, addr):
            # Conditional jump
            if core.scratch[cond] != 0:
                core.pc = addr
        case ("jump", addr):
            # Unconditional jump
            core.pc = addr
```

The `select` and `vselect` operations are crucial for [branchless programming](https://en.algorithmica.org/hpc/pipelining/branchless/). Instead of:
```
if (condition) x = a; else x = b;
```
You can do:
```
x = select(condition, a, b);
```

This avoids branch misprediction penalties (though in this simulator, there's no branch predictor - but it still saves cycles by avoiding jumps).

---

## Part 2: The Problem - Tree Traversal with Hashing

Now that we understand the machine, let's look at what we're actually computing. The problem is a **batched tree traversal with hashing**.

### The Tree Structure

From the [`Tree` dataclass](https://github.com/anthropics/original_performance_takehome/blob/main/problem.py#L405-L418):

```python
@dataclass
class Tree:
    """
    An implicit perfect balanced binary tree with values on the nodes.
    """
    height: int
    values: list[int]

    @staticmethod
    def generate(height: int):
        n_nodes = 2 ** (height + 1) - 1
        values = [random.randint(0, 2**30 - 1) for _ in range(n_nodes)]
        return Tree(height, values)
```

This is a [perfect binary tree](https://www.programiz.com/dsa/perfect-binary-tree) stored as an array. For a tree of height 10, that's `2^11 - 1 = 2047` nodes.

The indexing follows the standard array representation:
- Root is at index 0
- Left child of node `i` is at index `2*i + 1`
- Right child of node `i` is at index `2*i + 2`

### The Hash Function

A key part of the algorithm is a [32-bit hash function](https://github.com/anthropics/original_performance_takehome/blob/main/problem.py#L439-L464):

```python
HASH_STAGES = [
    ("+", 0x7ED55D16, "+", "<<", 12),
    ("^", 0xC761C23C, "^", ">>", 19),
    ("+", 0x165667B1, "+", "<<", 5),
    ("+", 0xD3A2646C, "^", "<<", 9),
    ("+", 0xFD7046C5, "+", "<<", 3),
    ("^", 0xB55A4F09, "^", ">>", 16),
]

def myhash(a: int) -> int:
    """A simple 32-bit hash function"""
    fns = {
        "+": lambda x, y: x + y,
        "^": lambda x, y: x ^ y,
        "<<": lambda x, y: x << y,
        ">>": lambda x, y: x >> y,
    }

    def r(x):
        return x % (2**32)

    for op1, val1, op2, op3, val3 in HASH_STAGES:
        a = r(fns[op2](r(fns[op1](a, val1)), r(fns[op3](a, val3))))

    return a
```

Each stage does: `a = (a op1 val1) op2 (a op3 val3)`

For example, the first stage: `a = (a + 0x7ED55D16) + (a << 12)`

This is similar to [Bob Jenkins' one-at-a-time hash](https://en.wikipedia.org/wiki/Jenkins_hash_function) or other simple mixing functions. The data-driven format (`HASH_STAGES`) makes it easy to implement in the kernel.

### The Reference Kernel

Here's the [actual algorithm we need to implement](https://github.com/anthropics/original_performance_takehome/blob/main/problem.py#L467-L484):

```python
def reference_kernel(t: Tree, inp: Input):
    """
    Reference implementation of the kernel.

    A parallel tree traversal where at each node we set
    cur_inp_val = myhash(cur_inp_val ^ node_val)
    and then choose the left branch if cur_inp_val is even.
    If we reach the bottom of the tree we wrap around to the top.
    """
    for h in range(inp.rounds):
        for i in range(len(inp.indices)):
            idx = inp.indices[i]
            val = inp.values[i]
            val = myhash(val ^ t.values[idx])
            idx = 2 * idx + (1 if val % 2 == 0 else 2)
            idx = 0 if idx >= len(t.values) else idx
            inp.values[i] = val
            inp.indices[i] = idx
```

In plain English:
1. We have a **batch** of `(index, value)` pairs (256 of them in the test)
2. For each **round** (16 rounds in the test):
   - For each item in the batch:
     - XOR the current value with the tree node value
     - Hash the result
     - Move to left child (if hash is even) or right child (if odd)
     - If we fall off the tree, wrap back to root

The test configuration is:
- Tree height: 10 (2047 nodes)
- Batch size: 256 items
- Rounds: 16

That's `256 * 16 = 4096` traversal steps, each involving a hash computation.

### Memory Layout

The [memory image](https://github.com/anthropics/original_performance_takehome/blob/main/problem.py#L487-L513) is laid out as:

```python
def build_mem_image(t: Tree, inp: Input) -> list[int]:
    # Header: 7 words
    mem[0] = inp.rounds
    mem[1] = len(t.values)      # n_nodes
    mem[2] = len(inp.indices)   # batch_size
    mem[3] = t.height
    mem[4] = forest_values_p    # Pointer to tree values
    mem[5] = inp_indices_p      # Pointer to indices array
    mem[6] = inp_values_p       # Pointer to values array
    
    # Data sections follow...
```

This is a classic C-style memory layout with pointers. The kernel needs to read this header to find where everything is.

---

## Part 3: The Baseline Kernel Implementation

Now let's look at the [baseline kernel implementation](https://github.com/anthropics/original_performance_takehome/blob/main/perf_takehome.py#L88-L175) that candidates need to optimize:

```python
def build_kernel(
    self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
):
    """
    Like reference_kernel2 but building actual instructions.
    Scalar implementation using only scalar ALU and load/store.
    """
    tmp1 = self.alloc_scratch("tmp1")
    tmp2 = self.alloc_scratch("tmp2")
    tmp3 = self.alloc_scratch("tmp3")
    # ... initialization code ...
    
    for round in range(rounds):
        for i in range(batch_size):
            i_const = self.scratch_const(i)
            # idx = mem[inp_indices_p + i]
            body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
            body.append(("load", ("load", tmp_idx, tmp_addr)))
            # ... many more instructions ...
```

### Why It's So Slow

The baseline implementation has several deliberate inefficiencies:

1. **Fully unrolled loops**: The entire `rounds * batch_size` computation is unrolled
2. **One operation per cycle**: Each slot is in its own instruction bundle
3. **No vector operations**: Everything uses scalar ALU, ignoring SIMD
4. **No instruction packing**: Each instruction bundle only uses one engine

Let me show you what I mean. Here's the [build() method](https://github.com/anthropics/original_performance_takehome/blob/main/perf_takehome.py#L51-L56):

```python
def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
    # Simple slot packing that just uses one slot per instruction bundle
    instrs = []
    for engine, slot in slots:
        instrs.append({engine: [slot]})
    return instrs
```

This creates one instruction bundle per operation. So instead of:
```python
{"alu": [op1, op2, op3], "load": [load1]}  # All in one cycle
```

You get:
```python
{"alu": [op1]}   # Cycle 1
{"alu": [op2]}   # Cycle 2
{"alu": [op3]}   # Cycle 3
{"load": [load1]} # Cycle 4
```

That's 4 cycles instead of 1!

### The Hash Implementation

Here's how the hash is [implemented in the kernel](https://github.com/anthropics/original_performance_takehome/blob/main/perf_takehome.py#L77-L86):

```python
def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
    slots = []
    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
        slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
        slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
        slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
        slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))
    return slots
```

Each hash stage requires 3 ALU operations. With 6 stages, that's 18 ALU operations per hash. Multiply by 4096 traversal steps = **73,728 ALU cycles just for hashing** (in the unoptimized version).

---

## Part 4: The Debugging and Testing Infrastructure

Anthropic included some really nice tooling for understanding and debugging the machine.

### Perfetto Trace Viewer

The simulator can output traces in [Chrome's Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview), viewable in [Perfetto](https://ui.perfetto.dev/). This is the same format used for Chrome DevTools performance traces and Android system traces.

From the [setup_trace method](https://github.com/anthropics/original_performance_takehome/blob/main/problem.py#L151-L176):

```python
def setup_trace(self):
    """
    The simulator generates traces in Chrome's Trace Event Format for
    visualization in Perfetto (or chrome://tracing if you prefer it).
    """
    self.trace = open("trace.json", "w")
    self.trace.write("[")
    # ... creates trace events for each engine slot ...
```

To use it:
```bash
python perf_takehome.py Tests.test_kernel_trace
python watch_trace.py  # Opens browser with live-reloading trace
```

This gives you a visual timeline showing which engine slots are being used each cycle. It's incredibly useful for spotting inefficiencies.

### Debug Instructions

The machine supports [debug instructions](https://github.com/anthropics/original_performance_takehome/blob/main/problem.py#L365-L382) that don't count toward cycle count:

```python
if name == "debug":
    for slot in slots:
        if slot[0] == "compare":
            loc, key = slot[1], slot[2]
            ref = self.value_trace[key]
            res = core.scratch[loc]
            assert res == ref, f"{res} != {ref} for {key} at pc={core.pc}"
```

You can sprinkle `("debug", ("compare", addr, key))` throughout your kernel to verify intermediate values against the reference implementation. Super helpful when your optimized kernel produces wrong results.

### The Watch Trace Server

The [`watch_trace.py`](https://github.com/anthropics/original_performance_takehome/blob/main/watch_trace.py) file is a clever little HTTP server that:

1. Serves the `trace.json` file
2. Detects when it changes
3. Automatically reloads the trace in Perfetto

```python
async def repoll(win, traceUrl, mtime):
    const newMtime = await getMtime();
    if (newMtime !== mtime) {
        logs.innerText += `Trace updated, fetching new version...\n`;
        // ... reload trace ...
    }
    setTimeout(() => repoll(win, traceUrl, newMtime), 500);
}
```

This creates a nice edit-run-visualize loop.

### Submission Tests

The [submission tests](https://github.com/anthropics/original_performance_takehome/blob/main/tests/submission_tests.py) verify both correctness and performance:

```python
class SpeedTests(unittest.TestCase):
    def test_kernel_speedup(self):
        assert cycles() < BASELINE  # 147,734

    def test_opus45_casual(self):
        # Claude Opus 4.5 casual session
        assert cycles() < 1790

    def test_opus45_11hr(self):
        # Claude Opus 4.5 after 11.5 hours
        assert cycles() < 1487
```

Note: The tests use a [frozen copy of problem.py](https://github.com/anthropics/original_performance_takehome/blob/main/tests/frozen_problem.py) to prevent "optimization by modifying the simulator" cheats.

---

## Part 5: Optimization Strategies

Now for the fun part - how would you actually optimize this? I won't give away solutions, but here are the key techniques to consider:

### 1. Instruction Packing (VLIW)

The most obvious win. Instead of one operation per instruction, pack multiple independent operations together:

```python
# Before: 4 cycles
{"alu": [op1]}
{"alu": [op2]}
{"alu": [op3]}
{"load": [load1]}

# After: 1 cycle
{"alu": [op1, op2, op3], "load": [load1]}
```

You can pack up to **12 ALU + 6 VALU + 2 load + 2 store + 1 flow** in a single cycle. That's a lot of parallelism to exploit.

### 2. SIMD Vectorization

Process 8 batch items at once instead of 1:

```python
# Before: 256 scalar operations
for i in range(256):
    val[i] = hash(val[i] ^ node[idx[i]])

# After: 32 vector operations
for i in range(0, 256, 8):
    val[i:i+8] = vhash(val[i:i+8] ^ node[idx[i:i+8]])
```

8x fewer iterations, and `valu` ops can do 8 elements in one slot.

### 3. Software Pipelining

Overlap computation of different rounds/iterations:

```python
# Instead of: load -> compute -> store -> load -> compute -> store
# Do: load[0], load[1], compute[0], load[2], compute[1], store[0], ...
```

This keeps all engine slots busy by interleaving different stages of the pipeline.

### 4. Loop Unrolling with Packing

Partially unroll loops to expose more instruction-level parallelism, then pack those operations together.

### 5. Reducing Memory Operations

With only 2 load slots per cycle, memory access is often the bottleneck. Strategies:
- Preload data into scratch space
- Use vector loads (`vload`) to get 8 elements per slot
- Reuse loaded values across iterations

### 6. Branchless Tree Traversal

The `select` instruction enables branchless computation:

```python
# Instead of:
if val % 2 == 0:
    idx = 2*idx + 1
else:
    idx = 2*idx + 2

# Use:
offset = select(val % 2 == 0, 1, 2)
idx = 2*idx + offset
```

### The Challenge: Data Dependencies

Here's what makes this hard. In the reference algorithm, you can't compute the next iteration until you know which tree node to visit. But you don't know that until you've:
1. Loaded the current node value
2. XORed it with the current value
3. Hashed the result
4. Checked if it's even or odd

That's a long [dependency chain](https://en.wikipedia.org/wiki/Instruction-level_parallelism#Data_dependency). The art is finding parallelism *across* different batch items and rounds, even when individual chains are sequential.

---

## Why This Test Is Brilliant

After studying this challenge, I have a deep appreciation for its design:

1. **Domain-specific architecture**: You can't just throw standard compiler optimizations at it. You need to understand the hardware.

2. **Multiple optimization dimensions**: VLIW packing, SIMD vectorization, memory access patterns, software pipelining - they all interact.

3. **Clear metrics**: Cycle count is unambiguous. No "well, it depends on the cache" ambiguity.

4. **Rich debugging tools**: The Perfetto integration is chef's kiss.

5. **Escalating difficulty**: The baseline is easy to beat (just pack instructions), but approaching Claude's results requires sophisticated techniques.

6. **Resistant to memorization**: The exact numbers change with random seeds, so memorizing solutions doesn't help.

It's essentially asking: "Can you think like a compiler backend for a novel architecture?" That's a genuinely hard problem that tests real understanding.

---

## Diagrams

For those who are visual learners, here are some diagrams that help understand the architecture:

### Machine Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         VLIW SIMD Machine                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────── Single Cycle ──────────────────────┐     │
│   │                                                        │     │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │     │
│   │  │   ALU    │  │   VALU   │  │   LOAD   │            │     │
│   │  │ 12 slots │  │ 6 slots  │  │ 2 slots  │            │     │
│   │  │          │  │ (VLEN=8) │  │          │            │     │
│   │  └────┬─────┘  └────┬─────┘  └────┬─────┘            │     │
│   │       │             │             │                   │     │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │     │
│   │  │  STORE   │  │   FLOW   │  │  DEBUG   │            │     │
│   │  │ 2 slots  │  │ 1 slot   │  │ (free)   │            │     │
│   │  └────┬─────┘  └────┬─────┘  └──────────┘            │     │
│   │       │             │                                 │     │
│   └───────┴─────────────┴─────────────────────────────────┘     │
│                         │                                        │
│   ┌─────────────────────┴─────────────────────────────────┐     │
│   │              SCRATCH SPACE (1536 words)                │     │
│   │         (registers + constants + cache)                │     │
│   └───────────────────────┬───────────────────────────────┘     │
│                           │                                      │
│   ┌───────────────────────┴───────────────────────────────┐     │
│   │                    MAIN MEMORY                         │     │
│   │   ┌─────────┬──────────────┬──────────┬──────────┐    │     │
│   │   │ Header  │ Tree Values  │ Indices  │  Values  │    │     │
│   │   │(7 words)│ (2047 words) │(256 words)│(256 words)│   │     │
│   │   └─────────┴──────────────┴──────────┴──────────┘    │     │
│   └───────────────────────────────────────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Instruction Bundle Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    INSTRUCTION BUNDLE                            │
│                     (1 clock cycle)                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "alu": [                     "valu": [                         │
│    ("+", dest, a, b),           ("*", vdest, va, vb),           │
│    ("-", dest, a, b),           ("+", vdest, va, vb),           │
│    ("*", dest, a, b),           ...up to 6 slots...             │
│    ...up to 12 slots...       ]                                  │
│  ]                                                               │
│                                                                  │
│  "load": [                    "store": [                        │
│    ("load", dest, addr),        ("store", addr, src),           │
│    ("vload", vdest, addr)       ("vstore", addr, vsrc)          │
│  ]                            ]                                  │
│                                                                  │
│  "flow": [                    "debug": [                        │
│    ("select", d, c, a, b)       ("compare", loc, key)           │
│  ]                            ]  (not counted)                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Tree Traversal Algorithm

```
                    Round 0                    Round 1
                       │                          │
        ┌──────────────┼──────────────┐          ...
        │              │              │
    ┌───▼───┐      ┌───▼───┐      ┌───▼───┐
    │Batch 0│      │Batch 1│      │Batch 2│     ... (256 items)
    └───┬───┘      └───┬───┘      └───┬───┘
        │              │              │
        ▼              ▼              ▼
   ┌─────────────────────────────────────────┐
   │  idx = indices[i]                       │
   │  val = values[i]                        │
   │  node_val = tree[idx]                   │
   │  val = hash(val ^ node_val)             │
   │  idx = 2*idx + (1 if val%2==0 else 2)   │
   │  if idx >= n_nodes: idx = 0             │
   │  indices[i] = idx                       │
   │  values[i] = val                        │
   └─────────────────────────────────────────┘
```

### Hash Function Pipeline

```
Input value: a
        │
        ▼
┌───────────────────────────────────────────────────┐
│ Stage 0: a = (a + 0x7ED55D16) + (a << 12)        │
├───────────────────────────────────────────────────┤
│ Stage 1: a = (a ^ 0xC761C23C) ^ (a >> 19)        │
├───────────────────────────────────────────────────┤
│ Stage 2: a = (a + 0x165667B1) + (a << 5)         │
├───────────────────────────────────────────────────┤
│ Stage 3: a = (a + 0xD3A2646C) ^ (a << 9)         │
├───────────────────────────────────────────────────┤
│ Stage 4: a = (a + 0xFD7046C5) + (a << 3)         │
├───────────────────────────────────────────────────┤
│ Stage 5: a = (a ^ 0xB55A4F09) ^ (a >> 16)        │
└───────────────────────────────────────────────────┘
        │
        ▼
   Output hash

Each stage requires 3 ALU operations:
  tmp1 = a op1 constant1    (e.g., a + 0x7ED55D16)
  tmp2 = a op3 constant2    (e.g., a << 12)
  a = tmp1 op2 tmp2         (e.g., tmp1 + tmp2)

Total: 6 stages × 3 ops = 18 ALU operations per hash
```

---

## Try It Yourself

If you want to take a crack at this:

```bash
git clone https://github.com/anthropics/original_performance_takehome.git
cd original_performance_takehome
python perf_takehome.py Tests.test_kernel_cycles
```

You should see output like:
```
forest_height=10, rounds=16, batch_size=256
CYCLES:  147734
Speedup over baseline:  1.0
```

Can you beat Claude Opus 4.5's 1,487 cycles? If you get below that, email performance-recruiting@anthropic.com - they're hiring!

---

## Further Reading

If this post piqued your interest, here are some resources to learn more:

- [VLIW Architecture - Wikipedia](https://en.wikipedia.org/wiki/Very_long_instruction_word) - The architectural paradigm this simulator is based on
- [SIMD Programming - Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) - Real-world SIMD programming
- [Software Pipelining - Wikipedia](https://en.wikipedia.org/wiki/Software_pipelining) - Key optimization technique
- [Perfetto UI](https://ui.perfetto.dev/) - The trace viewer used for debugging
- [Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview) - Format documentation
- [Computer Architecture: A Quantitative Approach](https://www.amazon.com/Computer-Architecture-Quantitative-Approach-Kaufmann/dp/0128119055) - The classic textbook on these topics
- [Branchless Programming Techniques](https://en.algorithmica.org/hpc/pipelining/branchless/) - Avoiding branches for performance

---

*Thanks for reading! If you found this interesting, follow me on [Twitter](https://twitter.com/trirpi) or [GitHub](https://github.com/trirpi) for more deep dives into performance engineering and AI systems.*
