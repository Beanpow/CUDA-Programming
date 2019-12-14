# My testing result

## CPU
| computation     | S | D |
|:------------|:---------|:---------|
| vector addition (chapter 5) | 77 ms  |  160 ms |
| arithmetic (chapter 5) | 320 ms |  450 ms |
| reduction (chapter 8) | 96 ms |  96 ms |
| neighbor (chapter 9) | 230 ms |  230 ms |

## GPU
| computation     | V100 (S) | V100 (D) | 2080ti (S) | 2080ti (D) | P100 (S) | P100 (D) |
|:------------|:---------|:---------|:---------|:---------|
| vector addition (chapter 5) | 1.5 ms | 3.0 ms |  2.1 ms |  4.3 ms | 2.2 ms |  4.3 ms |
| add+memcpy (chapter 5) | not used | not used | 130 ms  |  250 ms | not used | not used |
| arithmetic (chapter 5) | 11 ms |  28 ms | 15 ms | 450 ms |
| matrix copy (chapter 7) | 1.1 ms |  2.0 ms | 1.6 ms | 2.9 ms |
| transpose with coalesced read (chapter 7) | 4.5 ms |  6.2 ms | 5.3 ms | 5.4 ms |
| transpose with coalesced write (chapter 7) | 1.6 ms |  2.2 ms | 2.8 ms | 3.7 ms |
| transpose with ldg read (chapter 7) | 1.6 ms |  2.2 ms | 2.8 ms | 3.7 ms |
| transpose with bank conflict (chapter 8) | 1.8 ms | 2.6  ms | 3.5 ms | 4.3 ms |
| transpose without bank conflict (chapter 8) | 1.4 ms | 2.5  ms | 2.3 ms | 4.2 ms |
| reduction with global memory only (chapter 8) | not used | not used | 3.2 ms | 3.8 ms |
| reduction with static shared memory (chapter 8) | not used | not used | 2.9 ms | 4.8 ms |
| reduction with dynamic shared memory (chapter 8) | not used | not used | 2.9 ms | 4.8 ms |
| reduction with less blocks (chapter 8) | not used | not used | 1.0 ms | 1.7 ms |
| neighbor without atomicAdd (chapter 9) | 2.0 ms | 2.7  ms | 1.9 ms | 17 ms |
| neighbor with atomicAdd (chapter 9) | 1.8 ms | 2.6  ms | 1.9 ms | 11 ms |
