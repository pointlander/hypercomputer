# hypercomputer

Simulation of analog hypercomputation using exact rationals (`math/big.Rat`).

A real in [0, 1] has infinitely many binary digits. Analog operations
extract those digits exactly in ℚ: the Bernoulli map `2x mod 1` and
the Cantor (base-4) stack with saturated-linear neurons (Siegelmann–Sontag).
√, π, sin, and cos are truncated rational series at a working bit
precision `p`. `Truncate` rounds a value to a dyadic of width `p`,
which is the analog of a finite tape.

The numeric type is a `big.Rat` with an analog bit API. On top of it:

- a BSS-style real register machine
- 2-symbol Turing machines with analog Cantor-stack tapes
- a Zeno machine (step `n` takes time `2^{-(n+1)}`)
- bounded halt-set and Chaitin Ω oracles encoded as reals
- an analog recurrent net that pops a Cantor-encoded oracle
- a state-vector quantum circuit simulator (Clifford+T, rotations, QFT)
- Kolmogorov complexity of a bit string via the analog halt oracle

Finite machines cannot decide the true halting set. The oracle here is
the *bounded* halt set of small TMs, packed into a rational; as the
step bound and the readable bit depth grow, more of the genuine oracle
is visible.

## Usage

```bash
go test ./...
go run ./cmd/hypercomputer -demo=all -prec=256
go run ./cmd/hypercomputer -demo=quantum -prec=256
go run ./cmd/hypercomputer -demo=kcomplexity -kstring=1111 -kstates=2
```
