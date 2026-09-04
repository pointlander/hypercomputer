# hypercomputer

Simulation of analog hypercomputation using high-precision bit floats.

A real in [0, 1] has infinitely many binary digits. Analog operations
extract those digits: the Bernoulli map `2x mod 1` reads a binary
expansion, and a Cantor (base-4) stack with saturated-linear neurons
reads a noise-stable encoding (Siegelmann–Sontag). With infinite
precision this is a hypercomputer; with a `p`-bit float it is a
`p`-bit approximation — precision is the resource that stands in for
the missing infinite tape.

The numeric type is a binary float whose mantissa width is measured in
bits (`math/big.Float` plus analog bit operations). On top of it:

- a BSS-style real register machine
- 2-symbol Turing machines with analog Cantor-stack tapes
- a Zeno machine (step `n` takes time `2^{-(n+1)}`)
- bounded halt-set and Chaitin Ω oracles encoded as reals
- an analog recurrent net that pops a Cantor-encoded oracle
- a state-vector quantum circuit simulator (Clifford+T, rotations, QFT)
- Kolmogorov complexity of a bit string via the analog halt oracle

Finite machines cannot decide the true halting set. The oracle here is
the *bounded* halt set of small TMs, packed into a real; as the step
bound and the mantissa width grow, more of the genuine oracle is
visible.

## Usage

```bash
go test ./...
go run ./cmd/hypercomputer -demo=all -prec=256
go run ./cmd/hypercomputer -demo=quantum -prec=256
go run ./cmd/hypercomputer -demo=kcomplexity -kstring=1111 -kstates=2
```
