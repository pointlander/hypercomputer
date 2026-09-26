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
- a Zeno machine (step `n` takes time `2^{-(n+1)}`) with an ω-limit
  (halt-and-frozen, Cauchy tape, or diverge)
- bounded halt-set and Chaitin Ω oracles encoded as reals
- a prefix-free reference machine U shared by Ω_U and K_U
- an analog recurrent net that pops a Cantor-encoded oracle
- a state-vector quantum circuit simulator (Clifford+T, rotations, QFT)
- Kolmogorov complexity of a bit string via the analog halt oracle
- Chaitin reconstruction of K_U from analog bits of Ω_U
- a p×T resource sweep (analog Ω bits × dovetail bound)

Finite machines cannot decide the true halting set. The oracle here is
the *bounded* halt set of small TMs, packed into a rational; as the
step bound and the readable bit depth grow, more of the genuine oracle
is visible.

## Usage

```bash
go test ./...
go run ./cmd/hypercomputer -demo=all -prec=256
go run ./cmd/hypercomputer -demo=quantum -prec=256
go run ./cmd/hypercomputer -demo=kcomplexity -kstring=1111 -kbits=12
go run ./cmd/hypercomputer -demo=chaitin -kbits=8
go run ./cmd/hypercomputer -demo=sweep
go run ./cmd/hypercomputer -demo=lm -text=pg100.txt
```

`lm` compares two next-byte models on The Complete Works of William
Shakespeare. One gives each training window a short prefix-free program
and scores the next byte by that code length; an unseen continuation
keeps the listing, byte-run, or splice. The other is a one-hot window
of the same width. The last tenth of the book is validation for both.

Long bit strings go through `KDivideConquer`: each block is solved once
(listing, repeat, or the bounded search) and the prefix-free programs are
spliced. The reported length is an upper bound on `K_U`, checked by running `U`.

## Lean theory

`lean/Hyperuniverse.lean` models a universe whose machines are finite
enumerated Turing tables, with a code for every state count. A particle
stores infinitely many bits, and decay reads the halt bit of its query
(the Zeno ω-limit of that machine).
`Computable` means some finite table writes the bit function: parity is a
theorem, and `no_TM_decides_Halt` places the enumerated halt bit outside
`Computable`. A truncated `N`-bit lab particle decides `Halts` on coded
queries `< N`.

```bash
cd lean && lake build
```
