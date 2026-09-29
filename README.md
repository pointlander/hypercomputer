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
go run ./cmd/hypercomputer -demo=lm -text=pg100.txt -prompt='To be, or not to be' -gen=200
go run ./cmd/hypercomputer -demo=mcts -text=pg100.txt -gen=200
```

`lm` compares two next-byte models on The Complete Works of William
Shakespeare. One mixes the variable-order phrase code with the fixed
4-byte code, weighting a long suffix by how often it was seen. The
other is the 4-byte code alone. A training window's Shannon codeword is
a program K may select when it is shorter than a listing, byte-run, or
splice. The last tenth of the book is validation for both. After the
scores, each new byte is drawn from a Witten-Bell backoff distribution:
the longest counted suffix blended with the next-shorter one, down to
the unigram. The draw is conditioned on `-prompt` and then on the bytes
just written. Counts stay frozen. `-gen=0` skips the sample.

`mcts` is a second next-byte model with a 32-byte window. The exhaustive
halt oracle only names programs up to 12 bits, which cannot print a
window that wide. Monte Carlo tree search runs U on programs past that
limit and keeps a witness for the window. The next byte blends the
32-byte counts into the shorter mixture. The last tenth of the book
is validation. The printed continuation is itself a tree search:
each playout scores an 8-byte horizon by the geometric mean of the
model probabilities, and the byte with the best playout is kept.

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
