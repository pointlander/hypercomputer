// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package hypercomputer simulates analog hypercomputation using
// exact rationals (math/big.Rat).
//
// A real in [0, 1] carries infinitely many bits. Analog operations
// (the Bernoulli doubling map, Cantor-stack pop) extract those bits
// exactly in ℚ. Transcendentals used by the quantum simulator
// (√, π, sin, cos) are truncated rational series at a working bit
// precision p; Truncate rounds a value to a dyadic of width p.
//
// The numeric type is BitFloat, a big.Rat with an analog bit API.
// On top of it the package provides:
//
//   - a BSS-style real register machine
//   - Turing machines with analog (Cantor-stack) tapes
//   - a Zeno / accelerated machine that takes step n in time 2^{-n}
//   - halt-set and Chaitin Ω oracles encoded as reals
//   - a prefix-free reference machine U shared by Ω_U and K_U
//   - an analog recurrent net that reads a Cantor-encoded oracle
//   - a state-vector quantum circuit simulator over BitFloat complexes
//   - Kolmogorov complexity of bit strings via the analog halt oracle
//   - Chaitin reconstruction of K_U from analog bits of Ω_U
//   - a p×T resource sweep: analog Ω bits and dovetail bound vs certified K_U
package hypercomputer
