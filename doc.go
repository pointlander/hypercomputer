// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package hypercomputer simulates analog hypercomputation using
// high-precision binary floating-point numbers.
//
// A real in [0, 1] carries infinitely many bits. Analog operations
// (the Bernoulli doubling map, Cantor-stack pop) extract those bits.
// With infinite precision this is a genuine hypercomputer in the
// sense of Siegelmann–Sontag and Moore; with a p-bit float it is a
// p-bit approximation: the first p bits of an encoded oracle are
// readable, and p is a first-class resource.
//
// The numeric type is BitFloat, a binary float whose precision is
// measured in bits (math/big.Float with an analog bit API). On top of
// it the package provides:
//
//   - a BSS-style real register machine
//   - Turing machines with analog (Cantor-stack) tapes
//   - a Zeno / accelerated machine that takes step n in time 2^{-n}
//   - halt-set and Chaitin Ω oracles encoded as reals
//   - an analog recurrent net that reads a Cantor-encoded oracle
package hypercomputer
