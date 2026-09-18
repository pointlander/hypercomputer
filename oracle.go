// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

// HaltOracle encodes the bounded halting set of n-state 2-symbol TMs
// as a binary real: bit i is 1 iff TM i halts on a blank tape within
// bound steps. As bound → ∞ and precision → ∞ this approaches a
// genuine halt oracle (which is not Turing-computable). Finite bound
// and p-bit precision yield the first p bits of that approximation.
func HaltOracle(nstates, bound int, prec uint) (oracle *BitFloat, halted []bool) {
	n := NumTMs(nstates)
	halted = make([]bool, n)
	for i := 0; i < n; i++ {
		h, _, _ := TMFromIndex(i, nstates).Run(bound)
		halted[i] = h
	}
	if prec == 0 {
		prec = PrecBits(n)
	}
	return FromBits(prec, halted), halted
}

// CantorHaltOracle is HaltOracle using Cantor encoding, so analog
// satlin neurons can read it without dyadic-endpoint collisions.
func CantorHaltOracle(nstates, bound int, prec uint) (oracle *BitFloat, halted []bool) {
	n := NumTMs(nstates)
	halted = make([]bool, n)
	for i := 0; i < n; i++ {
		h, _, _ := TMFromIndex(i, nstates).Run(bound)
		halted[i] = h
	}
	if prec == 0 {
		prec = PrecBits(2 * n)
	}
	return FromCantor(prec, halted), halted
}

// OmegaTM is the older Ω over n-state TM indices (1^n 0 + n-bit index).
func OmegaTM(nstates, maxN, bound int, prec uint) *BitFloat {
	if prec == 0 {
		prec = PrecBits(2*maxN + 8)
	}
	omega := New(prec)
	ntm := NumTMs(nstates)
	for n := 0; n <= maxN; n++ {
		w := Pow2(prec, -(2*n + 1))
		limit := 1 << n
		if limit > ntm {
			limit = ntm
		}
		for i := 0; i < limit; i++ {
			h, _, _ := TMFromIndex(i, nstates).Run(bound)
			if h {
				omega.Add(omega, w)
			}
		}
	}
	return omega
}

// UHaltOracle encodes, as a binary real, whether each bit buffer of
// length 1..maxBits is a complete prefix-free U-program that halts
// in bound steps. Index is UBufferIndex(n, i).
func UHaltOracle(maxBits, bound int, prec uint) (oracle *BitFloat, halted []bool) {
	if maxBits < 1 {
		maxBits = DefaultUMaxBits
	}
	if bound <= 0 {
		bound = DefaultUBound
	}
	n := NumUBuffers(maxBits)
	halted = make([]bool, n)
	for nb := 1; nb <= maxBits; nb++ {
		limit := 1 << nb
		for i := 0; i < limit; i++ {
			src := IntBits(nb, i)
			res := RunU(src, bound)
			halted[UBufferIndex(nb, i)] = res.Status == UHalt && res.Read == nb
		}
	}
	if prec == 0 {
		prec = PrecBits(n)
	}
	return FromBits(prec, halted), halted
}

// Omega approximates Chaitin's constant for the prefix-free machine U:
//
//	Ω_U = Σ { 2^{-|p|} : U(p)↓ }
//
// summed over complete programs of length 1..maxBits that halt within
// bound steps. As maxBits, bound → ∞ this increases to Ω_U.
func Omega(maxBits, bound int, prec uint) *BitFloat {
	if maxBits < 1 {
		maxBits = DefaultUMaxBits
	}
	if bound <= 0 {
		bound = DefaultUBound
	}
	if prec == 0 {
		prec = PrecBits(maxBits + 8)
	}
	omega := New(prec)
	for n := 1; n <= maxBits; n++ {
		w := Pow2(prec, -n)
		limit := 1 << n
		for i := 0; i < limit; i++ {
			src := IntBits(n, i)
			res := RunU(src, bound)
			if res.Status == UHalt && res.Read == n {
				omega.Add(omega, w)
			}
		}
	}
	return omega
}
