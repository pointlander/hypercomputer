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

// Omega approximates Chaitin's constant for a toy prefix-free
// universal machine U:
//
//	programs are 1^n 0 followed by an n-bit TM index
//	U runs that n-state-enumeration TM on a blank tape
//	Ω_U = Σ { 2^{-|p|} : U(p)↓ }
//
// The inner TM is simulated for bound steps, so this is a left-
// computable approximation. More bits of Ω decide more of the
// halting problem (Levin–Chaitin).
func Omega(nstates, maxN, bound int, prec uint) *BitFloat {
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
