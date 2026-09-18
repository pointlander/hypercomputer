// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import "fmt"

// HaltedProg is a complete prefix-free U-program found while
// reconstructing the halt set from Ω.
type HaltedProg struct {
	P     []bool
	Out   []bool
	Steps int
}

// ReconstructResult is Chaitin's reconstruction of the short halt
// set from an analog prefix of Ω_U.
//
// Given the first M bits of Ω (ω ≤ Ω < ω+2^{−M}), dovetail U until
// the enumerated mass is ≥ ω. The leftover is then < 2^{−M}, so every
// still-unseen program of length ≤ M cannot halt (in the step bound).
type ReconstructResult struct {
	M           int
	MaxBits     int
	Bound       int
	OmegaBits   []bool
	OmegaHat    *BitFloat
	Sum         *BitFloat
	Caught      bool
	Stage       int
	Progs       []HaltedProg
	AnalogSteps uint64
}

// OmegaPrefix extracts the first m fractional bits of omega with the
// analog Bernoulli map (BSS QueryBit) and returns the dyadic
// ω = 0.b1…bm.
func OmegaPrefix(omega *BitFloat, m int) (bits []bool, hat *BitFloat, steps uint64) {
	if m < 0 {
		m = 0
	}
	mach := NewMachine(omega.Prec(), 2)
	_ = mach.Load(0, omega)
	bits = make([]bool, m)
	for i := 0; i < m; i++ {
		b, err := mach.QueryBit(0, i)
		if err != nil {
			break
		}
		bits[i] = b == 1
		steps += uint64(i + 1)
	}
	hat = FromBits(omega.Prec(), bits)
	return bits, hat, steps
}

// Reconstruct reads M analog bits of omega and dovetails U on all
// buffers of length 1..maxBits with a geometrically increasing step
// budget until the found mass is ≥ the Ω prefix (or bound is exhausted).
func Reconstruct(omega *BitFloat, m, maxBits, bound int) *ReconstructResult {
	if m < 1 {
		m = 1
	}
	if maxBits < 1 {
		maxBits = DefaultUMaxBits
	}
	if bound <= 0 {
		bound = DefaultUBound
	}
	bits, hat, ast := OmegaPrefix(omega, m)
	prec := omega.Prec()
	if prec == 0 {
		prec = DefaultPrec
	}
	r := &ReconstructResult{
		M:           m,
		MaxBits:     maxBits,
		Bound:       bound,
		OmegaBits:   bits,
		OmegaHat:    hat,
		Sum:         New(prec),
		AnalogSteps: ast,
	}
	nbuf := NumUBuffers(maxBits)
	seen := make([]bool, nbuf)
	for t := 1; ; {
		if t > bound {
			t = bound
		}
		r.Stage = t
		for n := 1; n <= maxBits; n++ {
			w := Pow2(prec, -n)
			for i := 0; i < 1<<n; i++ {
				idx := UBufferIndex(n, i)
				if seen[idx] {
					continue
				}
				src := IntBits(n, i)
				res := RunU(src, t)
				if res.Status != UHalt || res.Read != n {
					continue
				}
				seen[idx] = true
				r.Progs = append(r.Progs, HaltedProg{
					P:     append([]bool(nil), src...),
					Out:   append([]bool(nil), res.Out...),
					Steps: res.Steps,
				})
				r.Sum.Add(r.Sum, w)
			}
		}
		if r.Sum.Cmp(hat) >= 0 {
			r.Caught = true
			return r
		}
		if t == bound {
			return r
		}
		if t > bound/2 {
			t = bound
		} else {
			t *= 2
		}
	}
}

// KFromOmega reconstructs K_U(x) from M analog bits of Ω_U.
//
// If reconstruction catches, every program of length ≤ M that halts
// in bound steps is known. Then either K_U(x) is the shortest such
// program that outputs x, or K_U(x) > M (refused).
func KFromOmega(x []bool, m, maxBits, bound int, prec uint) *KResult {
	if prec == 0 {
		prec = DefaultPrec
	}
	omega := Omega(maxBits, bound, prec)
	return KFromOmegaValue(x, omega, m, maxBits, bound)
}

// KFromOmegaValue is KFromOmega with a supplied analog Ω register.
func KFromOmegaValue(x []bool, omega *BitFloat, m, maxBits, bound int) *KResult {
	if m < 1 {
		m = 1
	}
	if maxBits < 1 {
		maxBits = DefaultUMaxBits
	}
	if bound <= 0 {
		bound = DefaultUBound
	}
	rec := Reconstruct(omega, m, maxBits, bound)
	return rec.KOf(x)
}

// KOf classifies x from an already-reconstructed short halt set.
func (rec *ReconstructResult) KOf(x []bool) *KResult {
	r := &KResult{
		Bits:        append([]bool(nil), x...),
		K:           -1,
		How:         "chaitin-incomplete",
		Bound:       rec.Bound,
		MaxBits:     rec.MaxBits,
		Queries:     rec.Stage,
		AnalogSteps: rec.AnalogSteps,
		AnalogOK:    rec.Caught,
		OmegaBits:   rec.OmegaBits,
		M:           rec.M,
		Caught:      rec.Caught,
	}
	if !rec.Caught {
		return r
	}
	best := -1
	var prog []bool
	var steps int
	for _, p := range rec.Progs {
		if len(p.P) > rec.M {
			continue
		}
		if !bitsEq(p.Out, x) {
			continue
		}
		if best < 0 || len(p.P) < best {
			best = len(p.P)
			prog = p.P
			steps = p.Steps
		}
	}
	if best < 0 {
		r.How = "chaitin-beyond"
		return r
	}
	r.K = best
	r.Program = prog
	r.How = "chaitin"
	r.TMSteps = steps
	return r
}

func (r *KResult) chaitinString(s string) string {
	switch r.How {
	case "chaitin-beyond":
		return fmt.Sprintf("K_U(%s)>%d  via chaitin  Ω[0:%d]=%s",
			s, r.M, r.M, FormatBits(r.OmegaBits))
	case "chaitin-incomplete":
		return fmt.Sprintf("K_U(%s) uncertified  via chaitin  Ω[0:%d]=%s",
			s, r.M, FormatBits(r.OmegaBits))
	case "chaitin":
		return fmt.Sprintf("K_U(%s)=%d  via chaitin  |p|=%d  p=%s  Ω[0:%d]=%s",
			s, r.K, len(r.Program), FormatBits(r.Program), r.M, FormatBits(r.OmegaBits))
	}
	return ""
}
