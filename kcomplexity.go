// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import "fmt"

const analogISAQueryLimit = 64

// KResult is Kolmogorov complexity of a bit string.
//
// For KComplexity this is K_U relative to the prefix-free machine U
// (same U as Ω). For KComplexityTM it is complexity in the n-state
// TM enumeration (legacy).
type KResult struct {
	Bits        []bool
	K           int
	Program     []bool
	How         string
	PlainC      int
	PrefixK     int
	PrintBound  int
	ByPrint     bool
	TMIndex     int
	TMStates    int
	TMSteps     int
	Bound       int
	MaxBits     int
	Queries     int
	AnalogSteps uint64
	AnalogOK    bool
}

func (r *KResult) String() string {
	s := FormatBits(r.Bits)
	if s == "" {
		s = "ε"
	}
	if r.How != "" {
		return fmt.Sprintf("K_U(%s)=%d  via %s  |p|=%d  p=%s",
			s, r.K, r.How, len(r.Program), FormatBits(r.Program))
	}
	how := "print"
	if !r.ByPrint {
		how = fmt.Sprintf("TM #%d (%d-state, %d steps)", r.TMIndex, r.TMStates, r.TMSteps)
	}
	pk := "∞"
	c := "∞"
	if r.PrefixK >= 0 {
		pk = fmt.Sprintf("%d", r.PrefixK)
		c = fmt.Sprintf("%d", r.PlainC)
	}
	return fmt.Sprintf("K(%s)=%d  via %s  C=%s  prefix-K=%s  print≤%d",
		s, r.K, how, c, pk, r.PrintBound)
}

// FormatBits renders a bit slice as a string of 0s and 1s.
func FormatBits(bits []bool) string {
	b := make([]byte, len(bits))
	for i, v := range bits {
		if v {
			b[i] = '1'
		} else {
			b[i] = '0'
		}
	}
	return string(b)
}

// ParseBitString reads a string of 0s and 1s (whitespace ignored).
func ParseBitString(s string) []bool {
	b := make([]bool, 0, len(s))
	for _, r := range s {
		switch r {
		case '0':
			b = append(b, false)
		case '1':
			b = append(b, true)
		case ' ', '\t', '\n', '\r':
		default:
			panic("ParseBitString: non-bit character")
		}
	}
	return b
}

// BitLen is the number of bits needed to write n ≥ 0 in binary.
// BitLen(0) = 1.
func BitLen(n int) int {
	if n <= 0 {
		return 1
	}
	k := 0
	for n > 0 {
		k++
		n >>= 1
	}
	return k
}

// PrefixProgramLen is the length of the Ω-style program for TM index i:
// 1^n 0 followed by an n-bit index, n = min { n : i < 2^n }.
func PrefixProgramLen(i int) int {
	n := 0
	for 1<<n <= i {
		n++
	}
	return 2*n + 1
}

func bitsEq(a, b []bool) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func oracleHalt(m *Machine, oracle *BitFloat, index int, isa bool) (bool, uint64, error) {
	if isa {
		if err := m.Load(0, oracle); err != nil {
			return false, 0, err
		}
		m.R[1].SetInt64(int64(index))
		if err := m.Load(2, FromInt(m.Prec, 1)); err != nil {
			return false, 0, err
		}
		m.PC = 0
		m.Steps = 0
		if err := m.Run(HaltQuery(0, 1, 3, 2)); err != nil {
			return false, m.Steps, err
		}
		return m.R[3].Sign() != 0, m.Steps, nil
	}
	b, err := m.QueryBit(0, index)
	return b == 1, uint64(index + 1), err
}

func considerU(x, p []bool, bound int, how string, best *KResult) {
	if len(p) == 0 || (best.K >= 0 && len(p) >= best.K) {
		return
	}
	res := RunU(p, bound)
	if res.Status != UHalt || res.Read != len(p) || !bitsEq(res.Out, x) {
		return
	}
	best.K = len(p)
	best.Program = append([]bool(nil), p...)
	best.How = how
	best.TMSteps = res.Steps
}

// KComplexity computes K_U(x) for the prefix-free machine U (the same
// U as Omega). Listing and unary-repeat templates are always tried;
// programs of length 1..maxBits are then searched via the analog halt
// oracle. Finite bound and maxBits give K^T ≥ K_U.
func KComplexity(x []bool, maxBits, bound int, prec uint) *KResult {
	if maxBits < 1 {
		maxBits = DefaultUMaxBits
	}
	if bound <= 0 {
		bound = DefaultUBound
	}
	listing := ListingProgram(x)
	r := &KResult{
		Bits:    append([]bool(nil), x...),
		K:       -1,
		Bound:   bound,
		MaxBits: maxBits,
	}
	considerU(x, listing, bound, "listing", r)
	if bit, k, ok := unaryRun(x); ok {
		considerU(x, RepeatProgram(bit, k), bound, "repeat", r)
	}
	searchTo := maxBits
	if r.K >= 0 && r.K-1 < searchTo {
		searchTo = r.K - 1
	}
	oprec := PrecBits(NumUBuffers(maxBits))
	if prec > oprec {
		oprec = prec
	}
	oracle, halted := UHaltOracle(maxBits, bound, oprec)
	m := NewMachine(oracle.Prec(), 8)
	if err := m.Load(0, oracle); err != nil {
		panic(err)
	}
	isa := NumUBuffers(maxBits) <= analogISAQueryLimit
	for n := 1; n <= searchTo; n++ {
		if r.K >= 0 && n >= r.K {
			break
		}
		for i := 0; i < 1<<n; i++ {
			r.Queries++
			idx := UBufferIndex(n, i)
			var halt bool
			if isa {
				h, steps, err := oracleHalt(m, oracle, idx, true)
				if err != nil {
					panic(err)
				}
				r.AnalogSteps += steps
				halt = h
				if halt != halted[idx] {
					panic(fmt.Sprintf("U oracle bit %d disagrees with table", idx))
				}
			} else {
				halt = halted[idx]
			}
			if !halt {
				continue
			}
			src := IntBits(n, i)
			res := RunU(src, bound)
			if res.Status != UHalt || res.Read != n || !bitsEq(res.Out, x) {
				continue
			}
			r.K = n
			r.Program = src
			r.How = "search"
			r.TMSteps = res.Steps
			r.AnalogOK = analogWitness(m, halted, n, i, isa, r)
			return r
		}
	}
	if r.Program != nil && len(r.Program) <= maxBits {
		n := len(r.Program)
		r.AnalogOK = analogWitness(m, halted, n, bitsInt(r.Program), false, r)
	}
	return r
}

func analogWitness(m *Machine, halted []bool, n, i int, counted bool, r *KResult) bool {
	idx := UBufferIndex(n, i)
	if idx < 0 || idx >= len(halted) || !halted[idx] {
		return false
	}
	bit, err := m.QueryBit(0, idx)
	if !counted {
		r.AnalogSteps += uint64(idx + 1)
	}
	return err == nil && bit == 1
}

func unaryRun(x []bool) (bit bool, k int, ok bool) {
	if len(x) == 0 {
		return false, 0, false
	}
	b := x[0]
	for _, v := range x {
		if v != b {
			return false, 0, false
		}
	}
	return b, len(x), true
}

func bitsInt(b []bool) int {
	n := 0
	for _, v := range b {
		n <<= 1
		if v {
			n |= 1
		}
	}
	return n
}

// KComplexityTM computes Kolmogorov complexity of x using the analog
// halt oracle of n-state 2-symbol TMs simulated for bound steps.
func KComplexityTM(x []bool, nstates, bound int, prec uint) *KResult {
	if nstates < 1 {
		nstates = 1
	}
	if bound < 1 {
		bound = 32
	}
	printBound := len(x) + 1
	ntm := NumTMs(nstates)
	oprec := PrecBits(ntm)
	if prec > oprec {
		oprec = prec
	}
	oracle, halted := HaltOracle(nstates, bound, oprec)
	m := NewMachine(oracle.Prec(), 8)
	if err := m.Load(0, oracle); err != nil {
		panic(err)
	}
	isa := ntm <= analogISAQueryLimit
	r := &KResult{
		Bits:       append([]bool(nil), x...),
		K:          printBound,
		PlainC:     printBound,
		PrefixK:    -1,
		PrintBound: printBound,
		ByPrint:    true,
		TMIndex:    -1,
		TMStates:   nstates,
		Bound:      bound,
	}
	for i := 0; i < ntm; i++ {
		r.Queries++
		var halt bool
		if isa {
			h, steps, err := oracleHalt(m, oracle, i, true)
			if err != nil {
				panic(err)
			}
			r.AnalogSteps += steps
			halt = h
			if halt != halted[i] {
				panic(fmt.Sprintf("oracle bit %d disagrees with table", i))
			}
		} else {
			halt = halted[i]
		}
		if !halt {
			continue
		}
		h, tmSteps, cfg := TMFromIndex(i, nstates).Run(bound)
		if !h || !bitsEq(cfg.Output(), x) {
			continue
		}
		c := BitLen(i)
		pk := PrefixProgramLen(i)
		r.PlainC = c
		r.PrefixK = pk
		r.TMIndex = i
		r.TMSteps = tmSteps
		r.ByPrint = false
		if c <= printBound {
			r.K = c
		} else {
			r.K = printBound
			r.ByPrint = true
		}
		if !isa {
			bit, err := m.QueryBit(0, i)
			if err != nil {
				panic(err)
			}
			if bit != 1 {
				panic(fmt.Sprintf("analog oracle missed witness %d", i))
			}
			r.AnalogSteps += uint64(i + 1)
		}
		a := NewAnalogTM(oprec, TMFromIndex(i, nstates))
		ah, _ := a.Run(bound)
		r.AnalogOK = ah && bitsEq(a.Output(bound), x)
		return r
	}
	return r
}
