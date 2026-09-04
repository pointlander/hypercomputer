// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import "fmt"

const analogISAQueryLimit = 64

// KResult is Kolmogorov complexity of a bit string relative to the
// n-state TM enumeration, computed with the analog halt oracle.
//
// Plain C_U(x) is the bit-length of the smallest TM index whose machine
// outputs x. Prefix-free K_U matches the Ω program format 1^n 0 + n-bit
// index. PrintBound is the trivial program |x|+1 (a tag bit plus x).
// K is min(C_U, PrintBound). With a finite step bound this is K^T ≥ K;
// as bound and oracle precision → ∞ it decreases to true C_U.
type KResult struct {
	Bits        []bool
	K           int
	PlainC      int
	PrefixK     int
	PrintBound  int
	ByPrint     bool
	TMIndex     int
	TMStates    int
	TMSteps     int
	Bound       int
	Queries     int
	AnalogSteps uint64
	AnalogOK    bool
}

func (r *KResult) String() string {
	s := FormatBits(r.Bits)
	if s == "" {
		s = "ε"
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

// KComplexity computes Kolmogorov complexity of x using the analog
// halt oracle of n-state 2-symbol TMs simulated for bound steps.
func KComplexity(x []bool, nstates, bound int, prec uint) *KResult {
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
