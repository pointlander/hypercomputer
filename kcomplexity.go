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
	OmegaBits   []bool
	M           int
	Caught      bool
}

func (r *KResult) String() string {
	s := FormatBits(r.Bits)
	if s == "" {
		s = "ε"
	}
	if r.How == "chaitin" || r.How == "chaitin-beyond" || r.How == "chaitin-incomplete" {
		return r.chaitinString(s)
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

// splicePrograms drops a trailing HALT on p and continues with q.
// The result is accepted only when RunU prints the concatenation, so a
// program that does not end by executing HALT, or that leaves the
// register in a state q does not expect, is rejected.
func splicePrograms(p, q []bool) []bool {
	if len(p) < 3 || len(q) == 0 || p[len(p)-3] || p[len(p)-2] || p[len(p)-1] {
		return nil
	}
	out := make([]bool, 0, len(p)-3+len(q))
	out = append(out, p[:len(p)-3]...)
	out = append(out, q...)
	return out
}

// uProg is the shortest program of length ≤ maxBits that prints Out.
type uProg struct {
	bits  []bool
	n, i  int
	steps int
}

// uSearcher is one shared halt oracle and program catalog for every block
// of a divide-and-conquer search. Both are exponential in maxBits and are
// built once; each block is then a map lookup.
type uSearcher struct {
	maxBits     int
	bound       int
	oracle      *BitFloat
	halted      []bool
	m           *Machine
	isa         bool
	queries     int
	analogSteps uint64
	cat         map[string]uProg
}

func (s *uSearcher) init(prec uint) {
	oprec := PrecBits(NumUBuffers(s.maxBits))
	if prec > oprec {
		oprec = prec
	}
	var err error
	s.oracle, s.halted = UHaltOracle(s.maxBits, s.bound, oprec)
	s.m = NewMachine(s.oracle.Prec(), 8)
	if err = s.m.Load(0, s.oracle); err != nil {
		panic(err)
	}
	s.isa = NumUBuffers(s.maxBits) <= analogISAQueryLimit
}

// catalog records the shortest halting program for each output.
// Lengths increase, and i increases inside a length, so the first hit
// is the one the linear search would have returned.
func (s *uSearcher) catalog() map[string]uProg {
	if s.cat != nil {
		return s.cat
	}
	s.cat = make(map[string]uProg)
	for n := 1; n <= s.maxBits; n++ {
		for i := 0; i < 1<<n; i++ {
			s.queries++
			idx := UBufferIndex(n, i)
			var halt bool
			if s.isa {
				h, steps, err := oracleHalt(s.m, s.oracle, idx, true)
				if err != nil {
					panic(err)
				}
				s.analogSteps += steps
				halt = h
				if halt != s.halted[idx] {
					panic(fmt.Sprintf("U oracle bit %d disagrees with table", idx))
				}
			} else {
				halt = s.halted[idx]
			}
			if !halt {
				continue
			}
			src := IntBits(n, i)
			res := RunU(src, s.bound)
			if res.Status != UHalt || res.Read != n {
				continue
			}
			key := FormatBits(res.Out)
			if _, ok := s.cat[key]; ok {
				continue
			}
			s.cat[key] = uProg{bits: src, n: n, i: i, steps: res.Steps}
		}
	}
	return s.cat
}

// improve replaces r with the shortest program of length ≤ maxBits that
// prints x within s.bound, if that program is shorter than r.K.
func (s *uSearcher) improve(x []bool, r *KResult) {
	w, ok := s.catalog()[FormatBits(x)]
	if !ok || (r.K >= 0 && w.n >= r.K) {
		return
	}
	r.K = w.n
	r.Program = append([]bool(nil), w.bits...)
	r.How = "search"
	r.TMSteps = w.steps
	before := r.AnalogSteps
	r.AnalogOK = analogWitness(s.m, s.halted, w.n, w.i, s.isa, r)
	s.analogSteps += r.AnalogSteps - before
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
	r := &KResult{
		Bits:    append([]bool(nil), x...),
		K:       -1,
		Bound:   bound,
		MaxBits: maxBits,
	}
	considerU(x, ListingProgram(x), bound, "listing", r)
	if bit, k, ok := unaryRun(x); ok {
		considerU(x, RepeatProgram(bit, k), bound, "repeat", r)
	}
	s := &uSearcher{maxBits: maxBits, bound: bound}
	s.init(prec)
	s.improve(x, r)
	if r.How != "search" && r.Program != nil && len(r.Program) <= maxBits {
		before := r.AnalogSteps
		r.AnalogOK = analogWitness(s.m, s.halted, len(r.Program), bitsInt(r.Program), false, r)
		s.analogSteps += r.AnalogSteps - before
	}
	r.Queries = s.queries
	r.AnalogSteps = s.analogSteps
	return r
}

// DefaultKLeaf is the block size at which divide-and-conquer stops and
// runs the bounded program search.
const DefaultKLeaf = 16

// KDivideConquer is an upper bound on K_U(x) for long strings.
//
// The string is split in half until each block has at most leaf bits.
// Each distinct block is solved once — listing, unary repeat, then the
// analog halt-oracle search shared across blocks — and the resulting
// prefix-free programs are spliced. Identical blocks hit a cache, so a
// repeated file does not repeat the search. When len(x) ≤ leaf the
// result is KComplexity.
//
// Printing x takes Ω(|x|) steps of U. If bound is too small to run the
// witness, it is raised to 6|x|+64 and that budget is reported in
// Bound. The oracle search itself still uses the supplied bound.
// leaf ≤ 0 selects DefaultKLeaf.
func KDivideConquer(x []bool, maxBits, bound, leaf int, prec uint) *KResult {
	if maxBits < 1 {
		maxBits = DefaultUMaxBits
	}
	if bound <= 0 {
		bound = DefaultUBound
	}
	if leaf < 1 {
		leaf = DefaultKLeaf
	}
	if len(x) <= leaf {
		return KComplexity(x, maxBits, bound, prec)
	}
	prove := bound
	if need := len(x)*6 + 64; prove < need {
		prove = need
	}
	s := &uSearcher{maxBits: maxBits, bound: bound}
	s.init(prec)
	r := s.divide(x, leaf, prove, make(map[string]*KResult))
	s.improve(x, r)
	if r.How == "search" {
		r.Bound = bound
	} else {
		r.Bound = prove
		if r.Program != nil && len(r.Program) <= maxBits {
			before := r.AnalogSteps
			r.AnalogOK = analogWitness(s.m, s.halted, len(r.Program), bitsInt(r.Program), false, r)
			s.analogSteps += r.AnalogSteps - before
		}
	}
	r.Queries = s.queries
	r.AnalogSteps = s.analogSteps
	r.Bits = append([]bool(nil), x...)
	r.MaxBits = maxBits
	return r
}

func (s *uSearcher) divide(x []bool, leaf, prove int, cache map[string]*KResult) *KResult {
	key := FormatBits(x)
	if hit, ok := cache[key]; ok {
		return hit
	}
	r := &KResult{
		Bits:    append([]bool(nil), x...),
		K:       -1,
		Bound:   prove,
		MaxBits: s.maxBits,
	}
	considerU(x, ListingProgram(x), prove, "listing", r)
	if bit, k, ok := unaryRun(x); ok {
		considerU(x, RepeatProgram(bit, k), prove, "repeat", r)
		if r.How == "repeat" {
			// A splice of smaller repeats is longer. Still look for a
			// program shorter than this repeat under the bit cap.
			s.improve(x, r)
			cache[key] = r
			return r
		}
	}
	if len(x) <= leaf {
		s.improve(x, r)
		cache[key] = r
		return r
	}
	mid := len(x) / 2
	left := s.divide(x[:mid], leaf, prove, cache)
	right := s.divide(x[mid:], leaf, prove, cache)
	if sp := splicePrograms(left.Program, right.Program); sp != nil {
		considerU(x, sp, prove, "divide", r)
	}
	cache[key] = r
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
