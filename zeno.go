// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import "fmt"

// Zeno is an accelerated Turing machine: step n is allotted time
// 2^{-(n+1)}, so ω discrete steps complete in analog time 1.
type Zeno struct {
	Analog *AnalogTM
	Time   *BitFloat
	Steps  int
}

// NewZeno returns a Zeno machine for tm.
func NewZeno(prec uint, tm *TM) *Zeno {
	return &Zeno{
		Analog: NewAnalogTM(prec, tm),
		Time:   New(prec),
	}
}

// tick performs one accelerated step. dt = 2^{-(Steps+1)}.
func (z *Zeno) tick() bool {
	dt := Pow2(z.Analog.Prec, -(z.Steps + 1))
	running := z.Analog.Step()
	z.Time.Add(z.Time, dt)
	z.Steps++
	return running
}

// Run performs up to n accelerated steps. Once the TM halts the
// analog configuration freezes; time still records the partial sum
// of 1/2 + 1/4 + ... used so far.
func (z *Zeno) Run(n int) {
	for i := 0; i < n; i++ {
		if !z.tick() {
			return
		}
	}
}

// Halted reports whether the inner TM has halted.
func (z *Zeno) Halted() bool {
	return z.Analog.Halted()
}

// LimitKind is the ω-limit of a Zeno run.
type LimitKind int

const (
	// LimitHalt: finite control halted and the analog tapes are frozen.
	LimitHalt LimitKind = iota
	// LimitCauchy: not halted, but Left/Right/state agree between n and 2n.
	LimitCauchy
	// LimitDiverge: analog configuration still changes at n = 2^k.
	LimitDiverge
)

func (k LimitKind) String() string {
	switch k {
	case LimitHalt:
		return "halt"
	case LimitCauchy:
		return "cauchy"
	case LimitDiverge:
		return "diverge"
	default:
		return fmt.Sprintf("limit(%d)", k)
	}
}

// ZenoLimit is the ω-limit after a doubling schedule n = 1, 2, …, 2^k.
type ZenoLimit struct {
	Kind   LimitKind
	K      int
	N      int
	Steps  int
	Time   *BitFloat
	Halted bool
	Frozen bool
	Left   *BitFloat
	Right  *BitFloat
	Output []bool
}

func (lim *ZenoLimit) String() string {
	out := FormatBits(lim.Output)
	if out == "" {
		out = "ε"
	}
	fr := "n"
	if lim.Frozen {
		fr = "y"
	}
	return fmt.Sprintf("%-7s n=%-4d steps=%-4d frozen=%s time=%s out=%s",
		lim.Kind, lim.N, lim.Steps, fr, lim.Time.Text('g', 12), out)
}

type zenoSnap struct {
	state int
	bits  []bool
}

func (a *AnalogTM) snap(span int) zenoSnap {
	if span < 1 {
		span = 1
	}
	b := make([]bool, 2*span+1)
	for i, d := 0, -span; d <= span; d++ {
		b[i] = a.TapeBit(d) == 1
		i++
	}
	return zenoSnap{state: a.State, bits: b}
}

func (s zenoSnap) eq(o zenoSnap) bool {
	return s.state == o.state && bitsEq(s.bits, o.bits)
}

// Limit runs a doubling schedule n = 1, 2, …, 2^k and classifies the
// ω-limit of the analog configuration.
//
// Halt if finite control halted (tapes freeze after that). Cauchy if
// the machine is still running but Left, Right, and state agree
// between 2^{k-1} and 2^k. Diverge if the configuration still moves.
func (z *Zeno) Limit(k int) *ZenoLimit {
	if k < 1 {
		k = 1
	}
	span := 1 << k
	if span > 64 {
		span = 64
	}
	var prev zenoSnap
	frozen := false
	for stage := 0; stage <= k; stage++ {
		target := 1 << stage
		for z.Steps < target && !z.Halted() {
			z.tick()
		}
		cur := z.Analog.snap(span)
		if stage > 0 {
			frozen = cur.eq(prev)
		}
		prev = cur
	}
	kind := LimitDiverge
	if z.Halted() {
		kind = LimitHalt
		frozen = true
	} else if frozen {
		kind = LimitCauchy
	}
	outSpan := z.Steps
	if outSpan < 8 {
		outSpan = 8
	}
	if outSpan > 64 {
		outSpan = 64
	}
	return &ZenoLimit{
		Kind:   kind,
		K:      k,
		N:      1 << k,
		Steps:  z.Steps,
		Time:   z.Time.Copy(),
		Halted: z.Halted(),
		Frozen: frozen,
		Left:   z.Analog.Left.Copy(),
		Right:  z.Analog.Right.Copy(),
		Output: z.Analog.Output(outSpan),
	}
}
