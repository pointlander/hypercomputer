// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

// Halt is the TM halt state.
const Halt = -1

// Transition is one TM transition: write a symbol, move, next state.
type Transition struct {
	Write int // 0 or 1
	Move  int // -1 or +1
	Next  int // 0..States-1 or Halt
}

// TM is a 2-symbol Turing machine with a distinguished halt state.
type TM struct {
	States int
	Start  int
	Delta  [][2]Transition
}

// Config is a discrete TM configuration.
type Config struct {
	State int
	Head  int
	Tape  map[int]int
}

func (c *Config) read() int {
	if c.Tape == nil {
		return 0
	}
	return c.Tape[c.Head]
}

func (c *Config) clone() *Config {
	tape := make(map[int]int, len(c.Tape))
	for k, v := range c.Tape {
		tape[k] = v
	}
	return &Config{State: c.State, Head: c.Head, Tape: tape}
}

func transRadix(nstates int) int {
	return 2 * 2 * (nstates + 1)
}

// NumTMs is the number of n-state 2-symbol TMs (halt is extra).
func NumTMs(nstates int) int {
	if nstates < 1 {
		return 0
	}
	r := transRadix(nstates)
	n := 1
	for i := 0; i < 2*nstates; i++ {
		n *= r
	}
	return n
}

func decodeTrans(code, nstates int) Transition {
	write := code % 2
	code /= 2
	move := code % 2
	code /= 2
	next := code % (nstates + 1)
	mv := -1
	if move == 1 {
		mv = 1
	}
	nst := Halt
	if next > 0 {
		nst = next - 1
	}
	return Transition{Write: write, Move: mv, Next: nst}
}

func encodeTrans(t Transition, nstates int) int {
	write := t.Write & 1
	move := 0
	if t.Move > 0 {
		move = 1
	}
	next := 0
	if t.Next >= 0 {
		next = t.Next + 1
		if next > nstates {
			next = nstates
		}
	}
	return write + 2*move + 4*next
}

// Index is the inverse of TMFromIndex.
func (tm *TM) Index() int {
	r := transRadix(tm.States)
	idx, mul := 0, 1
	for s := 0; s < tm.States; s++ {
		for read := 0; read < 2; read++ {
			idx += encodeTrans(tm.Delta[s][read], tm.States) * mul
			mul *= r
		}
	}
	return idx
}

// TMFromIndex returns the n-state TM with the given enumeration index.
func TMFromIndex(index, nstates int) *TM {
	r := transRadix(nstates)
	tm := &TM{
		States: nstates,
		Start:  0,
		Delta:  make([][2]Transition, nstates),
	}
	code := index
	for s := 0; s < nstates; s++ {
		for read := 0; read < 2; read++ {
			t := code % r
			code /= r
			tm.Delta[s][read] = decodeTrans(t, nstates)
		}
	}
	return tm
}

// BusyBeaver2 is the 2-state 2-symbol champion (4 ones, 6 steps).
func BusyBeaver2() *TM {
	return &TM{
		States: 2,
		Start:  0,
		Delta: [][2]Transition{
			{{Write: 1, Move: 1, Next: 1}, {Write: 1, Move: -1, Next: 1}},
			{{Write: 1, Move: -1, Next: 0}, {Write: 1, Move: 1, Next: Halt}},
		},
	}
}

// Step applies one transition. It returns false if the machine is halted.
func (tm *TM) Step(c *Config) bool {
	if c.State < 0 {
		return false
	}
	if c.Tape == nil {
		c.Tape = make(map[int]int)
	}
	tr := tm.Delta[c.State][c.read()]
	c.Tape[c.Head] = tr.Write
	c.Head += tr.Move
	c.State = tr.Next
	return c.State >= 0
}

// Run simulates up to bound steps from a blank tape.
func (tm *TM) Run(bound int) (halted bool, steps int, cfg *Config) {
	cfg = &Config{State: tm.Start, Tape: make(map[int]int)}
	for steps = 0; steps < bound; {
		running := tm.Step(cfg)
		steps++
		if !running {
			return true, steps, cfg
		}
	}
	return false, steps, cfg
}

// Ones counts 1-symbols on the tape.
func (c *Config) Ones() int {
	n := 0
	for _, v := range c.Tape {
		if v == 1 {
			n++
		}
	}
	return n
}

// Output is the 1-span of the tape: bits from the leftmost 1 to the
// rightmost 1, zeros in between kept. A tape with no 1s is empty.
func (c *Config) Output() []bool {
	if c.Tape == nil {
		return nil
	}
	min, max := 0, 0
	found := false
	for i, v := range c.Tape {
		if v != 1 {
			continue
		}
		if !found || i < min {
			min = i
		}
		if !found || i > max {
			max = i
		}
		found = true
	}
	if !found {
		return nil
	}
	out := make([]bool, max-min+1)
	for i := min; i <= max; i++ {
		out[i-min] = c.Tape[i] == 1
	}
	return out
}

// AnalogTM is a TM whose tape is two Cantor stacks (left of the head,
// and the head plus the right). Finite control stays discrete; the
// tape is an analog real — Moore's generalized-shift machine.
type AnalogTM struct {
	Prec  uint
	TM    *TM
	State int
	Left  *BitFloat
	Right *BitFloat
}

// NewAnalogTM returns an analog TM on a blank (all-zero) tape.
func NewAnalogTM(prec uint, tm *TM) *AnalogTM {
	if prec == 0 {
		prec = DefaultPrec
	}
	return &AnalogTM{
		Prec:  prec,
		TM:    tm,
		State: tm.Start,
		Left:  CantorZeros(prec),
		Right: CantorZeros(prec),
	}
}

// Halted reports whether finite control is in the halt state.
func (a *AnalogTM) Halted() bool {
	return a.State < 0
}

// Read returns the analog head symbol.
func (a *AnalogTM) Read() int {
	return a.Right.CantorBit()
}

// Step performs one analog TM transition. It returns false if halted.
func (a *AnalogTM) Step() bool {
	if a.State < 0 {
		return false
	}
	read := a.Read()
	tr := a.TM.Delta[a.State][read]
	if tr.Move > 0 {
		rest := New(a.Prec)
		rest.CantorRest(a.Right)
		a.Right.Set(rest)
		a.Left.CantorPush(a.Left, tr.Write)
	} else {
		rest := New(a.Prec)
		rest.CantorRest(a.Right)
		written := New(a.Prec).CantorPush(rest, tr.Write)
		bit := a.Left.CantorBit()
		leftRest := New(a.Prec)
		leftRest.CantorRest(a.Left)
		a.Left.Set(leftRest)
		a.Right.CantorPush(written, bit)
	}
	a.State = tr.Next
	return a.State >= 0
}

// Run performs up to bound analog steps.
func (a *AnalogTM) Run(bound int) (halted bool, steps int) {
	for steps = 0; steps < bound; {
		running := a.Step()
		steps++
		if !running {
			return true, steps
		}
	}
	return a.Halted(), steps
}

// TapeBit returns the symbol at offset d from the analog head.
// d = 0 is the head, d > 0 is to the right, d < 0 to the left.
func (a *AnalogTM) TapeBit(d int) int {
	if d >= 0 {
		return a.Right.CantorBitAt(d)
	}
	return a.Left.CantorBitAt(-d - 1)
}

// Output is the 1-span of the analog tape over offsets [−span, span].
func (a *AnalogTM) Output(span int) []bool {
	if span < 1 {
		span = 64
	}
	min, max := 0, 0
	found := false
	for d := -span; d <= span; d++ {
		if a.TapeBit(d) != 1 {
			continue
		}
		if !found || d < min {
			min = d
		}
		if !found || d > max {
			max = d
		}
		found = true
	}
	if !found {
		return nil
	}
	out := make([]bool, max-min+1)
	for d := min; d <= max; d++ {
		out[d-min] = a.TapeBit(d) == 1
	}
	return out
}

// Zeno is an accelerated Turing machine: step n is allotted time
// 2^{-(n+1)}, so ω discrete steps complete in analog time 1.
// The configuration is an analog real pair; the limit as steps → ∞
// (precision permitting) is the supertask result.
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

// Run performs up to n accelerated steps. Once the TM halts the
// analog configuration freezes; time still records the partial sum
// of 1/2 + 1/4 + ... used so far.
func (z *Zeno) Run(n int) {
	dt := FromRat(z.Analog.Prec, 1, 2)
	half := FromRat(z.Analog.Prec, 1, 2)
	for i := 0; i < n; i++ {
		running := z.Analog.Step()
		z.Time.Add(z.Time, dt)
		dt.Mul(dt, half)
		z.Steps++
		if !running {
			return
		}
	}
}

// Halted reports whether the inner TM has halted.
func (z *Zeno) Halted() bool {
	return z.Analog.Halted()
}
