// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import (
	"fmt"
	"math/rand"
	"strings"
)

// MaxQubits is the largest state vector the simulator will allocate.
const MaxQubits = 16

// QState is a 2^n amplitude vector over high-precision complexes.
// Qubit 0 is the least significant bit of the basis index.
type QState struct {
	Prec   uint
	Qubits int
	Amp    []*Complex
}

// NewQState returns |0…0⟩ on n qubits.
func NewQState(prec uint, n int) *QState {
	if prec == 0 {
		prec = DefaultPrec
	}
	if n < 1 || n > MaxQubits {
		panic(fmt.Sprintf("qubits: want 1..%d, got %d", MaxQubits, n))
	}
	dim := 1 << n
	amp := make([]*Complex, dim)
	for i := range amp {
		amp[i] = NewComplex(prec)
	}
	amp[0].Re.SetInt64(1)
	return &QState{Prec: prec, Qubits: n, Amp: amp}
}

func (q *QState) check(qs ...int) {
	for _, b := range qs {
		if b < 0 || b >= q.Qubits {
			panic(fmt.Sprintf("qubit %d out of range 0..%d", b, q.Qubits-1))
		}
	}
}

func invSqrt2(prec uint) *BitFloat {
	return New(prec).Quo(FromInt(prec, 1), New(prec).Sqrt(FromInt(prec, 2)))
}

// apply1ctrl applies a 1-qubit matrix to target t on basis states
// whose bits in ctrlMask are all set. ctrlMask == 0 is an un-controlled gate.
func (q *QState) apply1ctrl(u00, u01, u10, u11 *Complex, t, ctrlMask int) {
	tmask := 1 << t
	tmp0 := NewComplex(q.Prec)
	tmp1 := NewComplex(q.Prec)
	ta := NewComplex(q.Prec)
	tb := NewComplex(q.Prec)
	n := len(q.Amp)
	for i := 0; i < n; i++ {
		if i&tmask != 0 {
			continue
		}
		if i&ctrlMask != ctrlMask {
			continue
		}
		j := i | tmask
		a, b := q.Amp[i], q.Amp[j]
		tmp0.Mul(u00, a)
		ta.Mul(u01, b)
		tmp0.Add(tmp0, ta)
		tmp1.Mul(u10, a)
		tb.Mul(u11, b)
		tmp1.Add(tmp1, tb)
		a.Set(tmp0)
		b.Set(tmp1)
	}
}

func (q *QState) apply1(u00, u01, u10, u11 *Complex, t int) {
	q.apply1ctrl(u00, u01, u10, u11, t, 0)
}

func matX(prec uint) (u00, u01, u10, u11 *Complex) {
	z, o := CZero(prec), COne(prec)
	return z, o, o.Copy(), z.Copy()
}

func matY(prec uint) (u00, u01, u10, u11 *Complex) {
	// [[0, -i], [i, 0]]
	mi := NewComplex(prec)
	mi.Im.SetInt64(-1)
	i := CI(prec)
	z := CZero(prec)
	return z, mi, i, z.Copy()
}

func matZ(prec uint) (u00, u01, u10, u11 *Complex) {
	o := COne(prec)
	m := COne(prec)
	m.Re.SetInt64(-1)
	z := CZero(prec)
	return o, z, z.Copy(), m
}

func matH(prec uint) (u00, u01, u10, u11 *Complex) {
	s := invSqrt2(prec)
	p := CReal(s)
	m := CReal(New(prec).Neg(s))
	return p, p.Copy(), p.Copy(), m
}

func matS(prec uint) (u00, u01, u10, u11 *Complex) {
	z := CZero(prec)
	return COne(prec), z, z.Copy(), CI(prec)
}

func matSdg(prec uint) (u00, u01, u10, u11 *Complex) {
	z := CZero(prec)
	mi := NewComplex(prec)
	mi.Im.SetInt64(-1)
	return COne(prec), z, z.Copy(), mi
}

func matT(prec uint) (u00, u01, u10, u11 *Complex) {
	// diag(1, (1+i)/√2)
	s := invSqrt2(prec)
	ph := NewComplex(prec)
	ph.Re.Set(s)
	ph.Im.Set(s)
	z := CZero(prec)
	return COne(prec), z, z.Copy(), ph
}

func matTdg(prec uint) (u00, u01, u10, u11 *Complex) {
	s := invSqrt2(prec)
	ph := NewComplex(prec)
	ph.Re.Set(s)
	ph.Im.Neg(s)
	z := CZero(prec)
	return COne(prec), z, z.Copy(), ph
}

// X is the Pauli-X (NOT) gate.
func (q *QState) X(t int) {
	q.check(t)
	u00, u01, u10, u11 := matX(q.Prec)
	q.apply1(u00, u01, u10, u11, t)
}

// Y is the Pauli-Y gate.
func (q *QState) Y(t int) {
	q.check(t)
	u00, u01, u10, u11 := matY(q.Prec)
	q.apply1(u00, u01, u10, u11, t)
}

// Z is the Pauli-Z gate.
func (q *QState) Z(t int) {
	q.check(t)
	u00, u01, u10, u11 := matZ(q.Prec)
	q.apply1(u00, u01, u10, u11, t)
}

// H is the Hadamard gate.
func (q *QState) H(t int) {
	q.check(t)
	u00, u01, u10, u11 := matH(q.Prec)
	q.apply1(u00, u01, u10, u11, t)
}

// S is the phase gate diag(1, i).
func (q *QState) S(t int) {
	q.check(t)
	u00, u01, u10, u11 := matS(q.Prec)
	q.apply1(u00, u01, u10, u11, t)
}

// Sdg is S†.
func (q *QState) Sdg(t int) {
	q.check(t)
	u00, u01, u10, u11 := matSdg(q.Prec)
	q.apply1(u00, u01, u10, u11, t)
}

// T is the π/8 gate diag(1, (1+i)/√2).
func (q *QState) T(t int) {
	q.check(t)
	u00, u01, u10, u11 := matT(q.Prec)
	q.apply1(u00, u01, u10, u11, t)
}

// Tdg is T†.
func (q *QState) Tdg(t int) {
	q.check(t)
	u00, u01, u10, u11 := matTdg(q.Prec)
	q.apply1(u00, u01, u10, u11, t)
}

// RX is a rotation about X by theta radians.
func (q *QState) RX(t int, theta *BitFloat) {
	q.check(t)
	half := New(q.Prec).Quo(theta, FromInt(q.Prec, 2))
	c := New(q.Prec).Cos(half)
	s := New(q.Prec).Sin(half)
	cc := CReal(c)
	mis := NewComplex(q.Prec)
	mis.Im.Neg(s)
	q.apply1(cc, mis, mis, cc, t)
}

// RY is a rotation about Y by theta radians.
func (q *QState) RY(t int, theta *BitFloat) {
	q.check(t)
	half := New(q.Prec).Quo(theta, FromInt(q.Prec, 2))
	c := New(q.Prec).Cos(half)
	s := New(q.Prec).Sin(half)
	cc := CReal(c)
	ms := CReal(New(q.Prec).Neg(s))
	ss := CReal(s)
	q.apply1(cc, ms, ss, cc.Copy(), t)
}

// RZ is a rotation about Z by theta radians.
func (q *QState) RZ(t int, theta *BitFloat) {
	q.check(t)
	half := New(q.Prec).Quo(theta, FromInt(q.Prec, 2))
	neg := New(q.Prec).Neg(half)
	e0 := CExpI(neg)
	e1 := CExpI(half)
	z := CZero(q.Prec)
	q.apply1(e0, z, z.Copy(), e1, t)
}

// P is the phase gate diag(1, e^{iθ}).
func (q *QState) P(t int, theta *BitFloat) {
	q.check(t)
	z := CZero(q.Prec)
	q.apply1(COne(q.Prec), z, z.Copy(), CExpI(theta), t)
}

// CNOT is the controlled-NOT (CX) gate.
func (q *QState) CNOT(c, t int) {
	q.check(c, t)
	if c == t {
		return
	}
	u00, u01, u10, u11 := matX(q.Prec)
	q.apply1ctrl(u00, u01, u10, u11, t, 1<<c)
}

// CY is the controlled-Y gate.
func (q *QState) CY(c, t int) {
	q.check(c, t)
	if c == t {
		return
	}
	u00, u01, u10, u11 := matY(q.Prec)
	q.apply1ctrl(u00, u01, u10, u11, t, 1<<c)
}

// CZ is the controlled-Z gate.
func (q *QState) CZ(c, t int) {
	q.check(c, t)
	if c == t {
		return
	}
	mask := (1 << c) | (1 << t)
	m := NewComplex(q.Prec)
	m.Re.SetInt64(-1)
	tmp := NewComplex(q.Prec)
	for i, a := range q.Amp {
		if i&mask == mask {
			tmp.Mul(a, m)
			a.Set(tmp)
		}
	}
}

// CP is the controlled-phase gate: e^{iθ} on |11⟩ of (c, t).
func (q *QState) CP(c, t int, theta *BitFloat) {
	q.check(c, t)
	if c == t {
		return
	}
	mask := (1 << c) | (1 << t)
	ph := CExpI(theta)
	tmp := NewComplex(q.Prec)
	for i, a := range q.Amp {
		if i&mask == mask {
			tmp.Mul(a, ph)
			a.Set(tmp)
		}
	}
}

// SWAP exchanges two qubits.
func (q *QState) SWAP(a, b int) {
	q.check(a, b)
	if a == b {
		return
	}
	q.CNOT(a, b)
	q.CNOT(b, a)
	q.CNOT(a, b)
}

// CCX is the Toffoli (CCNOT) gate.
func (q *QState) CCX(c1, c2, t int) {
	q.check(c1, c2, t)
	if c1 == t || c2 == t || c1 == c2 {
		return
	}
	u00, u01, u10, u11 := matX(q.Prec)
	q.apply1ctrl(u00, u01, u10, u11, t, (1<<c1)|(1<<c2))
}

// CCZ is the controlled-controlled-Z gate.
func (q *QState) CCZ(c1, c2, t int) {
	q.check(c1, c2, t)
	mask := (1 << c1) | (1 << c2) | (1 << t)
	m := NewComplex(q.Prec)
	m.Re.SetInt64(-1)
	tmp := NewComplex(q.Prec)
	for i, a := range q.Amp {
		if i&mask == mask {
			tmp.Mul(a, m)
			a.Set(tmp)
		}
	}
}

// QFT applies the quantum Fourier transform (little-endian, with swaps).
func (q *QState) QFT() {
	n := q.Qubits
	pi := Pi(q.Prec)
	for i := 0; i < n; i++ {
		q.H(i)
		ang := pi.Copy()
		for m := 1; i+m < n; m++ {
			ang.Quo(ang, FromInt(q.Prec, 2))
			q.CP(i+m, i, ang)
		}
	}
	for i := 0; i < n/2; i++ {
		q.SWAP(i, n-1-i)
	}
}

// Norm2 returns the squared Euclidean norm of the amplitude vector.
func (q *QState) Norm2() *BitFloat {
	s := New(q.Prec)
	for _, a := range q.Amp {
		s.Add(s, a.Abs2())
	}
	return s
}

// Prob returns the Born probability of a computational-basis state.
func (q *QState) Prob(basis int) *BitFloat {
	if basis < 0 || basis >= len(q.Amp) {
		return New(q.Prec)
	}
	return q.Amp[basis].Abs2()
}

// Probs returns Born probabilities for every basis state.
func (q *QState) Probs() []*BitFloat {
	p := make([]*BitFloat, len(q.Amp))
	for i, a := range q.Amp {
		p[i] = a.Abs2()
	}
	return p
}

// ProbQubit returns P(0) and P(1) for a single qubit.
func (q *QState) ProbQubit(bit int) (p0, p1 *BitFloat) {
	q.check(bit)
	p0, p1 = New(q.Prec), New(q.Prec)
	mask := 1 << bit
	for i, a := range q.Amp {
		n2 := a.Abs2()
		if i&mask != 0 {
			p1.Add(p1, n2)
		} else {
			p0.Add(p0, n2)
		}
	}
	return
}

// CollapseQubit projects qubit bit onto val ∈ {0,1} and renormalizes.
func (q *QState) CollapseQubit(bit, val int) {
	q.check(bit)
	if val != 0 {
		val = 1
	}
	mask := 1 << bit
	norm := New(q.Prec)
	for i, a := range q.Amp {
		on := 0
		if i&mask != 0 {
			on = 1
		}
		if on != val {
			a.Re.SetInt64(0)
			a.Im.SetInt64(0)
		} else {
			norm.Add(norm, a.Abs2())
		}
	}
	if norm.Sign() == 0 {
		return
	}
	s := New(q.Prec).Sqrt(norm)
	for _, a := range q.Amp {
		a.Re.Quo(a.Re, s)
		a.Im.Quo(a.Im, s)
	}
}

// Measure samples qubit bit in the computational basis and collapses.
// If rng is nil the more likely outcome is chosen (0 on a tie).
func (q *QState) Measure(bit int, rng *rand.Rand) int {
	p0, _ := q.ProbQubit(bit)
	val := 1
	if rng == nil {
		half := FromRat(q.Prec, 1, 2)
		if p0.Cmp(half) >= 0 {
			val = 0
		}
	} else {
		r := FromFloat64(q.Prec, rng.Float64())
		if r.Cmp(p0) < 0 {
			val = 0
		}
	}
	q.CollapseQubit(bit, val)
	return val
}

// Ket formats a basis index as |q_{n-1}…q_0⟩.
func Ket(n, basis int) string {
	b := make([]byte, n)
	for q := 0; q < n; q++ {
		if basis&(1<<q) != 0 {
			b[n-1-q] = '1'
		} else {
			b[n-1-q] = '0'
		}
	}
	return string(b)
}

func (q *QState) String() string {
	var b strings.Builder
	thresh := Pow2(q.Prec, -40)
	for i, a := range q.Amp {
		if a.Abs2().Cmp(thresh) > 0 {
			fmt.Fprintf(&b, "|%s⟩  %s  p=%s\n", Ket(q.Qubits, i), a, a.Abs2().Text('g', 12))
		}
	}
	return b.String()
}

// QCircuit is a unitary quantum circuit over n qubits.
type QCircuit struct {
	Prec   uint
	Qubits int
	ops    []qop
}

type qop struct {
	name  string
	qs    []int
	theta *BitFloat
}

// NewQCircuit returns an empty circuit on n qubits.
func NewQCircuit(prec uint, n int) *QCircuit {
	if prec == 0 {
		prec = DefaultPrec
	}
	if n < 1 || n > MaxQubits {
		panic(fmt.Sprintf("qubits: want 1..%d, got %d", MaxQubits, n))
	}
	return &QCircuit{Prec: prec, Qubits: n}
}

func (c *QCircuit) add(name string, theta *BitFloat, qs ...int) *QCircuit {
	c.ops = append(c.ops, qop{name: name, qs: qs, theta: theta})
	return c
}

func (c *QCircuit) H(q int) *QCircuit                { return c.add("H", nil, q) }
func (c *QCircuit) X(q int) *QCircuit                { return c.add("X", nil, q) }
func (c *QCircuit) Y(q int) *QCircuit                { return c.add("Y", nil, q) }
func (c *QCircuit) Z(q int) *QCircuit                { return c.add("Z", nil, q) }
func (c *QCircuit) S(q int) *QCircuit                { return c.add("S", nil, q) }
func (c *QCircuit) Sdg(q int) *QCircuit              { return c.add("Sdg", nil, q) }
func (c *QCircuit) T(q int) *QCircuit                { return c.add("T", nil, q) }
func (c *QCircuit) Tdg(q int) *QCircuit              { return c.add("Tdg", nil, q) }
func (c *QCircuit) RX(q int, th *BitFloat) *QCircuit { return c.add("RX", th, q) }
func (c *QCircuit) RY(q int, th *BitFloat) *QCircuit { return c.add("RY", th, q) }
func (c *QCircuit) RZ(q int, th *BitFloat) *QCircuit { return c.add("RZ", th, q) }
func (c *QCircuit) P(q int, th *BitFloat) *QCircuit  { return c.add("P", th, q) }
func (c *QCircuit) CNOT(c0, t int) *QCircuit         { return c.add("CNOT", nil, c0, t) }
func (c *QCircuit) CY(c0, t int) *QCircuit           { return c.add("CY", nil, c0, t) }
func (c *QCircuit) CZ(c0, t int) *QCircuit           { return c.add("CZ", nil, c0, t) }
func (c *QCircuit) CP(c0, t int, th *BitFloat) *QCircuit {
	return c.add("CP", th, c0, t)
}
func (c *QCircuit) SWAP(a, b int) *QCircuit     { return c.add("SWAP", nil, a, b) }
func (c *QCircuit) CCX(c1, c2, t int) *QCircuit { return c.add("CCX", nil, c1, c2, t) }
func (c *QCircuit) CCZ(c1, c2, t int) *QCircuit { return c.add("CCZ", nil, c1, c2, t) }

// QFT appends an n-qubit quantum Fourier transform.
func (c *QCircuit) QFT() *QCircuit {
	n := c.Qubits
	pi := Pi(c.Prec)
	for i := 0; i < n; i++ {
		c.H(i)
		ang := pi.Copy()
		for m := 1; i+m < n; m++ {
			ang = New(c.Prec).Quo(ang, FromInt(c.Prec, 2))
			c.CP(i+m, i, ang.Copy())
		}
	}
	for i := 0; i < n/2; i++ {
		c.SWAP(i, n-1-i)
	}
	return c
}

// Apply runs the circuit on an existing state.
func (c *QCircuit) Apply(s *QState) {
	if s.Qubits != c.Qubits {
		panic(fmt.Sprintf("circuit has %d qubits, state has %d", c.Qubits, s.Qubits))
	}
	for _, op := range c.ops {
		switch op.name {
		case "H":
			s.H(op.qs[0])
		case "X":
			s.X(op.qs[0])
		case "Y":
			s.Y(op.qs[0])
		case "Z":
			s.Z(op.qs[0])
		case "S":
			s.S(op.qs[0])
		case "Sdg":
			s.Sdg(op.qs[0])
		case "T":
			s.T(op.qs[0])
		case "Tdg":
			s.Tdg(op.qs[0])
		case "RX":
			s.RX(op.qs[0], op.theta)
		case "RY":
			s.RY(op.qs[0], op.theta)
		case "RZ":
			s.RZ(op.qs[0], op.theta)
		case "P":
			s.P(op.qs[0], op.theta)
		case "CNOT":
			s.CNOT(op.qs[0], op.qs[1])
		case "CY":
			s.CY(op.qs[0], op.qs[1])
		case "CZ":
			s.CZ(op.qs[0], op.qs[1])
		case "CP":
			s.CP(op.qs[0], op.qs[1], op.theta)
		case "SWAP":
			s.SWAP(op.qs[0], op.qs[1])
		case "CCX":
			s.CCX(op.qs[0], op.qs[1], op.qs[2])
		case "CCZ":
			s.CCZ(op.qs[0], op.qs[1], op.qs[2])
		default:
			panic("unknown gate " + op.name)
		}
	}
}

// Run returns the state obtained by applying the circuit to |0…0⟩.
func (c *QCircuit) Run() *QState {
	s := NewQState(c.Prec, c.Qubits)
	c.Apply(s)
	return s
}

// Bell returns the Bell pair (|00⟩+|11⟩)/√2.
func Bell(prec uint) *QState {
	return NewQCircuit(prec, 2).H(0).CNOT(0, 1).Run()
}

// GHZ returns an n-qubit Greenberger–Horne–Zeilinger state.
func GHZ(prec uint, n int) *QState {
	c := NewQCircuit(prec, n).H(0)
	for i := 1; i < n; i++ {
		c.CNOT(0, i)
	}
	return c.Run()
}

// Deutsch runs Deutsch's algorithm. balanced selects f(x)=x (CNOT)
// versus the constant-0 oracle. Qubit 0 is |1⟩ iff f is balanced.
func Deutsch(prec uint, balanced bool) *QState {
	c := NewQCircuit(prec, 2).X(1).H(0).H(1)
	if balanced {
		c.CNOT(0, 1)
	}
	return c.H(0).Run()
}

// Grover2 runs one Grover iteration on 2 qubits targeting |11⟩.
func Grover2(prec uint) *QState {
	return NewQCircuit(prec, 2).
		H(0).H(1).
		CZ(0, 1).
		H(0).H(1).X(0).X(1).CZ(0, 1).X(0).X(1).H(0).H(1).
		Run()
}

// Teleport moves qubit 0 onto qubit 2 by deferred-measurement teleportation.
func Teleport(prec uint, prep func(*QCircuit)) *QState {
	c := NewQCircuit(prec, 3)
	if prep != nil {
		prep(c)
	}
	c.H(1).CNOT(1, 2).
		CNOT(0, 1).H(0).
		CNOT(1, 2).CZ(0, 2)
	return c.Run()
}
