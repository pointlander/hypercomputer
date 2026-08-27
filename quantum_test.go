// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import "testing"

const qprec uint = 160

func assertProb(t *testing.T, s *QState, basis int, want *BitFloat, bits uint) {
	t.Helper()
	got := s.Prob(basis)
	if !got.ApproxEq(want, bits) {
		t.Fatalf("|%s⟩ p=%s want %s", Ket(s.Qubits, basis), got.Text('g', 20), want.Text('g', 20))
	}
}

func TestHadamardTwiceIsI(t *testing.T) {
	s := NewQState(qprec, 1)
	s.H(0)
	s.H(0)
	if !s.Amp[0].ApproxEq(COne(qprec), 80) || !s.Amp[1].ApproxEq(CZero(qprec), 80) {
		t.Fatalf("H²|0⟩ =\n%s", s)
	}
}

func TestPauliSquares(t *testing.T) {
	for _, gate := range []struct {
		name string
		fn   func(*QState, int)
	}{
		{"X", (*QState).X},
		{"Y", (*QState).Y},
		{"Z", (*QState).Z},
		{"S then Sdg", nil},
	} {
		s := NewQState(qprec, 1)
		s.H(0)
		if gate.fn != nil {
			gate.fn(s, 0)
			gate.fn(s, 0)
		} else {
			s.S(0)
			s.Sdg(0)
		}
		s.H(0)
		if !s.Amp[0].ApproxEq(COne(qprec), 70) {
			t.Fatalf("%s² did not restore |0⟩:\n%s", gate.name, s)
		}
	}
}

func TestTTdg(t *testing.T) {
	s := NewQState(qprec, 1)
	s.H(0)
	s.T(0)
	s.Tdg(0)
	s.H(0)
	if !s.Amp[0].ApproxEq(COne(qprec), 70) {
		t.Fatalf("T T†:\n%s", s)
	}
}

func TestBell(t *testing.T) {
	s := Bell(qprec)
	half := FromRat(qprec, 1, 2)
	zero := New(qprec)
	assertProb(t, s, 0, half, 80)
	assertProb(t, s, 1, zero, 80)
	assertProb(t, s, 2, zero, 80)
	assertProb(t, s, 3, half, 80)
	if !s.Norm2().ApproxEq(FromInt(qprec, 1), 80) {
		t.Fatal("bell not normalized")
	}
}

func TestGHZ(t *testing.T) {
	s := GHZ(qprec, 3)
	half := FromRat(qprec, 1, 2)
	assertProb(t, s, 0, half, 80)
	assertProb(t, s, 7, half, 80)
	for i := 1; i < 7; i++ {
		assertProb(t, s, i, New(qprec), 80)
	}
}

func TestDeutsch(t *testing.T) {
	c := Deutsch(qprec, false)
	_, p1 := c.ProbQubit(0)
	if !p1.ApproxEq(New(qprec), 70) {
		t.Fatalf("constant f: P(1)=%s want 0", p1)
	}
	b := Deutsch(qprec, true)
	_, p1 = b.ProbQubit(0)
	if !p1.ApproxEq(FromInt(qprec, 1), 70) {
		t.Fatalf("balanced f: P(1)=%s want 1", p1)
	}
}

func TestGrover2(t *testing.T) {
	s := Grover2(qprec)
	assertProb(t, s, 3, FromInt(qprec, 1), 60)
}

func TestToffoli(t *testing.T) {
	s := NewQState(qprec, 3)
	s.X(0)
	s.X(1)
	s.CCX(0, 1, 2)
	assertProb(t, s, 7, FromInt(qprec, 1), 80)
	s.CCX(0, 1, 2)
	assertProb(t, s, 3, FromInt(qprec, 1), 80)
}

func TestSWAP(t *testing.T) {
	s := NewQState(qprec, 2)
	s.X(0)
	s.SWAP(0, 1)
	assertProb(t, s, 2, FromInt(qprec, 1), 80)
}

func TestRXPiIsX(t *testing.T) {
	s := NewQState(qprec, 1)
	s.RX(0, Pi(qprec))
	// RX(π) = −i X, so |0⟩ ↦ −i |1⟩
	assertProb(t, s, 1, FromInt(qprec, 1), 60)
}

func TestQFTUniformOnZero(t *testing.T) {
	n := 3
	s := NewQState(qprec, n)
	s.QFT()
	want := Pow2(qprec, -n)
	for i := 0; i < 1<<n; i++ {
		assertProb(t, s, i, want, 60)
	}
	c := NewQCircuit(qprec, n).QFT().Run()
	for i := 0; i < 1<<n; i++ {
		assertProb(t, c, i, want, 60)
	}
}

func TestTeleportOne(t *testing.T) {
	s := Teleport(qprec, func(c *QCircuit) { c.X(0) })
	_, p1 := s.ProbQubit(2)
	if !p1.ApproxEq(FromInt(qprec, 1), 60) {
		t.Fatalf("teleported |1⟩: P(q2=1)=%s", p1)
	}
}

func TestTeleportPlus(t *testing.T) {
	s := Teleport(qprec, func(c *QCircuit) { c.H(0) })
	p0, p1 := s.ProbQubit(2)
	half := FromRat(qprec, 1, 2)
	if !p0.ApproxEq(half, 60) || !p1.ApproxEq(half, 60) {
		t.Fatalf("teleported |+⟩: p0=%s p1=%s", p0, p1)
	}
}

func TestMeasureZero(t *testing.T) {
	s := NewQState(qprec, 1)
	if s.Measure(0, nil) != 0 {
		t.Fatal("measure |0⟩")
	}
	assertProb(t, s, 0, FromInt(qprec, 1), 80)
}

func TestMeasurePlusCollapses(t *testing.T) {
	s := NewQState(qprec, 1)
	s.H(0)
	v := s.Measure(0, nil)
	assertProb(t, s, v, FromInt(qprec, 1), 80)
}

func TestCircuitBuilder(t *testing.T) {
	s := NewQCircuit(qprec, 2).H(0).CNOT(0, 1).Run()
	half := FromRat(qprec, 1, 2)
	assertProb(t, s, 0, half, 80)
	assertProb(t, s, 3, half, 80)
}

func TestCZPhase(t *testing.T) {
	s := NewQState(qprec, 2)
	s.H(0)
	s.H(1)
	s.CZ(0, 1)
	// |11⟩ amplitude should be negative
	if s.Amp[3].Re.Sign() >= 0 {
		t.Fatalf("CZ should flip |11⟩, got %s", s.Amp[3])
	}
}

func TestKet(t *testing.T) {
	if Ket(3, 1) != "001" || Ket(3, 5) != "101" {
		t.Fatalf("Ket: %q %q", Ket(3, 1), Ket(3, 5))
	}
}
