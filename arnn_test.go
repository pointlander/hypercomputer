// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import (
	"testing"
)

func TestCantorReader(t *testing.T) {
	msg := bitsOf("1100101000110111")
	q := FromCantor(PrecBits(2*len(msg)+16), msg)
	r := NewCantorReader(q)
	got := r.ReadN(len(msg))
	if bitsString(got) != bitsString(msg) {
		t.Fatalf("arnn read %s want %s", bitsString(got), bitsString(msg))
	}
}

func TestCantorReaderHaltOracle(t *testing.T) {
	oracle, halted := CantorHaltOracle(1, 32, 0)
	r := NewCantorReader(oracle)
	got := r.ReadN(len(halted))
	for i, h := range halted {
		if got[i] != h {
			t.Fatalf("neuron pop %d = %v want %v", i, got[i], h)
		}
	}
}

func TestARNNSatLinDecay(t *testing.T) {
	a := NewARNN(64, 1)
	a.W[0][0] = FromRat(64, 1, 2)
	a.X[0] = FromInt(64, 1)
	for i := 0; i < 8; i++ {
		a.Step()
	}
	if a.X[0].Float64() > 0.01 {
		t.Fatalf("x should decay under w=1/2, got %v", a.X[0].Float64())
	}
}

func TestARNNDoublingMapNeuron(t *testing.T) {
	// One analog step of T(x) = 2x mod 1 for x in [0, 0.5):
	// σ(2x) = 2x.
	a := NewARNN(128, 1)
	a.W[0][0] = FromInt(128, 2)
	a.X[0] = FromRat(128, 1, 4)
	a.Step()
	want := FromRat(128, 1, 2)
	if !a.X[0].ApproxEq(want, 40) {
		t.Fatalf("σ(2·1/4) = %s want 1/2", a.X[0])
	}
	a.Step()
	if a.X[0].Cmp(FromInt(128, 1)) != 0 {
		t.Fatalf("σ(2·1/2) should saturate at 1, got %s", a.X[0])
	}
}
