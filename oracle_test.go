// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import (
	"testing"
)

func TestHaltOracleBits(t *testing.T) {
	const (
		nstates = 1
		bound   = 32
	)
	oracle, halted := HaltOracle(nstates, bound, 0)
	if len(halted) != NumTMs(nstates) {
		t.Fatalf("len %d", len(halted))
	}
	for i, h := range halted {
		got := oracle.BitAt(i)
		want := 0
		if h {
			want = 1
		}
		if got != want {
			t.Fatalf("oracle bit %d = %d want %d", i, got, want)
		}
		sim, _, _ := TMFromIndex(i, nstates).Run(bound)
		if sim != h {
			t.Fatalf("stored halt bit %d disagrees with resimulation", i)
		}
	}
}

func TestCantorHaltOracle(t *testing.T) {
	oracle, halted := CantorHaltOracle(1, 32, 0)
	got := oracle.CantorBits(len(halted))
	for i, h := range halted {
		if got[i] != h {
			t.Fatalf("cantor bit %d = %v want %v", i, got[i], h)
		}
	}
}

func TestHaltOracleMachineQuery(t *testing.T) {
	oracle, halted := HaltOracle(1, 32, 0)
	prec := oracle.Prec()
	m := NewMachine(prec, 8)
	_ = m.Load(2, FromInt(prec, 1))
	for i, h := range halted {
		_ = m.Load(0, oracle)
		m.R[1].SetInt64(int64(i))
		m.PC = 0
		m.Steps = 0
		if err := m.Run(HaltQuery(0, 1, 3, 2)); err != nil {
			t.Fatal(err)
		}
		got := m.R[3].Sign() != 0
		if got != h {
			t.Fatalf("machine query TM %d: got %v want %v", i, got, h)
		}
	}
}

func TestOmegaMonotone(t *testing.T) {
	a := Omega(1, 2, 32, 256)
	b := Omega(1, 4, 32, 256)
	if b.Cmp(a) < 0 {
		t.Fatalf("more programs should not decrease Ω: %s vs %s", a, b)
	}
	one := FromInt(256, 1)
	if a.Cmp(one) >= 0 || b.Cmp(one) >= 0 {
		t.Fatal("Ω should be < 1")
	}
	if a.Sign() <= 0 {
		t.Fatal("some 1-state TMs halt, so Ω > 0")
	}
}

func TestTwoStateOraclePrefix(t *testing.T) {
	oracle, halted := HaltOracle(2, 32, PrecBits(256))
	if len(halted) != NumTMs(2) {
		t.Fatalf("len %d", len(halted))
	}
	for i := 0; i < 128; i++ {
		got := oracle.BitAt(i) == 1
		if got != halted[i] {
			t.Fatalf("bit %d: analog %v stored %v", i, got, halted[i])
		}
	}
}

func TestWriteOneHaltIsInOracle(t *testing.T) {
	_, halted := HaltOracle(1, 8, 0)
	found := false
	for i, h := range halted {
		if !h {
			continue
		}
		tm := TMFromIndex(i, 1)
		hh, steps, cfg := tm.Run(8)
		if hh && steps > 0 && cfg.Ones() >= 1 {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected at least one 1-state TM that writes and halts")
	}
}
