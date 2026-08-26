// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import (
	"testing"
)

func TestHaltQueryProgram(t *testing.T) {
	bits := bitsOf("1011001110001010")
	prec := PrecBits(len(bits))
	m := NewMachine(prec, 8)
	if err := m.Load(0, FromBits(prec, bits)); err != nil {
		t.Fatal(err)
	}
	if err := m.Load(2, FromInt(prec, 1)); err != nil {
		t.Fatal(err)
	}
	for i, b := range bits {
		m.PC = 0
		m.Steps = 0
		if err := m.Load(0, FromBits(prec, bits)); err != nil {
			t.Fatal(err)
		}
		m.R[1].SetInt64(int64(i))
		if err := m.Run(HaltQuery(0, 1, 3, 2)); err != nil {
			t.Fatal(err)
		}
		got := int(m.R[3].Float64())
		want := 0
		if b {
			want = 1
		}
		if got != want {
			t.Fatalf("bit %d: got %d want %d", i, got, want)
		}
	}
}

func TestNewtonSqrtProgram(t *testing.T) {
	prec := uint(256)
	m := NewMachine(prec, 8)
	m.R[0].SetInt64(2)  // x
	m.R[1].SetInt64(1)  // guess
	m.R[2].SetInt64(32) // iterations
	m.R[3].SetInt64(2)  // two
	m.R[4].SetInt64(1)  // one
	if err := m.Run(NewtonSqrt(0, 1, 2, 3, 4, 5)); err != nil {
		t.Fatal(err)
	}
	want := New(prec).Sqrt(FromInt(prec, 2))
	if !m.R[1].ApproxEq(want, 64) {
		t.Fatalf("newton sqrt(2) = %s want %s", m.R[1].Text('g', 40), want.Text('g', 40))
	}
}

func TestArithmetic(t *testing.T) {
	m := NewMachine(64, 4)
	prog := []Instr{
		{Op: OpLoad, D: 0, Imm: FromInt(64, 6)},
		{Op: OpLoad, D: 1, Imm: FromInt(64, 7)},
		{Op: OpMul, D: 2, A: 0, B: 1},
		{Op: OpHalt},
	}
	if err := m.Run(prog); err != nil {
		t.Fatal(err)
	}
	if m.R[2].Float64() != 42 {
		t.Fatalf("6*7 = %v", m.R[2].Float64())
	}
}

func TestStepLimit(t *testing.T) {
	m := NewMachine(32, 2)
	m.MaxSteps = 10
	prog := []Instr{
		{Op: OpJump, Addr: 0},
		{Op: OpHalt},
	}
	if err := m.Run(prog); err != ErrSteps {
		t.Fatalf("got %v want ErrSteps", err)
	}
}

func TestDivZero(t *testing.T) {
	m := NewMachine(32, 3)
	prog := []Instr{
		{Op: OpLoad, D: 0, Imm: FromInt(32, 1)},
		{Op: OpLoad, D: 1, Imm: FromInt(32, 0)},
		{Op: OpQuo, D: 2, A: 0, B: 1},
		{Op: OpHalt},
	}
	if err := m.Run(prog); err != ErrDivZero {
		t.Fatalf("got %v want ErrDivZero", err)
	}
}

func TestQueryBit(t *testing.T) {
	bits := bitsOf("01101")
	m := NewMachine(64, 1)
	_ = m.Load(0, FromBits(64, bits))
	for i, b := range bits {
		got, err := m.QueryBit(0, i)
		if err != nil {
			t.Fatal(err)
		}
		want := 0
		if b {
			want = 1
		}
		if got != want {
			t.Fatalf("bit %d: %d want %d", i, got, want)
		}
	}
}

func TestCantorStackOps(t *testing.T) {
	prec := uint(256)
	m := NewMachine(prec, 6)
	prog := []Instr{
		{Op: OpLoad, D: 0, Imm: CantorZeros(prec)},
		{Op: OpLoad, D: 1, Imm: FromInt(prec, 1)},
		{Op: OpCantorPush, D: 0, A: 0, B: 1},
		{Op: OpLoad, D: 1, Imm: FromInt(prec, 0)},
		{Op: OpCantorPush, D: 0, A: 0, B: 1},
		{Op: OpCantorPop, D: 2, A: 0, B: 3},
		{Op: OpCantorPop, D: 2, A: 2, B: 4},
		{Op: OpHalt},
	}
	if err := m.Run(prog); err != nil {
		t.Fatal(err)
	}
	if m.R[3].Float64() != 0 {
		t.Fatalf("first pop (last push) = %v want 0", m.R[3].Float64())
	}
	if m.R[4].Float64() != 1 {
		t.Fatalf("second pop = %v want 1", m.R[4].Float64())
	}
}

func TestFloorFracISA(t *testing.T) {
	m := NewMachine(64, 4)
	prog := []Instr{
		{Op: OpLoad, D: 0, Imm: FromFloat64(64, 3.75)},
		{Op: OpFloor, D: 1, A: 0},
		{Op: OpFrac, D: 2, A: 0},
		{Op: OpHalt},
	}
	if err := m.Run(prog); err != nil {
		t.Fatal(err)
	}
	if m.R[1].Float64() != 3 {
		t.Fatalf("floor = %v", m.R[1].Float64())
	}
	if m.R[2].Float64() != 0.75 {
		t.Fatalf("frac = %v", m.R[2].Float64())
	}
}
