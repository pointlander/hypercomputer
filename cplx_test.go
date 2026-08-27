// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import "testing"

func TestComplexArithmetic(t *testing.T) {
	const prec uint = 128
	a := NewComplex(prec)
	a.Re.SetInt64(2)
	a.Im.SetInt64(3)
	b := NewComplex(prec)
	b.Re.SetInt64(1)
	b.Im.SetInt64(-4)

	sum := NewComplex(prec).Add(a, b)
	if sum.Re.Float64() != 3 || sum.Im.Float64() != -1 {
		t.Fatalf("add %s", sum)
	}

	prod := NewComplex(prec).Mul(a, b)
	// (2+3i)(1-4i) = 2-8i+3i-12i² = 14-5i
	if prod.Re.Float64() != 14 || prod.Im.Float64() != -5 {
		t.Fatalf("mul %s", prod)
	}

	quot := NewComplex(prec).Quo(prod, b)
	if !quot.ApproxEq(a, 80) {
		t.Fatalf("quo %s want %s", quot, a)
	}

	cj := NewComplex(prec).Conj(a)
	if cj.Re.Float64() != 2 || cj.Im.Float64() != -3 {
		t.Fatalf("conj %s", cj)
	}

	if a.Abs2().Float64() != 13 {
		t.Fatalf("abs2 %s", a.Abs2())
	}
}

func TestCExpI(t *testing.T) {
	const prec uint = 160
	z := CExpI(Pi(prec))
	want := CReal(FromInt(prec, -1))
	if !z.ApproxEq(want, 80) {
		t.Fatalf("e^{iπ} = %s want -1", z)
	}
	half := New(prec).Quo(Pi(prec), FromInt(prec, 2))
	i := CExpI(half)
	if !i.ApproxEq(CI(prec), 80) {
		t.Fatalf("e^{iπ/2} = %s want i", i)
	}
}
