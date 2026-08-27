// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import (
	"math"
	"strings"
	"testing"
)

func TestPiDigits(t *testing.T) {
	s := Pi(256).Text('f', 50)
	want := "3.14159265358979323846264338327950288419716939937510"
	if !strings.HasPrefix(s, "3.141592653589793238462643383279502884197169399375") {
		t.Fatalf("π = %s want %s…", s, want)
	}
}

func TestSinCosSpecials(t *testing.T) {
	const prec uint = 192
	pi := Pi(prec)
	half := New(prec).Quo(pi, FromInt(prec, 2))
	zero := FromInt(prec, 0)
	one := FromInt(prec, 1)
	mone := FromInt(prec, -1)

	if !New(prec).Sin(zero).ApproxEq(zero, 80) {
		t.Fatal("sin(0)")
	}
	if !New(prec).Cos(zero).ApproxEq(one, 80) {
		t.Fatal("cos(0)")
	}
	if !New(prec).Sin(half).ApproxEq(one, 80) {
		t.Fatalf("sin(π/2) = %s", New(prec).Sin(half))
	}
	if !New(prec).Cos(pi).ApproxEq(mone, 80) {
		t.Fatalf("cos(π) = %s", New(prec).Cos(pi))
	}
	if !New(prec).Sin(pi).ApproxEq(zero, 80) {
		t.Fatalf("sin(π) = %s", New(prec).Sin(pi))
	}
}

func TestSinCosVsMath(t *testing.T) {
	const prec uint = 128
	for _, x := range []float64{0.1, -0.3, 1.2, -2.5, 3.1, 6.5, 10} {
		s := New(prec).Sin(FromFloat64(prec, x)).Float64()
		c := New(prec).Cos(FromFloat64(prec, x)).Float64()
		if math.Abs(s-math.Sin(x)) > 1e-12 {
			t.Fatalf("sin(%v) = %v want %v", x, s, math.Sin(x))
		}
		if math.Abs(c-math.Cos(x)) > 1e-12 {
			t.Fatalf("cos(%v) = %v want %v", x, c, math.Cos(x))
		}
	}
}
