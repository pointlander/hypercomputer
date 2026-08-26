// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import (
	"math"
	"math/big"
	"testing"
)

func bitsOf(s string) []bool {
	b := make([]bool, len(s))
	for i, c := range s {
		b[i] = c == '1'
	}
	return b
}

func bitsString(b []bool) string {
	s := make([]byte, len(b))
	for i, v := range b {
		if v {
			s[i] = '1'
		} else {
			s[i] = '0'
		}
	}
	return string(s)
}

func TestFromBitsRoundtrip(t *testing.T) {
	cases := []string{
		"",
		"0",
		"1",
		"10",
		"01",
		"1111",
		"1000",
		"0001",
		"1011001010110101",
		"1100100100001111110110101010001000100001011010001100001000110100", // pi bits
	}
	for _, s := range cases {
		bits := bitsOf(s)
		x := FromBits(PrecBits(len(bits)+8), bits)
		got := bitsString(x.Bits(len(bits)))
		if got != s {
			t.Fatalf("roundtrip %q: got %q", s, got)
		}
	}
}

func TestBitAtMatchesBits(t *testing.T) {
	bits := bitsOf("1011001110001111")
	x := FromBits(256, bits)
	for i, b := range bits {
		want := 0
		if b {
			want = 1
		}
		if x.BitAt(i) != want {
			t.Fatalf("bit %d: got %d want %d", i, x.BitAt(i), want)
		}
	}
}

func TestCantorRoundtrip(t *testing.T) {
	cases := []string{
		"",
		"0",
		"1",
		"10",
		"01",
		"1111",
		"1000",
		"0001",
		"1011001010110101",
		"0011010110010111",
	}
	for _, s := range cases {
		bits := bitsOf(s)
		x := FromCantor(PrecBits(2*len(bits)+16), bits)
		got := bitsString(x.CantorBits(len(bits)))
		if got != s {
			t.Fatalf("cantor roundtrip %q: got %q", s, got)
		}
	}
}

func TestCantorZerosStable(t *testing.T) {
	z := CantorZeros(256)
	if z.CantorBit() != 0 {
		t.Fatal("zeros stack top is not 0")
	}
	rest := New(256)
	if rest.CantorPop(z) != 0 {
		t.Fatal("pop zeros")
	}
	if !rest.ApproxEq(z, 200) {
		t.Fatalf("pop zeros should stay 1/3: %s", rest)
	}
	pushed := New(256).CantorPush(z, 0)
	if !pushed.ApproxEq(z, 200) {
		t.Fatalf("push 0 onto zeros should stay 1/3: %s", pushed)
	}
}

func TestFloor(t *testing.T) {
	prec := uint(128)
	cases := []struct {
		x, want float64
	}{
		{2.3, 2},
		{-2.3, -3},
		{-2, -2},
		{2, 2},
		{0.9, 0},
		{-0.5, -1},
		{0, 0},
	}
	for _, c := range cases {
		x := FromFloat64(prec, c.x)
		got := New(prec).Floor(x)
		if got.Float64() != c.want {
			t.Fatalf("floor(%v) = %v want %v", c.x, got.Float64(), c.want)
		}
	}
}

func TestShiftBernoulli(t *testing.T) {
	x := FromBits(256, bitsOf("10110"))
	if x.Bit() != 1 {
		t.Fatal("first bit")
	}
	y := New(256).Shift(x)
	if bitsString(y.Bits(4)) != "0110" {
		t.Fatalf("after shift: %s", bitsString(y.Bits(4)))
	}
}

func TestPrecisionLimitsOracleDepth(t *testing.T) {
	const n = 64
	bits := make([]bool, n)
	for i := range bits {
		bits[i] = i%3 == 0
	}
	full := FromBits(PrecBits(n), bits)
	if bitsString(full.Bits(n)) != bitsString(bits) {
		t.Fatal("full precision should recover all bits")
	}
	shallow := FromBits(12, bits)
	got := shallow.Bits(n)
	if bitsString(got[:6]) != bitsString(bits[:6]) {
		t.Fatalf("leading bits should survive 12-bit precision: got %s want %s",
			bitsString(got[:6]), bitsString(bits[:6]))
	}
	mismatch := 0
	for i := 24; i < n; i++ {
		if got[i] != bits[i] {
			mismatch++
		}
	}
	if mismatch == 0 {
		t.Fatal("low precision should lose later oracle bits")
	}
}

func TestNewtonMatchesBigSqrt(t *testing.T) {
	prec := uint(256)
	x := FromInt(prec, 2)
	got := New(prec).Sqrt(x)
	want := new(big.Float).SetPrec(prec)
	want.Sqrt(big.NewFloat(2).SetPrec(prec))
	if got.Big().Cmp(want) != 0 {
		t.Fatalf("sqrt(2) mismatch: %s vs %s", got, want.Text('g', 40))
	}
}

func TestSatLin(t *testing.T) {
	prec := uint(64)
	cases := []struct {
		x, want float64
	}{
		{-1, 0},
		{0, 0},
		{0.25, 0.25},
		{1, 1},
		{2, 1},
	}
	for _, c := range cases {
		got := New(prec).SatLin(FromFloat64(prec, c.x))
		if math.Abs(got.Float64()-c.want) > 1e-12 {
			t.Fatalf("satlin(%v) = %v want %v", c.x, got.Float64(), c.want)
		}
	}
}

func TestMantissaOfOne(t *testing.T) {
	x := FromInt(32, 1)
	sign, exp, bits := x.Mantissa()
	if sign != 1 {
		t.Fatalf("sign %d", sign)
	}
	if exp != 1 {
		t.Fatalf("exp %d", exp)
	}
	if !bits[0] {
		t.Fatal("mantissa of 1 should start with 1 (0.5)")
	}
}

func TestApproxEq(t *testing.T) {
	a := FromRat(128, 1, 3)
	b := FromRat(128, 1, 3)
	if !a.ApproxEq(b, 100) {
		t.Fatal("1/3 should equal itself")
	}
	c := FromRat(128, 1, 4)
	if a.ApproxEq(c, 10) {
		t.Fatal("1/3 should not approx 1/4")
	}
}

func TestFrac(t *testing.T) {
	x := FromFloat64(128, 3.25)
	f := New(128).Frac(x)
	if math.Abs(f.Float64()-0.25) > 1e-12 {
		t.Fatalf("frac(3.25) = %v", f.Float64())
	}
	y := FromFloat64(128, -1.25)
	g := New(128).Frac(y)
	if math.Abs(g.Float64()-0.75) > 1e-12 {
		t.Fatalf("frac(-1.25) = %v want 0.75", g.Float64())
	}
}
