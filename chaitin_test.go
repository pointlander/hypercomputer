// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import "testing"

func TestOmegaPrefixAnalog(t *testing.T) {
	om := Omega(8, 64, 128)
	const m = 8
	bits, hat, steps := OmegaPrefix(om, m)
	if steps == 0 {
		t.Fatal("no analog shifts")
	}
	if !bitsEq(bits, om.Bits(m)) {
		t.Fatalf("prefix %q want %q", FormatBits(bits), om.Binary(m))
	}
	if !hat.ApproxEq(FromBits(128, bits), 100) {
		t.Fatal("hat")
	}
}

func TestReconstructCatches(t *testing.T) {
	const (
		maxBits = 8
		bound   = 64
		m       = 6
	)
	om := Omega(maxBits, bound, 128)
	rec := Reconstruct(om, m, maxBits, bound)
	if !rec.Caught {
		t.Fatalf("not caught: sum=%s hat=%s", rec.Sum, rec.OmegaHat)
	}
	if rec.Sum.Cmp(rec.OmegaHat) < 0 {
		t.Fatal("sum < hat")
	}
	trueHalt := map[string]bool{}
	for n := 1; n <= m; n++ {
		for i := 0; i < 1<<n; i++ {
			src := IntBits(n, i)
			if ValidUProgram(src, bound) {
				trueHalt[FormatBits(src)] = true
			}
		}
	}
	got := map[string]bool{}
	for _, p := range rec.Progs {
		if len(p.P) <= m {
			got[FormatBits(p.P)] = true
		}
	}
	for s := range trueHalt {
		if !got[s] {
			t.Fatalf("missed short program %s", s)
		}
	}
}

func TestKFromOmegaEmptyAndOne(t *testing.T) {
	e := KFromOmega(nil, 3, 8, 64, 128)
	if e.How != "chaitin" || e.K != 3 || !e.Caught {
		t.Fatalf("%s", e)
	}
	one := KFromOmega(ParseBitString("1"), 6, 8, 64, 128)
	if one.How != "chaitin" || one.K != 6 {
		t.Fatalf("%s", one)
	}
	if FormatBits(one.Program) != "010000" {
		t.Fatalf("p=%q", FormatBits(one.Program))
	}
}

func TestKFromOmegaBeyond(t *testing.T) {
	r := KFromOmega(ParseBitString("1"), 5, 8, 64, 128)
	if r.How != "chaitin-beyond" || r.K != -1 {
		t.Fatalf("m=5 should refuse K(1)=6: %s", r)
	}
	if !r.Caught {
		t.Fatal("should still catch")
	}
}

func TestKFromOmegaNeedsBits(t *testing.T) {
	x := ParseBitString("11")
	full := KFromOmega(x, 9, 10, 64, 128)
	if full.How != "chaitin" || full.K != 9 {
		t.Fatalf("m=9: %s", full)
	}
	short := KFromOmega(x, 8, 10, 64, 128)
	if short.How != "chaitin-beyond" {
		t.Fatalf("m=8 should not certify K=9: %s", short)
	}
}

func TestKFromOmegaMatchesKComplexity(t *testing.T) {
	for _, s := range []string{"", "0", "1", "11"} {
		x := ParseBitString(s)
		want := KComplexity(x, 8, 64, 128)
		got := KFromOmega(x, 10, 10, 64, 128)
		if got.How != "chaitin" || got.K != want.K {
			t.Fatalf("%q: chaitin %s vs %s", s, got, want)
		}
	}
}

func TestFlippedOmegaIncomplete(t *testing.T) {
	om := Omega(8, 64, 128)
	if om.Cmp(FromRat(128, 1, 2)) >= 0 {
		t.Fatal("test needs Ω < 1/2")
	}
	bad := New(128).Add(om, FromRat(128, 1, 2))
	rec := Reconstruct(bad, 6, 8, 64)
	if rec.Caught {
		t.Fatal("inflated Ω should not catch")
	}
	r := KFromOmegaValue(ParseBitString("1"), bad, 6, 8, 64)
	if r.How != "chaitin-incomplete" {
		t.Fatalf("got %s", r)
	}
}

func TestReconstructPrefixFree(t *testing.T) {
	om := Omega(7, 64, 128)
	rec := Reconstruct(om, 7, 7, 64)
	if !rec.Caught {
		t.Fatal("not caught")
	}
	for i, p := range rec.Progs {
		for j, q := range rec.Progs {
			if i == j || len(p.P) >= len(q.P) {
				continue
			}
			if bitsEq(p.P, q.P[:len(p.P)]) {
				t.Fatalf("prefix %q of %q", FormatBits(p.P), FormatBits(q.P))
			}
		}
	}
}
