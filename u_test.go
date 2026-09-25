// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import (
	"testing"
	"time"
)

func TestRunUHaltEmpty(t *testing.T) {
	p := ListingProgram(nil)
	if FormatBits(p) != "000" {
		t.Fatalf("HALT encoding %q", FormatBits(p))
	}
	res := RunU(p, 16)
	if res.Status != UHalt || res.Read != 3 || len(res.Out) != 0 {
		t.Fatalf("%+v", res)
	}
}

func TestRunUListing(t *testing.T) {
	x := ParseBitString("101")
	p := ListingProgram(x)
	res := RunU(p, 32)
	if res.Status != UHalt || res.Read != len(p) || !bitsEq(res.Out, x) {
		t.Fatalf("listing: status=%d read=%d out=%q p=%q",
			res.Status, res.Read, FormatBits(res.Out), FormatBits(p))
	}
}

func TestRunURepeatOnes(t *testing.T) {
	for k := 1; k <= 16; k++ {
		x := make([]bool, k)
		for i := range x {
			x[i] = true
		}
		p := RepeatProgram(true, k)
		res := RunU(p, 256)
		if res.Status != UHalt || res.Read != len(p) {
			t.Fatalf("k=%d status=%d read=%d want %d p=%q", k, res.Status, res.Read, len(p), FormatBits(p))
		}
		if !bitsEq(res.Out, x) {
			t.Fatalf("k=%d out=%q", k, FormatBits(res.Out))
		}
	}
}

func TestRunUInfiniteLoopTimesOut(t *testing.T) {
	// SET 1, OUT1, JMP 0 (jump to self)
	var p []bool
	putBits(&p, uOpSet, 3)
	putBits(&p, 1, 4)
	putBits(&p, uOpOut1, 3)
	putBits(&p, uOpJmp, 3)
	putBits(&p, 0, 4) // pc -= 1, stay on JMP
	res := RunU(p, 32)
	if res.Status != UTimeout {
		t.Fatalf("status %d", res.Status)
	}
}

func TestRunUBlock(t *testing.T) {
	res := RunU([]bool{true}, 16) // 1 bit, need 3
	if res.Status != UBlock {
		t.Fatalf("status %d", res.Status)
	}
}

func TestPrefixFree(t *testing.T) {
	const maxN = 8
	var progs [][]bool
	for n := 1; n <= maxN; n++ {
		for i := 0; i < 1<<n; i++ {
			src := IntBits(n, i)
			if ValidUProgram(src, 64) {
				progs = append(progs, append([]bool(nil), src...))
			}
		}
	}
	if len(progs) == 0 {
		t.Fatal("no valid programs")
	}
	for i, p := range progs {
		for j, q := range progs {
			if i == j || len(p) >= len(q) {
				continue
			}
			if bitsEq(p, q[:len(p)]) {
				t.Fatalf("prefix: %q of %q", FormatBits(p), FormatBits(q))
			}
		}
	}
}

func TestOmegaEqualsHaltMass(t *testing.T) {
	const maxBits = 8
	w := Omega(maxBits, 64, 256)
	sum := New(256)
	for n := 1; n <= maxBits; n++ {
		for i := 0; i < 1<<n; i++ {
			src := IntBits(n, i)
			if ValidUProgram(src, 64) {
				sum.Add(sum, Pow2(256, -n))
			}
		}
	}
	if !w.ApproxEq(sum, 200) {
		t.Fatalf("Ω=%s sum=%s", w, sum)
	}
	if w.Sign() <= 0 || w.Cmp(FromInt(256, 1)) >= 0 {
		t.Fatal("Ω not in (0,1)")
	}
}

func TestUHaltOracleBits(t *testing.T) {
	oracle, halted := UHaltOracle(6, 64, 0)
	if len(halted) != NumUBuffers(6) {
		t.Fatalf("len %d", len(halted))
	}
	for i, h := range halted {
		if (oracle.BitAt(i) == 1) != h {
			t.Fatalf("bit %d", i)
		}
	}
}

func TestKUEmptyAndOne(t *testing.T) {
	e := KComplexity(nil, 8, 64, 128)
	if e.K != 3 || e.How != "listing" {
		t.Fatalf("%s", e)
	}
	one := KComplexity(ParseBitString("1"), 8, 64, 128)
	if one.K != 6 {
		t.Fatalf("K(1)=%d want 6 (%s)", one.K, one)
	}
	zero := KComplexity(ParseBitString("0"), 8, 64, 128)
	if zero.K != 6 {
		t.Fatalf("K(0)=%d want 6", zero.K)
	}
}

func TestKURepeatBeatsListing(t *testing.T) {
	x := make([]bool, 16)
	for i := range x {
		x[i] = true
	}
	r := KComplexity(x, 8, 256, 128)
	listing := len(ListingProgram(x))
	if r.K >= listing {
		t.Fatalf("repeat should beat listing %d, got %s", listing, r)
	}
	if r.How != "repeat" {
		t.Fatalf("how %s", r.How)
	}
	if !ValidUProgram(r.Program, 256) || !bitsEq(RunU(r.Program, 256).Out, x) {
		t.Fatal("witness does not print 1^16")
	}
}

func TestKUNoShorterThanK(t *testing.T) {
	x := ParseBitString("11")
	r := KComplexity(x, 10, 64, 128)
	for n := 1; n < r.K; n++ {
		for i := 0; i < 1<<n; i++ {
			src := IntBits(n, i)
			res := RunU(src, 64)
			if res.Status == UHalt && res.Read == n && bitsEq(res.Out, x) {
				t.Fatalf("shorter program %q outputs 11", FormatBits(src))
			}
		}
	}
}

func TestKDivideMatchesExact(t *testing.T) {
	for _, s := range []string{"", "0", "1", "11", "101", "1111", "1111111111111111"} {
		x := ParseBitString(s)
		a := KComplexity(x, 10, 256, 128)
		b := KDivideConquer(x, 10, 256, DefaultKLeaf, 128)
		if a.K != b.K || a.How != b.How || !bitsEq(a.Program, b.Program) {
			t.Fatalf("%q exact %s divide %s", s, a, b)
		}
	}
}

func TestKDivideSplitsRuns(t *testing.T) {
	x := make([]bool, 64)
	for i := 0; i < 32; i++ {
		x[i] = true
	}
	r := KDivideConquer(x, 12, 32, 8, 128)
	listing := len(ListingProgram(x))
	if r.How != "divide" || r.K <= 0 || r.K >= listing {
		t.Fatalf("listing %d got %s", listing, r)
	}
	if r.Bound < 32 {
		t.Fatalf("bound %d", r.Bound)
	}
	res := RunU(r.Program, r.Bound)
	if res.Status != UHalt || res.Read != len(r.Program) || !bitsEq(res.Out, x) {
		t.Fatalf("witness status %d read %d/%d out %d", res.Status, res.Read, len(r.Program), len(res.Out))
	}
}

func TestKDivideLarge(t *testing.T) {
	ones := make([]bool, 5000)
	for i := range ones {
		ones[i] = true
	}
	start := time.Now()
	r := KDivideConquer(ones, 12, 256, DefaultKLeaf, 64)
	if time.Since(start) > 10*time.Second {
		t.Fatalf("unary took %s", time.Since(start))
	}
	if r.K <= 0 || r.K >= len(ListingProgram(ones)) {
		t.Fatalf("listing %d got %s", len(ListingProgram(ones)), r)
	}
	res := RunU(r.Program, r.Bound)
	if res.Status != UHalt || !bitsEq(res.Out, ones) {
		t.Fatalf("unary witness status %d out %d", res.Status, len(res.Out))
	}

	// Alternating runs: two distinct blocks, spliced up the tree.
	x := make([]bool, 0, 2048)
	for len(x) < 2048 {
		for i := 0; i < 8; i++ {
			x = append(x, true)
		}
		for i := 0; i < 8; i++ {
			x = append(x, false)
		}
	}
	start = time.Now()
	r = KDivideConquer(x, 12, 256, 8, 64)
	if time.Since(start) > 10*time.Second {
		t.Fatalf("runs took %s", time.Since(start))
	}
	if r.How != "divide" || r.K >= len(ListingProgram(x)) {
		t.Fatalf("listing %d got %s", len(ListingProgram(x)), r)
	}
	res = RunU(r.Program, r.Bound)
	if res.Status != UHalt || res.Read != len(r.Program) || !bitsEq(res.Out, x) {
		t.Fatalf("runs witness status %d read %d/%d out %d", res.Status, res.Read, len(r.Program), len(res.Out))
	}
}

func TestUOmegaMonotone(t *testing.T) {
	a := Omega(4, 64, 128)
	b := Omega(8, 64, 128)
	if b.Cmp(a) < 0 {
		t.Fatalf("Ω decreased: %s vs %s", a, b)
	}
}
