// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import "testing"

func TestOmegaNondecreasingInT(t *testing.T) {
	prec := uint(128)
	var prev *BitFloat
	for _, bound := range []int{1, 2, 4, 8} {
		om := Omega(8, bound, prec)
		if prev != nil && om.Cmp(prev) < 0 {
			t.Fatalf("Ω_T decreased at T=%d: %s < %s", bound, om, prev)
		}
		prev = om
	}
}

func TestSweepPCertifiesK(t *testing.T) {
	s := SweepResources(DefaultSweepProbes(), []int{5, 6}, []int{8}, 9, 128)
	lo, ok := s.cell(5, 8)
	if !ok || !lo.Caught {
		t.Fatal("p=5")
	}
	if !contains(lo.Beyond, "1") {
		t.Fatalf("p=5 should refuse 1: %+v", lo)
	}
	hi, ok := s.cell(6, 8)
	if !ok || !hi.Caught {
		t.Fatal("p=6")
	}
	if !contains(hi.Certified, "1") {
		t.Fatalf("p=6 should certify 1: %+v", hi)
	}
}

func TestSweepTCertifiesK(t *testing.T) {
	s := SweepResources(DefaultSweepProbes(), []int{6}, []int{1, 2}, 8, 128)
	early, ok := s.cell(6, 1)
	if !ok || !early.Caught {
		t.Fatal("T=1")
	}
	if !contains(early.Beyond, "1") {
		t.Fatalf("T=1 should refuse 1 (listing needs 2 steps): %+v", early)
	}
	late, ok := s.cell(6, 2)
	if !ok || !late.Caught {
		t.Fatal("T=2")
	}
	if !contains(late.Certified, "1") {
		t.Fatalf("T=2 should certify 1: %+v", late)
	}
}

func TestSweepMonotoneCertified(t *testing.T) {
	s := SweepResources(DefaultSweepProbes(), []int{3, 6, 9}, []int{1, 2, 4}, 9, 128)
	n := func(p, bound int) int {
		c, ok := s.cell(p, bound)
		if !ok {
			t.Fatalf("missing %d %d", p, bound)
		}
		return len(c.Certified)
	}
	if n(3, 4) > n(6, 4) || n(6, 4) > n(9, 4) {
		t.Fatalf("certified set should grow with p: 3→%d 6→%d 9→%d",
			n(3, 4), n(6, 4), n(9, 4))
	}
	if n(9, 1) > n(9, 2) || n(9, 2) > n(9, 4) {
		t.Fatalf("certified set should grow with T: 1→%d 2→%d 4→%d",
			n(9, 1), n(9, 2), n(9, 4))
	}
}

func TestSweepTableNotEmpty(t *testing.T) {
	s := SweepResources(nil, nil, nil, 0, 64)
	if len(s.Cells) == 0 || s.String() == "" {
		t.Fatal("empty sweep")
	}
}

func contains(xs []string, v string) bool {
	for _, x := range xs {
		if x == v {
			return true
		}
	}
	return false
}
