// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import "testing"

func TestTMIndexRoundtrip(t *testing.T) {
	for nstates := 1; nstates <= 2; nstates++ {
		limit := NumTMs(nstates)
		if nstates == 2 {
			limit = 256
		}
		for i := 0; i < limit; i++ {
			got := TMFromIndex(i, nstates).Index()
			if got != i {
				t.Fatalf("nstates=%d i=%d Index=%d", nstates, i, got)
			}
		}
	}
	bb := BusyBeaver2()
	got := TMFromIndex(bb.Index(), 2)
	if got.Delta[0][0] != bb.Delta[0][0] || got.Delta[1][1] != bb.Delta[1][1] {
		t.Fatal("BB(2) index roundtrip")
	}
}

func TestTapeOutput(t *testing.T) {
	_, _, cfg := writeOneHalt().Run(8)
	if FormatBits(cfg.Output()) != "1" {
		t.Fatalf("write-one output %q", FormatBits(cfg.Output()))
	}
	_, _, bb := BusyBeaver2().Run(32)
	if FormatBits(bb.Output()) != "1111" {
		t.Fatalf("BB(2) output %q", FormatBits(bb.Output()))
	}
	if !bitsEq(bb.Output(), ParseBitString("1111")) {
		t.Fatal("parse")
	}
}

func TestAnalogOutputMatchesDiscrete(t *testing.T) {
	tm := BusyBeaver2()
	_, _, cfg := tm.Run(32)
	a := NewAnalogTM(256, tm)
	a.Run(32)
	if !bitsEq(a.Output(16), cfg.Output()) {
		t.Fatalf("analog %q discrete %q", FormatBits(a.Output(16)), FormatBits(cfg.Output()))
	}
}

func TestKEmptyAndOne(t *testing.T) {
	empty := KComplexity(nil, 1, 32, 128)
	if empty.ByPrint {
		t.Fatalf("empty string should have a 1-state TM: %s", empty)
	}
	if empty.K > empty.PrintBound {
		t.Fatal("K exceeds print bound")
	}
	if !empty.AnalogOK {
		t.Fatal("analog witness failed for ε")
	}
	one := KComplexity(ParseBitString("1"), 1, 32, 128)
	if one.ByPrint {
		t.Fatalf("\"1\" should have a 1-state TM: %s", one)
	}
	if !one.AnalogOK {
		t.Fatal("analog witness failed for 1")
	}
}

func TestKIncompressibleUsesPrint(t *testing.T) {
	x := ParseBitString("1011001110001010")
	r := KComplexity(x, 1, 32, 128)
	if !r.ByPrint {
		t.Fatalf("1-state TMs cannot emit a 16-bit string, got %s", r)
	}
	if r.K != len(x)+1 {
		t.Fatalf("print bound %d got %d", len(x)+1, r.K)
	}
	if r.Queries != NumTMs(1) {
		t.Fatalf("queries %d", r.Queries)
	}
}

func TestK1111TwoState(t *testing.T) {
	x := ParseBitString("1111")
	r := KComplexity(x, 2, 32, 256)
	if r.TMIndex < 0 {
		t.Fatalf("no 2-state TM for 1111: %s", r)
	}
	h, _, cfg := TMFromIndex(r.TMIndex, 2).Run(32)
	if !h || FormatBits(cfg.Output()) != "1111" {
		t.Fatalf("witness TM #%d output %q", r.TMIndex, FormatBits(cfg.Output()))
	}
	if !r.AnalogOK {
		t.Fatal("analog tape of witness does not match")
	}
	if r.PlainC != BitLen(r.TMIndex) {
		t.Fatal("plain C")
	}
	if r.PrefixK != PrefixProgramLen(r.TMIndex) {
		t.Fatal("prefix K")
	}
	if r.K > r.PrintBound {
		t.Fatal("K > print")
	}
}

func TestKNeverExceedsPrint(t *testing.T) {
	for _, s := range []string{"", "0", "1", "11", "101", "1111"} {
		r := KComplexity(ParseBitString(s), 1, 16, 64)
		if r.K > r.PrintBound {
			t.Fatalf("%s: K=%d print=%d", s, r.K, r.PrintBound)
		}
	}
}

func TestPrefixProgramLen(t *testing.T) {
	if PrefixProgramLen(0) != 1 {
		t.Fatal("i=0")
	}
	if PrefixProgramLen(1) != 3 {
		t.Fatal("i=1")
	}
	if PrefixProgramLen(2) != 5 {
		t.Fatal("i=2")
	}
	if BitLen(0) != 1 || BitLen(1) != 1 || BitLen(2) != 2 || BitLen(7) != 3 {
		t.Fatal("BitLen")
	}
}

func TestParseFormatBits(t *testing.T) {
	if FormatBits(ParseBitString(" 101 \n")) != "101" {
		t.Fatal("roundtrip")
	}
}
