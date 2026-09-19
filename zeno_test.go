// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import "testing"

func TestZenoLimitHalt(t *testing.T) {
	z := NewZeno(256, WriteOneHalt())
	lim := z.Limit(4)
	if lim.Kind != LimitHalt {
		t.Fatalf("kind %s", lim.Kind)
	}
	if !lim.Halted || !lim.Frozen {
		t.Fatal("should be halted and frozen")
	}
	if lim.Steps != 1 {
		t.Fatalf("steps %d", lim.Steps)
	}
	if FormatBits(lim.Output) != "1" {
		t.Fatalf("out %q", FormatBits(lim.Output))
	}
}

func TestZenoLimitBB2(t *testing.T) {
	lim := NewZeno(256, BusyBeaver2()).Limit(4)
	if lim.Kind != LimitHalt {
		t.Fatalf("kind %s", lim.Kind)
	}
	if lim.Steps != 6 {
		t.Fatalf("steps %d", lim.Steps)
	}
	if FormatBits(lim.Output) != "1111" {
		t.Fatalf("out %q", FormatBits(lim.Output))
	}
}

func TestZenoLimitCauchy(t *testing.T) {
	z := NewZeno(256, IdleBlank())
	lim := z.Limit(5)
	if lim.Kind != LimitCauchy {
		t.Fatalf("kind %s", lim.Kind)
	}
	if lim.Halted {
		t.Fatal("idle should not halt")
	}
	if !lim.Frozen {
		t.Fatal("visible tape should freeze")
	}
	if len(lim.Output) != 0 {
		t.Fatalf("out %q", FormatBits(lim.Output))
	}
	one := FromInt(256, 1)
	if !lim.Time.ApproxEq(one, uint(lim.N)) {
		t.Fatalf("time %s", lim.Time)
	}
}

func TestZenoLimitDiverge(t *testing.T) {
	z := NewZeno(256, PaintRight())
	lim := z.Limit(5)
	if lim.Kind != LimitDiverge {
		t.Fatalf("kind %s", lim.Kind)
	}
	if lim.Halted || lim.Frozen {
		t.Fatal("paint should keep moving")
	}
	if lim.Steps != 1<<5 {
		t.Fatalf("steps %d", lim.Steps)
	}
	if len(lim.Output) != 32 {
		t.Fatalf("painted %d ones", len(lim.Output))
	}
}

func TestZenoLimitTimeBelowOne(t *testing.T) {
	lim := NewZeno(256, PaintRight()).Limit(6)
	if lim.Time.Cmp(FromInt(256, 1)) >= 0 {
		t.Fatal("time should stay below 1")
	}
}
