// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import (
	"testing"
)

func TestBusyBeaver2(t *testing.T) {
	tm := BusyBeaver2()
	halted, steps, cfg := tm.Run(32)
	if !halted {
		t.Fatal("BB(2) should halt")
	}
	if steps != 6 {
		t.Fatalf("BB(2) steps = %d want 6", steps)
	}
	if cfg.Ones() != 4 {
		t.Fatalf("BB(2) ones = %d want 4", cfg.Ones())
	}
}

func TestAnalogMatchesDiscrete(t *testing.T) {
	tms := []*TM{
		BusyBeaver2(),
		paintRight(),
		writeOneHalt(),
	}
	for i, tm := range tms {
		d := &Config{State: tm.Start, Tape: make(map[int]int)}
		a := NewAnalogTM(512, tm)
		for step := 0; step < 16; step++ {
			if tm.Step(d) != a.Step() {
				t.Fatalf("tm %d step %d: running mismatch (d state=%d a state=%d)", i, step, d.State, a.State)
			}
			if d.State != a.State {
				t.Fatalf("tm %d step %d: state d=%d a=%d", i, step, d.State, a.State)
			}
			for off := -step - 1; off <= step+1; off++ {
				want := d.Tape[d.Head+off]
				got := a.TapeBit(off)
				if got != want {
					t.Fatalf("tm %d step %d off %d: analog %d discrete %d (head %d)",
						i, step, off, got, want, d.Head)
				}
			}
			if d.State < 0 {
				break
			}
		}
	}
}

func paintRight() *TM {
	return &TM{
		States: 1,
		Start:  0,
		Delta: [][2]Transition{
			{{Write: 1, Move: 1, Next: 0}, {Write: 1, Move: 1, Next: 0}},
		},
	}
}

func writeOneHalt() *TM {
	return &TM{
		States: 1,
		Start:  0,
		Delta: [][2]Transition{
			{{Write: 1, Move: 1, Next: Halt}, {Write: 1, Move: 1, Next: Halt}},
		},
	}
}

func TestZenoHaltingFreezes(t *testing.T) {
	z := NewZeno(256, writeOneHalt())
	z.Run(32)
	if !z.Halted() {
		t.Fatal("should halt")
	}
	if z.Steps != 1 {
		t.Fatalf("steps %d want 1", z.Steps)
	}
	left := z.Analog.Left.Copy()
	right := z.Analog.Right.Copy()
	z.Analog.Step()
	if z.Analog.Left.Cmp(left) != 0 || z.Analog.Right.Cmp(right) != 0 {
		t.Fatal("halted analog tape should freeze")
	}
}

func TestZenoNonHaltingKeepsWriting(t *testing.T) {
	z := NewZeno(512, paintRight())
	z.Run(8)
	if z.Halted() {
		t.Fatal("paint should not halt")
	}
	if z.Steps != 8 {
		t.Fatalf("steps %d", z.Steps)
	}
	ones := 0
	for i := 1; i <= 8; i++ {
		if z.Analog.TapeBit(-i) == 1 {
			ones++
		}
	}
	if ones != 8 {
		t.Fatalf("wrote %d ones want 8 (left of head)", ones)
	}
}

func TestZenoTimeApproachesOne(t *testing.T) {
	z := NewZeno(256, paintRight())
	z.Run(40)
	one := FromInt(256, 1)
	if !z.Time.ApproxEq(one, 40) {
		t.Fatalf("time after 40 steps = %s, should be 1-2^{-40}", z.Time)
	}
	if z.Time.Cmp(one) >= 0 {
		t.Fatal("partial geometric sum should stay below 1")
	}
}

func TestTMFromIndexRoundtripSize(t *testing.T) {
	if NumTMs(1) != 64 {
		t.Fatalf("1-state TMs = %d want 64", NumTMs(1))
	}
	if NumTMs(2) != 20736 {
		t.Fatalf("2-state TMs = %d want 20736", NumTMs(2))
	}
	seen := map[string]int{}
	for i := 0; i < NumTMs(1); i++ {
		tm := TMFromIndex(i, 1)
		key := transKey(tm)
		if j, ok := seen[key]; ok {
			t.Fatalf("duplicate encoding %d and %d", j, i)
		}
		seen[key] = i
	}
	if len(seen) != 64 {
		t.Fatalf("enumerated %d unique 1-state TMs", len(seen))
	}
}

func transKey(tm *TM) string {
	s := make([]byte, 0, 32)
	for st := 0; st < tm.States; st++ {
		for r := 0; r < 2; r++ {
			tr := tm.Delta[st][r]
			s = append(s, byte('0'+tr.Write), byte('0'+tr.Move+1), byte('A'+tr.Next+1))
		}
	}
	return string(s)
}
