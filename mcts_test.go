// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import (
	"bytes"
	"math"
	"math/rand/v2"
	"testing"
)

func TestMCTSOracleAgreesWithRunU(t *testing.T) {
	o := NewMCTSOracle(48, 256, 1)
	if _, _, _ = o.Search(nil, 40); o.Visits() != 40 {
		t.Fatalf("visits %d", o.Visits())
	}
	long := 0
	halted := 0
	for prog, halt := range o.halt {
		bits := ParseBitString(prog)
		if halt {
			halted++
			if !ValidUProgram(bits, o.Bound) {
				t.Fatalf("oracle says halt, U does not: %s", prog)
			}
		}
		if len(bits) > DefaultUMaxBits {
			long++
		}
	}
	if halted == 0 || long == 0 {
		t.Fatalf("halts %d programs longer than %d bits: %d", halted, DefaultUMaxBits, long)
	}
	packed := o.OracleBits()
	if len(packed) != len(o.halt) {
		t.Fatalf("packed %d evals %d", len(packed), len(o.halt))
	}
}

func TestMCTSWitnessExceedsBruteForce(t *testing.T) {
	raw := bytes.Repeat([]byte{' '}, 16)
	target := windowBits(raw)
	o := NewMCTSOracle(64, 32*16+64, 1)
	prog, k, how := o.Witness(target)
	listing := 3 * (len(target) + 1)
	if k <= DefaultUMaxBits || k >= listing {
		t.Fatalf("K=%d how=%s listing=%d |p|=%d", k, how, listing, len(prog))
	}
	halt, known := o.Halts(prog)
	if !known || !halt {
		t.Fatalf("known %v halt %v", known, halt)
	}
	res := RunU(prog, o.Bound)
	if res.Status != UHalt || !bitsEq(res.Out, target) {
		t.Fatalf("status %d out %d", res.Status, len(res.Out))
	}
}

func TestMCTSWindowSeesPastSixteen(t *testing.T) {
	shared := []byte("abcdefghijklmnop")
	leftZ := []byte("AAAAAAAA")
	leftY := []byte("BBBBBBBB")
	var text []byte
	for i := 0; i < 40; i++ {
		text = append(text, leftZ...)
		text = append(text, shared...)
		text = append(text, 'Z')
		text = append(text, leftY...)
		text = append(text, shared...)
		text = append(text, 'Y')
	}
	cfg := LMConfig{Window: 24, ValidFrac: 0.1, MaxBits: 8, Bound: 64, Prec: 64, Seed: 1, Sims: 16}
	if _, _, err := TrainDeltaKLM(text, cfg); err != errDKWindow {
		t.Fatalf("delta window: %v", err)
	}
	m, rep, _, err := TrainMCTSLM(text, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Kind != "mcts" || rep.Window != 24 {
		t.Fatalf("%+v", rep)
	}
	prompt := append(append([]byte{}, leftZ...), shared...)
	prog, k, how := m.ContextProgram(prompt)
	if k <= DefaultUMaxBits || len(prog) != k {
		t.Fatalf("witness K=%d how=%s |p|=%d", k, how, len(prog))
	}
	alphabet, p := m.Distribution(prompt)
	_, back := m.back.Distribution(prompt)
	pz, bz := -1.0, -1.0
	for i, b := range alphabet {
		if b == 'Z' {
			pz = p[i]
			bz = back[i]
		}
	}
	if pz < 0.8 || pz <= bz {
		t.Fatalf("P(Z|24)=%g P(Z|mix)=%g", pz, bz)
	}
	if got := m.Greedy(prompt, 1); string(got) != string(prompt)+"Z" {
		t.Fatalf("greedy %q", got)
	}
	var sum float64
	for _, x := range p {
		sum += x
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Fatalf("sum %g", sum)
	}
}

func TestMCTSDecodeFollowsLongCycle(t *testing.T) {
	text := bytes.Repeat([]byte("    .\n"), 80)
	m, _, _, err := TrainMCTSLM(text, LMConfig{
		Window: 24, ValidFrac: 0.1, MaxBits: 8, Bound: 64, Prec: 64, Seed: 1, Sims: 8,
	})
	if err != nil {
		t.Fatal(err)
	}
	prompt := []byte("    ")
	got := m.Generate(prompt, 6, 1)
	want := m.Greedy(prompt, 6)
	if string(got) != "    .\n    " || string(got) != string(want) {
		t.Fatalf("mcts %q greedy %q", got, want)
	}
	if again := m.Generate(prompt, 6, 1); string(again) != string(got) {
		t.Fatalf("redrew %q then %q", got, again)
	}
	long := m.Generate(prompt, 18, 1)
	if string(long) != "    .\n    .\n    .\n    " {
		t.Fatalf("cycle %q", long)
	}
	sample := m.extend(prompt, 6, func(alphabet []byte, p []float64) byte {
		return pickByte(alphabet, p, rand.New(rand.NewPCG(9, 1)).Float64())
	})
	if seqLog(m, got)+1e-9 < seqLog(m, sample) {
		t.Fatalf("mcts logp %g sample logp %g (%q)", seqLog(m, got), seqLog(m, sample), sample)
	}
}

func TestMCTSDecodeBreaksRepeatedPhrase(t *testing.T) {
	phrase := []byte("and with my lord, ")
	alt := []byte("or else depart.\n")
	var text []byte
	for i := 0; i < 40; i++ {
		text = append(text, phrase...)
		if i%3 == 2 {
			text = append(text, alt...)
		}
	}
	m, _, _, err := TrainMCTSLM(text, LMConfig{
		Window: 24, ValidFrac: 0.1, MaxBits: 8, Bound: 64, Prec: 64, Seed: 1, Sims: 8,
	})
	if err != nil {
		t.Fatal(err)
	}
	got := m.Generate(phrase, 80, 1)
	rest := got[len(phrase):]
	if !bytes.HasPrefix(rest, alt[:len("or else")]) {
		t.Fatalf("echoed %q", got)
	}
	double := append(append([]byte{}, phrase...), phrase...)
	if bytes.Contains(got, double) {
		t.Fatalf("repeated %q", got)
	}
	againPrompt := append(append([]byte{}, phrase...), phrase...)
	continued := m.Generate(againPrompt, 8, 1)
	if continued[len(againPrompt)] == phrase[0] {
		t.Fatalf("extended the copy %q", continued)
	}
	if again := m.Generate(phrase, 80, 1); string(again) != string(got) {
		t.Fatalf("redrew %q then %q", got, again)
	}
}

func TestLoopedPlayoutSkipsShortCycle(t *testing.T) {
	phrase := []byte("and with my lord, ")
	if loopedPlayout(phrase, phrase[:mctsRepeatLen-1]) {
		t.Fatal("short echo")
	}
	if !loopedPlayout(phrase, phrase[:mctsRepeatLen]) {
		t.Fatal("missed a new copy")
	}
	cycle := bytes.Repeat([]byte("    .\n"), 4)
	if loopedPlayout(cycle, []byte("    .\n")) {
		t.Fatal("penalized a short cycle")
	}
	stuck := bytes.Repeat(phrase, 2)
	if !loopedPlayout(stuck, phrase[:mctsRepeatLen]) {
		t.Fatal("missed a continued copy")
	}
	long := []byte(" not with my lord, and with you shall have no more they are not with a soldiers, and with all have")
	if !loopedPlayout(long, long[:mctsRepeatLen]) {
		t.Fatalf("missed period %d", len(long))
	}
}

func seqLog(m *MCTSLM, text []byte) float64 {
	p := make([]float64, len(m.Alphabet))
	var s float64
	for i := 0; i < len(text); i++ {
		m.probs(text[:i], p)
		pb := 1e-15
		for j, b := range m.Alphabet {
			if b == text[i] && j < len(p) && p[j] > pb {
				pb = p[j]
			}
		}
		s += math.Log(pb)
	}
	return s
}
