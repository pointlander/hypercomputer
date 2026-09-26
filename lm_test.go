// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import "testing"

func TestByteMachineCodeRoundtrip(t *testing.T) {
	prog, _, dim := byteMachineCodes(12, 256, 64)
	if dim < 8 {
		t.Fatalf("dim %d", dim)
	}
	seen := map[string]byte{}
	for b := 0; b < 256; b++ {
		res := RunU(prog[b], 256)
		if res.Status != UHalt || !bitsEq(res.Out, byteBits(byte(b))) {
			t.Fatalf("byte %d status %d out %d", b, res.Status, len(res.Out))
		}
		key := FormatBits(prog[b])
		if prev, ok := seen[key]; ok {
			t.Fatalf("bytes %d and %d share program %s", prev, b, key)
		}
		seen[key] = byte(b)
	}
}

func TestLMLearnsAlternation(t *testing.T) {
	var text []byte
	for i := 0; i < 4000; i++ {
		text = append(text, 'a', 'b')
	}
	m, rep, err := TrainLanguageModel(text, LMConfig{
		Window: 1, Epochs: 4, Rate: 0.5, ValidFrac: 0.1,
		MaxBits: 8, Bound: 64, Prec: 64, Seed: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if m.Dim < 1 || len(m.Program['a']) == 0 {
		t.Fatal("missing embedding")
	}
	if rep.ValidAcc < 0.9 {
		t.Fatalf("valid acc %.3f ppl %.2f", rep.ValidAcc, rep.ValidPPL)
	}
	if rep.ValidPPL >= 2 {
		t.Fatalf("valid ppl %.3f", rep.ValidPPL)
	}
}

func TestLMShakespeareSlice(t *testing.T) {
	text, err := LoadCorpus("pg100.txt")
	if err != nil {
		t.Skip(err)
	}
	if len(text) < 100000 {
		t.Fatalf("body %d", len(text))
	}
	// A prefix long enough to learn letter frequencies, short enough for the test.
	slice := text[:60000]
	_, rep, err := TrainLanguageModel(slice, LMConfig{
		Window: 4, Epochs: 1, Rate: 0.1, ValidFrac: 0.1,
		MaxBits: 8, Bound: 64, Prec: 64, Seed: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if rep.ValidN < 1000 || rep.TrainN < 1000 {
		t.Fatalf("%+v", rep)
	}
	// Uniform over ~80 symbols is perplexity ~80 and accuracy ~0.01.
	if rep.ValidPPL >= 80 || rep.ValidAcc < 0.05 {
		t.Fatalf("valid ppl %.2f acc %.3f", rep.ValidPPL, rep.ValidAcc)
	}
}
