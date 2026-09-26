// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import (
	"bytes"
	"testing"
)

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

func TestSpanProgramCompressesBlankBits(t *testing.T) {
	cfg := LMConfig{Window: 4, MaxBits: 12, Bound: 64, Prec: 64, Seed: 1, ValidFrac: 0.1, Epochs: 1, Rate: 0.1}
	cfg.norm()
	text := bytes.Repeat([]byte{0}, 32)
	m, _, err := TrainSpanLM(text, cfg)
	if err != nil {
		t.Fatal(err)
	}
	w := bytes.Repeat([]byte{0}, 4)
	prog, how, k := m.WindowProgram(w)
	res := RunU(prog, m.prove)
	if res.Status != UHalt || !bitsEq(res.Out, windowBits(w)) {
		t.Fatalf("how %s status %d out %d", how, res.Status, len(res.Out))
	}
	if k >= len(ListingProgram(windowBits(w))) {
		t.Fatalf("blank window did not compress: %s K=%d listing=%d", how, k, len(ListingProgram(windowBits(w))))
	}
}

func TestSpanByteRunSpaces(t *testing.T) {
	text := bytes.Repeat([]byte{' ', '\n'}, 32)
	m, _, err := TrainSpanLM(text, LMConfig{
		Window: 4, Epochs: 1, Rate: 0.1, ValidFrac: 0.1,
		MaxBits: 8, Bound: 64, Prec: 64, Seed: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, ex := range [][]byte{
		{' ', ' ', ' ', ' '},
		{'\n', '\n', '\n', '\n'},
	} {
		prog, how, k := m.WindowProgram(ex)
		bits := windowBits(ex)
		res := RunU(prog, m.prove)
		if res.Status != UHalt || !bitsEq(res.Out, bits) {
			t.Fatalf("%q how %s status %d", ex, how, res.Status)
		}
		if how != "repeat" || k >= len(ListingProgram(bits)) {
			t.Fatalf("%q how %s K=%d listing %d", ex, how, k, len(ListingProgram(bits)))
		}
	}
}

func TestSpanVsOneHotLearnsCycle(t *testing.T) {
	var text []byte
	for i := 0; i < 3000; i++ {
		text = append(text, 'a', 'b', 'c', 'd')
	}
	cfg := LMConfig{
		Window: 2, Epochs: 3, Rate: 0.4, ValidFrac: 0.1,
		MaxBits: 8, Bound: 64, Prec: 64, Seed: 1,
	}
	_, span, hot, err := CompareContextModels(text, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if span.Kind != "span" || hot.Kind != "onehot" {
		t.Fatalf("%s %s", span.Kind, hot.Kind)
	}
	if span.ValidAcc < 0.9 || hot.ValidAcc < 0.9 {
		t.Fatalf("span acc %.3f ppl %.2f  onehot acc %.3f ppl %.2f",
			span.ValidAcc, span.ValidPPL, hot.ValidAcc, hot.ValidPPL)
	}
}

func TestSpanVsOneHotShakespeareSlice(t *testing.T) {
	text, err := LoadCorpus("pg100.txt")
	if err != nil {
		t.Skip(err)
	}
	slice := text[:40000]
	_, span, hot, err := CompareContextModels(slice, LMConfig{
		Window: 4, Epochs: 1, Rate: 0.1, ValidFrac: 0.1,
		MaxBits: 8, Bound: 64, Prec: 64, Seed: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if span.ValidN < 1000 || hot.ValidN != span.ValidN {
		t.Fatalf("span %+v onehot %+v", span, hot)
	}
	if span.ValidPPL >= 80 || hot.ValidPPL >= 80 {
		t.Fatalf("span ppl %.2f onehot ppl %.2f", span.ValidPPL, hot.ValidPPL)
	}
}
