// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import (
	"bytes"
	"math"
	"math/rand/v2"
	"os"
)

const (
	// DefaultLMWindow is how many previous bytes the model reads.
	DefaultLMWindow = 4
	// DefaultLMEpochs is one pass over the training split.
	DefaultLMEpochs = 1
	// DefaultLMRate is the SGD step size.
	DefaultLMRate = 0.1
	// DefaultLMValid is the fraction of the corpus held out at the end.
	DefaultLMValid = 0.1
)

// LangModel predicts the next byte from the machine-code embeddings of
// the previous Window bytes. Each byte's embedding is the bit string of
// the shortest U program found for that byte.
type LangModel struct {
	Window   int
	Dim      int // program width
	InDim    int // Window * Dim
	ClassOf  [256]int
	Alphabet []byte
	Program  [256][]bool
	How      [256]string
	Emb      []float32 // [256 * Dim]
	W        []float32 // [nClass * InDim]
	B        []float32 // [nClass]
}

// LMReport is training and validation loss for one fit.
type LMReport struct {
	TrainN   int
	ValidN   int
	TrainNLL float64 // nats per predicted byte
	ValidNLL float64
	TrainPPL float64
	ValidPPL float64
	TrainAcc float64
	ValidAcc float64
	Dim      int
	Window   int
	Epochs   int
}

// LMConfig selects the corpus split and the K search used for embeddings.
type LMConfig struct {
	Window    int
	Epochs    int
	Rate      float32
	ValidFrac float64
	MaxBits   int
	Bound     int
	Prec      uint
	Seed      uint64
}

func (c *LMConfig) norm() {
	if c.Window < 1 {
		c.Window = DefaultLMWindow
	}
	if c.Epochs < 1 {
		c.Epochs = DefaultLMEpochs
	}
	if c.Rate <= 0 {
		c.Rate = DefaultLMRate
	}
	if c.ValidFrac <= 0 || c.ValidFrac >= 1 {
		c.ValidFrac = DefaultLMValid
	}
	if c.MaxBits < 1 {
		c.MaxBits = DefaultUMaxBits
	}
	if c.Bound <= 0 {
		c.Bound = DefaultUBound
	}
	if c.Prec == 0 {
		c.Prec = 64
	}
	if c.Seed == 0 {
		c.Seed = 1
	}
}

// GutenbergBody returns the text between the Project Gutenberg start
// and end markers. Text without those markers is returned unchanged.
func GutenbergBody(text []byte) []byte {
	const (
		start = "*** START OF THE PROJECT GUTENBERG EBOOK"
		end   = "*** END OF THE PROJECT GUTENBERG EBOOK"
	)
	i := bytes.Index(text, []byte(start))
	if i < 0 {
		return text
	}
	nl := bytes.IndexByte(text[i:], '\n')
	if nl < 0 {
		return text
	}
	body := text[i+nl+1:]
	if j := bytes.Index(body, []byte(end)); j >= 0 {
		body = body[:j]
	}
	return body
}

// LoadCorpus reads path and returns the Gutenberg body when the file
// has one.
func LoadCorpus(path string) ([]byte, error) {
	text, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return GutenbergBody(text), nil
}

func byteBits(b byte) []bool {
	out := make([]bool, 8)
	for i := 0; i < 8; i++ {
		out[i] = b&(1<<(7-i)) != 0
	}
	return out
}

func programVec(p []bool, dim int) []float32 {
	v := make([]float32, dim)
	n := len(p)
	if n > dim {
		n = dim
	}
	for i := 0; i < n; i++ {
		if p[i] {
			v[i] = 1
		}
	}
	return v
}

// byteMachineCodes is the K_U program for every byte, one shared oracle.
func byteMachineCodes(maxBits, bound int, prec uint) (prog [256][]bool, how [256]string, dim int) {
	s := &uSearcher{maxBits: maxBits, bound: bound}
	s.init(prec)
	for b := 0; b < 256; b++ {
		bits := byteBits(byte(b))
		r := &KResult{Bits: bits, K: -1, Bound: bound, MaxBits: maxBits}
		considerU(bits, ListingProgram(bits), bound, "listing", r)
		if bit, k, ok := unaryRun(bits); ok {
			considerU(bits, RepeatProgram(bit, k), bound, "repeat", r)
		}
		s.improve(bits, r)
		prog[b] = append([]bool(nil), r.Program...)
		how[b] = r.How
		if len(r.Program) > dim {
			dim = len(r.Program)
		}
	}
	return prog, how, dim
}

// TrainLanguageModel fits a next-byte model on text. The last ValidFrac
// of the bytes are validation and are not used for the weight update.
// Each input coordinate is a bit of the U program for that byte.
func TrainLanguageModel(text []byte, cfg LMConfig) (*LangModel, LMReport, error) {
	cfg.norm()
	var rep LMReport
	if len(text) < cfg.Window+2 {
		return nil, rep, errCorpusShort
	}
	nValid := int(float64(len(text)) * cfg.ValidFrac)
	if nValid < cfg.Window+1 {
		nValid = cfg.Window + 1
	}
	if nValid >= len(text)-cfg.Window {
		return nil, rep, errCorpusShort
	}
	train := text[:len(text)-nValid]
	valid := text[len(text)-nValid:]

	prog, how, dim := byteMachineCodes(cfg.MaxBits, cfg.Bound, cfg.Prec)
	m := &LangModel{
		Window:  cfg.Window,
		Dim:     dim,
		InDim:   cfg.Window * dim,
		Program: prog,
		How:     how,
		Emb:     make([]float32, 256*dim),
	}
	for b := 0; b < 256; b++ {
		m.ClassOf[b] = -1
		copy(m.Emb[b*dim:(b+1)*dim], programVec(prog[b], dim))
	}
	for _, b := range train {
		if m.ClassOf[b] >= 0 {
			continue
		}
		m.ClassOf[b] = len(m.Alphabet)
		m.Alphabet = append(m.Alphabet, b)
	}
	nClass := len(m.Alphabet)
	m.W = make([]float32, nClass*m.InDim)
	m.B = make([]float32, nClass)
	rng := rand.New(rand.NewPCG(cfg.Seed, cfg.Seed^0x9e3779b97f4a7c15))
	scale := float32(0.02)
	for i := range m.W {
		m.W[i] = (rng.Float32()*2 - 1) * scale
	}

	var trainNLL float64
	var trainCorrect, trainScored, trainTotal int
	for ep := 0; ep < cfg.Epochs; ep++ {
		nll, correct, scored, total := m.pass(train, cfg.Rate, true)
		trainNLL += nll
		trainCorrect += correct
		trainScored += scored
		trainTotal += total
	}
	validNLL, validCorrect, validScored, validTotal := m.pass(valid, 0, false)
	if trainScored == 0 || validScored == 0 {
		return nil, rep, errCorpusShort
	}
	rep = LMReport{
		TrainN:   trainTotal,
		ValidN:   validTotal,
		TrainNLL: trainNLL / float64(trainScored),
		ValidNLL: validNLL / float64(validScored),
		TrainPPL: math.Exp(trainNLL / float64(trainScored)),
		ValidPPL: math.Exp(validNLL / float64(validScored)),
		TrainAcc: float64(trainCorrect) / float64(trainTotal),
		ValidAcc: float64(validCorrect) / float64(validTotal),
		Dim:      dim,
		Window:   cfg.Window,
		Epochs:   cfg.Epochs,
	}
	return m, rep, nil
}

// pass walks text one byte at a time. update writes the SGD step at rate.
// nll is the total nats over scored bytes. total counts every predicted
// byte, including targets absent from the training alphabet.
func (m *LangModel) pass(text []byte, rate float32, update bool) (nll float64, correct, scored, total int) {
	if len(text) <= m.Window {
		return 0, 0, 0, 0
	}
	nClass := len(m.Alphabet)
	ctx := make([]float32, m.InDim)
	logits := make([]float32, nClass)
	probs := make([]float32, nClass)
	for i := m.Window; i < len(text); i++ {
		total++
		class, ok := m.class(text[i])
		if !ok {
			continue
		}
		scored++
		m.fillCtx(ctx, text[i-m.Window:i])
		m.forward(ctx, logits)
		loss, hit := softmaxStep(logits, probs, class)
		nll += float64(loss)
		if hit {
			correct++
		}
		if update {
			m.update(ctx, probs, class, rate)
		}
	}
	return nll, correct, scored, total
}

func (m *LangModel) class(b byte) (int, bool) {
	c := m.ClassOf[b]
	return c, c >= 0
}

func (m *LangModel) fillCtx(ctx []float32, window []byte) {
	dim := m.Dim
	for t, b := range window {
		copy(ctx[t*dim:(t+1)*dim], m.Emb[int(b)*dim:int(b)*dim+dim])
	}
}

func (m *LangModel) forward(ctx, logits []float32) {
	inDim := m.InDim
	for c := range logits {
		row := m.W[c*inDim : (c+1)*inDim]
		var s float32
		for d, x := range ctx {
			s += x * row[d]
		}
		logits[c] = s + m.B[c]
	}
}

func (m *LangModel) update(ctx, probs []float32, class int, rate float32) {
	inDim := m.InDim
	for c, p := range probs {
		g := p
		if c == class {
			g -= 1
		}
		g *= rate
		row := m.W[c*inDim : (c+1)*inDim]
		for d, x := range ctx {
			row[d] -= g * x
		}
		m.B[c] -= g
	}
}

// softmaxStep returns the nats of the true class and whether it was argmax.
func softmaxStep(logits, probs []float32, class int) (nll float32, hit bool) {
	max := logits[0]
	for _, z := range logits[1:] {
		if z > max {
			max = z
		}
	}
	var sum float32
	for i, z := range logits {
		e := float32(math.Exp(float64(z - max)))
		probs[i] = e
		sum += e
	}
	inv := 1 / sum
	best, arg := probs[0], 0
	for i := range probs {
		probs[i] *= inv
		if probs[i] > best {
			best = probs[i]
			arg = i
		}
	}
	p := probs[class]
	if p < 1e-12 {
		p = 1e-12
	}
	return -float32(math.Log(float64(p))), arg == class
}

type lmError string

func (e lmError) Error() string { return string(e) }

const errCorpusShort lmError = "corpus shorter than the context window"
