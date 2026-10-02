// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import (
	"bytes"
	"math"
	"math/rand/v2"
	"os"
	"sort"
	"strings"
	"sync"
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
	Kind     string
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
	// Sims is the MCTS halt-oracle budget. Zero selects the default.
	Sims int
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
	train, valid, err := splitCorpus(text, cfg)
	if err != nil {
		return nil, rep, err
	}

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
		Kind:     "byte",
	}
	return m, rep, nil
}

func splitCorpus(text []byte, cfg LMConfig) (train, valid []byte, err error) {
	if len(text) < cfg.Window+2 {
		return nil, nil, errCorpusShort
	}
	nValid := int(float64(len(text)) * cfg.ValidFrac)
	if nValid < cfg.Window+1 {
		nValid = cfg.Window + 1
	}
	if nValid >= len(text)-cfg.Window {
		return nil, nil, errCorpusShort
	}
	cut := len(text) - nValid
	return text[:cut], text[cut:], nil
}

func howIndex(h string) int {
	switch h {
	case "repeat":
		return 1
	case "divide":
		return 2
	case "search":
		return 3
	default:
		return 0
	}
}

func windowBits(w []byte) []bool {
	out := make([]bool, 0, 8*len(w))
	for _, b := range w {
		out = append(out, byteBits(b)...)
	}
	return out
}

func opcodeRates(p []bool) [8]float32 {
	var c [8]float32
	in := 0
	var n float32
	for in < len(p) {
		inst, ok := uFetchInst(p, &in)
		if !ok {
			break
		}
		if inst.op >= 0 && inst.op < 8 {
			c[inst.op]++
			n++
		}
	}
	if n > 0 {
		for i := range c {
			c[i] /= n
		}
	}
	return c
}

// SpanLM reads one U program for the whole context window.
// The coordinates are the program bits, its length, how it was built
// (listing, repeat, divide, search), and its opcode rates.
type SpanLM struct {
	Window   int
	ListLen  int
	Dim      int
	ClassOf  [256]int
	Alphabet []byte
	W        []float32
	B        []float32
	search   *uSearcher
	kcache   map[string]*KResult
	vcache   map[string][]float32
	prove    int
}

// WindowProgram is the U program for the bit string of w.
func (m *SpanLM) WindowProgram(w []byte) (prog []bool, how string, k int) {
	bits := windowBits(w)
	r := m.search.divide(bits, DefaultKLeaf, m.prove, m.kcache)
	return r.Program, r.How, r.K
}

func (m *SpanLM) embed(w []byte) []float32 {
	key := string(w)
	if v, ok := m.vcache[key]; ok {
		return v
	}
	prog, how, k := m.WindowProgram(w)
	v := make([]float32, m.Dim)
	n := len(prog)
	if n > m.ListLen {
		n = m.ListLen
	}
	for i := 0; i < n; i++ {
		if prog[i] {
			v[i] = 1
		}
	}
	v[m.ListLen] = float32(k) / float32(m.ListLen)
	v[m.ListLen+1+howIndex(how)] = 1
	rates := opcodeRates(prog)
	for i, r := range rates {
		v[m.ListLen+5+i] = r
	}
	m.vcache[key] = v
	return v
}

func (m *SpanLM) pass(text []byte, rate float32, update bool) (nll float64, correct, scored, total int) {
	if len(text) <= m.Window {
		return 0, 0, 0, 0
	}
	nClass := len(m.Alphabet)
	logits := make([]float32, nClass)
	probs := make([]float32, nClass)
	for i := m.Window; i < len(text); i++ {
		total++
		class := m.ClassOf[text[i]]
		if class < 0 {
			continue
		}
		scored++
		ctx := m.embed(text[i-m.Window : i])
		m.forward(ctx, logits)
		loss, hit := softmaxStep(logits, probs, class)
		nll += float64(loss)
		if hit {
			correct++
		}
		if update {
			m.step(ctx, probs, class, rate)
		}
	}
	return nll, correct, scored, total
}

func (m *SpanLM) forward(ctx, logits []float32) {
	dim := m.Dim
	for c := range logits {
		row := m.W[c*dim : (c+1)*dim]
		var s float32
		for d, x := range ctx {
			s += x * row[d]
		}
		logits[c] = s + m.B[c]
	}
}

func (m *SpanLM) step(ctx, probs []float32, class int, rate float32) {
	dim := m.Dim
	for c, p := range probs {
		g := p
		if c == class {
			g -= 1
		}
		g *= rate
		row := m.W[c*dim : (c+1)*dim]
		for d, x := range ctx {
			row[d] -= g * x
		}
		m.B[c] -= g
	}
}

// OneHotLM is the same linear next-byte model on a one-hot window.
type OneHotLM struct {
	Window   int
	ClassOf  [256]int
	Alphabet []byte
	W        []float32 // [nClass * Window * nClass], index (c, pos, byte)
	B        []float32
}

func (m *OneHotLM) pass(text []byte, rate float32, update bool) (nll float64, correct, scored, total int) {
	if len(text) <= m.Window {
		return 0, 0, 0, 0
	}
	nClass := len(m.Alphabet)
	logits := make([]float32, nClass)
	probs := make([]float32, nClass)
	for i := m.Window; i < len(text); i++ {
		total++
		class := m.ClassOf[text[i]]
		if class < 0 {
			continue
		}
		scored++
		window := text[i-m.Window : i]
		m.forward(window, logits)
		loss, hit := softmaxStep(logits, probs, class)
		nll += float64(loss)
		if hit {
			correct++
		}
		if update {
			m.step(window, probs, class, rate)
		}
	}
	return nll, correct, scored, total
}

func (m *OneHotLM) forward(window []byte, logits []float32) {
	n := len(m.Alphabet)
	for c := range logits {
		s := m.B[c]
		base := c * m.Window * n
		for t, b := range window {
			a := m.ClassOf[b]
			if a < 0 {
				continue
			}
			s += m.W[base+t*n+a]
		}
		logits[c] = s
	}
}

func (m *OneHotLM) step(window []byte, probs []float32, class int, rate float32) {
	n := len(m.Alphabet)
	for c, p := range probs {
		g := p
		if c == class {
			g -= 1
		}
		g *= rate
		m.B[c] -= g
		base := c * m.Window * n
		for t, b := range window {
			a := m.ClassOf[b]
			if a < 0 {
				continue
			}
			m.W[base+t*n+a] -= g
		}
	}
}

func newAlphabet(train []byte) (classOf [256]int, alphabet []byte) {
	for i := range classOf {
		classOf[i] = -1
	}
	for _, b := range train {
		if classOf[b] >= 0 {
			continue
		}
		classOf[b] = len(alphabet)
		alphabet = append(alphabet, b)
	}
	return classOf, alphabet
}

func finishReport(kind string, cfg LMConfig, dim int, nll float64, correct, scored, total int, vnll float64, vcorrect, vscored, vtotal int) (LMReport, error) {
	if scored == 0 || vscored == 0 {
		return LMReport{}, errCorpusShort
	}
	return LMReport{
		TrainN:   total,
		ValidN:   vtotal,
		TrainNLL: nll / float64(scored),
		ValidNLL: vnll / float64(vscored),
		TrainPPL: math.Exp(nll / float64(scored)),
		ValidPPL: math.Exp(vnll / float64(vscored)),
		TrainAcc: float64(correct) / float64(total),
		ValidAcc: float64(vcorrect) / float64(vtotal),
		Dim:      dim,
		Window:   cfg.Window,
		Epochs:   cfg.Epochs,
		Kind:     kind,
	}, nil
}

func runEpochs(epochs int, rate float32, train, valid []byte, pass func([]byte, float32, bool) (float64, int, int, int)) (trNLL float64, trC, trS, trT int, vaNLL float64, vaC, vaS, vaT int) {
	for ep := 0; ep < epochs; ep++ {
		nll, c, s, t := pass(train, rate, true)
		trNLL += nll
		trC += c
		trS += s
		trT += t
	}
	vaNLL, vaC, vaS, vaT = pass(valid, 0, false)
	return
}

// TrainSpanLM fits the next-byte model whose input is the single U
// program of the context window.
func TrainSpanLM(text []byte, cfg LMConfig) (*SpanLM, LMReport, error) {
	cfg.norm()
	train, valid, err := splitCorpus(text, cfg)
	if err != nil {
		return nil, LMReport{}, err
	}
	classOf, alphabet := newAlphabet(train)
	listLen := 3 * (8*cfg.Window + 1)
	m := &SpanLM{
		Window:   cfg.Window,
		ListLen:  listLen,
		Dim:      listLen + 1 + 4 + 8,
		ClassOf:  classOf,
		Alphabet: alphabet,
		W:        make([]float32, len(alphabet)*(listLen+1+4+8)),
		B:        make([]float32, len(alphabet)),
		search:   &uSearcher{maxBits: cfg.MaxBits, bound: cfg.Bound},
		kcache:   make(map[string]*KResult),
		vcache:   make(map[string][]float32),
		prove:    cfg.Bound,
	}
	need := 6*(8*cfg.Window) + 64
	if m.prove < need {
		m.prove = need
	}
	m.search.init(cfg.Prec)
	initWeights(m.W, cfg.Seed)
	trNLL, trC, trS, trT, vaNLL, vaC, vaS, vaT := runEpochs(cfg.Epochs, cfg.Rate, train, valid, m.pass)
	rep, err := finishReport("span", cfg, m.Dim, trNLL, trC, trS, trT, vaNLL, vaC, vaS, vaT)
	return m, rep, err
}

// TrainOneHotLM fits the same linear model on a one-hot encoding of the
// same window.
func TrainOneHotLM(text []byte, cfg LMConfig) (*OneHotLM, LMReport, error) {
	cfg.norm()
	train, valid, err := splitCorpus(text, cfg)
	if err != nil {
		return nil, LMReport{}, err
	}
	classOf, alphabet := newAlphabet(train)
	n := len(alphabet)
	m := &OneHotLM{
		Window:   cfg.Window,
		ClassOf:  classOf,
		Alphabet: alphabet,
		W:        make([]float32, n*cfg.Window*n),
		B:        make([]float32, n),
	}
	initWeights(m.W, cfg.Seed)
	trNLL, trC, trS, trT, vaNLL, vaC, vaS, vaT := runEpochs(cfg.Epochs, cfg.Rate, train, valid, m.pass)
	rep, err := finishReport("onehot", cfg, cfg.Window*n, trNLL, trC, trS, trT, vaNLL, vaC, vaS, vaT)
	return m, rep, err
}

// DeltaKLM scores the next byte by ΔK = K(wb) − K(w). The templates are
// the same ones the span model uses (listing, bit-repeat, byte-run,
// splice). Windows longer than 7 bytes are not packed into the cache.
type DeltaKLM struct {
	Window   int
	ClassOf  [256]int
	Alphabet []byte
	search   *uSearcher
	kcache   map[string]*KResult
	kPack    map[byteKey]int
	prove    int
}

type byteKey struct {
	n byte
	b [8]byte
}

func byteKeyOf(w []byte) byteKey {
	var k byteKey
	k.n = byte(len(w))
	copy(k.b[:], w)
	return k
}

func (m *DeltaKLM) complexity(w []byte) int {
	key := byteKeyOf(w)
	if k, ok := m.kPack[key]; ok {
		return k
	}
	bits := windowBits(w)
	r := m.search.divide(bits, DefaultKLeaf, m.prove, m.kcache)
	k := r.K
	if k < 0 {
		k = 3 * (len(bits) + 1)
	}
	m.kPack[key] = k
	return k
}

// Delta is K(w followed by b) − K(w). A smaller value is a cheaper continuation.
func (m *DeltaKLM) Delta(w []byte, b byte) int {
	buf := make([]byte, len(w)+1)
	copy(buf, w)
	buf[len(w)] = b
	return m.complexity(buf) - m.complexity(w)
}

func (m *DeltaKLM) pass(text []byte) (nll float64, correct, scored, total int) {
	if len(text) <= m.Window {
		return 0, 0, 0, 0
	}
	nClass := len(m.Alphabet)
	logits := make([]float32, nClass)
	probs := make([]float32, nClass)
	ln2 := float32(math.Ln2)
	var buf [8]byte
	for i := m.Window; i < len(text); i++ {
		total++
		class := m.ClassOf[text[i]]
		if class < 0 {
			continue
		}
		scored++
		copy(buf[:m.Window], text[i-m.Window:i])
		kw := m.complexity(buf[:m.Window])
		for c, b := range m.Alphabet {
			buf[m.Window] = b
			delta := m.complexity(buf[:m.Window+1]) - kw
			logits[c] = -float32(delta) * ln2
		}
		loss, hit := softmaxStep(logits, probs, class)
		nll += float64(loss)
		if hit {
			correct++
		}
	}
	return nll, correct, scored, total
}

// TrainDeltaKLM evaluates the parameter-free ΔK distribution on the
// same split as the linear models. The train columns are that
// distribution's loss on the training bytes, not an SGD fit.
func TrainDeltaKLM(text []byte, cfg LMConfig) (*DeltaKLM, LMReport, error) {
	cfg.norm()
	if cfg.Window > 7 {
		return nil, LMReport{}, errDKWindow
	}
	train, valid, err := splitCorpus(text, cfg)
	if err != nil {
		return nil, LMReport{}, err
	}
	classOf, alphabet := newAlphabet(train)
	m := &DeltaKLM{
		Window:   cfg.Window,
		ClassOf:  classOf,
		Alphabet: alphabet,
		search:   &uSearcher{maxBits: cfg.MaxBits, bound: cfg.Bound},
		kcache:   make(map[string]*KResult),
		kPack:    make(map[byteKey]int),
		prove:    cfg.Bound,
	}
	need := 6*8*(cfg.Window+1) + 64
	if m.prove < need {
		m.prove = need
	}
	m.search.init(cfg.Prec)
	nll, c, s, t := m.pass(train)
	vnll, vc, vs, vt := m.pass(valid)
	cfg.Epochs = 1
	rep, err := finishReport("deltak", cfg, 1, nll, c, s, t, vnll, vc, vs, vt)
	return m, rep, err
}

// CompareDeltaK evaluates ΔK and trains the one-hot model on the same split.
func CompareDeltaK(text []byte, cfg LMConfig) (*DeltaKLM, LMReport, LMReport, error) {
	dk, drep, err := TrainDeltaKLM(text, cfg)
	if err != nil {
		return nil, LMReport{}, LMReport{}, err
	}
	_, hot, err := TrainOneHotLM(text, cfg)
	if err != nil {
		return nil, LMReport{}, LMReport{}, err
	}
	return dk, drep, hot, nil
}

// PhraseKLM gives every training window a short prefix-free program.
// A context's continuations use a Witten-Bell code: a byte that followed
// the window gets a Shannon length from its count, and a byte that never
// did is an escape plus today's listing, byte-run, or splice. ΔK is that
// conditional program length.
type PhraseKLM struct {
	Window   int
	ClassOf  [256]int
	Alphabet []byte
	ctx      map[byteKey]*phraseCond
	tmplMu   sync.RWMutex
	tmpl     map[byteKey]int
	book     *PhraseBook
}

type phraseCond struct {
	total int
	succ  map[byte]int
}

func (m *PhraseKLM) templateK(w []byte) int {
	key := byteKeyOf(w)
	m.tmplMu.RLock()
	if k, ok := m.tmpl[key]; ok {
		m.tmplMu.RUnlock()
		return k
	}
	m.tmplMu.RUnlock()
	nbits := 8 * len(w)
	best := 3 * (nbits + 1)
	if len(w) >= 2 {
		same := true
		for _, b := range w[1:] {
			if b != w[0] {
				same = false
				break
			}
		}
		if same {
			extra := 0
			if len(w) > 15 {
				extra = len(w) - 15
			}
			if br := 47 + 3*extra; br < best {
				best = br
			}
		}
	}
	if unaryByteBits(w) && nbits > 0 {
		extra := 0
		if nbits > 15 {
			extra = nbits - 15
		}
		if br := 26 + 3*extra; br < best {
			best = br
		}
	}
	if len(w) >= 4 && len(w)%2 == 0 {
		mid := len(w) / 2
		if sp := m.templateK(w[:mid]) + m.templateK(w[mid:]) - 3; sp < best {
			best = sp
		}
	}
	if m.book != nil {
		if n := m.book.BitLen(w); n > 0 && n < best {
			best = n
		}
	}
	m.tmplMu.Lock()
	if k, ok := m.tmpl[key]; ok {
		m.tmplMu.Unlock()
		return k
	}
	m.tmpl[key] = best
	m.tmplMu.Unlock()
	return best
}

// PhraseBook is a prefix-free code for the training windows. Each
// codeword is a real bit string; looking it up yields the window bits.
// K may use the codeword when it is shorter than a listing, byte-run, or splice.
type PhraseBook struct {
	prog map[byteKey][]bool
	dec  map[string][]bool
}

// NewPhraseBook assigns Shannon codewords to every window of length
// window and window+1 in text.
func NewPhraseBook(text []byte, window int) *PhraseBook {
	counts := make(map[byteKey]int)
	rawOf := make(map[byteKey][]byte)
	add := func(w []byte) {
		if len(w) == 0 || len(w) > 8 {
			return
		}
		key := byteKeyOf(w)
		if counts[key] == 0 {
			rawOf[key] = append([]byte(nil), w...)
		}
		counts[key]++
	}
	if window < 1 {
		window = 4
	}
	for _, n := range []int{window, window + 1} {
		if n > 8 || len(text) < n {
			continue
		}
		for i := 0; i+n <= len(text); i++ {
			add(text[i : i+n])
		}
	}
	type item struct {
		key   byteKey
		count int
		ell   int
		raw   []byte
	}
	total := 0
	items := make([]item, 0, len(counts))
	for key, c := range counts {
		total += c
		items = append(items, item{key: key, count: c, raw: rawOf[key]})
	}
	if total == 0 {
		return &PhraseBook{prog: map[byteKey][]bool{}, dec: map[string][]bool{}}
	}
	for i := range items {
		items[i].ell = codeLen(float64(items[i].count) / float64(total))
	}
	sort.Slice(items, func(i, j int) bool {
		if items[i].ell != items[j].ell {
			return items[i].ell < items[j].ell
		}
		return bytes.Compare(items[i].raw, items[j].raw) < 0
	})
	book := &PhraseBook{
		prog: make(map[byteKey][]bool, len(items)),
		dec:  make(map[string][]bool, len(items)),
	}
	var code uint64
	prev := 0
	for i, it := range items {
		if i == 0 {
			prev = it.ell
		} else {
			code++
			if it.ell > prev {
				code <<= uint(it.ell - prev)
			}
			prev = it.ell
		}
		word := u64Bits(code, it.ell)
		book.prog[it.key] = word
		book.dec[FormatBits(word)] = windowBits(it.raw)
	}
	return book
}

// BitLen is the phrase-codeword length of w, or 0 if w is not in the book.
func (b *PhraseBook) BitLen(w []byte) int {
	if b == nil {
		return 0
	}
	prog, ok := b.prog[byteKeyOf(w)]
	if !ok {
		return 0
	}
	return len(prog)
}

// Program is the codeword for w.
func (b *PhraseBook) Program(w []byte) []bool {
	if b == nil {
		return nil
	}
	return b.prog[byteKeyOf(w)]
}

// Run looks up a codeword and returns the window bits it stands for.
func (b *PhraseBook) Run(prog []bool) ([]bool, bool) {
	if b == nil || len(prog) == 0 {
		return nil, false
	}
	out, ok := b.dec[FormatBits(prog)]
	if !ok {
		return nil, false
	}
	return append([]bool(nil), out...), true
}

// PrefixFree reports whether no codeword is a prefix of another.
func (b *PhraseBook) PrefixFree() bool {
	words := make([]string, 0, len(b.dec))
	for w := range b.dec {
		words = append(words, w)
	}
	for i := 0; i < len(words); i++ {
		for j := 0; j < len(words); j++ {
			if i != j && strings.HasPrefix(words[j], words[i]) {
				return false
			}
		}
	}
	return true
}

func (b *PhraseBook) consider(x []bool, r *KResult) {
	w, ok := bitsToBytes(x)
	if !ok {
		return
	}
	prog := b.Program(w)
	if len(prog) == 0 || (r.K >= 0 && len(prog) >= r.K) {
		return
	}
	got, ok := b.Run(prog)
	if !ok || !bitsEq(got, x) {
		return
	}
	r.K = len(prog)
	r.Program = append([]bool(nil), prog...)
	r.How = "phrase"
	r.TMSteps = 1
}

// WindowK is K of w using the U templates and this model's phrase code.
func (m *PhraseKLM) WindowK(w []byte) (prog []bool, k int, how string) {
	bits := windowBits(w)
	r := &KResult{Bits: bits, K: -1, Bound: 256}
	considerTemplates(bits, 256, r)
	if m.book != nil {
		m.book.consider(bits, r)
	}
	return r.Program, r.K, r.How
}

func u64Bits(v uint64, n int) []bool {
	if n < 1 {
		n = 1
	}
	out := make([]bool, n)
	for i := 0; i < n; i++ {
		shift := uint(n - 1 - i)
		if shift < 64 && v&(uint64(1)<<shift) != 0 {
			out[i] = true
		}
	}
	return out
}

func bitsToBytes(x []bool) ([]byte, bool) {
	if len(x) == 0 || len(x)%8 != 0 {
		return nil, false
	}
	out := make([]byte, len(x)/8)
	for i := range out {
		var b byte
		for bit := 0; bit < 8; bit++ {
			if x[i*8+bit] {
				b |= 1 << uint(7-bit)
			}
		}
		out[i] = b
	}
	return out, true
}

func unaryByteBits(w []byte) bool {
	if len(w) == 0 {
		return false
	}
	for _, b := range w {
		if b != 0 && b != 0xff {
			return false
		}
		if b != w[0] {
			return false
		}
	}
	return true
}

func codeLen(p float64) int {
	if p >= 1 {
		return 1
	}
	if p <= 0 {
		return 62
	}
	l := int(math.Ceil(-math.Log2(p) - 1e-9))
	if l < 1 {
		return 1
	}
	if l > 62 {
		return 62
	}
	return l
}

// CodeLen is the conditional program length of b after w, in bits.
func (m *PhraseKLM) CodeLen(w []byte, b byte) int {
	ell := make([]int, len(m.Alphabet))
	m.codeLens(w, ell)
	for i, a := range m.Alphabet {
		if a == b {
			return ell[i]
		}
	}
	return 62
}

func (m *PhraseKLM) codeLens(w []byte, ell []int) {
	p := make([]float64, len(ell))
	m.probs(w, p)
	for i := range ell {
		ell[i] = codeLen(p[i])
	}
}

// probs writes the 4-byte continuation distribution. It sums to one.
func (m *PhraseKLM) probs(w []byte, p []float64) {
	for i := range p {
		p[i] = 0
	}
	cc := m.ctx[byteKeyOf(w)]
	if cc == nil || cc.total == 0 || len(cc.succ) == 0 {
		m.templateProbs(w, p)
		return
	}
	t := len(cc.succ)
	nUnseen := 0
	for _, b := range m.Alphabet {
		if cc.succ[b] == 0 {
			nUnseen++
		}
	}
	denom := float64(cc.total + t)
	if nUnseen == 0 {
		for i, b := range m.Alphabet {
			p[i] = float64(cc.succ[b]) / float64(cc.total)
		}
		return
	}
	pEsc := float64(t) / denom
	var buf [8]byte
	copy(buf[:len(w)], w)
	tw := m.templateK(buf[:len(w)])
	weight := make([]float64, len(p))
	var z float64
	for i, b := range m.Alphabet {
		if cc.succ[b] > 0 {
			continue
		}
		buf[len(w)] = b
		dt := m.templateK(buf[:len(w)+1]) - tw
		weight[i] = math.Exp2(-float64(dt))
		z += weight[i]
	}
	if z == 0 {
		z = 1
	}
	for i, b := range m.Alphabet {
		if c := cc.succ[b]; c > 0 {
			p[i] = float64(c) / denom
		} else {
			p[i] = pEsc * weight[i] / z
		}
	}
	normProbs(p)
}

func (m *PhraseKLM) templateProbs(w []byte, p []float64) {
	var buf [8]byte
	copy(buf[:len(w)], w)
	tw := m.templateK(buf[:len(w)])
	var z float64
	for i, b := range m.Alphabet {
		buf[len(w)] = b
		dt := m.templateK(buf[:len(w)+1]) - tw
		p[i] = math.Exp2(-float64(dt))
		z += p[i]
	}
	if z == 0 {
		z = 1
	}
	for i := range p {
		p[i] /= z
	}
}

func (m *PhraseKLM) templateLens(w []byte, ell []int) {
	p := make([]float64, len(ell))
	m.templateProbs(w, p)
	for i := range ell {
		ell[i] = codeLen(p[i])
	}
}

func normProbs(p []float64) {
	var z float64
	for _, x := range p {
		z += x
	}
	if z <= 0 {
		u := 1 / float64(len(p))
		for i := range p {
			p[i] = u
		}
		return
	}
	for i := range p {
		p[i] /= z
	}
}

func (m *PhraseKLM) pass(text []byte) (nll float64, correct, scored, total int) {
	if len(text) <= m.Window {
		return 0, 0, 0, 0
	}
	nClass := len(m.Alphabet)
	logits := make([]float32, nClass)
	probs := make([]float32, nClass)
	ell := make([]int, nClass)
	ln2 := float32(math.Ln2)
	var buf [8]byte
	for i := m.Window; i < len(text); i++ {
		total++
		class := m.ClassOf[text[i]]
		if class < 0 {
			continue
		}
		scored++
		copy(buf[:m.Window], text[i-m.Window:i])
		m.codeLens(buf[:m.Window], ell)
		for c, bits := range ell {
			logits[c] = -float32(bits) * ln2
		}
		loss, hit := softmaxStep(logits, probs, class)
		nll += float64(loss)
		if hit {
			correct++
		}
	}
	return nll, correct, scored, total
}

// TrainPhraseKLM counts training windows and scores the next byte by the
// length of its phrase program. The train columns are that code's loss
// on the training bytes, not an SGD fit.
func TrainPhraseKLM(text []byte, cfg LMConfig) (*PhraseKLM, LMReport, error) {
	cfg.norm()
	if cfg.Window > 7 {
		return nil, LMReport{}, errDKWindow
	}
	train, valid, err := splitCorpus(text, cfg)
	if err != nil {
		return nil, LMReport{}, err
	}
	classOf, alphabet := newAlphabet(train)
	m := &PhraseKLM{
		Window:   cfg.Window,
		ClassOf:  classOf,
		Alphabet: alphabet,
		ctx:      make(map[byteKey]*phraseCond),
		tmpl:     make(map[byteKey]int),
		book:     NewPhraseBook(train, cfg.Window),
	}
	for i := cfg.Window; i < len(train); i++ {
		key := byteKeyOf(train[i-cfg.Window : i])
		cc := m.ctx[key]
		if cc == nil {
			cc = &phraseCond{succ: make(map[byte]int)}
			m.ctx[key] = cc
		}
		cc.total++
		cc.succ[train[i]]++
	}
	nll, c, s, t := m.pass(train)
	vnll, vc, vs, vt := m.pass(valid)
	cfg.Epochs = 1
	rep, err := finishReport("phrase", cfg, 1, nll, c, s, t, vnll, vc, vs, vt)
	return m, rep, err
}

// ComparePhraseK scores the phrase code and trains the one-hot model
// on the same split.
func ComparePhraseK(text []byte, cfg LMConfig) (*PhraseKLM, LMReport, LMReport, error) {
	ph, prep, err := TrainPhraseKLM(text, cfg)
	if err != nil {
		return nil, LMReport{}, LMReport{}, err
	}
	_, hot, err := TrainOneHotLM(text, cfg)
	if err != nil {
		return nil, LMReport{}, LMReport{}, err
	}
	return ph, prep, hot, nil
}

// DefaultVarOrder is the longest context suffix of the variable-order code.
const DefaultVarOrder = 16

// ppmMinCount is the smallest number of times a longer context must have
// occurred before its program is used. A suffix seen once is skipped so
// the escape does not stop on a unique string.
const ppmMinCount = 2

// VarPhrase is a variable-order phrase code. The next byte is coded from
// the longest context suffix that occurred at least ppmMinCount times in
// training. A byte never seen there escapes to the next-shorter suffix.
// The empty suffix is the unigram.
type VarPhrase struct {
	Max      int
	ClassOf  [256]int
	Alphabet []byte
	ctx      map[ctxKey]tally
	mix      map[ctxKey]map[byte]uint32
	low      map[ctxKey]tally
	lowMix   map[ctxKey]map[byte]uint32
	base     *PhraseKLM
}

// ctxKey is a context of at most DefaultVarOrder bytes.
type ctxKey struct {
	n uint8
	b [DefaultVarOrder]byte
}

type tally struct {
	total uint32
	sym   byte
	pure  bool
}

func suffixKey(ctx []byte, k int) ctxKey {
	var key ctxKey
	key.n = uint8(k)
	copy(key.b[:k], ctx[len(ctx)-k:])
	return key
}

func addTally(tallies map[ctxKey]tally, mixed map[ctxKey]map[byte]uint32, key ctxKey, s byte) {
	t := tallies[key]
	if t.total == 0 {
		tallies[key] = tally{total: 1, sym: s, pure: true}
		return
	}
	t.total++
	if t.pure {
		if t.sym != s {
			mixed[key] = map[byte]uint32{t.sym: t.total - 1, s: 1}
			t.pure = false
		}
	} else {
		mixed[key][s]++
	}
	tallies[key] = t
}

func (m *VarPhrase) observe(text []byte) {
	maxO := m.Max
	tallies := make(map[ctxKey]tally, len(text))
	mixed := make(map[ctxKey]map[byte]uint32)
	for i := 0; i < len(text); i++ {
		s := text[i]
		lim := i
		if lim > maxO {
			lim = maxO
		}
		// The scored code keeps orders 1..4 in the phrase table.
		// Only longer suffixes are counted here.
		for k := 5; k <= lim; k++ {
			addTally(tallies, mixed, suffixKey(text[i-k:i], k), s)
		}
	}
	kept := make(map[ctxKey]tally)
	keptMix := make(map[ctxKey]map[byte]uint32)
	for key, t := range tallies {
		if t.total < ppmMinCount {
			continue
		}
		kept[key] = t
		if mm, ok := mixed[key]; ok {
			keptMix[key] = mm
		}
	}
	m.ctx = kept
	m.mix = keptMix

	// Orders 0..3 back the sampler up. They are not used by the scored
	// code, which already has order 4 and only consults suffixes of
	// length 5 and up.
	lowT := make(map[ctxKey]tally)
	lowM := make(map[ctxKey]map[byte]uint32)
	for i := 0; i < len(text); i++ {
		s := text[i]
		lim := i
		if lim > 3 {
			lim = 3
		}
		for k := 0; k <= lim; k++ {
			var key ctxKey
			if k > 0 {
				key = suffixKey(text[i-k:i], k)
			}
			addTally(lowT, lowM, key, s)
		}
	}
	m.low = lowT
	m.lowMix = lowM
}

// CodeLen is the conditional program length of b after w, in bits.
// w may be longer than Max; only its Max-byte suffix is used.
func (m *VarPhrase) CodeLen(w []byte, b byte) int {
	if len(w) > m.Max {
		w = w[len(w)-m.Max:]
	}
	ell := make([]int, len(m.Alphabet))
	m.codeLens(w, ell)
	for i, a := range m.Alphabet {
		if a == b {
			return ell[i]
		}
	}
	return 62
}

func (m *VarPhrase) codeLens(ctx []byte, ell []int) {
	n := len(m.Alphabet)
	baseCtx := ctx
	if len(baseCtx) > 4 {
		baseCtx = baseCtx[len(baseCtx)-4:]
	}
	if m.base != nil && len(baseCtx) == 4 {
		m.base.codeLens(baseCtx, ell)
	} else {
		u := codeLen(1 / float64(n))
		for i := range ell {
			ell[i] = u
		}
	}
	lim := len(ctx)
	if lim > m.Max {
		lim = m.Max
	}
	var used tally
	var mix map[byte]uint32
	found := false
	for k := lim; k >= 5; k-- {
		t, ok := m.ctx[suffixKey(ctx, k)]
		if !ok || t.total < ppmMinCount {
			continue
		}
		used = t
		mix = m.mix[suffixKey(ctx, k)]
		found = true
		break
	}
	if !found {
		return
	}
	kinds := 1
	if !used.pure {
		kinds = len(mix)
	}
	denom := float64(used.total + uint32(kinds))
	esc := float64(kinds) / denom
	raw := make([]float64, n)
	if used.pure {
		if idx := m.ClassOf[used.sym]; idx >= 0 {
			raw[idx] = float64(used.total) / denom
		}
	} else {
		for b, c := range mix {
			if idx := m.ClassOf[b]; idx >= 0 && c > 0 {
				raw[idx] = float64(c) / denom
			}
		}
	}
	for i := range raw {
		if raw[i] == 0 {
			raw[i] = esc * math.Exp2(-float64(ell[i]))
		}
	}
	var sum float64
	for _, p := range raw {
		sum += p
	}
	if sum <= 0 {
		sum = 1
	}
	for i := range ell {
		ell[i] = codeLen(raw[i] / sum)
	}
}

func (m *VarPhrase) pass(text []byte) (nll float64, correct, scored, total int) {
	if len(text) == 0 {
		return 0, 0, 0, 0
	}
	nClass := len(m.Alphabet)
	logits := make([]float32, nClass)
	probs := make([]float32, nClass)
	ell := make([]int, nClass)
	ln2 := float32(math.Ln2)
	for i := 4; i < len(text); i++ {
		total++
		class := m.ClassOf[text[i]]
		if class < 0 {
			continue
		}
		scored++
		start := 0
		if i > m.Max {
			start = i - m.Max
		}
		m.codeLens(text[start:i], ell)
		for c, bits := range ell {
			logits[c] = -float32(bits) * ln2
		}
		loss, hit := softmaxStep(logits, probs, class)
		nll += float64(loss)
		if hit {
			correct++
		}
	}
	return nll, correct, scored, total
}

// TrainVarPhrase builds the variable-order code on the training half.
// It also returns the fixed 4-byte code trained on that same split.
func TrainVarPhrase(text []byte, cfg LMConfig) (*VarPhrase, LMReport, *PhraseKLM, LMReport, error) {
	cfg.norm()
	train, _, err := splitCorpus(text, cfg)
	if err != nil {
		return nil, LMReport{}, nil, LMReport{}, err
	}
	classOf, alphabet := newAlphabet(train)
	fixedCfg := cfg
	fixedCfg.Window = 4
	base, brep, err := TrainPhraseKLM(text, fixedCfg)
	if err != nil {
		return nil, LMReport{}, nil, LMReport{}, err
	}
	m := &VarPhrase{
		Max:      DefaultVarOrder,
		ClassOf:  classOf,
		Alphabet: alphabet,
		base:     base,
	}
	m.observe(train)
	nll, c, s, t := m.pass(train)
	vnll, vc, vs, vt := m.pass(validSplit(text, cfg))
	cfg.Window = m.Max
	cfg.Epochs = 1
	rep, err := finishReport("varphrase", cfg, m.Max, nll, c, s, t, vnll, vc, vs, vt)
	return m, rep, base, brep, err
}

func validSplit(text []byte, cfg LMConfig) []byte {
	_, valid, err := splitCorpus(text, cfg)
	if err != nil {
		return nil
	}
	return valid
}

// CompareVarPhrase scores the variable-order code against the fixed
// 4-byte phrase code on the same split.
func CompareVarPhrase(text []byte, cfg LMConfig) (*VarPhrase, LMReport, *PhraseKLM, LMReport, error) {
	return TrainVarPhrase(text, cfg)
}

// MixPhrase scores the next byte by the ideal length of
// λ P_long + (1-λ) P_4, with λ = c_long / (c_long + c_4).
type MixPhrase struct {
	long *VarPhrase
	base *PhraseKLM
}

// CodeBits is -log2 of the mixture probability, in bits.
func (m *MixPhrase) CodeBits(w []byte, b byte) float64 {
	if len(w) > m.long.Max {
		w = w[len(w)-m.long.Max:]
	}
	p := make([]float64, len(m.base.Alphabet))
	m.probs(w, p)
	for i, a := range m.base.Alphabet {
		if a == b {
			if p[i] <= 0 {
				return 62
			}
			return -math.Log2(p[i])
		}
	}
	return 62
}

func (m *MixPhrase) probs(ctx []byte, out []float64) {
	n := len(out)
	p4 := make([]float64, n)
	baseCtx := ctx
	if len(baseCtx) > 4 {
		baseCtx = baseCtx[len(baseCtx)-4:]
	}
	if len(baseCtx) == 4 {
		m.base.probs(baseCtx, p4)
	} else {
		u := 1 / float64(n)
		for i := range p4 {
			p4[i] = u
		}
	}
	p16 := make([]float64, n)
	c16 := m.longProbs(ctx, p16)
	c4 := 0
	if len(baseCtx) == 4 {
		if cc := m.base.ctx[byteKeyOf(baseCtx)]; cc != nil {
			c4 = cc.total
		}
	}
	lam := 0.0
	if c16+c4 > 0 {
		lam = float64(c16) / float64(c16+c4)
	}
	for i := range out {
		out[i] = lam*p16[i] + (1-lam)*p4[i]
	}
	normProbs(out)
}

// longProbs writes the longest repeated suffix distribution and returns
// its count. The count is zero when no suffix longer than 4 bytes qualifies.
func (m *MixPhrase) longProbs(ctx []byte, p []float64) int {
	for i := range p {
		p[i] = 0
	}
	v := m.long
	lim := len(ctx)
	if lim > v.Max {
		lim = v.Max
	}
	var used tally
	var mix map[byte]uint32
	found := false
	for k := lim; k >= 5; k-- {
		t, ok := v.ctx[suffixKey(ctx, k)]
		if !ok || t.total < ppmMinCount {
			continue
		}
		used = t
		mix = v.mix[suffixKey(ctx, k)]
		found = true
		break
	}
	if !found {
		return 0
	}
	kinds := 1
	if !used.pure {
		kinds = len(mix)
	}
	if kinds < 1 {
		kinds = 1
	}
	denom := float64(used.total) + float64(kinds)
	if used.pure {
		for i, b := range v.Alphabet {
			if b == used.sym {
				p[i] = float64(used.total) / denom
			}
		}
	} else {
		for b, c := range mix {
			idx := v.ClassOf[b]
			if idx >= 0 && c > 0 {
				p[idx] = float64(c) / denom
			}
		}
	}
	nUn := 0
	var assigned float64
	for _, x := range p {
		if x == 0 {
			nUn++
		} else {
			assigned += x
		}
	}
	esc := 1 - assigned
	if esc < 0 {
		esc = 0
	}
	if nUn > 0 {
		share := esc / float64(nUn)
		for i := range p {
			if p[i] == 0 {
				p[i] = share
			}
		}
	}
	normProbs(p)
	return int(used.total)
}

func (m *MixPhrase) pass(text []byte) (nll float64, correct, scored, total int) {
	if len(text) <= 4 {
		return 0, 0, 0, 0
	}
	n := len(m.base.Alphabet)
	p := make([]float64, n)
	for i := 4; i < len(text); i++ {
		total++
		class := m.base.ClassOf[text[i]]
		if class < 0 {
			continue
		}
		scored++
		start := 0
		if i > m.long.Max {
			start = i - m.long.Max
		}
		m.probs(text[start:i], p)
		pb := p[class]
		if pb < 1e-15 {
			pb = 1e-15
		}
		nll += -math.Log(pb)
		best, arg := p[0], 0
		for j := 1; j < n; j++ {
			if p[j] > best {
				best = p[j]
				arg = j
			}
		}
		if arg == class {
			correct++
		}
	}
	return nll, correct, scored, total
}

// CompareMixPhrase scores the mixture against the fixed 4-byte code
// on the same split.
func CompareMixPhrase(text []byte, cfg LMConfig) (*MixPhrase, LMReport, LMReport, error) {
	cfg.norm()
	vp, _, base, frep, err := TrainVarPhrase(text, cfg)
	if err != nil {
		return nil, LMReport{}, LMReport{}, err
	}
	mix := &MixPhrase{long: vp, base: base}
	train, valid, err := splitCorpus(text, cfg)
	if err != nil {
		return nil, LMReport{}, LMReport{}, err
	}
	nll, c, s, t := mix.pass(train)
	vnll, vc, vs, vt := mix.pass(valid)
	cfg.Window = vp.Max
	cfg.Epochs = 1
	rep, err := finishReport("mix", cfg, vp.Max, nll, c, s, t, vnll, vc, vs, vt)
	return mix, rep, frep, err
}

// Generate draws n bytes from the backoff distribution. Each byte is
// conditioned on the prompt and on the bytes already drawn. seed selects
// the PCG stream. The result is the prompt followed by the sample. Counts
// are not updated.
func (m *MixPhrase) Generate(prompt []byte, n int, seed uint64) []byte {
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	return m.extend(prompt, n, func(alphabet []byte, p []float64) byte {
		return pickByte(alphabet, p, rng.Float64())
	})
}

// Distribution writes the sampling distribution of the byte that follows
// ctx, in alphabet order. The entries sum to one.
func (m *MixPhrase) Distribution(ctx []byte) (alphabet []byte, probs []float64) {
	if m == nil || m.base == nil || len(m.base.Alphabet) == 0 {
		return nil, nil
	}
	alphabet = append([]byte(nil), m.base.Alphabet...)
	probs = make([]float64, len(alphabet))
	m.sampleProbs(ctx, probs)
	return alphabet, probs
}

// Greedy writes the mode of the sampling distribution for n bytes after
// prompt. Ties keep the earlier alphabet byte. Counts are not updated.
func (m *MixPhrase) Greedy(prompt []byte, n int) []byte {
	return m.extend(prompt, n, modeByte)
}

func (m *MixPhrase) extend(prompt []byte, n int, next func(alphabet []byte, p []float64) byte) []byte {
	if n < 0 {
		n = 0
	}
	out := make([]byte, len(prompt), len(prompt)+n)
	copy(out, prompt)
	if n == 0 || m == nil || m.base == nil || m.long == nil || len(m.base.Alphabet) == 0 {
		return out
	}
	p := make([]float64, len(m.base.Alphabet))
	max := m.long.Max
	if max < 4 {
		max = 4
	}
	for i := 0; i < n; i++ {
		ctx := out
		if len(ctx) > max {
			ctx = ctx[len(ctx)-max:]
		}
		m.sampleProbs(ctx, p)
		out = append(out, next(m.base.Alphabet, p))
	}
	return out
}

// sampleProbs is the distribution a generated byte is drawn from.
// Order k blends its counts into the next-shorter order,
// P_k(b) = c(b)/(c+t) + (t/(c+t)) P_{k-1}(b). Orders 5..16 are the
// repeated long suffixes, order 4 is the phrase table, and orders 0..3
// run from the unigram up through the trigram. A missing order keeps
// the shorter distribution. The empty context is uniform over the
// training alphabet, and the unigram blends into that.
func (m *MixPhrase) sampleProbs(ctx []byte, out []float64) {
	n := len(out)
	if n == 0 {
		return
	}
	u := 1 / float64(n)
	for i := range out {
		out[i] = u
	}
	if m == nil || m.long == nil {
		return
	}
	lim := len(ctx)
	if lim > m.long.Max {
		lim = m.long.Max
	}
	scratch := make([]float64, n)
	for k := 0; k <= lim; k++ {
		if !m.blend(ctx, k, scratch, out) {
			continue
		}
		copy(out, scratch)
	}
	normProbs(out)
}

// blend writes order k's Witten-Bell mixture of src into dst.
// It reports false when that order was not counted.
func (m *MixPhrase) blend(ctx []byte, k int, dst, src []float64) bool {
	if m.base == nil {
		return false
	}
	total, kinds, visit, ok := m.order(ctx, k)
	if !ok || total < 1 || kinds < 1 {
		return false
	}
	esc := float64(kinds) / float64(total+kinds)
	scale := 1 / float64(total+kinds)
	for i := range dst {
		dst[i] = esc * src[i]
	}
	classOf := &m.base.ClassOf
	visit(func(b byte, c int) {
		if c <= 0 {
			return
		}
		idx := classOf[b]
		if idx >= 0 && idx < len(dst) {
			dst[idx] += float64(c) * scale
		}
	})
	return true
}

func (m *MixPhrase) order(ctx []byte, k int) (total, kinds int, visit func(func(byte, int)), ok bool) {
	v := m.long
	if k >= 5 {
		if len(ctx) < k {
			return 0, 0, nil, false
		}
		key := suffixKey(ctx, k)
		t, found := v.ctx[key]
		if !found || t.total < ppmMinCount {
			return 0, 0, nil, false
		}
		if t.pure {
			sym, tot := t.sym, int(t.total)
			return tot, 1, func(fn func(byte, int)) { fn(sym, tot) }, true
		}
		succ := v.mix[key]
		if len(succ) == 0 {
			return 0, 0, nil, false
		}
		return int(t.total), len(succ), func(fn func(byte, int)) {
			for b, c := range succ {
				fn(b, int(c))
			}
		}, true
	}
	if k == 4 {
		if m.base == nil || len(ctx) < 4 {
			return 0, 0, nil, false
		}
		cc := m.base.ctx[byteKeyOf(ctx[len(ctx)-4:])]
		if cc == nil || cc.total == 0 || len(cc.succ) == 0 {
			return 0, 0, nil, false
		}
		return cc.total, len(cc.succ), func(fn func(byte, int)) {
			for b, c := range cc.succ {
				fn(b, c)
			}
		}, true
	}
	if v.low == nil || (k > 0 && len(ctx) < k) {
		return 0, 0, nil, false
	}
	var key ctxKey
	if k > 0 {
		key = suffixKey(ctx, k)
	}
	t, found := v.low[key]
	if !found || t.total == 0 {
		return 0, 0, nil, false
	}
	if t.pure {
		sym, tot := t.sym, int(t.total)
		return tot, 1, func(fn func(byte, int)) { fn(sym, tot) }, true
	}
	succ := v.lowMix[key]
	if len(succ) == 0 {
		return 0, 0, nil, false
	}
	return int(t.total), len(succ), func(fn func(byte, int)) {
		for b, c := range succ {
			fn(b, int(c))
		}
	}, true
}

func pickByte(alphabet []byte, p []float64, u float64) byte {
	var c float64
	for i, x := range p {
		if i >= len(alphabet) {
			break
		}
		c += x
		if u < c {
			return alphabet[i]
		}
	}
	return alphabet[len(alphabet)-1]
}

func modeByte(alphabet []byte, p []float64) byte {
	best, arg := -1.0, 0
	for i, x := range p {
		if i >= len(alphabet) {
			break
		}
		if x > best {
			best, arg = x, i
		}
	}
	return alphabet[arg]
}

// CompareContextModels trains the span-program model and the one-hot
// model on the same split, with the same rate, epochs, and seed.
func CompareContextModels(text []byte, cfg LMConfig) (*SpanLM, LMReport, LMReport, error) {
	spanM, span, err := TrainSpanLM(text, cfg)
	if err != nil {
		return nil, LMReport{}, LMReport{}, err
	}
	_, hot, err := TrainOneHotLM(text, cfg)
	if err != nil {
		return nil, LMReport{}, LMReport{}, err
	}
	return spanM, span, hot, nil
}

func initWeights(w []float32, seed uint64) {
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	const scale = 0.02
	for i := range w {
		w[i] = (rng.Float32()*2 - 1) * scale
	}
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
const errDKWindow lmError = "ΔK cache packs at most 7 context bytes"
