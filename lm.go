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
	tmpl     map[byteKey]int
}

type phraseCond struct {
	total int
	succ  map[byte]int
}

func (m *PhraseKLM) templateK(w []byte) int {
	key := byteKeyOf(w)
	if k, ok := m.tmpl[key]; ok {
		return k
	}
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
	m.tmpl[key] = best
	return best
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
	cc := m.ctx[byteKeyOf(w)]
	if cc == nil {
		m.templateLens(w, ell)
		return
	}
	t := len(cc.succ)
	if t == 0 || cc.total == 0 {
		m.templateLens(w, ell)
		return
	}
	nUnseen := 0
	for _, b := range m.Alphabet {
		if cc.succ[b] == 0 {
			nUnseen++
		}
	}
	denom := float64(cc.total + t)
	if nUnseen == 0 {
		for i, b := range m.Alphabet {
			ell[i] = codeLen(float64(cc.succ[b]) / float64(cc.total))
		}
		return
	}
	pEsc := float64(t) / denom
	var buf [8]byte
	copy(buf[:len(w)], w)
	tw := m.templateK(buf[:len(w)])
	weight := make([]float64, len(ell))
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
			ell[i] = codeLen(float64(c) / denom)
		} else {
			ell[i] = codeLen(pEsc * weight[i] / z)
		}
	}
}

func (m *PhraseKLM) templateLens(w []byte, ell []int) {
	var buf [8]byte
	copy(buf[:len(w)], w)
	tw := m.templateK(buf[:len(w)])
	weight := make([]float64, len(ell))
	var z float64
	for i, b := range m.Alphabet {
		buf[len(w)] = b
		dt := m.templateK(buf[:len(w)+1]) - tw
		weight[i] = math.Exp2(-float64(dt))
		z += weight[i]
	}
	if z == 0 {
		z = 1
	}
	for i := range ell {
		ell[i] = codeLen(weight[i] / z)
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
