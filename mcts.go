// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import (
	"math"
	"math/rand/v2"
	"sort"
)

const (
	// DefaultMCTSWindow is the context width of the MCTS language model.
	// It is wider than the 4-byte phrase window and the 7-byte ΔK cache.
	DefaultMCTSWindow = 32
	// maxMCTSWindow is the widest context winKey can hold.
	maxMCTSWindow = 32
	// DefaultMCTSBudget is how many tree-search playouts training may spend.
	DefaultMCTSBudget = 128
	// DefaultMCTSMaxBits is the longest program the tree search will write.
	// The exhaustive halt oracle stops at DefaultUMaxBits (12).
	DefaultMCTSMaxBits = 64
)

const errMCTSWindow lmError = "MCTS window is at most 32 bytes"

// mctsAction is one U instruction in the search tree.
type mctsAction struct {
	op, arg int
}

func mctsActions() []mctsAction {
	a := make([]mctsAction, 0, 38)
	for op := 0; op <= uOpJz; op++ {
		a = append(a, mctsAction{op: op})
	}
	for arg := 0; arg < 16; arg++ {
		a = append(a, mctsAction{op: uOpJmp, arg: arg})
		a = append(a, mctsAction{op: uOpSet, arg: arg})
	}
	return a
}

func (a mctsAction) bits() []bool {
	p := make([]bool, 0, 7)
	putBits(&p, a.op, 3)
	if a.op == uOpJmp || a.op == uOpSet {
		putBits(&p, a.arg, 4)
	}
	return p
}

type mctsNode struct {
	bits   []bool
	kids   []*mctsNode
	order  []int
	next   int
	visits int
	value  float64
	dead   bool
}

type mctsHit struct {
	prog []bool
	k    int
	how  string
}

// MCTSOracle is a halt oracle for U grown by Monte Carlo tree search.
// Each playout writes a prefix-free program and runs U. Programs longer
// than the exhaustive 12-bit oracle are ordinary nodes in the tree, so a
// context of dozens of bytes can still be given a halting witness.
type MCTSOracle struct {
	MaxBits int
	Bound   int
	C       float64
	Left    int
	Per     int
	root    *mctsNode
	halt    map[string]bool
	best    map[string]mctsHit
	actions []mctsAction
	rng     *rand.Rand
	Evals   int
	HaltN   int
}

// NewMCTSOracle searches programs of at most maxBits and runs each one
// for at most bound steps. seed fixes the playout stream.
func NewMCTSOracle(maxBits, bound int, seed uint64) *MCTSOracle {
	if maxBits < DefaultUMaxBits+1 {
		maxBits = DefaultMCTSMaxBits
	}
	if bound < 1 {
		bound = DefaultUBound
	}
	if seed == 0 {
		seed = 1
	}
	return &MCTSOracle{
		MaxBits: maxBits,
		Bound:   bound,
		C:       math.Sqrt2,
		Per:     8,
		root:    &mctsNode{},
		halt:    make(map[string]bool),
		best:    make(map[string]mctsHit),
		actions: mctsActions(),
		rng:     rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15)),
	}
}

// Visits is the number of playouts that have backed up through the root.
func (o *MCTSOracle) Visits() int {
	if o == nil || o.root == nil {
		return 0
	}
	return o.root.visits
}

// Halts reports whether prog was evaluated, and whether that run was a
// complete halting program.
func (o *MCTSOracle) Halts(prog []bool) (halt, known bool) {
	if o == nil || len(prog) == 0 {
		return false, false
	}
	h, ok := o.halt[FormatBits(prog)]
	return h, ok
}

// OracleBits packs the evaluated halt bits in lexicographic order of
// the program bit strings. Bit i is 1 when that program halts.
func (o *MCTSOracle) OracleBits() []bool {
	if o == nil || len(o.halt) == 0 {
		return nil
	}
	keys := make([]string, 0, len(o.halt))
	for k := range o.halt {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	bits := make([]bool, len(keys))
	for i, k := range keys {
		bits[i] = o.halt[k]
	}
	return bits
}

// Search runs sims playouts aimed at printing target. A nil target
// rewards every halting program, which fills the oracle. The result is
// the shortest witness found for target, including listing and byte-run.
func (o *MCTSOracle) Search(target []bool, sims int) (prog []bool, k int, how string) {
	prog, k, how = o.considerTarget(target)
	if sims < 1 {
		return prog, k, how
	}
	for i := 0; i < sims; i++ {
		o.simulate(target)
	}
	if hit, ok := o.best[FormatBits(target)]; ok {
		return hit.prog, hit.k, hit.how
	}
	return prog, k, how
}

// Witness returns a certified program for target, spending a slice of
// the remaining search budget when the tree might shorten the template.
func (o *MCTSOracle) Witness(target []bool) (prog []bool, k int, how string) {
	prog, k, how = o.considerTarget(target)
	n := o.Per
	if n > o.Left {
		n = o.Left
	}
	if n < 1 || len(target) == 0 {
		return prog, k, how
	}
	o.Left -= n
	for i := 0; i < n; i++ {
		o.simulate(target)
	}
	if hit, ok := o.best[FormatBits(target)]; ok {
		return hit.prog, hit.k, hit.how
	}
	return prog, k, how
}

func (o *MCTSOracle) considerTarget(target []bool) (prog []bool, k int, how string) {
	if len(target) == 0 {
		return nil, -1, ""
	}
	key := FormatBits(target)
	if hit, ok := o.best[key]; ok {
		return hit.prog, hit.k, hit.how
	}
	r := &KResult{K: -1, Bound: o.Bound}
	considerTemplates(target, o.Bound, r)
	if r.Program == nil || r.K < 0 {
		return nil, -1, ""
	}
	o.halt[FormatBits(r.Program)] = true
	o.Evals++
	o.HaltN++
	o.best[key] = mctsHit{prog: append([]bool(nil), r.Program...), k: r.K, how: r.How}
	return o.best[key].prog, r.K, r.How
}

func (o *MCTSOracle) simulate(target []bool) {
	path := []*mctsNode{o.root}
	n := o.root
	for {
		if n.dead || len(n.bits) >= o.MaxBits {
			break
		}
		if n.order == nil {
			n.order = o.feasible(n.bits)
		}
		if n.next < len(n.order) {
			child := o.expand(n)
			path = append(path, child)
			n = child
			break
		}
		if len(n.kids) == 0 {
			break
		}
		n = o.bestChild(n)
		path = append(path, n)
	}
	reward := o.rollout(n.bits, n.dead, target)
	for _, node := range path {
		node.visits++
		node.value += reward
	}
}

func (o *MCTSOracle) feasible(prefix []bool) []int {
	order := make([]int, 0, len(o.actions))
	for i, a := range o.actions {
		if len(prefix)+len(a.bits()) <= o.MaxBits {
			order = append(order, i)
		}
	}
	o.rng.Shuffle(len(order), func(i, j int) {
		order[i], order[j] = order[j], order[i]
	})
	return order
}

func (o *MCTSOracle) expand(n *mctsNode) *mctsNode {
	a := o.actions[n.order[n.next]]
	n.next++
	bits := make([]bool, 0, len(n.bits)+7)
	bits = append(bits, n.bits...)
	bits = append(bits, a.bits()...)
	child := &mctsNode{bits: bits, dead: a.op == uOpHalt}
	n.kids = append(n.kids, child)
	return child
}

func (o *MCTSOracle) bestChild(n *mctsNode) *mctsNode {
	logN := math.Log(float64(max(n.visits, 1)))
	best := n.kids[0]
	bestS := math.Inf(-1)
	for _, c := range n.kids {
		s := math.Inf(1)
		if c.visits > 0 {
			s = c.value/float64(c.visits) + o.C*math.Sqrt(logN/float64(c.visits))
		}
		if s > bestS {
			bestS = s
			best = c
		}
	}
	return best
}

func (o *MCTSOracle) rollout(start []bool, complete bool, target []bool) float64 {
	prog := append([]bool(nil), start...)
	if !complete {
		for len(prog) < o.MaxBits {
			if len(prog) > DefaultUMaxBits && o.rng.IntN(4) == 0 {
				if hb := (mctsAction{op: uOpHalt}).bits(); len(prog)+len(hb) <= o.MaxBits {
					prog = append(prog, hb...)
				}
				break
			}
			a := o.actions[o.rng.IntN(len(o.actions))]
			if a.op == uOpHalt && len(prog) <= DefaultUMaxBits {
				a = mctsAction{op: uOpOut0}
			}
			b := a.bits()
			if len(prog)+len(b) > o.MaxBits {
				if hb := (mctsAction{op: uOpHalt}).bits(); len(prog)+len(hb) <= o.MaxBits {
					prog = append(prog, hb...)
				}
				break
			}
			prog = append(prog, b...)
			if a.op == uOpHalt {
				break
			}
		}
	}
	res := RunU(prog, o.Bound)
	o.note(prog, res)
	if res.Status == UHalt && res.Read == len(prog) && len(target) > 0 && bitsEq(res.Out, target) {
		o.save(target, prog, "mcts")
		return 1 + float64(o.MaxBits-len(prog))/float64(o.MaxBits)
	}
	if len(target) == 0 {
		if res.Status == UHalt && res.Read == len(prog) {
			return 1
		}
		return 0
	}
	return float64(prefixMatch(res.Out, target)) / float64(len(target)+1)
}

func (o *MCTSOracle) note(prog []bool, res UResult) {
	if len(prog) == 0 {
		return
	}
	ok := res.Status == UHalt && res.Read == len(prog)
	key := FormatBits(prog)
	if _, seen := o.halt[key]; seen {
		return
	}
	o.halt[key] = ok
	o.Evals++
	if ok {
		o.HaltN++
	}
}

func (o *MCTSOracle) save(target, prog []bool, how string) {
	if len(target) == 0 || len(prog) == 0 {
		return
	}
	key := FormatBits(target)
	if hit, ok := o.best[key]; ok && len(prog) >= hit.k {
		return
	}
	o.best[key] = mctsHit{prog: append([]bool(nil), prog...), k: len(prog), how: how}
}

func prefixMatch(out, target []bool) int {
	n := len(out)
	if len(target) < n {
		n = len(target)
	}
	i := 0
	for i < n && out[i] == target[i] {
		i++
	}
	return i
}

// mctxKey is a context of at most maxMCTSWindow bytes.
type mctxKey struct {
	n uint8
	b [maxMCTSWindow]byte
}

func mctxOf(w []byte) mctxKey {
	var key mctxKey
	key.n = uint8(len(w))
	copy(key.b[:], w)
	return key
}

func addMCTX(tallies map[mctxKey]tally, mixed map[mctxKey]map[byte]uint32, key mctxKey, s byte) {
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

// MCTSLM predicts the next byte from a window wider than the exhaustive
// halt oracle can index. A 32-byte suffix is used only after MCTSOracle
// certifies a program that prints it. That count blends into the
// shorter mixture.
type MCTSLM struct {
	Window   int
	ClassOf  [256]int
	Alphabet []byte
	ctx      map[mctxKey]tally
	mixc     map[mctxKey]map[byte]uint32
	back     *MixPhrase
	oracle   *MCTSOracle
}

// TrainMCTSLM fits the wide-window model on the same split as the mixture.
// The second report is that mixture.
func TrainMCTSLM(text []byte, cfg LMConfig) (*MCTSLM, LMReport, LMReport, error) {
	cfg.norm()
	if cfg.Window < 1 {
		cfg.Window = DefaultMCTSWindow
	}
	if cfg.Window > maxMCTSWindow {
		return nil, LMReport{}, LMReport{}, errMCTSWindow
	}
	back, brep, _, err := CompareMixPhrase(text, cfg)
	if err != nil {
		return nil, LMReport{}, LMReport{}, err
	}
	train, valid, err := splitCorpus(text, cfg)
	if err != nil {
		return nil, LMReport{}, LMReport{}, err
	}
	bound := cfg.Bound
	if need := 32*cfg.Window + 64; bound < need {
		bound = need
	}
	maxBits := cfg.MaxBits
	if maxBits < DefaultMCTSMaxBits {
		maxBits = DefaultMCTSMaxBits
	}
	oracle := NewMCTSOracle(maxBits, bound, cfg.Seed)
	budget := cfg.Sims
	if budget < 1 {
		budget = DefaultMCTSBudget
	}
	oracle.Left = budget
	m := &MCTSLM{
		Window:   cfg.Window,
		ClassOf:  back.base.ClassOf,
		Alphabet: append([]byte(nil), back.base.Alphabet...),
		back:     back,
		oracle:   oracle,
	}
	m.observe(train)
	nll, c, s, t := m.pass(train)
	vnll, vc, vs, vt := m.pass(valid)
	cfg.Epochs = 1
	rep, err := finishReport("mcts", cfg, m.Window, nll, c, s, t, vnll, vc, vs, vt)
	return m, rep, brep, err
}

// OracleVisits is the number of tree-search playouts.
func (m *MCTSLM) OracleVisits() int {
	if m == nil || m.oracle == nil {
		return 0
	}
	return m.oracle.Visits()
}

// OracleEvals is how many distinct programs the oracle has run.
func (m *MCTSLM) OracleEvals() int {
	if m == nil || m.oracle == nil {
		return 0
	}
	return m.oracle.Evals
}

// OracleHalts is how many of those programs halted.
func (m *MCTSLM) OracleHalts() int {
	if m == nil || m.oracle == nil {
		return 0
	}
	return m.oracle.HaltN
}

func (m *MCTSLM) observe(text []byte) {
	w := m.Window
	tallies := make(map[mctxKey]tally)
	mixed := make(map[mctxKey]map[byte]uint32)
	for i := w; i < len(text); i++ {
		addMCTX(tallies, mixed, mctxOf(text[i-w:i]), text[i])
	}
	kept := make(map[mctxKey]tally)
	keptMix := make(map[mctxKey]map[byte]uint32)
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
	m.mixc = keptMix
}

// ContextProgram is the halting program the oracle assigns to the
// Window-byte suffix of w.
func (m *MCTSLM) ContextProgram(w []byte) (prog []bool, k int, how string) {
	if m == nil || m.oracle == nil || len(w) < m.Window {
		return nil, -1, ""
	}
	suf := w[len(w)-m.Window:]
	return m.oracle.Witness(windowBits(suf))
}

// Distribution writes the sampling distribution of the byte after ctx.
func (m *MCTSLM) Distribution(ctx []byte) (alphabet []byte, probs []float64) {
	if m == nil || len(m.Alphabet) == 0 {
		return nil, nil
	}
	alphabet = append([]byte(nil), m.Alphabet...)
	probs = make([]float64, len(alphabet))
	m.probs(ctx, probs)
	return alphabet, probs
}

// Generate draws n bytes from the wide-window distribution.
func (m *MCTSLM) Generate(prompt []byte, n int, seed uint64) []byte {
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	return m.extend(prompt, n, func(alphabet []byte, p []float64) byte {
		return pickByte(alphabet, p, rng.Float64())
	})
}

// Greedy writes the mode of the wide-window distribution.
func (m *MCTSLM) Greedy(prompt []byte, n int) []byte {
	return m.extend(prompt, n, modeByte)
}

func (m *MCTSLM) extend(prompt []byte, n int, next func(alphabet []byte, p []float64) byte) []byte {
	if n < 0 {
		n = 0
	}
	out := make([]byte, len(prompt), len(prompt)+n)
	copy(out, prompt)
	if n == 0 || m == nil || len(m.Alphabet) == 0 {
		return out
	}
	p := make([]float64, len(m.Alphabet))
	for i := 0; i < n; i++ {
		m.probs(out, p)
		out = append(out, next(m.Alphabet, p))
	}
	return out
}

func (m *MCTSLM) probs(ctx []byte, out []float64) {
	if m.back != nil {
		m.back.probs(ctx, out)
	} else {
		u := 1 / float64(len(out))
		for i := range out {
			out[i] = u
		}
	}
	if len(ctx) < m.Window {
		return
	}
	suf := ctx[len(ctx)-m.Window:]
	t, ok := m.ctx[mctxOf(suf)]
	if !ok || t.total < ppmMinCount {
		return
	}
	prog, k, _ := m.oracle.Witness(windowBits(suf))
	if k < 0 || len(prog) == 0 {
		return
	}
	halt, known := m.oracle.Halts(prog)
	if !known || !halt {
		return
	}
	scratch := make([]float64, len(out))
	copy(scratch, out)
	applyWide(out, scratch, t, m.mixc[mctxOf(suf)], &m.ClassOf)
	normProbs(out)
}

func applyWide(dst, src []float64, t tally, succ map[byte]uint32, classOf *[256]int) {
	kinds := 1
	if !t.pure {
		kinds = len(succ)
	}
	if t.total < 1 || kinds < 1 {
		copy(dst, src)
		return
	}
	esc := float64(kinds) / float64(int(t.total)+kinds)
	scale := 1 / float64(int(t.total)+kinds)
	for i := range dst {
		dst[i] = esc * src[i]
	}
	if t.pure {
		if idx := classOf[t.sym]; idx >= 0 && idx < len(dst) {
			dst[idx] += float64(t.total) * scale
		}
		return
	}
	for b, c := range succ {
		if c == 0 {
			continue
		}
		if idx := classOf[b]; idx >= 0 && idx < len(dst) {
			dst[idx] += float64(c) * scale
		}
	}
}

func (m *MCTSLM) pass(text []byte) (nll float64, correct, scored, total int) {
	if len(text) <= m.Window {
		return 0, 0, 0, 0
	}
	n := len(m.Alphabet)
	p := make([]float64, n)
	for i := m.Window; i < len(text); i++ {
		total++
		class := m.ClassOf[text[i]]
		if class < 0 {
			continue
		}
		scored++
		m.probs(text[:i], p)
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
