// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	hc "github.com/pointlander/hypercomputer"
)

func main() {
	demo := flag.String("demo", "all", "bits, oracle, zeno, omega, arnn, newton, quantum, kcomplexity, chaitin, sweep, lm, or all")
	prec := flag.Uint("prec", 256, "mantissa precision in bits")
	kstring := flag.String("kstring", "", "bit string for k-complexity (e.g. 1111)")
	kbits := flag.Int("kbits", 12, "max U-program length for k-complexity search")
	text := flag.String("text", "pg100.txt", "corpus for -demo=lm (Project Gutenberg text)")
	flag.Parse()

	switch *demo {
	case "bits":
		demoBits(*prec)
	case "oracle":
		demoOracle(*prec)
	case "zeno":
		demoZeno(*prec)
	case "omega":
		demoOmega(*prec)
	case "arnn":
		demoARNN(*prec)
	case "newton":
		demoNewton(*prec)
	case "quantum":
		demoQuantum(*prec)
	case "kcomplexity":
		demoKComplexity(*prec, *kstring, *kbits)
	case "chaitin":
		demoChaitin(*prec, *kstring, *kbits)
	case "sweep":
		demoSweep(*prec)
	case "lm":
		demoLM(*text, *prec, *kbits)
	case "all":
		demoBits(*prec)
		demoOracle(*prec)
		demoZeno(*prec)
		demoOmega(*prec)
		demoARNN(*prec)
		demoNewton(*prec)
		demoQuantum(*prec)
		demoKComplexity(*prec, *kstring, *kbits)
		demoChaitin(*prec, *kstring, *kbits)
		demoSweep(*prec)
	default:
		fmt.Fprintf(os.Stderr, "unknown demo %q\n", *demo)
		flag.Usage()
		os.Exit(2)
	}
}

func demoBits(prec uint) {
	fmt.Println("== analog bit tape ==")
	msg := "110010010000111111011010"
	bits := make([]bool, len(msg))
	for i, c := range msg {
		bits[i] = c == '1'
	}
	x := hc.FromBits(prec, bits)
	fmt.Printf("prec=%d  0.%s\n", prec, x.Binary(len(bits)))
	fmt.Printf("decoded  0.%s\n", bitsString(x.Bits(len(bits))))

	c := hc.FromCantor(prec, bits)
	fmt.Printf("cantor   0.%s  (value %s)\n", bitsString(c.CantorBits(len(bits))), c.Text('g', 24))
	fmt.Println()
}

func demoOracle(prec uint) {
	fmt.Println("== bounded halt oracle as a real ==")
	const (
		nstates = 1
		bound   = 32
	)
	oracle, halted := hc.HaltOracle(nstates, bound, prec)
	n := 0
	for _, h := range halted {
		if h {
			n++
		}
	}
	fmt.Printf("%d-state 2-symbol TMs: %d  halted within %d steps: %d\n",
		nstates, len(halted), bound, n)
	fmt.Printf("oracle 0.%s...\n", oracle.Binary(64))

	m := hc.NewMachine(oracle.Prec(), 8)
	_ = m.Load(2, hc.FromInt(oracle.Prec(), 1))
	fmt.Println("BSS queries:")
	shown := 0
	for i, h := range halted {
		if shown >= 8 && !h {
			continue
		}
		_ = m.Load(0, oracle)
		m.R[1].SetInt64(int64(i))
		m.PC = 0
		m.Steps = 0
		if err := m.Run(hc.HaltQuery(0, 1, 3, 2)); err != nil {
			panic(err)
		}
		got := m.R[3].Sign() != 0
		if got != h {
			panic(fmt.Sprintf("query mismatch at %d", i))
		}
		if shown < 8 || h {
			fmt.Printf("  TM %2d halt=%v  (analog bit, %d steps)\n", i, got, m.Steps)
			shown++
		}
		if shown >= 12 {
			break
		}
	}
	fmt.Println()
}

func demoZeno(prec uint) {
	fmt.Println("== Zeno ω-limit (doubling n=1,2,…,2^k) ==")
	k := 5
	type row struct {
		name string
		tm   *hc.TM
	}
	for _, r := range []row{
		{"BB(2)", hc.BusyBeaver2()},
		{"write-1", hc.WriteOneHalt()},
		{"idle", hc.IdleBlank()},
		{"paint", hc.PaintRight()},
	} {
		lim := hc.NewZeno(prec, r.tm).Limit(k)
		fmt.Printf("%-8s %s\n", r.name, lim)
	}
	fmt.Println()
}

func demoOmega(prec uint) {
	fmt.Println("== Chaitin Ω_U (prefix-free U) ==")
	for _, n := range []int{4, 6, 8} {
		w := hc.Omega(n, 256, prec)
		fmt.Printf("|p|≤%d: Ω ≈ %s\n", n, w.Text('g', 24))
	}
	fmt.Println()
}

func demoARNN(prec uint) {
	fmt.Println("== analog net reading a Cantor oracle ==")
	msg := "101100111000"
	bits := make([]bool, len(msg))
	for i, c := range msg {
		bits[i] = c == '1'
	}
	q := hc.FromCantor(prec, bits)
	r := hc.NewCantorReader(q)
	got := r.ReadN(len(bits))
	fmt.Printf("encoded %s\n", msg)
	fmt.Printf("satlin  %s\n", bitsString(got))
	fmt.Println()
}

func demoNewton(prec uint) {
	fmt.Println("== BSS Newton sqrt(2) ==")
	m := hc.NewMachine(prec, 8)
	m.R[0].SetInt64(2)
	m.R[1].SetInt64(1)
	m.R[2].SetInt64(16)
	m.R[3].SetInt64(2)
	m.R[4].SetInt64(1)
	if err := m.Run(hc.NewtonSqrt(0, 1, 2, 3, 4, 5)); err != nil {
		panic(err)
	}
	want := hc.New(prec).Sqrt(hc.FromInt(prec, 2))
	fmt.Printf("newton  %s\n", m.R[1].Text('g', 48))
	fmt.Printf("sqrt    %s\n", want.Text('g', 48))
	fmt.Println()
}

func demoQuantum(prec uint) {
	fmt.Println("== quantum circuits ==")
	bell := hc.Bell(prec)
	fmt.Println("Bell (|00⟩+|11⟩)/√2:")
	fmt.Print(bell)

	ghz := hc.GHZ(prec, 3)
	fmt.Println("GHZ₃:")
	fmt.Print(ghz)

	g := hc.Grover2(prec)
	fmt.Println("Grover₂ targeting |11⟩:")
	fmt.Print(g)

	c := hc.Deutsch(prec, false)
	_, p1 := c.ProbQubit(0)
	fmt.Printf("Deutsch constant  P(q0=1)=%s\n", p1.Text('g', 12))
	b := hc.Deutsch(prec, true)
	_, p1 = b.ProbQubit(0)
	fmt.Printf("Deutsch balanced  P(q0=1)=%s\n", p1.Text('g', 12))

	qft := hc.NewQState(prec, 3)
	qft.QFT()
	fmt.Println("QFT|000⟩ (uniform):")
	fmt.Print(qft)

	tel := hc.Teleport(prec, func(c *hc.QCircuit) { c.X(0) })
	_, p1 = tel.ProbQubit(2)
	fmt.Printf("teleport |1⟩ → q2  P(q2=1)=%s\n", p1.Text('g', 12))
	fmt.Println()
}

func demoSweep(prec uint) {
	fmt.Println("== p,T resource sweep (analog Ω bits × dovetail bound) ==")
	fmt.Println("entry = certified K_U, or >p if refused, or ? if Ω prefix not caught")
	s := hc.SweepResources(nil, nil, nil, 0, prec)
	fmt.Print(s)
	fmt.Println()
}

func demoChaitin(prec uint, kstring string, kbits int) {
	fmt.Println("== Chaitin reconstruction of K_U from analog Ω ==")
	if kbits < 1 {
		kbits = 8
	}
	maxB := kbits
	if kstring == "" && maxB < 9 {
		maxB = 9
	}
	m := maxB
	if m > 10 {
		m = 10
	}
	om := hc.Omega(maxB, 256, prec)
	bits, hat, steps := hc.OmegaPrefix(om, m)
	fmt.Printf("Ω first %d bits = %s  (ω=%s, analog-shifts=%d)\n",
		m, hc.FormatBits(bits), hat.Text('g', 12), steps)
	rec := hc.Reconstruct(om, m, maxB, 256)
	fmt.Printf("dovetail caught=%v stage=%d programs=%d sum=%s\n",
		rec.Caught, rec.Stage, len(rec.Progs), rec.Sum.Text('g', 12))
	run := func(s string, bitsM int) {
		x := hc.ParseBitString(s)
		r := hc.KFromOmegaValue(x, om, bitsM, maxB, 256)
		fmt.Println(r)
	}
	if kstring != "" {
		run(kstring, m)
		if m > 1 {
			fmt.Printf("one fewer Ω bit: ")
			run(kstring, m-1)
		}
		fmt.Println()
		return
	}
	run("", 3)
	run("1", 6)
	run("11", 9)
	run("1", 5)
	bad := hc.New(prec).Add(om, hc.FromRat(prec, 1, 2))
	fmt.Println(hc.KFromOmegaValue(hc.ParseBitString("1"), bad, 6, maxB, 256))
	fmt.Println()
}

func demoKComplexity(prec uint, kstring string, kbits int) {
	fmt.Println("== K_U via prefix-free U (same U as Ω) ==")
	if kbits < 1 {
		kbits = 12
	}
	run := func(s string) {
		x := hc.ParseBitString(s)
		r := hc.KDivideConquer(x, kbits, 256, 0, prec)
		fmt.Println(r)
		if r.AnalogOK {
			fmt.Printf("  analog oracle agrees, queries=%d analog-steps=%d\n",
				r.Queries, r.AnalogSteps)
		}
	}
	if kstring != "" {
		run(kstring)
		fmt.Println()
		return
	}
	for _, s := range []string{"", "0", "1", "11", "101", "1111", "1111111111111111"} {
		run(s)
	}
	fmt.Println()
}

func demoLM(path string, prec uint, kbits int) {
	fmt.Println("== ΔK vs one-hot next-byte LM ==")
	body, err := hc.LoadCorpus(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "lm: %v\n", err)
		os.Exit(1)
	}
	if kbits < 1 {
		kbits = 12
	}
	dk, dkRep, hotRep, err := hc.CompareDeltaK(body, hc.LMConfig{
		MaxBits: kbits,
		Prec:    prec,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "lm: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("corpus %s  body %d bytes  alphabet %d  window %d\n",
		path, len(body), len(dk.Alphabet), dk.Window)
	for _, ex := range []struct {
		ctx string
		b   byte
	}{
		{"    ", ' '},
		{"    ", 'e'},
		{"the ", ' '},
		{"\n\n\n\n", '\n'},
	} {
		fmt.Printf("  ΔK(%q | %q) = %d\n", ex.b, ex.ctx, dk.Delta([]byte(ex.ctx), ex.b))
	}
	fmt.Printf("ΔK      train ppl %.2f acc %.3f  valid ppl %.2f acc %.3f\n",
		dkRep.TrainPPL, dkRep.TrainAcc, dkRep.ValidPPL, dkRep.ValidAcc)
	fmt.Printf("one-hot train ppl %.2f acc %.3f  valid ppl %.2f acc %.3f\n",
		hotRep.TrainPPL, hotRep.TrainAcc, hotRep.ValidPPL, hotRep.ValidAcc)
	fmt.Println()
}

func bitsString(b []bool) string {
	var s strings.Builder
	s.Grow(len(b))
	for _, v := range b {
		if v {
			s.WriteByte('1')
		} else {
			s.WriteByte('0')
		}
	}
	return s.String()
}
