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
	demo := flag.String("demo", "all", "bits, oracle, zeno, omega, arnn, newton, or all")
	prec := flag.Uint("prec", 256, "mantissa precision in bits")
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
	case "all":
		demoBits(*prec)
		demoOracle(*prec)
		demoZeno(*prec)
		demoOmega(*prec)
		demoARNN(*prec)
		demoNewton(*prec)
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
	fmt.Println("== Zeno machine (accelerated TM) ==")
	bb := hc.BusyBeaver2()
	z := hc.NewZeno(prec, bb)
	z.Run(32)
	fmt.Printf("BB(2): halted=%v steps=%d analog-time=%s\n", z.Halted(), z.Steps, z.Time.Text('g', 20))
	var ones []byte
	for d := -8; d <= 8; d++ {
		ones = append(ones, byte('0'+z.Analog.TapeBit(d)))
	}
	fmt.Printf("analog tape around head: %s (head at offset 0)\n", ones)

	paint := &hc.TM{
		States: 1,
		Start:  0,
		Delta: [][2]hc.Transition{
			{{Write: 1, Move: 1, Next: 0}, {Write: 1, Move: 1, Next: 0}},
		},
	}
	z2 := hc.NewZeno(prec, paint)
	z2.Run(24)
	fmt.Printf("non-halting paint: halted=%v steps=%d time→%s\n", z2.Halted(), z2.Steps, z2.Time.Text('g', 20))
	fmt.Println()
}

func demoOmega(prec uint) {
	fmt.Println("== Chaitin Ω approximation ==")
	for _, n := range []int{2, 4, 6} {
		w := hc.Omega(1, n, 32, prec)
		fmt.Printf("max index bits %d: Ω ≈ %s\n", n, w.Text('g', 24))
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
