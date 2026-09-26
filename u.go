// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

// Prefix-free reference machine U.
//
// Programs are bit strings consumed from a one-way input. U fetches
// instructions on demand; |p| is the number of bits read at halt.
// A string that is a proper extension of a halting program is not
// itself a program (the shorter one already halted). This is a
// prefix-free code, so Σ 2^{-|p|} ≤ 1 over valid p.
//
// ISA (MSB first):
//
//	000       HALT
//	001       OUT 0
//	010       OUT 1
//	011       INC
//	100       DEC          (saturate at 0)
//	101       JZ           skip the next instruction if R = 0
//	110 rrrr  JMP          pc -= r+1
//	111 vvvv  SET          R = v (0..15)
//
// Non-halting programs (JMP 0, JZ/JMP loops, REP-style SET 0
// never taken) make the analog halt oracle necessary.

const (
	uOpHalt = 0
	uOpOut0 = 1
	uOpOut1 = 2
	uOpInc  = 3
	uOpDec  = 4
	uOpJz   = 5
	uOpJmp  = 6
	uOpSet  = 7

	// uOutLimit caps a looping OUT so a huge step bound cannot
	// allocate without limit. It sits well above file-sized inputs.
	uOutLimit = 1 << 22

	// DefaultUMaxBits is the default brute-force program length.
	DefaultUMaxBits = 12
	// DefaultUBound is the default U step bound.
	DefaultUBound = 256
)

// UStatus is the result of running U on a bit buffer.
type UStatus int

const (
	// UHalt: halted after reading Read bits.
	UHalt UStatus = iota
	// UBlock: tried to fetch past the end of the buffer.
	UBlock
	// UTimeout: exceeded the step bound.
	UTimeout
)

// UResult is one run of the prefix-free machine U.
type UResult struct {
	Status UStatus
	Out    []bool
	Read   int
	Steps  int
}

type uInst struct {
	op, arg int
}

func uFetch(src []bool, in *int, k int) (int, bool) {
	if *in+k > len(src) {
		return 0, false
	}
	v := 0
	for i := 0; i < k; i++ {
		v <<= 1
		if src[*in] {
			v |= 1
		}
		*in++
	}
	return v, true
}

func uFetchInst(src []bool, in *int) (uInst, bool) {
	op, ok := uFetch(src, in, 3)
	if !ok {
		return uInst{}, false
	}
	inst := uInst{op: op}
	if op == uOpJmp || op == uOpSet {
		inst.arg, ok = uFetch(src, in, 4)
		if !ok {
			return uInst{}, false
		}
	}
	return inst, true
}

// RunU runs the prefix-free machine U on src for at most bound steps.
// If bound ≤ 0, DefaultUBound is used.
func RunU(src []bool, bound int) UResult {
	if bound <= 0 {
		bound = DefaultUBound
	}
	var (
		in    int
		r     int
		out   []bool
		code  []uInst
		pc    int
		steps int
	)
	for steps = 0; steps < bound; steps++ {
		if pc < 0 {
			pc = 0
		}
		for pc >= len(code) {
			inst, ok := uFetchInst(src, &in)
			if !ok {
				return UResult{Status: UBlock, Out: out, Read: in, Steps: steps}
			}
			code = append(code, inst)
		}
		ins := code[pc]
		switch ins.op {
		case uOpHalt:
			return UResult{Status: UHalt, Out: out, Read: in, Steps: steps + 1}
		case uOpOut0:
			if len(out) < uOutLimit {
				out = append(out, false)
			}
			pc++
		case uOpOut1:
			if len(out) < uOutLimit {
				out = append(out, true)
			}
			pc++
		case uOpInc:
			r++
			pc++
		case uOpDec:
			if r > 0 {
				r--
			}
			pc++
		case uOpJz:
			if r == 0 {
				pc += 2
			} else {
				pc++
			}
		case uOpJmp:
			pc -= ins.arg + 1
		case uOpSet:
			r = ins.arg
			pc++
		default:
			pc++
		}
	}
	return UResult{Status: UTimeout, Out: out, Read: in, Steps: steps}
}

// ValidUProgram reports whether src is a complete prefix-free program:
// U halts and consumes every bit.
func ValidUProgram(src []bool, bound int) bool {
	res := RunU(src, bound)
	return res.Status == UHalt && res.Read == len(src)
}

func putBits(p *[]bool, v, k int) {
	for i := k - 1; i >= 0; i-- {
		*p = append(*p, (v>>i)&1 == 1)
	}
}

// ListingProgram is the print program for x: OUT each bit, then HALT.
// |p| = 3(|x| + 1).
func ListingProgram(x []bool) []bool {
	p := make([]bool, 0, 3*len(x)+3)
	for _, b := range x {
		if b {
			putBits(&p, uOpOut1, 3)
		} else {
			putBits(&p, uOpOut0, 3)
		}
	}
	putBits(&p, uOpHalt, 3)
	return p
}

// RepeatProgram emits bit `one` exactly k times using SET/DEC/JZ/JMP.
// k = 0 is HALT (empty output).
func RepeatProgram(one bool, k int) []bool {
	if k <= 0 {
		p := make([]bool, 0, 3)
		putBits(&p, uOpHalt, 3)
		return p
	}
	p := make([]bool, 0, 32+3*k)
	extra := 0
	setv := k
	if setv > 15 {
		extra = setv - 15
		setv = 15
	}
	putBits(&p, uOpSet, 3)
	putBits(&p, setv, 4)
	for i := 0; i < extra; i++ {
		putBits(&p, uOpInc, 3)
	}
	out := uOpOut0
	if one {
		out = uOpOut1
	}
	putBits(&p, out, 3)
	putBits(&p, uOpDec, 3)
	putBits(&p, uOpJz, 3)
	putBits(&p, uOpJmp, 3)
	// JMP is at index 2+extra+2 = extra+4; OUT is at extra+1.
	// pc -= arg+1 lands on OUT: (extra+4) - (arg+1) = extra+1 ⇒ arg = 2.
	putBits(&p, 2, 4)
	putBits(&p, uOpHalt, 3)
	return p
}

// ByteRunProgram emits the 8 bits of b exactly k times, then halts.
// k = 0 is HALT. For k ≤ 15 the program is 47 bits, shorter than a
// listing once the byte is repeated.
func ByteRunProgram(b byte, k int) []bool {
	if k <= 0 {
		p := make([]bool, 0, 3)
		putBits(&p, uOpHalt, 3)
		return p
	}
	p := make([]bool, 0, 64+3*k)
	extra := 0
	setv := k
	if setv > 15 {
		extra = setv - 15
		setv = 15
	}
	putBits(&p, uOpSet, 3)
	putBits(&p, setv, 4)
	for i := 0; i < extra; i++ {
		putBits(&p, uOpInc, 3)
	}
	for i := 7; i >= 0; i-- {
		if b&(1<<uint(i)) != 0 {
			putBits(&p, uOpOut1, 3)
		} else {
			putBits(&p, uOpOut0, 3)
		}
	}
	putBits(&p, uOpDec, 3)
	putBits(&p, uOpJz, 3)
	putBits(&p, uOpJmp, 3)
	// First OUT is instruction extra+1, JMP is extra+11.
	// (extra+11) - (arg+1) = extra+1 ⇒ arg = 9.
	putBits(&p, 9, 4)
	putBits(&p, uOpHalt, 3)
	return p
}

// IntBits is the n-bit big-endian representation of i.
func IntBits(n, i int) []bool {
	b := make([]bool, n)
	for j := 0; j < n; j++ {
		b[n-1-j] = (i>>j)&1 == 1
	}
	return b
}

// NumUBuffers is the number of bit strings of length 1..maxBits.
func NumUBuffers(maxBits int) int {
	if maxBits < 1 {
		return 0
	}
	return (1 << (maxBits + 1)) - 2
}

// UBufferIndex is the analog-oracle index of the n-bit string i
// (n ≥ 1, 0 ≤ i < 2^n): Σ_{k=1}^{n-1} 2^k + i = 2^n − 2 + i.
func UBufferIndex(n, i int) int {
	return (1 << n) - 2 + i
}
