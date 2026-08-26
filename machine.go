// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import (
	"errors"
	"fmt"
)

// Op is a BSS / analog register-machine opcode.
type Op uint8

const (
	OpHalt Op = iota
	OpLoad
	OpMove
	OpAdd
	OpSub
	OpMul
	OpQuo
	OpSatLin
	OpFloor
	OpFrac
	OpShift
	OpBit
	OpCantorBit
	OpCantorPop
	OpCantorPush
	OpJump
	OpJZ
	OpJNZ
	OpJGE
	OpJLT
)

func (op Op) String() string {
	names := [...]string{
		"halt", "load", "move", "add", "sub", "mul", "quo",
		"satlin", "floor", "frac", "shift", "bit",
		"cantorbit", "cantorpop", "cantorpush",
		"jump", "jz", "jnz", "jge", "jlt",
	}
	if int(op) < len(names) {
		return names[op]
	}
	return fmt.Sprintf("op(%d)", op)
}

// Instr is one analog-machine instruction.
//
//	Load        rD = Imm
//	Move        rD = rA
//	Add/Sub/Mul/Quo  rD = rA ∘ rB
//	SatLin/Floor/Frac/Shift/Bit/CantorBit  rD = f(rA)
//	CantorPop   rD = rest(rA), rB = bit(rA)
//	CantorPush  rD = push(rA, bit rB)  (rB treated as 0/1)
//	Jump        PC = Addr
//	JZ/JNZ      PC = Addr if rA is zero / nonzero
//	JGE/JLT     PC = Addr if rA ≥ 0 / rA < 0
type Instr struct {
	Op   Op
	D    int
	A    int
	B    int
	Imm  *BitFloat
	Addr int
}

func (in Instr) String() string {
	switch in.Op {
	case OpHalt:
		return "halt"
	case OpLoad:
		return fmt.Sprintf("r%d = %s", in.D, in.Imm)
	case OpJump:
		return fmt.Sprintf("jump %d", in.Addr)
	case OpJZ, OpJNZ, OpJGE, OpJLT:
		return fmt.Sprintf("%s r%d, %d", in.Op, in.A, in.Addr)
	case OpAdd, OpSub, OpMul, OpQuo, OpCantorPop, OpCantorPush:
		return fmt.Sprintf("%s r%d, r%d, r%d", in.Op, in.D, in.A, in.B)
	default:
		return fmt.Sprintf("%s r%d, r%d", in.Op, in.D, in.A)
	}
}

// DefaultMaxSteps is the default instruction budget.
const DefaultMaxSteps uint64 = 10_000_000

var (
	// ErrPC is returned when the program counter is out of range.
	ErrPC = errors.New("program counter out of range")
	// ErrReg is returned when a register index is out of range.
	ErrReg = errors.New("register out of range")
	// ErrDivZero is returned on division by zero.
	ErrDivZero = errors.New("division by zero")
	// ErrSteps is returned when the step limit is exceeded.
	ErrSteps = errors.New("step limit exceeded")
)

// Machine is a BSS-style register machine over BitFloats.
// Comparison branches and analog bit extraction are the
// hypercomputation primitives: they read bits of real registers.
type Machine struct {
	Prec     uint
	R        []*BitFloat
	PC       int
	Steps    uint64
	MaxSteps uint64
}

// NewMachine returns a machine with nregs registers at the given
// bit precision.
func NewMachine(prec uint, nregs int) *Machine {
	if prec == 0 {
		prec = DefaultPrec
	}
	if nregs < 1 {
		nregs = 8
	}
	r := make([]*BitFloat, nregs)
	for i := range r {
		r[i] = New(prec)
	}
	return &Machine{
		Prec:     prec,
		R:        r,
		MaxSteps: DefaultMaxSteps,
	}
}

// Reset clears registers, PC, and the step counter.
func (m *Machine) Reset() {
	for i := range m.R {
		m.R[i] = New(m.Prec)
	}
	m.PC = 0
	m.Steps = 0
}

func (m *Machine) reg(i int) (*BitFloat, error) {
	if i < 0 || i >= len(m.R) {
		return nil, ErrReg
	}
	return m.R[i], nil
}

// Run executes prog until Halt or the step limit.
func (m *Machine) Run(prog []Instr) error {
	m.PC = 0
	m.Steps = 0
	for {
		if m.MaxSteps > 0 && m.Steps >= m.MaxSteps {
			return ErrSteps
		}
		halted, err := m.Step(prog)
		if err != nil {
			return err
		}
		if halted {
			return nil
		}
	}
}

// Step executes one instruction. halted is true on OpHalt.
func (m *Machine) Step(prog []Instr) (halted bool, err error) {
	if m.PC < 0 || m.PC >= len(prog) {
		return false, ErrPC
	}
	in := prog[m.PC]
	switch in.Op {
	case OpHalt:
		return true, nil
	case OpLoad:
		d, err := m.reg(in.D)
		if err != nil {
			return false, err
		}
		if in.Imm == nil {
			d.SetInt64(0)
		} else {
			d.Set(in.Imm)
			d.SetPrec(m.Prec)
		}
		m.PC++
	case OpMove:
		d, a, err := m.unary(in)
		if err != nil {
			return false, err
		}
		d.Set(a)
		m.PC++
	case OpAdd:
		err = m.binary(in, func(d, a, b *BitFloat) { d.Add(a, b) })
	case OpSub:
		err = m.binary(in, func(d, a, b *BitFloat) { d.Sub(a, b) })
	case OpMul:
		err = m.binary(in, func(d, a, b *BitFloat) { d.Mul(a, b) })
	case OpQuo:
		d, a, b, e := m.ternary(in)
		if e != nil {
			return false, e
		}
		if b.Sign() == 0 {
			return false, ErrDivZero
		}
		d.Quo(a, b)
		m.PC++
	case OpSatLin:
		d, a, err := m.unary(in)
		if err != nil {
			return false, err
		}
		d.SatLin(a)
		m.PC++
	case OpFloor:
		d, a, err := m.unary(in)
		if err != nil {
			return false, err
		}
		d.Floor(a)
		m.PC++
	case OpFrac:
		d, a, err := m.unary(in)
		if err != nil {
			return false, err
		}
		d.Frac(a)
		m.PC++
	case OpShift:
		d, a, err := m.unary(in)
		if err != nil {
			return false, err
		}
		d.Shift(a)
		m.PC++
	case OpBit:
		d, a, err := m.unary(in)
		if err != nil {
			return false, err
		}
		d.SetInt64(int64(a.Bit()))
		m.PC++
	case OpCantorBit:
		d, a, err := m.unary(in)
		if err != nil {
			return false, err
		}
		d.SetInt64(int64(a.CantorBit()))
		m.PC++
	case OpCantorPop:
		d, a, b, e := m.ternary(in)
		if e != nil {
			return false, e
		}
		bit := d.CantorPop(a)
		b.SetInt64(int64(bit))
		m.PC++
	case OpCantorPush:
		d, a, b, e := m.ternary(in)
		if e != nil {
			return false, e
		}
		bit := 0
		if b.Bit() == 1 || b.Cmp(FromInt(m.Prec, 1)) >= 0 {
			bit = 1
		}
		d.CantorPush(a, bit)
		m.PC++
	case OpJump:
		m.PC = in.Addr
	case OpJZ:
		a, e := m.reg(in.A)
		if e != nil {
			return false, e
		}
		if a.Sign() == 0 {
			m.PC = in.Addr
		} else {
			m.PC++
		}
	case OpJNZ:
		a, e := m.reg(in.A)
		if e != nil {
			return false, e
		}
		if a.Sign() != 0 {
			m.PC = in.Addr
		} else {
			m.PC++
		}
	case OpJGE:
		a, e := m.reg(in.A)
		if e != nil {
			return false, e
		}
		if a.Sign() >= 0 {
			m.PC = in.Addr
		} else {
			m.PC++
		}
	case OpJLT:
		a, e := m.reg(in.A)
		if e != nil {
			return false, e
		}
		if a.Sign() < 0 {
			m.PC = in.Addr
		} else {
			m.PC++
		}
	default:
		return false, fmt.Errorf("unknown opcode %v", in.Op)
	}
	if err != nil {
		return false, err
	}
	m.Steps++
	return false, nil
}

func (m *Machine) unary(in Instr) (d, a *BitFloat, err error) {
	d, err = m.reg(in.D)
	if err != nil {
		return
	}
	a, err = m.reg(in.A)
	return
}

func (m *Machine) ternary(in Instr) (d, a, b *BitFloat, err error) {
	d, err = m.reg(in.D)
	if err != nil {
		return
	}
	a, err = m.reg(in.A)
	if err != nil {
		return
	}
	b, err = m.reg(in.B)
	return
}

func (m *Machine) binary(in Instr, fn func(d, a, b *BitFloat)) error {
	d, a, b, err := m.ternary(in)
	if err != nil {
		return err
	}
	fn(d, a, b)
	m.PC++
	return nil
}

// QueryBit extracts binary expansion bit k of register src by
// iterating the Bernoulli map. src is not modified.
func (m *Machine) QueryBit(src, k int) (int, error) {
	a, err := m.reg(src)
	if err != nil {
		return 0, err
	}
	return a.BitAt(k), nil
}

// Load sets register d to v (copied at machine precision).
func (m *Machine) Load(d int, v *BitFloat) error {
	r, err := m.reg(d)
	if err != nil {
		return err
	}
	r.Set(v)
	r.SetPrec(m.Prec)
	return nil
}

// HaltQuery is a BSS program that writes bit r[idx] of the binary
// oracle in r[src] into r[dst]. It destroys r[src] and r[idx].
// r[one] must hold 1.
func HaltQuery(src, idx, dst, one int) []Instr {
	return []Instr{
		{Op: OpJZ, A: idx, Addr: 4},
		{Op: OpShift, D: src, A: src},
		{Op: OpSub, D: idx, A: idx, B: one},
		{Op: OpJump, Addr: 0},
		{Op: OpBit, D: dst, A: src},
		{Op: OpHalt},
	}
}

// NewtonSqrt is a BSS program for Newton's method:
//
//	r[guess] ← (r[guess] + r[x]/r[guess]) / 2
//
// iterated r[n] times. r[two] must hold 2, r[one] must hold 1.
// r[scratch] is scratch. Destroys r[n].
func NewtonSqrt(x, guess, n, two, one, scratch int) []Instr {
	return []Instr{
		{Op: OpJZ, A: n, Addr: 6},
		{Op: OpQuo, D: scratch, A: x, B: guess},
		{Op: OpAdd, D: scratch, A: guess, B: scratch},
		{Op: OpQuo, D: guess, A: scratch, B: two},
		{Op: OpSub, D: n, A: n, B: one},
		{Op: OpJump, Addr: 0},
		{Op: OpHalt},
	}
}
