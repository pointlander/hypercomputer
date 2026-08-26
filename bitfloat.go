// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import (
	"math/big"
)

const (
	// DefaultPrec is the default mantissa precision in bits.
	DefaultPrec uint = 256
	// GuardBits is extra mantissa bits kept so analog shifts and
	// Cantor pops do not round away readable oracle bits.
	GuardBits uint = 64
)

// BitFloat is a high-precision binary floating-point number.
// Precision is the mantissa width in bits. Analog hypercomputation
// primitives (Bernoulli shift, Cantor stack) treat the fractional
// bits as an infinite tape.
type BitFloat struct {
	f *big.Float
}

// New returns a BitFloat 0 with the given mantissa precision.
func New(prec uint) *BitFloat {
	if prec == 0 {
		prec = DefaultPrec
	}
	return &BitFloat{
		f: new(big.Float).SetPrec(prec).SetMode(big.ToNearestEven),
	}
}

// PrecBits returns a precision large enough to hold n analog bits
// plus guard bits.
func PrecBits(n int) uint {
	if n < 0 {
		n = 0
	}
	return uint(n) + GuardBits
}

// FromInt returns n as a BitFloat.
func FromInt(prec uint, n int64) *BitFloat {
	z := New(prec)
	z.f.SetInt64(n)
	return z
}

// FromRat returns num/den as a BitFloat.
func FromRat(prec uint, num, den int64) *BitFloat {
	z := New(prec)
	z.f.SetRat(big.NewRat(num, den))
	return z
}

// FromFloat64 returns x as a BitFloat.
func FromFloat64(prec uint, x float64) *BitFloat {
	z := New(prec)
	z.f.SetFloat64(x)
	return z
}

// FromBits encodes bits as the binary fraction 0.b0 b1 b2 ...
// with trailing zeros. Exact when prec >= len(bits).
func FromBits(prec uint, bits []bool) *BitFloat {
	if prec == 0 {
		prec = PrecBits(len(bits))
	}
	z := New(prec)
	if len(bits) == 0 {
		return z
	}
	n := new(big.Int)
	for _, b := range bits {
		n.Lsh(n, 1)
		if b {
			n.SetBit(n, 0, 1)
		}
	}
	num := new(big.Float).SetPrec(prec).SetInt(n)
	den := new(big.Float).SetPrec(prec).SetInt(new(big.Int).Lsh(big.NewInt(1), uint(len(bits))))
	z.f.Quo(num, den)
	return z
}

// CantorZeros is the Cantor stack of infinite zeros: Σ 4^{-i} = 1/3.
func CantorZeros(prec uint) *BitFloat {
	return FromRat(prec, 1, 3)
}

// FromCantor encodes bits as a base-4 Cantor stack
//
//	x = Σ_i (2 b_i + 1) / 4^{i+1}
//
// with an infinite tail of zeros (value 1/3). Each trit is 1 or 3,
// so the representation is unique and stable under analog satlin
// extraction (Siegelmann–Sontag).
func FromCantor(prec uint, bits []bool) *BitFloat {
	if prec == 0 {
		prec = PrecBits(2 * len(bits))
	}
	z := CantorZeros(prec)
	for i := len(bits) - 1; i >= 0; i-- {
		bit := 0
		if bits[i] {
			bit = 1
		}
		z.CantorPush(z, bit)
	}
	return z
}

// Prec returns the mantissa precision in bits.
func (x *BitFloat) Prec() uint {
	if x == nil || x.f == nil {
		return 0
	}
	return x.f.Prec()
}

// SetPrec sets the mantissa precision in bits.
func (x *BitFloat) SetPrec(prec uint) *BitFloat {
	x.ensure(prec)
	return x
}

func (z *BitFloat) ensure(prec uint) {
	if z.f == nil {
		z.f = new(big.Float).SetMode(big.ToNearestEven)
	}
	if prec > 0 && z.f.Prec() != prec {
		z.f.SetPrec(prec)
	}
}

func (z *BitFloat) fit(xs ...*BitFloat) {
	p := z.Prec()
	for _, x := range xs {
		if x != nil && x.Prec() > p {
			p = x.Prec()
		}
	}
	z.ensure(p)
}

// Copy returns a deep copy.
func (x *BitFloat) Copy() *BitFloat {
	z := New(x.Prec())
	z.f.Copy(x.f)
	return z
}

// Set copies x into z.
func (z *BitFloat) Set(x *BitFloat) *BitFloat {
	z.fit(x)
	z.f.Copy(x.f)
	return z
}

// SetInt64 sets z to n.
func (z *BitFloat) SetInt64(n int64) *BitFloat {
	z.ensure(z.Prec())
	z.f.SetInt64(n)
	return z
}

// Sign returns -1, 0, or 1.
func (x *BitFloat) Sign() int {
	return x.f.Sign()
}

// Cmp compares x and y.
func (x *BitFloat) Cmp(y *BitFloat) int {
	return x.f.Cmp(y.f)
}

// IsInt reports whether x is an integer.
func (x *BitFloat) IsInt() bool {
	return x.f.IsInt()
}

// Acc returns the mantissa as a big.Float for inspection.
func (x *BitFloat) Big() *big.Float {
	return new(big.Float).Copy(x.f)
}

// Float64 returns the nearest float64 value.
func (x *BitFloat) Float64() float64 {
	v, _ := x.f.Float64()
	return v
}

// String returns a decimal representation with digits matching the
// mantissa precision.
func (x *BitFloat) String() string {
	digits := int(x.Prec()) * 301 / 1000
	if digits < 8 {
		digits = 8
	}
	return x.f.Text('g', digits)
}

// Text is math/big.Float.Text.
func (x *BitFloat) Text(format byte, prec int) string {
	return x.f.Text(format, prec)
}

// Add sets z to x + y.
func (z *BitFloat) Add(x, y *BitFloat) *BitFloat {
	z.fit(x, y)
	z.f.Add(x.f, y.f)
	return z
}

// Sub sets z to x - y.
func (z *BitFloat) Sub(x, y *BitFloat) *BitFloat {
	z.fit(x, y)
	z.f.Sub(x.f, y.f)
	return z
}

// Mul sets z to x * y.
func (z *BitFloat) Mul(x, y *BitFloat) *BitFloat {
	z.fit(x, y)
	z.f.Mul(x.f, y.f)
	return z
}

// Quo sets z to x / y.
func (z *BitFloat) Quo(x, y *BitFloat) *BitFloat {
	z.fit(x, y)
	z.f.Quo(x.f, y.f)
	return z
}

// Neg sets z to -x.
func (z *BitFloat) Neg(x *BitFloat) *BitFloat {
	z.fit(x)
	z.f.Neg(x.f)
	return z
}

// Abs sets z to |x|.
func (z *BitFloat) Abs(x *BitFloat) *BitFloat {
	z.fit(x)
	z.f.Abs(x.f)
	return z
}

// Sqrt sets z to √x.
func (z *BitFloat) Sqrt(x *BitFloat) *BitFloat {
	z.fit(x)
	z.f.Sqrt(x.f)
	return z
}

// Floor sets z to ⌊x⌋ (toward −∞).
func (z *BitFloat) Floor(x *BitFloat) *BitFloat {
	z.fit(x)
	if x.f.IsInf() {
		z.f.Copy(x.f)
		return z
	}
	i, acc := x.f.Int(nil)
	if x.f.Sign() < 0 && acc != big.Exact {
		i.Sub(i, big.NewInt(1))
	}
	z.f.SetInt(i)
	return z
}

// Frac sets z to x − ⌊x⌋, in [0, 1).
func (z *BitFloat) Frac(x *BitFloat) *BitFloat {
	fl := New(x.Prec()).Floor(x)
	return z.Sub(x, fl)
}

// SatLin sets z to the saturated-linear activation
//
//	0    if x < 0
//	x    if 0 ≤ x ≤ 1
//	1    if x > 1
//
// used by analog recurrent nets.
func (z *BitFloat) SatLin(x *BitFloat) *BitFloat {
	z.fit(x)
	zero := FromInt(x.Prec(), 0)
	one := FromInt(x.Prec(), 1)
	if x.Cmp(zero) <= 0 {
		z.f.SetInt64(0)
		return z
	}
	if x.Cmp(one) >= 0 {
		z.f.SetInt64(1)
		return z
	}
	z.f.Copy(x.f)
	return z
}

// Bit returns the leading fractional bit of x: 1 if x ≥ 1/2, else 0.
// For x in [0, 1) this is the first bit of the binary expansion.
func (x *BitFloat) Bit() int {
	half := FromRat(x.Prec(), 1, 2)
	if x.Cmp(half) >= 0 {
		return 1
	}
	return 0
}

// Shift sets z to the Bernoulli map T(x) = 2x − ⌊2x⌋ (2x mod 1).
// Iterating Shift and Bit reads the binary expansion of x.
func (z *BitFloat) Shift(x *BitFloat) *BitFloat {
	z.fit(x)
	two := FromInt(x.Prec(), 2)
	z.Mul(two, x)
	fl := New(z.Prec()).Floor(z)
	return z.Sub(z, fl)
}

// Bits returns the first n bits of the binary expansion of x,
// assuming x ∈ [0, 1). Uses a copy; x is not modified.
func (x *BitFloat) Bits(n int) []bool {
	y := x.Copy()
	tmp := New(x.Prec())
	bits := make([]bool, n)
	for i := 0; i < n; i++ {
		bits[i] = y.Bit() == 1
		tmp.Shift(y)
		y.Set(tmp)
	}
	return bits
}

// BitAt returns bit k of the binary expansion of x (0-based).
func (x *BitFloat) BitAt(k int) int {
	y := x.Copy()
	tmp := New(x.Prec())
	for i := 0; i < k; i++ {
		tmp.Shift(y)
		y.Set(tmp)
	}
	return y.Bit()
}

// Binary returns the first n fractional bits as a string of '0'/'1'.
func (x *BitFloat) Binary(n int) string {
	bits := x.Bits(n)
	b := make([]byte, n)
	for i, bit := range bits {
		if bit {
			b[i] = '1'
		} else {
			b[i] = '0'
		}
	}
	return string(b)
}

// Mantissa returns sign, binary exponent, and mantissa bits such that
//
//	x = sign × 2^exp × 0.bits
//
// with 0.5 ≤ |mantissa| < 1 for x ≠ 0 (IEEE-style).
func (x *BitFloat) Mantissa() (sign int, exp int, bits []bool) {
	sign = x.Sign()
	if sign == 0 {
		return 0, 0, make([]bool, int(x.Prec()))
	}
	m := new(big.Float).SetPrec(x.Prec())
	exp = x.f.MantExp(m)
	if sign < 0 {
		m.Neg(m)
	}
	return sign, exp, (&BitFloat{f: m}).Bits(int(x.Prec()))
}

// CantorBit returns the top bit of a Cantor-encoded stack:
// σ(4x − 2) ∈ {0, 1} for a well-formed encoding.
func (x *BitFloat) CantorBit() int {
	t := New(x.Prec()).Mul(FromInt(x.Prec(), 4), x)
	t.Sub(t, FromInt(x.Prec(), 2))
	t.SatLin(t)
	if t.Cmp(FromRat(x.Prec(), 1, 2)) >= 0 {
		return 1
	}
	return 0
}

// CantorRest sets z to the stack x with its top bit popped:
//
//	z = 4x − (2·bit + 1)
func (z *BitFloat) CantorRest(x *BitFloat) *BitFloat {
	bit := x.CantorBit()
	fourx := New(x.Prec()).Mul(FromInt(x.Prec(), 4), x)
	return z.Sub(fourx, FromInt(x.Prec(), int64(2*bit+1)))
}

// CantorPop returns the top bit of x and sets z to the remainder.
func (z *BitFloat) CantorPop(x *BitFloat) int {
	bit := x.CantorBit()
	z.CantorRest(x)
	return bit
}

// CantorPush sets z to the stack x with bit pushed on top:
//
//	z = (x + 2·bit + 1) / 4
func (z *BitFloat) CantorPush(x *BitFloat, bit int) *BitFloat {
	if bit < 0 {
		bit = 0
	}
	if bit > 1 {
		bit = 1
	}
	z.fit(x)
	z.Add(x, FromInt(x.Prec(), int64(2*bit+1)))
	return z.Quo(z, FromInt(x.Prec(), 4))
}

// CantorBits returns the first n bits of a Cantor-encoded stack.
func (x *BitFloat) CantorBits(n int) []bool {
	y := x.Copy()
	tmp := New(x.Prec())
	bits := make([]bool, n)
	for i := 0; i < n; i++ {
		bits[i] = tmp.CantorPop(y) == 1
		y.Set(tmp)
	}
	return bits
}

// CantorBitAt returns Cantor-stack bit k of x (0-based).
func (x *BitFloat) CantorBitAt(k int) int {
	y := x.Copy()
	tmp := New(x.Prec())
	var bit int
	for i := 0; i <= k; i++ {
		bit = tmp.CantorPop(y)
		y.Set(tmp)
	}
	return bit
}

// ApproxEq reports whether |x − y| < 2^{-bits}.
func (x *BitFloat) ApproxEq(y *BitFloat, bits uint) bool {
	d := New(max(x.Prec(), y.Prec())).Sub(x, y)
	d.Abs(d)
	eps := New(d.Prec())
	eps.f.SetMantExp(big.NewFloat(1).SetPrec(d.Prec()), -int(bits))
	return d.Cmp(eps) <= 0
}

// Pow2 returns 2^e as a BitFloat.
func Pow2(prec uint, e int) *BitFloat {
	z := New(prec)
	z.f.SetMantExp(big.NewFloat(1).SetPrec(prec), e)
	return z
}
