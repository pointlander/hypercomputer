// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import (
	"math/big"
)

const (
	// DefaultPrec is the default working precision in bits for
	// truncated series (√, π, sin, cos) and dyadic rounding.
	DefaultPrec uint = 256
	// GuardBits is extra bits kept while summing truncated series.
	GuardBits uint = 64
)

// BitFloat is a rational number (math/big.Rat) used as an analog
// register. Field operations and the Bernoulli / Cantor maps are
// exact in ℚ. Prec is the working bit precision for truncated
// series that leave ℚ (√, π, sin, cos) and for Truncate.
type BitFloat struct {
	r    *big.Rat
	prec uint
}

// New returns a BitFloat 0 with the given working precision.
func New(prec uint) *BitFloat {
	if prec == 0 {
		prec = DefaultPrec
	}
	return &BitFloat{r: new(big.Rat), prec: prec}
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
	z.r.SetInt64(n)
	return z
}

// FromRat returns num/den as a BitFloat.
func FromRat(prec uint, num, den int64) *BitFloat {
	z := New(prec)
	z.r.SetFrac64(num, den)
	return z
}

// FromFloat64 returns x as a BitFloat.
func FromFloat64(prec uint, x float64) *BitFloat {
	z := New(prec)
	z.r.SetFloat64(x)
	return z
}

// FromBits encodes bits as the binary fraction 0.b0 b1 b2 ...
// with trailing zeros. Exact in ℚ.
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
	den := new(big.Int).Lsh(big.NewInt(1), uint(len(bits)))
	z.r.SetFrac(n, den)
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
// with an infinite tail of zeros (value 1/3). Exact in ℚ.
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

// Prec returns the working precision in bits.
func (x *BitFloat) Prec() uint {
	if x == nil {
		return 0
	}
	return x.prec
}

// SetPrec sets the working precision in bits. The rational value
// is unchanged; use Truncate to round to a dyadic of that width.
func (x *BitFloat) SetPrec(prec uint) *BitFloat {
	x.ensure(prec)
	x.prec = prec
	return x
}

func (z *BitFloat) ensure(prec uint) {
	if z.r == nil {
		z.r = new(big.Rat)
	}
	if prec > z.prec {
		z.prec = prec
	}
}

func (z *BitFloat) cap() {
	if z.prec > 0 && z.r != nil {
		truncToBits(z.r, z.prec)
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
	z.r.Set(x.r)
	return z
}

// Set copies x into z.
func (z *BitFloat) Set(x *BitFloat) *BitFloat {
	z.fit(x)
	z.r.Set(x.r)
	return z
}

// SetInt64 sets z to n.
func (z *BitFloat) SetInt64(n int64) *BitFloat {
	z.ensure(z.Prec())
	z.r.SetInt64(n)
	return z
}

// Sign returns -1, 0, or 1.
func (x *BitFloat) Sign() int {
	return x.r.Sign()
}

// Cmp compares x and y.
func (x *BitFloat) Cmp(y *BitFloat) int {
	return x.r.Cmp(y.r)
}

// IsInt reports whether x is an integer.
func (x *BitFloat) IsInt() bool {
	return x.r.IsInt()
}

// Big returns a copy of the underlying rational.
func (x *BitFloat) Big() *big.Rat {
	return new(big.Rat).Set(x.r)
}

// Float64 returns the nearest float64 value.
func (x *BitFloat) Float64() float64 {
	f, _ := x.r.Float64()
	return f
}

// String returns a decimal representation with digits matching the
// working precision.
func (x *BitFloat) String() string {
	digits := int(x.Prec()) * 301 / 1000
	if digits < 8 {
		digits = 8
	}
	return x.Text('g', digits)
}

// Text returns a decimal string with prec digits after the point.
func (x *BitFloat) Text(format byte, prec int) string {
	_ = format
	if prec < 0 {
		prec = 6
	}
	if x.Sign() == 0 {
		if prec == 0 {
			return "0"
		}
		return x.r.FloatString(prec)
	}
	return x.r.FloatString(prec)
}

// Add sets z to x + y.
func (z *BitFloat) Add(x, y *BitFloat) *BitFloat {
	z.fit(x, y)
	z.r.Add(x.r, y.r)
	return z
}

// Sub sets z to x - y.
func (z *BitFloat) Sub(x, y *BitFloat) *BitFloat {
	z.fit(x, y)
	z.r.Sub(x.r, y.r)
	return z
}

// Mul sets z to x * y, then rounds to working precision.
func (z *BitFloat) Mul(x, y *BitFloat) *BitFloat {
	z.fit(x, y)
	z.r.Mul(x.r, y.r)
	z.cap()
	return z
}

// Quo sets z to x / y, then rounds to working precision.
func (z *BitFloat) Quo(x, y *BitFloat) *BitFloat {
	z.fit(x, y)
	z.r.Quo(x.r, y.r)
	z.cap()
	return z
}

// Neg sets z to -x.
func (z *BitFloat) Neg(x *BitFloat) *BitFloat {
	z.fit(x)
	z.r.Neg(x.r)
	return z
}

// Abs sets z to |x|.
func (z *BitFloat) Abs(x *BitFloat) *BitFloat {
	z.fit(x)
	z.r.Abs(x.r)
	return z
}

// Sqrt sets z to √x by Newton iteration in ℚ, truncated to working
// precision after each step.
func (z *BitFloat) Sqrt(x *BitFloat) *BitFloat {
	z.fit(x)
	if x.Sign() < 0 {
		z.r.SetInt64(0)
		return z
	}
	if x.Sign() == 0 {
		z.r.SetInt64(0)
		return z
	}
	p := z.Prec()
	wp := p + GuardBits
	g := new(big.Rat).SetInt64(1)
	two := big.NewRat(2, 1)
	tmp := new(big.Rat)
	eps := pow2Rat(-(int(p) + 8))
	for n := 0; n < int(p)+16; n++ {
		tmp.Quo(x.r, g)
		tmp.Add(g, tmp)
		tmp.Quo(tmp, two)
		diff := new(big.Rat).Sub(tmp, g)
		g.Set(tmp)
		truncToBits(g, wp)
		if ratAbsCmp(diff, eps) <= 0 {
			break
		}
	}
	truncToBits(g, p)
	z.r.Set(g)
	return z
}

// Floor sets z to ⌊x⌋ (toward −∞).
func (z *BitFloat) Floor(x *BitFloat) *BitFloat {
	z.fit(x)
	num := new(big.Int).Set(x.r.Num())
	den := new(big.Int).Set(x.r.Denom())
	q := new(big.Int)
	rem := new(big.Int)
	q.QuoRem(num, den, rem)
	if x.r.Sign() < 0 && rem.Sign() != 0 {
		q.Sub(q, big.NewInt(1))
	}
	z.r.SetInt(q)
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
func (z *BitFloat) SatLin(x *BitFloat) *BitFloat {
	z.fit(x)
	if x.Sign() <= 0 {
		z.r.SetInt64(0)
		return z
	}
	one := FromInt(x.Prec(), 1)
	if x.Cmp(one) >= 0 {
		z.r.SetInt64(1)
		return z
	}
	z.r.Set(x.r)
	return z
}

// Bit returns the leading fractional bit of x: 1 if x ≥ 1/2, else 0.
func (x *BitFloat) Bit() int {
	if x.Cmp(FromRat(x.Prec(), 1, 2)) >= 0 {
		return 1
	}
	return 0
}

// Shift sets z to the Bernoulli map T(x) = 2x − ⌊2x⌋ (2x mod 1).
func (z *BitFloat) Shift(x *BitFloat) *BitFloat {
	z.fit(x)
	z.Mul(FromInt(x.Prec(), 2), x)
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
// with 0.5 ≤ |mantissa| < 1 for x ≠ 0.
func (x *BitFloat) Mantissa() (sign int, exp int, bits []bool) {
	sign = x.Sign()
	if sign == 0 {
		return 0, 0, make([]bool, int(x.Prec()))
	}
	y := x.Copy()
	y.Abs(y)
	one := FromInt(y.Prec(), 1)
	half := FromRat(y.Prec(), 1, 2)
	two := FromInt(y.Prec(), 2)
	for y.Cmp(one) >= 0 {
		y.Quo(y, two)
		exp++
	}
	for y.Cmp(half) < 0 {
		y.Mul(y, two)
		exp--
	}
	return sign, exp, y.Bits(int(x.Prec()))
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

// ApproxEq reports whether |x − y| ≤ 2^{−bits}.
func (x *BitFloat) ApproxEq(y *BitFloat, bits uint) bool {
	d := New(max(x.Prec(), y.Prec())).Sub(x, y)
	d.Abs(d)
	return d.Cmp(Pow2(d.Prec(), -int(bits))) <= 0
}

// Truncate rounds z to the nearest multiple of 2^{−bits} (a dyadic
// rational of that width) and sets the working precision to bits.
func (z *BitFloat) Truncate(bits uint) *BitFloat {
	z.ensure(bits)
	truncToBits(z.r, bits)
	z.prec = bits
	return z
}

// Pow2 returns 2^e as a BitFloat.
func Pow2(prec uint, e int) *BitFloat {
	z := New(prec)
	z.r.Set(pow2Rat(e))
	return z
}

func pow2Rat(e int) *big.Rat {
	r := new(big.Rat)
	if e >= 0 {
		r.SetInt(new(big.Int).Lsh(big.NewInt(1), uint(e)))
		return r
	}
	r.SetFrac(big.NewInt(1), new(big.Int).Lsh(big.NewInt(1), uint(-e)))
	return r
}

func truncToBits(r *big.Rat, bits uint) {
	if bits == 0 || r.Sign() == 0 {
		if r.Sign() == 0 {
			r.SetInt64(0)
		}
		return
	}
	pow := new(big.Int).Lsh(big.NewInt(1), bits)
	scaled := new(big.Rat).Mul(r, new(big.Rat).SetInt(pow))
	n := roundRatToInt(scaled)
	r.SetFrac(n, pow)
}

func roundRatToInt(r *big.Rat) *big.Int {
	num := new(big.Int).Set(r.Num())
	den := new(big.Int).Set(r.Denom())
	if den.Cmp(big.NewInt(1)) == 0 {
		return num
	}
	q := new(big.Int)
	rem := new(big.Int)
	q.QuoRem(num, den, rem)
	absRem := new(big.Int).Abs(rem)
	absRem.Lsh(absRem, 1)
	if absRem.Cmp(den) >= 0 {
		if num.Sign() >= 0 {
			q.Add(q, big.NewInt(1))
		} else {
			q.Sub(q, big.NewInt(1))
		}
	}
	return q
}

func ratAbsCmp(x, y *big.Rat) int {
	ax := new(big.Rat).Abs(x)
	ay := new(big.Rat).Abs(y)
	return ax.Cmp(ay)
}
