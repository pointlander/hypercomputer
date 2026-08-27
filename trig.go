// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import "sync"

var (
	piMu     sync.Mutex
	piByPrec = map[uint]*BitFloat{}
)

// Pi returns π at the given mantissa precision.
func Pi(prec uint) *BitFloat {
	if prec == 0 {
		prec = DefaultPrec
	}
	piMu.Lock()
	defer piMu.Unlock()
	if p, ok := piByPrec[prec]; ok {
		return p.Copy()
	}
	p := machinPi(prec)
	piByPrec[prec] = p
	return p.Copy()
}

// machinPi uses Machin's formula: π/4 = 4 arctan(1/5) − arctan(1/239).
func machinPi(prec uint) *BitFloat {
	wp := prec + 32
	a := atanInv(wp, 5)
	b := atanInv(wp, 239)
	t := New(wp).Mul(FromInt(wp, 4), a)
	t.Sub(t, b)
	t.Mul(t, FromInt(wp, 4))
	out := New(prec)
	out.f.SetPrec(prec)
	out.f.Set(t.f)
	return out
}

// atanInv returns arctan(1/n) by Taylor series.
func atanInv(prec uint, n int64) *BitFloat {
	x := FromRat(prec, 1, n)
	xx := New(prec).Mul(x, x)
	term := x.Copy()
	sum := x.Copy()
	eps := Pow2(prec, -int(prec)+4)
	neg := true
	for k := int64(1); k < int64(prec)+16; k++ {
		term.Mul(term, xx)
		t := New(prec).Quo(term, FromInt(prec, 2*k+1))
		if neg {
			sum.Sub(sum, t)
		} else {
			sum.Add(sum, t)
		}
		neg = !neg
		if New(prec).Abs(t).Cmp(eps) <= 0 {
			break
		}
	}
	return sum
}

// Sin sets z to sin(x), using argument reduction and a Taylor series.
func (z *BitFloat) Sin(x *BitFloat) *BitFloat {
	p := max(z.Prec(), x.Prec())
	if p == 0 {
		p = DefaultPrec
	}
	wp := p + 16
	pi := Pi(wp)
	twoPi := New(wp).Mul(FromInt(wp, 2), pi)
	halfPi := New(wp).Quo(pi, FromInt(wp, 2))

	y := New(wp).Set(x)
	y.SetPrec(wp)
	q := New(wp).Quo(y, twoPi)
	y.Sub(y, New(wp).Mul(New(wp).Floor(q), twoPi))

	sign := 1
	if y.Cmp(pi) > 0 {
		y.Sub(y, pi)
		sign = -1
	}
	if y.Cmp(halfPi) > 0 {
		y.Sub(pi, y)
	}
	s := sinSeries(y)
	if sign < 0 {
		s.Neg(s)
	}
	z.ensure(p)
	z.f.Set(s.f)
	z.f.SetPrec(p)
	return z
}

// Cos sets z to cos(x) = sin(x + π/2).
func (z *BitFloat) Cos(x *BitFloat) *BitFloat {
	p := max(z.Prec(), x.Prec())
	if p == 0 {
		p = DefaultPrec
	}
	halfPi := New(p).Quo(Pi(p), FromInt(p, 2))
	return z.Sin(New(p).Add(x, halfPi))
}

// sinSeries evaluates sin(x) for x in [0, π/2].
func sinSeries(x *BitFloat) *BitFloat {
	prec := x.Prec()
	sum := x.Copy()
	term := x.Copy()
	xx := New(prec).Mul(x, x)
	eps := Pow2(prec, -int(prec)+4)
	for n := int64(1); n < int64(prec)+16; n++ {
		term.Mul(term, xx)
		term.Neg(term)
		term.Quo(term, FromInt(prec, 2*n))
		term.Quo(term, FromInt(prec, 2*n+1))
		sum.Add(sum, term)
		if New(prec).Abs(term).Cmp(eps) <= 0 {
			break
		}
	}
	return sum
}
