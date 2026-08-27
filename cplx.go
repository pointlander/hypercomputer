// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

// Complex is a high-precision complex number with BitFloat parts.
type Complex struct {
	Re, Im *BitFloat
}

// NewComplex returns 0+0i at the given precision.
func NewComplex(prec uint) *Complex {
	return &Complex{Re: New(prec), Im: New(prec)}
}

// CReal returns x + 0i.
func CReal(x *BitFloat) *Complex {
	return &Complex{Re: x.Copy(), Im: New(x.Prec())}
}

// CImag returns 0 + x i.
func CImag(x *BitFloat) *Complex {
	return &Complex{Re: New(x.Prec()), Im: x.Copy()}
}

// CZero returns 0+0i.
func CZero(prec uint) *Complex {
	return NewComplex(prec)
}

// COne returns 1+0i.
func COne(prec uint) *Complex {
	z := NewComplex(prec)
	z.Re.SetInt64(1)
	return z
}

// CI returns 0+1i.
func CI(prec uint) *Complex {
	z := NewComplex(prec)
	z.Im.SetInt64(1)
	return z
}

// CExpI returns e^{iθ} = cos θ + i sin θ.
func CExpI(theta *BitFloat) *Complex {
	p := theta.Prec()
	return &Complex{
		Re: New(p).Cos(theta),
		Im: New(p).Sin(theta),
	}
}

// Prec returns the mantissa precision in bits.
func (x *Complex) Prec() uint {
	if x == nil || x.Re == nil {
		return 0
	}
	return x.Re.Prec()
}

// Copy returns a deep copy.
func (x *Complex) Copy() *Complex {
	return &Complex{Re: x.Re.Copy(), Im: x.Im.Copy()}
}

// Set copies x into z.
func (z *Complex) Set(x *Complex) *Complex {
	if z.Re == nil {
		z.Re = New(x.Prec())
	}
	if z.Im == nil {
		z.Im = New(x.Prec())
	}
	z.Re.Set(x.Re)
	z.Im.Set(x.Im)
	return z
}

// Add sets z to x + y.
func (z *Complex) Add(x, y *Complex) *Complex {
	z.ensure(x, y)
	z.Re.Add(x.Re, y.Re)
	z.Im.Add(x.Im, y.Im)
	return z
}

// Sub sets z to x − y.
func (z *Complex) Sub(x, y *Complex) *Complex {
	z.ensure(x, y)
	z.Re.Sub(x.Re, y.Re)
	z.Im.Sub(x.Im, y.Im)
	return z
}

// Mul sets z to x y.
func (z *Complex) Mul(x, y *Complex) *Complex {
	z.ensure(x, y)
	ac := New(z.Prec()).Mul(x.Re, y.Re)
	bd := New(z.Prec()).Mul(x.Im, y.Im)
	ad := New(z.Prec()).Mul(x.Re, y.Im)
	bc := New(z.Prec()).Mul(x.Im, y.Re)
	z.Re.Sub(ac, bd)
	z.Im.Add(ad, bc)
	return z
}

// Scale sets z to s x (real scalar).
func (z *Complex) Scale(x *Complex, s *BitFloat) *Complex {
	z.ensure(x)
	z.Re.Mul(x.Re, s)
	z.Im.Mul(x.Im, s)
	return z
}

// Quo sets z to x / y.
func (z *Complex) Quo(x, y *Complex) *Complex {
	z.ensure(x, y)
	n2 := y.Abs2()
	ac := New(z.Prec()).Mul(x.Re, y.Re)
	bd := New(z.Prec()).Mul(x.Im, y.Im)
	bc := New(z.Prec()).Mul(x.Im, y.Re)
	ad := New(z.Prec()).Mul(x.Re, y.Im)
	z.Re.Add(ac, bd)
	z.Re.Quo(z.Re, n2)
	z.Im.Sub(bc, ad)
	z.Im.Quo(z.Im, n2)
	return z
}

// Conj sets z to the conjugate of x.
func (z *Complex) Conj(x *Complex) *Complex {
	z.ensure(x)
	z.Re.Set(x.Re)
	z.Im.Neg(x.Im)
	return z
}

// Neg sets z to −x.
func (z *Complex) Neg(x *Complex) *Complex {
	z.ensure(x)
	z.Re.Neg(x.Re)
	z.Im.Neg(x.Im)
	return z
}

// Abs2 returns |x|² = re² + im².
func (x *Complex) Abs2() *BitFloat {
	p := x.Prec()
	re2 := New(p).Mul(x.Re, x.Re)
	im2 := New(p).Mul(x.Im, x.Im)
	return New(p).Add(re2, im2)
}

// ApproxEq reports whether |x − y|² ≤ 2^{−bits}.
func (x *Complex) ApproxEq(y *Complex, bits uint) bool {
	d := NewComplex(max(x.Prec(), y.Prec())).Sub(x, y)
	return d.Abs2().ApproxEq(New(d.Prec()), bits)
}

func (z *Complex) ensure(xs ...*Complex) {
	p := z.Prec()
	for _, x := range xs {
		if x != nil && x.Prec() > p {
			p = x.Prec()
		}
	}
	if z.Re == nil {
		z.Re = New(p)
	} else {
		z.Re.ensure(p)
	}
	if z.Im == nil {
		z.Im = New(p)
	} else {
		z.Im.ensure(p)
	}
}

func (x *Complex) String() string {
	const d = 12
	re0, im0 := x.Re.Sign() == 0, x.Im.Sign() == 0
	switch {
	case re0 && im0:
		return "0"
	case im0:
		return x.Re.Text('g', d)
	case re0:
		return x.Im.Text('g', d) + "i"
	case x.Im.Sign() < 0:
		return x.Re.Text('g', d) + x.Im.Text('g', d) + "i"
	default:
		return x.Re.Text('g', d) + "+" + x.Im.Text('g', d) + "i"
	}
}
