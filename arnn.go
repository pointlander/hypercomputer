// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

// ARNN is a fully connected analog recurrent net with saturated-
// linear activations (Siegelmann–Sontag). Real weights can encode
// a non-recursive oracle; finite-precision weights encode a finite
// prefix of one.
type ARNN struct {
	Prec uint
	W    [][]*BitFloat
	B    []*BitFloat
	X    []*BitFloat
}

// NewARNN returns a zero net with n neurons.
func NewARNN(prec uint, n int) *ARNN {
	if prec == 0 {
		prec = DefaultPrec
	}
	w := make([][]*BitFloat, n)
	b := make([]*BitFloat, n)
	x := make([]*BitFloat, n)
	for i := 0; i < n; i++ {
		w[i] = make([]*BitFloat, n)
		for j := 0; j < n; j++ {
			w[i][j] = New(prec)
		}
		b[i] = New(prec)
		x[i] = New(prec)
	}
	return &ARNN{Prec: prec, W: w, B: b, X: x}
}

// N returns the number of neurons.
func (a *ARNN) N() int {
	return len(a.X)
}

// Step performs one synchronous saturated-linear update
//
//	x' = σ(W x + b)
func (a *ARNN) Step() {
	n := a.N()
	y := make([]*BitFloat, n)
	for i := 0; i < n; i++ {
		s := New(a.Prec)
		for j := 0; j < n; j++ {
			p := New(a.Prec).Mul(a.W[i][j], a.X[j])
			s.Add(s, p)
		}
		s.Add(s, a.B[i])
		y[i] = New(a.Prec).SatLin(s)
	}
	a.X = y
}

// CantorReader is a two-layer analog net that pops bits from a
// Cantor-encoded oracle:
//
//	bit = σ(4q − 2)
//	q'  = σ(4q − 2·bit − 1)
//
// One Read is one pop. This is the Siegelmann stack-head.
type CantorReader struct {
	Prec uint
	Q    *BitFloat
	Bit  *BitFloat
}

// NewCantorReader reads bits from oracle q.
func NewCantorReader(q *BitFloat) *CantorReader {
	return &CantorReader{
		Prec: q.Prec(),
		Q:    q.Copy(),
		Bit:  New(q.Prec()),
	}
}

// Read pops the next Cantor bit using satlin neurons.
func (r *CantorReader) Read() int {
	t := New(r.Prec).Mul(FromInt(r.Prec, 4), r.Q)
	t.Sub(t, FromInt(r.Prec, 2))
	r.Bit.SatLin(t)
	b := 0
	if r.Bit.Cmp(FromRat(r.Prec, 1, 2)) >= 0 {
		b = 1
		r.Bit.SetInt64(1)
	} else {
		r.Bit.SetInt64(0)
	}
	t.Mul(FromInt(r.Prec, 4), r.Q)
	twoBit := New(r.Prec).Mul(FromInt(r.Prec, 2), r.Bit)
	t.Sub(t, twoBit)
	t.Sub(t, FromInt(r.Prec, 1))
	r.Q.SatLin(t)
	return b
}

// ReadN pops n bits.
func (r *CantorReader) ReadN(n int) []bool {
	bits := make([]bool, n)
	for i := 0; i < n; i++ {
		bits[i] = r.Read() == 1
	}
	return bits
}
