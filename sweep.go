// Copyright 2026 The HyperComputer Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hypercomputer

import (
	"fmt"
	"strings"
)

// SweepProbe is a bit string named for a resource table.
type SweepProbe struct {
	Name string
	Bits []bool
}

// SweepCell is one (p, T) entry: p analog bits of Ω_T, dovetail bound T.
type SweepCell struct {
	P, T       int
	OmegaBits  []bool
	Caught     bool
	Stage      int
	Results    []*KResult
	Certified  []string
	Beyond     []string
	Incomplete []string
}

// ResourceSweep is a grid of Chaitin reconstructions over analog
// precision p and step bound T.
type ResourceSweep struct {
	MaxBits int
	Probes  []SweepProbe
	P       []int
	T       []int
	Cells   []SweepCell
}

// DefaultSweepProbes are strings whose K_U is known and small.
func DefaultSweepProbes() []SweepProbe {
	return []SweepProbe{
		{Name: "ε", Bits: nil},
		{Name: "0", Bits: ParseBitString("0")},
		{Name: "1", Bits: ParseBitString("1")},
		{Name: "11", Bits: ParseBitString("11")},
	}
}

// CertLabel is a compact certification mark for a table cell.
func (r *KResult) CertLabel() string {
	switch r.How {
	case "chaitin":
		return fmt.Sprintf("%d", r.K)
	case "chaitin-beyond":
		return fmt.Sprintf(">%d", r.M)
	case "chaitin-incomplete":
		return "?"
	default:
		return r.How
	}
}

// SweepResources reconstructs K_U from the first p analog bits of
// Ω_U built at bound T, for every pair (p, T) and every probe.
//
// Ω is rebuilt per T (so Ω_T is nondecreasing). Each cell reuses one
// dovetail for all probes. p is the Bernoulli prefix length, not
// Truncate: rounding Ω up can make the prefix uncatchable.
func SweepResources(probes []SweepProbe, ps, ts []int, maxBits int, prec uint) *ResourceSweep {
	if len(probes) == 0 {
		probes = DefaultSweepProbes()
	}
	if len(ps) == 0 {
		ps = []int{3, 5, 6, 8, 9}
	}
	if len(ts) == 0 {
		ts = []int{1, 2, 4, 8}
	}
	need := 0
	for _, p := range ps {
		if p > need {
			need = p
		}
	}
	if maxBits < 1 {
		maxBits = need
	}
	if maxBits < need {
		maxBits = need
	}
	if prec == 0 {
		prec = DefaultPrec
	}
	s := &ResourceSweep{
		MaxBits: maxBits,
		Probes:  probes,
		P:       append([]int(nil), ps...),
		T:       append([]int(nil), ts...),
	}
	for _, t := range ts {
		om := Omega(maxBits, t, prec)
		for _, p := range ps {
			rec := Reconstruct(om, p, maxBits, t)
			cell := SweepCell{
				P:         p,
				T:         t,
				OmegaBits: rec.OmegaBits,
				Caught:    rec.Caught,
				Stage:     rec.Stage,
			}
			for _, pr := range probes {
				kr := rec.KOf(pr.Bits)
				cell.Results = append(cell.Results, kr)
				switch kr.How {
				case "chaitin":
					cell.Certified = append(cell.Certified, pr.Name)
				case "chaitin-beyond":
					cell.Beyond = append(cell.Beyond, pr.Name)
				default:
					cell.Incomplete = append(cell.Incomplete, pr.Name)
				}
			}
			s.Cells = append(s.Cells, cell)
		}
	}
	return s
}

func (c SweepCell) labels() []string {
	out := make([]string, len(c.Results))
	for i, r := range c.Results {
		out[i] = fmt.Sprintf("%-3s", r.CertLabel())
	}
	return out
}

// String is a fixed-width table: p, T, Ω bits, caught, then one
// column per probe.
func (s *ResourceSweep) String() string {
	var b strings.Builder
	names := make([]string, len(s.Probes))
	for i, p := range s.Probes {
		names[i] = p.Name
	}
	fmt.Fprintf(&b, "p  T  Ωbits%-*s  catch  %s\n",
		max(0, s.maxP()-5), "", strings.Join(names, "  "))
	for _, c := range s.Cells {
		catch := "n"
		if c.Caught {
			catch = "y"
		}
		fmt.Fprintf(&b, "%-2d %-2d %-*s  %s     %s\n",
			c.P, c.T, s.maxP(), FormatBits(c.OmegaBits), catch,
			strings.Join(c.labels(), "  "))
	}
	return b.String()
}

func (s *ResourceSweep) maxP() int {
	m := 5
	for _, p := range s.P {
		if p > m {
			m = p
		}
	}
	return m
}

func (s *ResourceSweep) cell(p, t int) (SweepCell, bool) {
	for _, c := range s.Cells {
		if c.P == p && c.T == t {
			return c, true
		}
	}
	return SweepCell{}, false
}
