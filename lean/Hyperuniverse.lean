/-
  Hyperuniverse.lean
  ──────────────────────────────────────────────────────────────────────────
  A toy physical theory in which:

  1. The universe is a hypercomputer (a sound halt oracle; equivalently,
     a completed Zeno / Malament–Hogarth ω-run of every program).
  2. A particle is a computational system whose store is an infinite
     bit tape (a real / Cantor stack).
  3. Decay is a readout of that store at the particle's query address,
     and therefore solves one instance of the halting problem.

  This is the infinite-precision limit of the Go analog hypercomputer:
  `Store` is the analog tape, `decay` is a one-bit measurement,
  `cosmos` is the MH/Zeno clock. Finite-energy truncation is a
  different theory and is not assumed here.

  Primitives: `Halts`, `haltStore` with its specification, infinitude
  of halt and loop, `cosmos`, and a Zeno-limit oracle. Everything else
  is definition or theorem.
-/

namespace Hyperuniverse

/-! ## 0. Programs and the halting property -/

/-- Gödel codes of programs. -/
abbrev Program := Nat

/-- The (classically undecidable) halting property. -/
axiom Halts : Program → Prop

/-! ## 1. Infinite stores -/

/-- An infinite bit tape: cell `n` holds one bit. This is the analog
    real / Cantor stack of the Go simulator. -/
abbrev Store := Nat → Bool

/-- The tape is eventually all zeros. -/
def FiniteSupport (s : Store) : Prop :=
  ∃ N : Nat, ∀ n, N ≤ n → s n = false

/-- Infinitely much information: arbitrarily late `true` bits. -/
def InfiniteStore (s : Store) : Prop := ¬ FiniteSupport s

/-- Characteristic tape of `Halts`. Not Turing-computable. -/
axiom haltStore : Store

axiom haltStore_spec : ∀ n : Program, haltStore n = true ↔ Halts n

/-- Infinitely many programs halt. -/
axiom infinitely_many_halt : ∀ N : Nat, ∃ n, N ≤ n ∧ Halts n

/-- Infinitely many programs do not halt. -/
axiom infinitely_many_loop : ∀ N : Nat, ∃ n, N ≤ n ∧ ¬ Halts n

theorem haltStore_infinite : InfiniteStore haltStore := by
  rintro ⟨N, hN⟩
  obtain ⟨n, hn, hH⟩ := infinitely_many_halt N
  have ht : haltStore n = true := (haltStore_spec n).mpr hH
  have hf : haltStore n = false := hN n hn
  simp [ht] at hf

theorem haltStore_coinfinite :
    ¬ FiniteSupport (fun n => Bool.not (haltStore n)) := by
  rintro ⟨N, hN⟩
  obtain ⟨n, hn, hL⟩ := infinitely_many_loop N
  have hf : haltStore n = false :=
    Bool.eq_false_iff.mpr (mt (haltStore_spec n).mp hL)
  have : Bool.not (haltStore n) = false := hN n hn
  simp [hf] at this

/-! ## 2. Particles as computational systems -/

/-- A computational system: discrete dynamics plus an infinite store. -/
structure ComputationalSystem where
  /-- Internal program (the particle's "dynamics"). -/
  machine : Program
  /-- Infinite information (analog tape). -/
  store : Store

/-- A particle carries a halt-query: the program whose Halt-bit is
    written at address `query` on the store. -/
structure Particle extends ComputationalSystem where
  query : Program

/-- Physical particles store infinitely much information. -/
class InfiniteInformation (p : Particle) : Prop where
  infinite : InfiniteStore p.store

/-- The store answers `Halts` at the query address. -/
class HaltEncoded (p : Particle) : Prop where
  lookup : p.store p.query = true ↔ Halts p.query

/-- The store is the full halt-set characteristic tape. -/
class OracleParticle (p : Particle) : Prop where
  encodes : ∀ n : Program, p.store n = true ↔ Halts n

instance (p : Particle) [o : OracleParticle p] : HaltEncoded p where
  lookup := o.encodes p.query

instance (p : Particle) [o : OracleParticle p] : InfiniteInformation p where
  infinite := by
    rintro ⟨N, hN⟩
    obtain ⟨n, hn, hH⟩ := infinitely_many_halt N
    have ht : p.store n = true := (o.encodes n).mpr hH
    have hf : p.store n = false := hN n hn
    simp [ht] at hf

/-- The canonical particle for program `e`: tape `χ_Halt`, query `e`. -/
noncomputable def particleFor (e : Program) : Particle :=
  { machine := e, store := haltStore, query := e }

instance (e : Program) : OracleParticle (particleFor e) where
  encodes := haltStore_spec

/-! ## 3. The universe is a hypercomputer -/

/-- A sound halt oracle: a hypercomputer in the sense of this repo. -/
structure Hypercomputer where
  oracle : Program → Bool
  correct : ∀ p, oracle p = true ↔ Halts p

/-- Spacetime provides a completed ω-run of every program
    (Zeno schedule / Malament–Hogarth observer). -/
axiom cosmos : Hypercomputer

theorem cosmos_decides_Halt (p : Program) :
    cosmos.oracle p = true ↔ Halts p :=
  cosmos.correct p

/-- Turing-computable functions on programs. -/
axiom Computable : (Program → Bool) → Prop

/-- Church–Turing: no Turing-computable function decides `Halts`. -/
axiom no_TM_decides_Halt :
  ¬ ∃ f, Computable f ∧ ∀ p, f p = true ↔ Halts p

theorem cosmos_is_not_Turing : ¬ Computable cosmos.oracle := by
  intro h
  exact no_TM_decides_Halt ⟨cosmos.oracle, h, cosmos.correct⟩

/-! ## 4. Decay solves a halt instance -/

/-- Two-body (or two-channel) decay: the physical bit. -/
inductive Decay where
  | yes
  | no
deriving DecidableEq, Repr

def Decay.ofBool : Bool → Decay
  | true  => .yes
  | false => .no

@[simp] theorem Decay.ofBool_true : Decay.ofBool true = .yes := rfl
@[simp] theorem Decay.ofBool_false : Decay.ofBool false = .no := rfl

@[simp] theorem Decay.ofBool_eq_yes (b : Bool) :
    Decay.ofBool b = .yes ↔ b = true := by
  cases b <;> simp [Decay.ofBool]

/-- Decay is a readout of the infinite store at the query address. -/
def decay (p : Particle) : Decay :=
  Decay.ofBool (p.store p.query)

/-- **Main law.** A halt-encoded particle solves `Halts` on its query
    when it decays. -/
theorem decay_solves_halting (p : Particle) [hp : HaltEncoded p] :
    decay p = .yes ↔ Halts p.query := by
  simp [decay, hp.lookup]

/-- Decay agrees with the cosmic hypercomputer. -/
theorem decay_eq_cosmos (p : Particle) [HaltEncoded p] :
    decay p = .yes ↔ cosmos.oracle p.query = true := by
  rw [decay_solves_halting, cosmos.correct]

/-- Every program has a particle whose decay decides it. -/
theorem particle_for_decides (e : Program) :
    decay (particleFor e) = .yes ↔ Halts e :=
  decay_solves_halting (particleFor e)

/-- The family of all particles is a halt oracle. -/
noncomputable def laboratory (e : Program) : Bool :=
  match decay (particleFor e) with
  | .yes => true
  | .no  => false

theorem laboratory_is_hypercomputer (e : Program) :
    laboratory e = true ↔ Halts e := by
  unfold laboratory
  have d := particle_for_decides e
  cases h : decay (particleFor e) <;> simp_all

/-! ## 5. Decay as a Zeno ω-limit -/

/-- Kind of ω-limit of an accelerated run (cf. Go `LimitKind`). -/
inductive LimitKind where
  | halt
  | cauchy
  | diverge
deriving DecidableEq, Repr

/-- The universe completes an ω-run of a program and returns the
    limit kind. Halt iff the program actually halts. -/
axiom zenoLimit : Program → LimitKind

axiom zenoLimit_halt :
  ∀ p, zenoLimit p = LimitKind.halt ↔ Halts p

/-- If the particle's machine *is* its query, decay is the Zeno
    ω-limit of that machine. -/
theorem decay_is_zeno_limit (p : Particle)
    (h : p.machine = p.query) [HaltEncoded p] :
    decay p = .yes ↔ zenoLimit p.machine = LimitKind.halt := by
  rw [h, zenoLimit_halt, decay_solves_halting]

/-- Cosmic oracle = Zeno ω-limit bit. -/
theorem cosmos_eq_zeno (p : Program) :
    cosmos.oracle p = true ↔ zenoLimit p = LimitKind.halt := by
  rw [cosmos.correct, zenoLimit_halt]

/-! ## 6. Information: one particle, infinitely many bits -/

/-- An oracle particle's tape is not finitely supported. -/
theorem oracle_particle_infinite (p : Particle) [OracleParticle p] :
    InfiniteStore p.store :=
  InfiniteInformation.infinite

/-- Canonical particles store the whole halt set, hence infinitely
    much information. -/
theorem particleFor_infinite (e : Program) :
    InfiniteStore (particleFor e).store :=
  haltStore_infinite

/-- Looking up cell `n` of an oracle particle decides `Halts n`. -/
theorem oracle_lookup (p : Particle) [o : OracleParticle p] (n : Program) :
    p.store n = true ↔ Halts n :=
  o.encodes n

end Hyperuniverse
