/-
  Hyperuniverse.lean
  ──────────────────────────────────────────────────────────────────────────
  Finite 2-symbol tables (`TMFromIndex` / `decode`), `step` / `run`, and
  `Halts`. The Zeno ω-limit compares the tape window at 2^{k-1} and 2^k
  (halt / Cauchy / diverge).

  `Computable` means some finite table writes the bit function on a unary
  tape. Parity is computable. `haltStore` is the halt bit of `decode`;
  the axiom `no_TM_decides_Halt` places that function outside `Computable`.

  The universe is a classical halt oracle. A particle carries an infinite
  store; decay reads cell 0. A truncated `N`-bit lab particle decides
  `Halts` exactly on coded queries `< N`.
-/

namespace Hyperuniverse

/-! ## 0. Turing machines -/

structure Trans where
  write : Bool
  moveRight : Bool
  next : Option Nat
deriving Repr, DecidableEq

def Trans.halt : Trans :=
  { write := false, moveRight := true, next := none }

structure TM where
  nstates : Nat
  start : Nat
  table : List Trans

def TM.slot (s : Nat) (b : Bool) : Nat :=
  2 * s + if b then 1 else 0

def TM.get (tm : TM) (s : Nat) (b : Bool) : Trans :=
  if s < tm.nstates then tm.table.getD (TM.slot s b) Trans.halt else Trans.halt

structure Config where
  state : Option Nat
  head : Int
  tape : Int → Bool

def blankTape : Int → Bool := fun _ => false

def Config.init (tm : TM) (tape : Int → Bool := blankTape) : Config :=
  { state := if tm.start < tm.nstates then some tm.start else none
    head := 0
    tape }

def step (tm : TM) (c : Config) : Config :=
  match c.state with
  | none => c
  | some s =>
    let tr := tm.get s (c.tape c.head)
    { state := tr.next
      head := if tr.moveRight then c.head + 1 else c.head - 1
      tape := fun i => if i == c.head then tr.write else c.tape i }

def run (tm : TM) : Nat → Config → Config
  | 0, c => c
  | n + 1, c => run tm n (step tm c)

theorem run_zero (tm : TM) (c : Config) : run tm 0 c = c := rfl

theorem run_succ (tm : TM) (n : Nat) (c : Config) :
    run tm (n + 1) c = run tm n (step tm c) := rfl

theorem run_step (tm : TM) (n : Nat) (c : Config) :
    run tm (n + 1) c = step tm (run tm n c) := by
  induction n generalizing c with
  | zero => rfl
  | succ n ih =>
    calc
      run tm (n + 2) c = run tm (n + 1) (step tm c) := rfl
      _ = step tm (run tm n (step tm c)) := ih _
      _ = step tm (run tm (n + 1) c) := by rw [run_succ]

theorem step_halted (tm : TM) (c : Config) (h : c.state = none) :
    step tm c = c := by
  simp [step, h]

theorem run_halted (tm : TM) (n : Nat) (c : Config) (h : c.state = none) :
    run tm n c = c := by
  induction n with
  | zero => rfl
  | succ n ih => rw [run_succ, step_halted tm c h, ih]

def HaltsFrom (tm : TM) (c : Config) : Prop :=
  ∃ n, (run tm n c).state = none

def Halts (tm : TM) : Prop :=
  HaltsFrom tm (Config.init tm)

theorem run_add_halt (tm : TM) (c : Config) (n k : Nat)
    (hn : (run tm n c).state = none) :
    (run tm (n + k) c).state = none := by
  induction k with
  | zero => simpa using hn
  | succ k ih =>
    rw [Nat.add_succ, run_step, step_halted tm (run tm (n + k) c) ih]
    exact ih

theorem run_ge_halt (tm : TM) (c : Config) (n m : Nat)
    (hn : (run tm n c).state = none) (hm : n ≤ m) :
    (run tm m c).state = none := by
  have := run_add_halt tm c n (m - n) hn
  rwa [Nat.add_sub_of_le hm] at this

/-! ## 1. Named machines -/

def haltNow : TM where
  nstates := 1
  start := 0
  table := [Trans.halt, Trans.halt]

def idleTrans : Trans :=
  { write := false, moveRight := true, next := some 0 }

def idleBlank : TM where
  nstates := 1
  start := 0
  table := [idleTrans, idleTrans]

def paintTrans : Trans :=
  { write := true, moveRight := true, next := some 0 }

def paintRight : TM where
  nstates := 1
  start := 0
  table := [paintTrans, paintTrans]

@[simp] theorem haltNow_get (b : Bool) : haltNow.get 0 b = Trans.halt := by
  cases b <;> rfl

@[simp] theorem idleBlank_get (b : Bool) : idleBlank.get 0 b = idleTrans := by
  cases b <;> rfl

@[simp] theorem paintRight_get (b : Bool) : paintRight.get 0 b = paintTrans := by
  cases b <;> rfl

theorem haltNow_run_one :
    (run haltNow 1 (Config.init haltNow)).state = none := by
  rw [run_succ, run_zero]
  have hs : (Config.init haltNow).state = some 0 := by
    simp [Config.init, haltNow]
  simp [step, hs, haltNow_get, Trans.halt]

theorem haltNow_halts : Halts haltNow := ⟨1, haltNow_run_one⟩

theorem haltNow_run_ge_one (n : Nat) (h : 1 ≤ n) :
    (run haltNow n (Config.init haltNow)).state = none :=
  run_ge_halt haltNow (Config.init haltNow) 1 n haltNow_run_one h

theorem idle_step_state (c : Config) (h : c.state = some 0) :
    (step idleBlank c).state = some 0 := by
  simp [step, h, idleBlank_get, idleTrans]

theorem idle_run_state (n : Nat) :
    (run idleBlank n (Config.init idleBlank)).state = some 0 := by
  induction n with
  | zero => simp [run, Config.init, idleBlank]
  | succ n ih => rw [run_step, idle_step_state _ ih]

theorem step_idle_tape (c : Config) (hs : c.state = some 0) (i : Int) :
    (step idleBlank c).tape i =
      if i == c.head then false else c.tape i := by
  simp [step, hs, idleBlank_get, idleTrans]

theorem idle_run_tape (n : Nat) (i : Int) :
    (run idleBlank n (Config.init idleBlank)).tape i = false := by
  induction n generalizing i with
  | zero => simp [run, Config.init, blankTape]
  | succ n ih =>
    rw [run_step, step_idle_tape _ (idle_run_state n)]
    split <;> simp [ih]

theorem idle_not_halts : ¬ Halts idleBlank := by
  rintro ⟨n, hn⟩
  have := idle_run_state n
  simp [this] at hn

theorem paint_step_state (c : Config) (h : c.state = some 0) :
    (step paintRight c).state = some 0 := by
  simp [step, h, paintRight_get, paintTrans]

theorem paint_run_state (n : Nat) :
    (run paintRight n (Config.init paintRight)).state = some 0 := by
  induction n with
  | zero => simp [run, Config.init, paintRight]
  | succ n ih => rw [run_step, paint_step_state _ ih]

theorem paint_not_halts : ¬ Halts paintRight := by
  rintro ⟨n, hn⟩
  have := paint_run_state n
  simp [this] at hn

theorem step_paint_head (c : Config) (hs : c.state = some 0) :
    (step paintRight c).head = c.head + 1 := by
  simp [step, hs, paintRight_get, paintTrans]

theorem paint_run_head (n : Nat) :
    (run paintRight n (Config.init paintRight)).head = (n : Int) := by
  induction n with
  | zero => simp [run, Config.init]
  | succ n ih =>
    rw [run_step, step_paint_head _ (paint_run_state n), ih]
    simp

theorem step_paint_tape (c : Config) (hs : c.state = some 0) (i : Int) :
    (step paintRight c).tape i =
      if i == c.head then true else c.tape i := by
  simp [step, hs, paintRight_get, paintTrans]

theorem paint_run_tape (n : Nat) (i : Int) :
    (run paintRight n (Config.init paintRight)).tape i = true ↔
      0 ≤ i ∧ i < (n : Int) := by
  induction n generalizing i with
  | zero =>
    simp [run, Config.init, blankTape]
  | succ n ih =>
    rw [run_step, step_paint_tape _ (paint_run_state n), paint_run_head n]
    by_cases hi : i = (n : Int)
    · subst hi
      simp
      omega
    · have hbeq : (i == (n : Int)) = false := by
        simp [hi]
      simp [hbeq]
      constructor
      · intro h
        have := (ih i).mp h
        omega
      · intro h
        apply (ih i).mpr
        omega

/-! ## 2. Finite enumerated tables -/

/-- Base of the Go-style encoding: write × move × (nstates+1 next-codes). -/
def transRadix (nstates : Nat) : Nat := 4 * (nstates + 1)

def decodeTrans (code nstates : Nat) : Trans :=
  let write := code % 2 == 1
  let moveRight := (code / 2) % 2 == 1
  let nextCode := (code / 4) % (nstates + 1)
  { write := write
    moveRight := moveRight
    next := if nextCode = 0 then none else some (nextCode - 1) }

def numSlots (nstates : Nat) : Nat := 2 * nstates

/-- `index` in base `transRadix`, one digit per (state, bit) slot. -/
def TMFromIndex (index nstates : Nat) : TM where
  nstates := nstates
  start := 0
  table := List.ofFn fun (slot : Fin (numSlots nstates)) =>
    decodeTrans ((index / transRadix nstates ^ slot.val) % transRadix nstates) nstates

/-- Eight state-counts `1..8`, index in the high bits (mod table size). -/
def nstatesOf (n : Nat) : Nat := n % 8 + 1
def indexOf (n : Nat) : Nat := n / 8

def numTMs (nstates : Nat) : Nat :=
  transRadix nstates ^ numSlots nstates

def decode (n : Nat) : TM :=
  TMFromIndex (indexOf n % numTMs (nstatesOf n)) (nstatesOf n)

/-- Codes `512k` are the 1-state all-halt machine (`TMFromIndex 0 1`). -/
def halterCode (k : Nat) : Nat := 512 * k

/-- Codes `512k+48` are the 1-state blank-stay machine (`TMFromIndex 6 1`). -/
def looperCode (k : Nat) : Nat := 512 * k + 48

theorem halterCode_decode (k : Nat) : decode (halterCode k) = TMFromIndex 0 1 := by
  have hmod : halterCode k % 8 = 0 := by
    rw [halterCode, show 512 = 8 * 64 by decide, Nat.mul_assoc]
    exact Nat.mul_mod_right 8 (64 * k)
  have hdiv : halterCode k / 8 = 64 * k := by
    rw [halterCode, show 512 = 8 * 64 by decide, Nat.mul_assoc]
    exact Nat.mul_div_cancel_left (64 * k) (by decide : 0 < 8)
  have hns : nstatesOf (halterCode k) = 1 := by
    simp [nstatesOf, hmod]
  have hnum : numTMs 1 = 64 := by decide
  have hidx : indexOf (halterCode k) % numTMs 1 = 0 := by
    rw [indexOf, hdiv, hnum]
    exact Nat.mul_mod_right 64 k
  simp [decode, hns, hidx]

theorem looperCode_decode (k : Nat) : decode (looperCode k) = TMFromIndex 6 1 := by
  have hmod : looperCode k % 8 = 0 := by
    rw [looperCode, show 512 = 8 * 64 by decide, show 48 = 8 * 6 by decide,
      Nat.mul_assoc, ← Nat.mul_add]
    exact Nat.mul_mod_right 8 (64 * k + 6)
  have hdiv : looperCode k / 8 = 64 * k + 6 := by
    rw [looperCode, show 512 = 8 * 64 by decide, show 48 = 8 * 6 by decide,
      Nat.mul_assoc, ← Nat.mul_add]
    exact Nat.mul_div_cancel_left (64 * k + 6) (by decide : 0 < 8)
  have hns : nstatesOf (looperCode k) = 1 := by
    simp [nstatesOf, hmod]
  have hnum : numTMs 1 = 64 := by decide
  have hidx : indexOf (looperCode k) % numTMs 1 = 6 := by
    rw [indexOf, hdiv, hnum, Nat.mul_add_mod_self_left]
  simp [decode, hns, hidx]

/-- Index 0 writes blank and halts in place (move bit clear). `Trans.halt`
    moves right, so the transitions are equal only in `.next`. -/
theorem index0_next (b : Bool) : ((TMFromIndex 0 1).get 0 b).next = none := by
  cases b <;> native_decide

theorem index6_get_blank : (TMFromIndex 6 1).get 0 false = idleTrans := by
  native_decide

abbrev Store := Nat → Bool

def FiniteSupport (s : Store) : Prop :=
  ∃ N : Nat, ∀ n, N ≤ n → s n = false

def InfiniteStore (s : Store) : Prop := ¬ FiniteSupport s

/-- Characteristic bit of `Halts ∘ decode` on the finite-table enumeration.
    Not the parity function. -/
noncomputable def haltStore (n : Nat) : Bool :=
  haveI := Classical.propDecidable (Halts (decode n))
  decide (Halts (decode n))

theorem haltStore_spec (n : Nat) :
    haltStore n = true ↔ Halts (decode n) := by
  simp [haltStore]

theorem fromIndex0_run_one :
    (run (TMFromIndex 0 1) 1 (Config.init (TMFromIndex 0 1))).state = none := by
  rw [run_succ, run_zero]
  have hs : (Config.init (TMFromIndex 0 1)).state = some 0 := by
    simp [Config.init, TMFromIndex]
  simp [step, hs, index0_next]

theorem fromIndex0_halts : Halts (TMFromIndex 0 1) :=
  ⟨1, fromIndex0_run_one⟩

theorem stay_run (tm : TM) (n : Nat) (c : Config)
    (hc : c.state = some 0) (ht : ∀ i, c.tape i = false)
    (hget : tm.get 0 false = idleTrans) :
    (run tm n c).state = some 0 ∧ ∀ i, (run tm n c).tape i = false := by
  induction n with
  | zero => exact ⟨hc, ht⟩
  | succ n ih =>
    rw [run_step]
    obtain ⟨hs, htape⟩ := ih
    have hread : (run tm n c).tape (run tm n c).head = false := htape _
    have hget' : tm.get 0 ((run tm n c).tape (run tm n c).head) = idleTrans := by
      simpa [hread] using hget
    refine ⟨?_, ?_⟩
    · simp [step, hs, hget', idleTrans]
    · intro i
      simp [step, hs, hget, idleTrans, htape]

theorem fromIndex6_not_halts : ¬ Halts (TMFromIndex 6 1) := by
  rintro ⟨n, hn⟩
  have hinit_st : (Config.init (TMFromIndex 6 1)).state = some 0 := by
    simp [Config.init, TMFromIndex]
  have hinit_tp : ∀ i, (Config.init (TMFromIndex 6 1)).tape i = false := by
    intro i; simp [Config.init, blankTape]
  have h := stay_run (TMFromIndex 6 1) n (Config.init (TMFromIndex 6 1))
    hinit_st hinit_tp index6_get_blank
  simp [h.1] at hn

theorem infinitely_many_halt : ∀ N : Nat, ∃ n, N ≤ n ∧ Halts (decode n) := by
  intro N
  refine ⟨halterCode N, ?_, ?_⟩
  · simpa [halterCode] using Nat.le_mul_of_pos_left N (by decide : 0 < 512)
  · rw [halterCode_decode]
    exact fromIndex0_halts

theorem infinitely_many_loop : ∀ N : Nat, ∃ n, N ≤ n ∧ ¬ Halts (decode n) := by
  intro N
  refine ⟨looperCode N, ?_, ?_⟩
  · simp [looperCode]; omega
  · rw [looperCode_decode]
    exact fromIndex6_not_halts

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

/-! ## 3. Zeno ω-limit from `run` -/

inductive LimitKind where
  | halt
  | cauchy
  | diverge
deriving DecidableEq, Repr

def zenoK (k : Nat) : Nat := max k 1

def window (c : Config) (span : Nat) : List Bool :=
  List.range (2 * span + 1) |>.map fun i =>
    c.tape (c.head + (Int.ofNat i - Int.ofNat span))

def zenoEnd (tm : TM) (k : Nat) : Config :=
  run tm (2 ^ zenoK k) (Config.init tm)

def zenoMid (tm : TM) (k : Nat) : Config :=
  run tm (2 ^ (zenoK k - 1)) (Config.init tm)

def zenoFrozenB (tm : TM) (k : Nat) : Bool :=
  ((zenoMid tm k).state == (zenoEnd tm k).state) &&
    (window (zenoMid tm k) (2 ^ zenoK k) ==
      window (zenoEnd tm k) (2 ^ zenoK k))

def zenoLimitK (tm : TM) (k : Nat) : LimitKind :=
  if (zenoEnd tm k).state.isNone then LimitKind.halt
  else if zenoFrozenB tm k then LimitKind.cauchy
  else LimitKind.diverge

theorem zenoLimitK_eq_halt_iff (tm : TM) (k : Nat) :
    zenoLimitK tm k = LimitKind.halt ↔ (zenoEnd tm k).state = none := by
  cases h : (zenoEnd tm k).state
  · simp [zenoLimitK, h, Option.isNone]
  · simp [zenoLimitK, h, Option.isNone]
    split <;> simp

theorem two_pow_ge_self (n : Nat) : n ≤ 2 ^ n := by
  induction n with
  | zero => simp
  | succ n ih =>
    have : n + 1 ≤ 2 ^ n + 1 := Nat.succ_le_succ ih
    have h1 : 1 ≤ 2 ^ n := Nat.one_le_pow n 2 (by decide)
    have : 2 ^ n + 1 ≤ 2 ^ n + 2 ^ n := Nat.add_le_add_left h1 _
    have hpow : 2 ^ (n + 1) = 2 ^ n * 2 := Nat.pow_succ 2 n
    have : 2 ^ n * 2 = 2 ^ n + 2 ^ n := by
      rw [Nat.mul_two]
    omega

theorem le_two_pow_zenoK (n : Nat) : n ≤ 2 ^ zenoK n := by
  unfold zenoK
  cases n with
  | zero => decide
  | succ n =>
    rw [Nat.max_eq_left (Nat.succ_le_succ (Nat.zero_le n))]
    exact two_pow_ge_self (n + 1)

theorem zenoLimitK_haltNow (k : Nat) : zenoLimitK haltNow k = LimitKind.halt := by
  rw [zenoLimitK_eq_halt_iff, zenoEnd]
  exact haltNow_run_ge_one _ (Nat.one_le_pow (zenoK k) 2 (by decide))

theorem Halts_of_zenoLimitK_halt (tm : TM) (k : Nat)
    (h : zenoLimitK tm k = LimitKind.halt) : Halts tm := by
  rw [zenoLimitK_eq_halt_iff] at h
  exact ⟨2 ^ zenoK k, h⟩

theorem zenoLimitK_halt_of_Halts (tm : TM) (h : Halts tm) :
    ∃ k, zenoLimitK tm k = LimitKind.halt := by
  obtain ⟨n, hn⟩ := h
  refine ⟨n, ?_⟩
  rw [zenoLimitK_eq_halt_iff]
  exact run_ge_halt tm (Config.init tm) n (2 ^ zenoK n) hn (le_two_pow_zenoK n)

theorem Halts_iff_zeno_eventually_halt (tm : TM) :
    Halts tm ↔ ∃ k, zenoLimitK tm k = LimitKind.halt :=
  ⟨zenoLimitK_halt_of_Halts tm, fun ⟨k, hk⟩ => Halts_of_zenoLimitK_halt tm k hk⟩

theorem window_idle (n span : Nat) :
    window (run idleBlank n (Config.init idleBlank)) span =
      window (run idleBlank 0 (Config.init idleBlank)) span := by
  simp [window, idle_run_tape]

theorem zenoLimitK_idle (k : Nat) : zenoLimitK idleBlank k = LimitKind.cauchy := by
  have hne : (zenoEnd idleBlank k).state.isNone = false := by
    simp [zenoEnd, idle_run_state]
  have hf : zenoFrozenB idleBlank k = true := by
    simp [zenoFrozenB, zenoMid, zenoEnd, idle_run_state, window, idle_run_tape]
  simp [zenoLimitK, hne, hf]

theorem zenoLimitK_paint_ne_halt (k : Nat) :
    zenoLimitK paintRight k ≠ LimitKind.halt := by
  intro h
  have := (zenoLimitK_eq_halt_iff paintRight k).mp h
  simp [zenoEnd, paint_run_state] at this

/-! ## 4. Particles, cosmos, decay -/

structure ComputationalSystem where
  machine : TM
  store : Store

structure Particle extends ComputationalSystem where
  query : TM

class InfiniteInformation (p : Particle) : Prop where
  infinite : InfiniteStore p.store

class HaltEncoded (p : Particle) : Prop where
  lookup : p.store 0 = true ↔ Halts p.query

noncomputable def haltBit (tm : TM) : Bool :=
  haveI := Classical.propDecidable (Halts tm)
  decide (Halts tm)

theorem haltBit_spec (tm : TM) : haltBit tm = true ↔ Halts tm := by
  simp [haltBit]

noncomputable def particleFor (e : TM) : Particle :=
  { machine := e
    store := fun n => if n = 0 then haltBit e else haltStore n
    query := e }

instance (e : TM) : HaltEncoded (particleFor e) where
  lookup := by
    simp [particleFor, haltBit_spec]

instance (e : TM) : InfiniteInformation (particleFor e) where
  infinite := by
    rintro ⟨N, hN⟩
    obtain ⟨n, hn, hH⟩ := infinitely_many_halt (N + 1)
    have hn0 : n ≠ 0 :=
      Nat.pos_iff_ne_zero.mp (Nat.lt_of_lt_of_le (Nat.succ_pos N) hn)
    have ht : (particleFor e).store n = true := by
      simp [particleFor, hn0]
      exact (haltStore_spec n).mpr hH
    have hf : (particleFor e).store n = false := hN n (Nat.le_trans (Nat.le_succ N) hn)
    simp [ht] at hf

structure Hypercomputer where
  oracle : TM → Bool
  correct : ∀ p, oracle p = true ↔ Halts p

noncomputable def cosmos : Hypercomputer where
  oracle := haltBit
  correct := haltBit_spec

/-- Input `n` is `n` leading ones on a blank tape. -/
def unaryTape (n : Nat) : Int → Bool :=
  fun i => if 0 ≤ i ∧ i < (n : Int) then true else false

/-- `tm` computes `f` if, on `unaryTape n`, it halts with cell `n` equal to `f n`. -/
def Computes (tm : TM) (f : Nat → Bool) : Prop :=
  ∀ n, ∃ k,
    let c := run tm k (Config.init tm (unaryTape n))
    c.state = none ∧ c.tape (n : Int) = f n

/-- A bit function is computable when some finite TM computes it. -/
def Computable (f : Nat → Bool) : Prop :=
  ∃ tm : TM, Computes tm f

/-- No finite TM decides `Halts` on the enumerated codes. -/
axiom no_TM_decides_Halt : ¬ Computable haltStore

theorem haltStore_not_computable : ¬ Computable haltStore :=
  no_TM_decides_Halt

inductive Decay where
  | yes
  | no
deriving DecidableEq, Repr

def Decay.ofBool : Bool → Decay
  | true => .yes
  | false => .no

@[simp] theorem Decay.ofBool_eq_yes (b : Bool) :
    Decay.ofBool b = .yes ↔ b = true := by
  cases b <;> simp [Decay.ofBool]

def decay (p : Particle) : Decay :=
  Decay.ofBool (p.store 0)

theorem decay_solves_halting (p : Particle) [hp : HaltEncoded p] :
    decay p = .yes ↔ Halts p.query := by
  simp [decay, hp.lookup]

theorem decay_eq_cosmos (p : Particle) [HaltEncoded p] :
    decay p = .yes ↔ cosmos.oracle p.query = true := by
  rw [decay_solves_halting, cosmos.correct]

theorem particle_for_decides (e : TM) :
    decay (particleFor e) = .yes ↔ Halts e :=
  decay_solves_halting (particleFor e)

/-- If the particle's machine is its query, decay is the ω-limit halt bit. -/
theorem decay_is_zeno_limit (p : Particle)
    (h : p.machine = p.query) [HaltEncoded p] :
    decay p = .yes ↔ ∃ k, zenoLimitK p.machine k = LimitKind.halt := by
  rw [h, decay_solves_halting, Halts_iff_zeno_eventually_halt]

theorem cosmos_eq_zeno (tm : TM) :
    cosmos.oracle tm = true ↔ ∃ k, zenoLimitK tm k = LimitKind.halt := by
  rw [cosmos.correct, Halts_iff_zeno_eventually_halt]

/-! ## 5. Truncated particles (finite-\(p\) laboratory) -/

/-- The first `N` cells of two stores agree. -/
def prefixEq (s t : Store) (N : Nat) : Prop :=
  ∀ n, n < N → s n = t n

/-- Zero the tail: an `N`-bit analog tape. -/
def truncateStore (s : Store) (N : Nat) : Store :=
  fun n => if n < N then s n else false

theorem truncateStore_prefix (s : Store) (N : Nat) :
    prefixEq (truncateStore s N) s N := by
  intro n hn
  simp [truncateStore, hn]

theorem truncateStore_tail (s : Store) (N n : Nat) (hn : N ≤ n) :
    truncateStore s N n = false := by
  simp [truncateStore, Nat.not_lt.mpr hn]

theorem truncateStore_finite (s : Store) (N : Nat) :
    FiniteSupport (truncateStore s N) :=
  ⟨N, fun n hn => truncateStore_tail s N n hn⟩

/-- A laboratory particle: the query is a program *code*, read at
    that address on the store. -/
structure LabParticle where
  machine : TM
  store : Store
  query : Nat

/-- Decay reads the store at the query index (finite-\(p\) readout). -/
def labDecay (p : LabParticle) : Decay :=
  Decay.ofBool (p.store p.query)

/-- The store matches `haltStore` on cells `0 .. N-1`. -/
class Truncated (p : LabParticle) (N : Nat) : Prop where
  agree : prefixEq p.store haltStore N

/-- Infinite-\(p\) lab particle: full `haltStore`. -/
noncomputable def labParticleFor (e : Nat) : LabParticle :=
  { machine := decode e, store := haltStore, query := e }

/-- Finite-\(p\) particle: `haltStore` truncated to `N` bits. -/
noncomputable def truncatedParticle (e N : Nat) : LabParticle :=
  { machine := decode e, store := truncateStore haltStore N, query := e }

noncomputable instance (e N : Nat) : Truncated (truncatedParticle e N) N where
  agree := truncateStore_prefix haltStore N

theorem truncatedParticle_finite (e N : Nat) :
    FiniteSupport (truncatedParticle e N).store :=
  truncateStore_finite haltStore N

/-- In range, decay decides `Halts` on the coded query. -/
theorem truncated_sound {p : LabParticle} {N : Nat} [hp : Truncated p N]
    (h : p.query < N) :
    labDecay p = .yes ↔ Halts (decode p.query) := by
  have heq : p.store p.query = haltStore p.query := hp.agree p.query h
  simp [labDecay, heq, haltStore_spec]

theorem truncatedParticle_sound (e N : Nat) (h : e < N) :
    labDecay (truncatedParticle e N) = .yes ↔ Halts (decode e) :=
  truncated_sound (p := truncatedParticle e N) h

/-- Out of range, a truncated tape reads `false` even if the program
    actually halts. -/
theorem truncatedParticle_out_of_range (e N : Nat) (he : N ≤ e) :
    labDecay (truncatedParticle e N) = .no := by
  simp [labDecay, truncatedParticle, truncateStore, Nat.not_lt.mpr he,
    Decay.ofBool]

/-- `N = 0`: no query is in range. -/
theorem truncated_zero_no_query (e : Nat) :
    ¬ e < 0 :=
  Nat.not_lt_zero e

/-- There is always a halting query the `N`-bit tape misses. -/
theorem truncated_misses_a_halter (N : Nat) :
    ∃ e, N ≤ e ∧ Halts (decode e) ∧
      labDecay (truncatedParticle e N) = .no := by
  obtain ⟨e, he, hH⟩ := infinitely_many_halt N
  exact ⟨e, he, hH, truncatedParticle_out_of_range e N he⟩

/-- One extra analog bit decides a previously truncated halter:
    `halterCode N` is missed at depth `halterCode N` and certified at
    depth `halterCode N + 1`. -/
theorem extra_bit_certifies (N : Nat) :
    labDecay (truncatedParticle (halterCode N) (halterCode N + 1)) = .yes ∧
      labDecay (truncatedParticle (halterCode N) (halterCode N)) = .no := by
  constructor
  · have hlt : halterCode N < halterCode N + 1 := Nat.lt_succ_self _
    have hH : Halts (decode (halterCode N)) := by
      rw [halterCode_decode]; exact fromIndex0_halts
    exact (truncatedParticle_sound (halterCode N) (halterCode N + 1) hlt).mpr hH
  · exact truncatedParticle_out_of_range (halterCode N) (halterCode N) (Nat.le_refl _)

/-! ## 6. Computable = some finite TM writes the bits -/

/-- Even/odd parity. This is *not* `haltStore`. -/
def parity (n : Nat) : Bool := decide (n % 2 = 0)

/-- Walk the unary block, toggling parity; write the bit on the first blank. -/
def parityTM : TM where
  nstates := 2
  start := 0
  table := [
    { write := true, moveRight := false, next := none },
    { write := true, moveRight := true, next := some 1 },
    { write := false, moveRight := false, next := none },
    { write := true, moveRight := true, next := some 0 }
  ]

@[simp] theorem parity_get_0_false :
    parityTM.get 0 false = { write := true, moveRight := false, next := none } := rfl

@[simp] theorem parity_get_0_true :
    parityTM.get 0 true = { write := true, moveRight := true, next := some 1 } := rfl

@[simp] theorem parity_get_1_false :
    parityTM.get 1 false = { write := false, moveRight := false, next := none } := rfl

@[simp] theorem parity_get_1_true :
    parityTM.get 1 true = { write := true, moveRight := true, next := some 0 } := rfl

theorem parity_step_one (c : Config) (hs0 : c.state = some 0) (hb : c.tape c.head = true) :
    (step parityTM c).state = some 1 ∧ (step parityTM c).head = c.head + 1 ∧
      ∀ i, (step parityTM c).tape i = c.tape i := by
  simp [step, hs0, hb, parity_get_0_true]

theorem parity_step_zero_even (c : Config) (hs0 : c.state = some 0) (hb : c.tape c.head = false) :
    (step parityTM c).state = none ∧ (step parityTM c).tape c.head = true ∧
      ∀ i, i ≠ c.head → (step parityTM c).tape i = c.tape i := by
  refine ⟨?_, ?_, ?_⟩
  · simp [step, hs0, hb, parity_get_0_false]
  · simp [step, hs0, hb, parity_get_0_false]
  · intro i hi
    simp [step, hs0, hb, parity_get_0_false]
    intro h
    exact (hi h).elim

theorem parity_step_one_odd (c : Config) (hs : c.state = some 1) (hb : c.tape c.head = true) :
    (step parityTM c).state = some 0 ∧ (step parityTM c).head = c.head + 1 ∧
      ∀ i, (step parityTM c).tape i = c.tape i := by
  simp [step, hs, hb, parity_get_1_true]

theorem parity_step_blank_odd (c : Config) (hs : c.state = some 1) (hb : c.tape c.head = false) :
    (step parityTM c).state = none ∧ (step parityTM c).tape c.head = false ∧
      ∀ i, i ≠ c.head → (step parityTM c).tape i = c.tape i := by
  refine ⟨?_, ?_, ?_⟩
  · simp [step, hs, hb, parity_get_1_false]
  · simp [step, hs, hb, parity_get_1_false]
  · intro i hi
    simp [step, hs, hb, parity_get_1_false]
    exact fun _ => hi

theorem parity_before (n j : Nat) (hj : j ≤ n) :
    (run parityTM j (Config.init parityTM (unaryTape n))).head = (j : Int) ∧
      (run parityTM j (Config.init parityTM (unaryTape n))).state = some (j % 2) ∧
        ∀ i, (run parityTM j (Config.init parityTM (unaryTape n))).tape i =
          unaryTape n i := by
  induction j with
  | zero =>
    simp [run, Config.init, parityTM, unaryTape]
  | succ j ih =>
    have hj' : j ≤ n := Nat.le_of_succ_le hj
    have hjn : j < n := Nat.lt_of_lt_of_le (Nat.lt_succ_self j) hj
    obtain ⟨hhead, hst, htape⟩ := ih hj'
    have hbit : (run parityTM j (Config.init parityTM (unaryTape n))).tape (j : Int) = true := by
      rw [htape, unaryTape]
      have hlt : (j : Int) < (n : Int) := by exact_mod_cast hjn
      simp [Int.natCast_nonneg, hlt]
    rw [run_step]
    by_cases he : j % 2 = 0
    · have hs : (run parityTM j (Config.init parityTM (unaryTape n))).state = some 0 := by
        simp [hst, he]
      obtain ⟨hs1, hh1, ht1⟩ := parity_step_one _ hs (by simpa [hhead] using hbit)
      refine ⟨?_, ?_, ?_⟩
      · simp [hh1, hhead, Int.natCast_succ]
      · have : (j + 1) % 2 = 1 := by omega
        simp [hs1, this]
      · intro i
        rw [ht1, htape]
    · have hs : (run parityTM j (Config.init parityTM (unaryTape n))).state = some 1 := by
        have : j % 2 = 1 := by omega
        simp [hst, this]
      obtain ⟨hs0, hh0, ht0⟩ := parity_step_one_odd _ hs (by simpa [hhead] using hbit)
      refine ⟨?_, ?_, ?_⟩
      · simp [hh0, hhead, Int.natCast_succ]
      · have : (j + 1) % 2 = 0 := by omega
        simp [hs0, this]
      · intro i
        rw [ht0, htape]

theorem parity_computes : Computes parityTM parity := by
  intro n
  refine ⟨n + 1, ?_⟩
  have ⟨hhead, hst, htape⟩ := parity_before n n (Nat.le_refl n)
  rw [run_step]
  have hblank : (run parityTM n (Config.init parityTM (unaryTape n))).tape (n : Int) = false := by
    rw [htape, unaryTape]
    simp [Int.lt_irrefl (n : Int)]
  by_cases he : n % 2 = 0
  · have hs : (run parityTM n (Config.init parityTM (unaryTape n))).state = some 0 := by
      simp [hst, he]
    obtain ⟨hst', ht', _⟩ := parity_step_zero_even _ hs (by simpa [hhead] using hblank)
    have htn :
        (step parityTM (run parityTM n (Config.init parityTM (unaryTape n)))).tape (n : Int) =
          true := by
      rw [← hhead]; exact ht'
    simp [parity, he, hst', htn]
  · have hs : (run parityTM n (Config.init parityTM (unaryTape n))).state = some 1 := by
      have : n % 2 = 1 := by omega
      simp [hst, this]
    obtain ⟨hst', ht', _⟩ := parity_step_blank_odd _ hs (by simpa [hhead] using hblank)
    have htn :
        (step parityTM (run parityTM n (Config.init parityTM (unaryTape n)))).tape (n : Int) =
          false := by
      rw [← hhead]; exact ht'
    simp [parity, he, hst', htn]

theorem parity_computable : Computable parity :=
  ⟨parityTM, parity_computes⟩

end Hyperuniverse

