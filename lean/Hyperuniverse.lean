/-
  Hyperuniverse.lean
  ──────────────────────────────────────────────────────────────────────────
  A 2-symbol TM (`step` / `run`) is the dynamics. `Halts` is ∃ n, halted
  after n steps. The Zeno ω-limit is defined by comparing the tape window
  at 2^{k-1} and 2^k (halt / Cauchy / diverge).

  The universe is a classical halt oracle. A particle carries an infinite
  store; decay reads cell 0. When `machine = query`, that bit is the
  ω-limit halt bit of the internal TM.
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
  delta : Nat → Bool → Trans

def TM.get (tm : TM) (s : Nat) (b : Bool) : Trans :=
  if s < tm.nstates then tm.delta s b else Trans.halt

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
  delta := fun _ _ => Trans.halt

def idleBlank : TM where
  nstates := 1
  start := 0
  delta := fun _ _ =>
    { write := false, moveRight := true, next := some 0 }

def paintRight : TM where
  nstates := 1
  start := 0
  delta := fun _ _ =>
    { write := true, moveRight := true, next := some 0 }

theorem haltNow_run_one :
    (run haltNow 1 (Config.init haltNow)).state = none := by
  simp [run, step, Config.init, TM.get, haltNow, Trans.halt]

theorem haltNow_halts : Halts haltNow := ⟨1, haltNow_run_one⟩

theorem haltNow_run_ge_one (n : Nat) (h : 1 ≤ n) :
    (run haltNow n (Config.init haltNow)).state = none :=
  run_ge_halt haltNow (Config.init haltNow) 1 n haltNow_run_one h

theorem idle_step_state (c : Config) (h : c.state = some 0) :
    (step idleBlank c).state = some 0 := by
  simp [step, idleBlank, TM.get, h]

theorem idle_run_state (n : Nat) :
    (run idleBlank n (Config.init idleBlank)).state = some 0 := by
  induction n with
  | zero => simp [run, Config.init, idleBlank]
  | succ n ih => rw [run_step, idle_step_state _ ih]

theorem step_idle_tape (c : Config) (hs : c.state = some 0) (i : Int) :
    (step idleBlank c).tape i =
      if i == c.head then false else c.tape i := by
  simp [step, idleBlank, TM.get, hs]

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
  simp [step, paintRight, TM.get, h]

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
  simp [step, paintRight, TM.get, hs]

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
  simp [step, paintRight, TM.get, hs]

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

/-! ## 2. Codes and χ_Halt on a two-behaviour family -/

def decode (n : Nat) : TM :=
  if n % 2 = 0 then haltNow else idleBlank

theorem decode_even (k : Nat) : decode (2 * k) = haltNow := by
  simp [decode, Nat.mul_mod_right]

theorem decode_odd (k : Nat) : decode (2 * k + 1) = idleBlank := by
  simp [decode, Nat.add_mod, Nat.mul_mod_right]

abbrev Store := Nat → Bool

def FiniteSupport (s : Store) : Prop :=
  ∃ N : Nat, ∀ n, N ≤ n → s n = false

def InfiniteStore (s : Store) : Prop := ¬ FiniteSupport s

def haltStore (n : Nat) : Bool := decide (n % 2 = 0)

theorem haltStore_spec (n : Nat) :
    haltStore n = true ↔ Halts (decode n) := by
  simp only [haltStore, decode]
  by_cases h : n % 2 = 0
  · simp [h, haltNow_halts]
  · simp [h, idle_not_halts]

theorem infinitely_many_halt : ∀ N : Nat, ∃ n, N ≤ n ∧ Halts (decode n) := by
  intro N
  refine ⟨2 * N, Nat.le_mul_of_pos_left N (by decide : 0 < 2), ?_⟩
  simpa [decode_even] using haltNow_halts

theorem infinitely_many_loop : ∀ N : Nat, ∃ n, N ≤ n ∧ ¬ Halts (decode n) := by
  intro N
  refine ⟨2 * N + 1, by omega, ?_⟩
  simpa [decode_odd] using idle_not_halts

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

axiom Computable : (TM → Bool) → Prop

axiom no_TM_decides_Halt :
  ¬ ∃ f, Computable f ∧ ∀ t, f t = true ↔ Halts t

theorem cosmos_is_not_Turing : ¬ Computable cosmos.oracle := by
  intro h
  exact no_TM_decides_Halt ⟨cosmos.oracle, h, cosmos.correct⟩

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

end Hyperuniverse
