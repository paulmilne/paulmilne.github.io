/-! The pq-system from Hofstadter's *Gödel, Escher, Bach*, chapter II.
    A well-formed string is  x p y q z  with x, y, z strings of hyphens.
    A string of hyphens is determined by its length, so we record a string
    by the three lengths (x, y, z). -/

/-- Axioms: x p - q x-  for any x of at least one hyphen.
    Rule:   from x p y q z, derive x p y- q z-. -/
inductive PQ : Nat → Nat → Nat → Prop
  | ax {x} : 0 < x → PQ x 1 (x + 1)
  | rule {x y z} : PQ x y z → PQ x (y + 1) (z + 1)

/-- Soundness of the addition reading: every theorem is a true sum. -/
theorem PQ.sound {x y z : Nat} (h : PQ x y z) : x + y = z ∧ 0 < x ∧ 0 < y := by
  induction h with
  | ax hx => omega
  | rule _ ih => omega

/-- Completeness: every true sum of positive numbers is a theorem. -/
theorem PQ.complete {x y : Nat} (hx : 0 < x) (hy : 0 < y) : PQ x y (x + y) := by
  induction y with
  | zero => omega
  | succ y ih =>
    cases y with
    | zero => exact PQ.ax hx
    | succ y =>
      have := PQ.rule (ih (by omega))
      simpa [Nat.add_assoc] using this

/-- So the theorems are exactly the true additions. -/
theorem PQ.iff {x y z : Nat} : PQ x y z ↔ (0 < x ∧ 0 < y ∧ x + y = z) := by
  constructor
  · intro h; have := h.sound; omega
  · rintro ⟨hx, hy, rfl⟩; exact PQ.complete hx hy

/-- The same theorems under a second reading: p means "equals" and
    q means "taken from", so x p y q z says x = z − y. -/
theorem PQ.sound_subtraction {x y z : Nat} (h : PQ x y z) : x = z - y := by
  have := h.sound; omega

/-- A decision procedure, extracted from the characterisation. -/
instance (x y z : Nat) : Decidable (PQ x y z) :=
  decidable_of_iff _ PQ.iff.symm

example : PQ 2 3 5 := by decide        -- --p---q-----   (2 + 3 = 5)
example : ¬ PQ 2 2 5 := by decide      -- --p--q-----    (2 + 2 ≠ 5)
