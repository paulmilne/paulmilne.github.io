/-! The MIU system from Hofstadter's *Gödel, Escher, Bach*, chapter I.
    A theorem is `M` followed by a string of `I`s and `U`s; we drop the
    leading `M` and work with the rest. -/

inductive Sym | I | U
  deriving DecidableEq, Repr

open Sym

/-- The four rules, with `MI` as the only axiom. -/
inductive Derivable : List Sym → Prop
  | mi : Derivable [I]
  | rule1 {x} : Derivable (x ++ [I]) → Derivable (x ++ [I, U])          -- xI → xIU
  | rule2 {x} : Derivable x → Derivable (x ++ x)                         -- Mx → Mxx
  | rule3 {x y} : Derivable (x ++ [I, I, I] ++ y) → Derivable (x ++ [U] ++ y)  -- III → U
  | rule4 {x y} : Derivable (x ++ [U, U] ++ y) → Derivable (x ++ y)      -- UU → (nothing)

/-- Number of `I`s in a string. -/
def countI : List Sym → Nat
  | [] => 0
  | I :: s => countI s + 1
  | U :: s => countI s

theorem countI_append (a b : List Sym) : countI (a ++ b) = countI a + countI b := by
  induction a with
  | nil => simp [countI]
  | cons c a ih => cases c <;> simp [countI, ih] <;> omega

/-- The invariant: the number of `I`s is never a multiple of 3. -/
theorem countI_mod3 {s : List Sym} (h : Derivable s) : countI s % 3 ≠ 0 := by
  induction h with
  | mi => decide
  | rule1 _ ih => simp only [countI_append, countI] at *; omega
  | rule2 _ ih => simp only [countI_append] at *; omega
  | rule3 _ ih => simp only [countI_append, countI] at *; omega
  | rule4 _ ih => simp only [countI_append, countI] at *; omega

/-- Hofstadter's question, answered: `MU` is not a theorem. -/
theorem MU_not_derivable : ¬ Derivable [U] := by
  intro h
  have := countI_mod3 h
  simp [countI] at this
