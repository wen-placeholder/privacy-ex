# Exercise Sheet 1 — Manual
**Privacy-Preserving Methods for Data Science and Distributed Systems, Spring 2026**

---

## 1. How to Run the Code

### Requirements

Python 3 with the following packages:

```
pandas, numpy, scipy, pytest
```

All source files and the dataset must be in the same working directory:

```
ex1/
├── e1.py
├── adult_with_pii (1).csv
└── manual.md
```

### Run the main script

```bash
python3 e1.py
```

This executes the `__main__` block, which prints results for Questions 1–7 to stdout.

### Run the tests

```bash
python3 -m pytest e1.py -v
```

Expected output: **6 passed**.

---

## 2. Interpreting the Output

### Question 1 — Checking for k-anonymity

```
=== k-anonymity check for k in [1,10] ===
  k=1: True
  k=2: False
  ...
```

`adult_small` contains the first 100 rows of the adult dataset with columns
`Education`, `Marital Status`, and `Target`.

**Column roles:**

| Column | Role |
|---|---|
| (Name, SSN, DOB — not in adult_small) | Identifiers |
| Education, Marital Status | Quasi-identifiers |
| Target | Sensitive attribute |

**k-anonymity results:** `adult_small` satisfies only k = 1. For k ≥ 2 it fails because there exist
(Education, Marital Status) combinations that appear exactly once in the 100 rows — for example,
a row with `Bachelors` + `Never-married` may be the sole record in its equivalence class.
Since at least one equivalence class has size 1, no k ≥ 2 is satisfied.

---

### Question 2 — Generalization and suppression

```
=== Equivalence class sizes after generalization ===
Education  Marital Status  Target
< HS       Married         <=50K     11
           Not Married     <=50K      3
>= HS      Married         <=50K     27
                           >50K      20
           Not Married     <=50K     34
                           >50K       5
Rows after suppression: 100 (original: 100)
```

After applying:
- Education → `< HS` or `>= HS`
- Marital Status → `Married` or `Not Married`

all four equivalence classes already have size ≥ 2, so no rows need to be suppressed.

**Homogeneity attack:** A homogeneity attack is possible when all records in an equivalence class
share the same value of the sensitive attribute `Target`. From the output above:

- `(< HS, Married, <=50K)` — all 11 records have `Target = <=50K`
- `(< HS, Not Married, <=50K)` — all 3 records have `Target = <=50K`

An attacker who knows a person belongs to either of these groups can infer their income with
certainty, despite k = 2 being satisfied. The groups with mixed `Target` values
(`>= HS, Married` and `>= HS, Not Married`) are not vulnerable to this attack.

---

### Question 3 — Generalization (numeric)

```
generalize_numeric(47401, 0) = 47401
generalize_numeric(47401, 2) = 47400
generalize_numeric(47401, 4) = 40000
```

The function zeros the last `n` digits of a zip code using integer arithmetic:
`(zip // 10^n) * 10^n`.

---

### Question 4 — k-anonymity on a large dataset

```
=== Q4: k-anonymity on full adult dataset (QIs: Zip, Sex, Age) ===
  Zip n=0, Age n=0 -> suppress for k=3: 32561, k=7: 32561
  Zip n=0, Age n=1 -> suppress for k=3: 32519, k=7: 32561
  Zip n=2, Age n=0 -> suppress for k=3: 30517, k=7: 32561
  Zip n=2, Age n=1 -> suppress for k=3: 7245,  k=7: 24808
  Zip n=3, Age n=0 -> suppress for k=3: 6188,  k=7: 24420
  Zip n=3, Age n=1 -> suppress for k=3: 255,   k=7: 1143
```

`suppress_count(k, qis, df)` returns the number of rows belonging to equivalence classes
of size < k — i.e., the rows that must be deleted to achieve k-anonymity.

**How many digits to generalize?**
Without generalization, virtually the entire dataset (32 561 rows) must be suppressed because
raw zip codes combined with exact age create near-unique records. Generalizing Zip by 3 digits
and Age by 1 digit reduces suppression to 255 rows for k = 3 and 1 143 rows for k = 7.
This is the recommended setting: it preserves utility while making suppression tractable.
Further generalization (e.g., Zip n = 4) would increase information loss without proportional
privacy gain for typical k values.

---

### Question 5 — Checking for ℓ-diversity

`is_l_diverse(l, qis, sens_col, df, type)` checks whether every equivalence class satisfies
ℓ-diversity:

- **Probabilistic ℓ-diversity:** the most frequent sensitive value in each class must have
  frequency ≤ 1/ℓ.
- **Entropy ℓ-diversity:** the Shannon entropy of sensitive values in each class must be ≥ log(ℓ).

---

### Question 6 — ℓ-diversity of the adult dataset

```
=== Q6: l-diversity of adult dataset ===
  QIs: Education, Marital Status, Sex | Sensitive: DOB
  l=2: probabilistic=False, entropy=False
  l=3: probabilistic=False, entropy=False

  Groups preventing probabilistic 2-diversity (max_freq > 0.5):
    ('Assoc-voc', 'Married-AF-spouse', 'Male'): size=1, max_freq=1.000
    ('Preschool', 'Divorced', 'Male'): size=1, max_freq=1.000
    ('Preschool', 'Separated', 'Female'): size=1, max_freq=1.000
    ('Preschool', 'Widowed', 'Male'): size=1, max_freq=1.000
    ('Prof-school', 'Married-spouse-absent', 'Female'): size=1, max_freq=1.000
    ('Some-college', 'Married-AF-spouse', 'Male'): size=1, max_freq=1.000
```

The dataset is **not** ℓ-diverse for any ℓ ≥ 2. Six equivalence classes contain exactly one
record, meaning their single DOB value has frequency 1.0 — violating the condition for both
probabilistic and entropy ℓ-diversity.

**Difference between probabilistic and entropy ℓ-diversity:**
Both variants fail here for the same reason (singleton groups). In general, probabilistic
ℓ-diversity only constrains the *most common* sensitive value, while entropy ℓ-diversity
considers the *entire distribution*. A group where one value appears with frequency just below
1/ℓ but all remaining values are concentrated in a second value can satisfy probabilistic
ℓ-diversity while failing entropy ℓ-diversity. Conversely, entropy ℓ-diversity can accept a
higher effective ℓ when values are spread more evenly, as seen in Question 7.

---

### Question 7 — How diverse is df?

```
=== Q7: max_l on full adult dataset generalized with Q2 rules ===
  max_l (probabilistic): 97
  max_l (entropy): 193
```

After applying the Q2 generalizations (Education → `< HS` / `>= HS`,
Marital Status → `Married` / `Not Married`) to the full adult dataset, all equivalence classes
become large and DOB values are spread across many distinct dates.

`max_l` iterates over all equivalence classes and computes the maximum ℓ each class can support:
- Probabilistic: floor(group_size / max_count)
- Entropy: floor(exp(entropy))

The overall max_l is the minimum across all groups (bottleneck group).

**Why is entropy max_l larger than probabilistic max_l?**
Entropy ℓ-diversity measures the full spread of the distribution: even if one DOB value appears
slightly more often than 1/97 of the group, the overall entropy can still be high if the
remaining values are uniformly spread. Probabilistic ℓ-diversity is a stricter, worst-case
measure that is entirely determined by the single most frequent value, ignoring the distribution
of all other values. This is why entropy ℓ-diversity consistently reports a higher (more
optimistic) ℓ.

---

## 3. Test Case Rationale

### `test_is_k_anonymous`

| Case | Justification |
|---|---|
| k=1, all unique values | k=1 is trivially always satisfied — baseline correctness check |
| k=2, all unique values | Edge case: every group has size 1, must return False |
| k=2, one group of size 1 | Mixed groups: a single undersized group must trigger failure |
| k=2, two balanced groups of size 2 | Minimal passing case for k=2 |
| k=1 and k=2 on adult_small | Integration test against real data; confirms k=2 fails |

These cases together cover: trivially true, trivially false, mixed (boundary), and real data.

### `test_generalize_categorical`

| Case | Justification |
|---|---|
| Return type is DataFrame | Smoke test — function must return usable data |
| Education values ⊆ {`< HS`, `>= HS`} | Verifies mapping is exhaustive and correct |
| Marital Status values ⊆ {`Married`, `Not Married`} | Same for the second column |
| is_k_anonymous(2, ...) is True | End-to-end: the stated goal of the function |
| len(df) ≤ len(adult_small) | Suppression may only remove rows, never add them |

### `test_generalize_numeric`

| Case | Justification |
|---|---|
| n=0 | Identity: no generalization should leave the value unchanged |
| n=2, n=4 | Required by the problem statement (exact assert values given) |
| n=5 (all digits) | Boundary: full suppression to 0 |
| Already-zero suffix | Idempotency: zeroing already-zero digits must not change value |
| Different base value (99999) | Confirms formula is not hard-coded for 47401 |

### `test_suppress_count`

| Case | Justification |
|---|---|
| One undersized group of size 1 | Only that one row should be counted |
| k=1 | No suppression needed — zero is the floor |
| k larger than all groups | All rows must be suppressed |

### `test_is_l_diverse`

| Case | Justification |
|---|---|
| Two balanced groups, l=2 (True) | Minimal passing configuration for both variants |
| l=3 (False) | The smaller group (size 2) cannot satisfy l=3 — failure boundary |
| Entropy variant, l=2 (True) / l=3 (False) | Verifies both code paths in the same data |

Using the same DataFrame for both variants ensures the difference in behavior is due to the
algorithm, not the data.

### `test_max_l`

| Case | Justification |
|---|---|
| Mixed groups → max_l=1 | Bottleneck group (2/3 frequency) must cap the result at 1 |
| Perfectly uniform group of 4 → max_l=4 | Upper-bound check: best-case diversity |

Both probabilistic and entropy variants are tested on both DataFrames to confirm that the
correct formula is used for each variant.
