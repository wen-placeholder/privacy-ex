# Load the data and libraries
import pandas as pd
import numpy as np
import scipy.stats
import pytest

# Load dataset
adult = pd.read_csv("adult_with_pii (1).csv")

# Question 1: Create adult_small (first 100 rows, Education, Marital Status, Target)
adult_small = adult[["Education", "Marital Status", "Target"]].head(100).copy()


def is_k_anonymous(k, qis, df):
    """Returns true if df satisfies k-Anonymity for the quasi-identifiers
    qis. Returns false otherwise."""
    group_sizes = df.groupby(qis).size()
    return bool((group_sizes >= k).all())


def generalize_categorical():
    """Generalize adult_small to achieve k=2 anonymity.
    - Education: below HS-grad -> '< HS', otherwise '>= HS'
    - Marital Status: -> 'Married' or 'Not Married'
    - Suppression: delete rows whose equivalence class still has size < 2
    Returns the generalized and suppressed dataframe."""
    df = adult_small.copy()

    # Education levels below HS graduation
    below_hs = {"Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th"}
    df["Education"] = df["Education"].apply(lambda x: "< HS" if x in below_hs else ">= HS")

    # Marital status generalization
    married_values = {"Married-civ-spouse", "Married-AF-spouse", "Married-spouse-absent"}
    df["Marital Status"] = df["Marital Status"].apply(
        lambda x: "Married" if x in married_values else "Not Married"
    )

    # Suppression: remove equivalence classes with size < 2
    qis = ["Education", "Marital Status"]
    group_sizes = df.groupby(qis).transform("count")["Target"]
    df = df[group_sizes >= 2].reset_index(drop=True)

    return df


def generalize_numeric(zip, n):
    """Generalize a zip code by zeroing the last n digits.
    generalize_numeric(47401, 0) == 47401
    generalize_numeric(47401, 2) == 47400
    generalize_numeric(47401, 4) == 40000
    """
    factor = 10 ** n
    return (zip // factor) * factor


def suppress_count(k, qis, df):
    """Returns the number of rows that need to be suppressed to achieve k-anonymity."""
    group_sizes = df.groupby(qis).transform("count").iloc[:, 0]
    return int((group_sizes < k).sum())


def is_l_diverse(l, qis, sens_col, df, type='probabilistic'):
    """Returns True if df satisfies l-diversity for the given quasi-identifiers
    and sensitive column. Supports 'probabilistic' and 'entropy' variants."""
    for _, group in df.groupby(qis):
        vals = group[sens_col]
        n = len(vals)
        counts = vals.value_counts()
        if type == 'probabilistic':
            # Most frequent sensitive value must have frequency <= 1/l
            if counts.iloc[0] / n > 1 / l:
                return False
        else:  # entropy
            # Entropy of sensitive values must be >= log(l)
            probs = counts / n
            entropy = -np.sum(probs * np.log(probs))
            if entropy < np.log(l):
                return False
    return True


def max_l(qis, sens_col, df, type='probabilistic'):
    """Returns the largest l for which df satisfies l-diversity."""
    min_l = float('inf')
    for _, group in df.groupby(qis):
        vals = group[sens_col]
        n = len(vals)
        counts = vals.value_counts()
        if type == 'probabilistic':
            max_freq = counts.iloc[0] / n
            group_max = int(1 / max_freq)
        else:  # entropy
            probs = counts / n
            entropy = -np.sum(probs * np.log(probs))
            group_max = int(np.exp(entropy))
        min_l = min(min_l, group_max)
    return min_l if min_l != float('inf') else 0


# ── Tests ────────────────────────────────────────────────────────────────────

def test_is_k_anonymous():
    # k=1 always true; all-unique fails k=2; mixed group fails k=2; adult_small only satisfies k=1
    assert is_k_anonymous(1, ["A"], pd.DataFrame({"A": ["x", "y", "z"]})) is True
    assert is_k_anonymous(2, ["A"], pd.DataFrame({"A": ["a", "b", "c"]})) is False
    assert is_k_anonymous(2, ["A"], pd.DataFrame({"A": ["x", "x", "y"]})) is False
    assert is_k_anonymous(2, ["A"], pd.DataFrame({"A": ["x", "x", "y", "y"]})) is True
    assert is_k_anonymous(1, ["Education", "Marital Status"], adult_small) is True
    assert is_k_anonymous(2, ["Education", "Marital Status"], adult_small) is False


def test_generalize_categorical():
    df = generalize_categorical()
    assert isinstance(df, pd.DataFrame)
    assert set(df["Education"].unique()).issubset({"< HS", ">= HS"})
    assert set(df["Marital Status"].unique()).issubset({"Married", "Not Married"})
    assert is_k_anonymous(2, ["Education", "Marital Status"], df) is True
    assert len(df) <= len(adult_small)


def test_generalize_numeric():
    assert generalize_numeric(47401, 0) == 47401
    assert generalize_numeric(47401, 2) == 47400
    assert generalize_numeric(47401, 4) == 40000
    assert generalize_numeric(47401, 5) == 0
    assert generalize_numeric(99999, 3) == 99000


def test_suppress_count():
    df = pd.DataFrame({"A": ["x", "x", "y"], "S": [1, 2, 3]})
    assert suppress_count(2, ["A"], df) == 1   # group 'y' has 1 row
    assert suppress_count(1, ["A"], df) == 0   # all groups satisfy k=1
    assert suppress_count(3, ["A"], df) == 3   # all rows need suppression


def test_is_l_diverse():
    # Group a: [s1, s2]      -> max_freq=0.5,   entropy=log(2)
    # Group b: [s1, s2, s3]  -> max_freq=0.333, entropy=log(3)
    df = pd.DataFrame({
        "Q": ["a", "a", "b", "b", "b"],
        "S": ["s1", "s2", "s1", "s2", "s3"],
    })
    assert is_l_diverse(2, ["Q"], "S", df, type='probabilistic') is True   # both groups ok
    assert is_l_diverse(3, ["Q"], "S", df, type='probabilistic') is False  # group a: 0.5 > 1/3
    assert is_l_diverse(2, ["Q"], "S", df, type='entropy') is True
    assert is_l_diverse(3, ["Q"], "S", df, type='entropy') is False        # group a: entropy=log(2)<log(3)


def test_max_l():
    # Group a: [s1, s2] -> prob max_l=2, entropy max_l=2
    # Group b: [s1, s1, s2] -> prob max_l=1, entropy max_l=1 (exp(entropy)<2)
    df = pd.DataFrame({
        "Q": ["a", "a", "b", "b", "b"],
        "S": ["s1", "s2", "s1", "s1", "s2"],
    })
    assert max_l(["Q"], "S", df, type='probabilistic') == 1
    assert max_l(["Q"], "S", df, type='entropy') == 1
    # Perfectly diverse group
    df2 = pd.DataFrame({"Q": ["a"] * 4, "S": ["s1", "s2", "s3", "s4"]})
    assert max_l(["Q"], "S", df2, type='probabilistic') == 4
    assert max_l(["Q"], "S", df2, type='entropy') == 4


if __name__ == "__main__":
    # ── Question 1 ──────────────────────────────────────────────────────────
    print("=== adult_small (first 5 rows) ===")
    print(adult_small.head())
    print()

    print("=== k-anonymity check for k in [1,10] ===")
    qis = ["Education", "Marital Status"]
    for k in range(1, 11):
        result = is_k_anonymous(k, qis, adult_small)
        print(f"  k={k}: {result}")
    print()

    # ── Question 2 ──────────────────────────────────────────────────────────
    df_gen = generalize_categorical()
    print("=== Generalized adult_small (first 5 rows) ===")
    print(df_gen.head())
    print(f"Rows after suppression: {len(df_gen)} (original: {len(adult_small)})")
    print()
    print("=== Equivalence class sizes after generalization ===")
    print(df_gen.groupby(["Education", "Marital Status", "Target"]).size().to_string())
    print()

    # ── Question 3 ──────────────────────────────────────────────────────────
    print("=== generalize_numeric examples ===")
    for zip_val, n in [(47401, 0), (47401, 2), (47401, 4)]:
        print(f"  generalize_numeric({zip_val}, {n}) = {generalize_numeric(zip_val, n)}")
    print()

    # ── Question 4 ──────────────────────────────────────────────────────────
    print("=== Q4: k-anonymity on full adult dataset (QIs: Zip, Sex, Age) ===")
    qis4 = ["Zip", "Sex", "Age"]
    for zip_digits in [0, 2, 3]:
        for age_digits in [0, 1]:
            df4 = adult.copy()
            df4["Zip"] = df4["Zip"].apply(lambda z: generalize_numeric(z, zip_digits))
            df4["Age"] = df4["Age"].apply(lambda a: generalize_numeric(a, age_digits))
            sc3 = suppress_count(3, qis4, df4)
            sc7 = suppress_count(7, qis4, df4)
            print(f"  Zip n={zip_digits}, Age n={age_digits} -> suppress for k=3: {sc3}, k=7: {sc7}")
    print()

    # ── Question 5 & 6 ──────────────────────────────────────────────────────
    print("=== Q6: l-diversity of adult dataset ===")
    print("  QIs: Education, Marital Status, Sex | Sensitive: DOB")
    qis6 = ["Education", "Marital Status", "Sex"]
    sens6 = "DOB"
    for l in [2, 3]:
        prob = is_l_diverse(l, qis6, sens6, adult, type='probabilistic')
        entr = is_l_diverse(l, qis6, sens6, adult, type='entropy')
        print(f"  l={l}: probabilistic={prob}, entropy={entr}")

    # Show which groups prevent 2-diversity (probabilistic)
    print("\n  Groups preventing probabilistic 2-diversity (max_freq > 0.5):")
    for name, group in adult.groupby(qis6):
        vals = group[sens6]
        n = len(vals)
        max_freq = vals.value_counts().iloc[0] / n
        if max_freq > 0.5:
            print(f"    {name}: size={n}, max_freq={max_freq:.3f}")
    print()

    # ── Question 7 ──────────────────────────────────────────────────────────
    print("=== Q7: max_l on full adult dataset generalized with Q2 rules ===")
    below_hs = {"Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th"}
    married_values = {"Married-civ-spouse", "Married-AF-spouse", "Married-spouse-absent"}
    df7 = adult.copy()
    df7["Education"] = df7["Education"].apply(lambda x: "< HS" if x in below_hs else ">= HS")
    df7["Marital Status"] = df7["Marital Status"].apply(
        lambda x: "Married" if x in married_values else "Not Married"
    )
    qis7 = ["Education", "Marital Status", "Sex"]
    for variant in ['probabilistic', 'entropy']:
        ml = max_l(qis7, sens6, df7, type=variant)
        print(f"  max_l ({variant}): {ml}")
