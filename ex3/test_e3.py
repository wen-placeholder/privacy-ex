# Imports and definitions
import numpy as np
from collections import defaultdict
import galois
import pytest

from e3 import shamir_share, reconstruct, add_shares, add_const, mult_const, run_bgw, field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recon(shares, t):
    """Reconstruct from the first t shares."""
    return int(reconstruct(shares[:t]))


# ---------------------------------------------------------------------------
# shamir_share / reconstruct
# ---------------------------------------------------------------------------

class TestShamirShare:
    def test_returns_n_shares(self):
        shares = shamir_share(10, 2, 5)
        assert len(shares) == 5

    def test_share_x_values_are_1_to_n(self):
        n = 6
        shares = shamir_share(7, 3, n)
        xs = [int(x) for x, _ in shares]
        assert xs == list(range(1, n + 1))

    def test_reconstruct_exact_t_shares(self):
        # Exactly t shares must be sufficient
        secret, t, n = 99, 3, 5
        shares = shamir_share(secret, t, n)
        assert _recon(shares, t) == secret

    def test_reconstruct_more_than_t_shares(self):
        # More than t shares must also work
        secret, t, n = 55, 2, 7
        shares = shamir_share(secret, t, n)
        assert _recon(shares, n) == secret

    def test_reconstruct_all_shares(self):
        secret, t, n = 1234, 4, 7
        shares = shamir_share(secret, t, n)
        assert _recon(shares, n) == secret

    def test_secret_zero(self):
        shares = shamir_share(0, 2, 4)
        assert _recon(shares, 2) == 0

    def test_secret_one(self):
        shares = shamir_share(1, 2, 4)
        assert _recon(shares, 2) == 1

    def test_threshold_one(self):
        # t=1 means any single share reveals the secret (constant polynomial)
        secret = 77
        shares = shamir_share(secret, 1, 5)
        assert _recon(shares, 1) == secret

    def test_randomness_produces_different_shares(self):
        # Two sharings of the same secret should (almost surely) differ
        s1 = shamir_share(42, 3, 5)
        s2 = shamir_share(42, 3, 5)
        vals1 = [int(v) for _, v in s1]
        vals2 = [int(v) for _, v in s2]
        assert vals1 != vals2

    def test_fewer_than_t_shares_cannot_reconstruct(self):
        # With t-1 shares the reconstruction should give a wrong result
        # (with overwhelming probability due to randomness of polynomial)
        secret, t, n = 42, 3, 5
        wrong_count = 0
        for _ in range(10):
            shares = shamir_share(secret, t, n)
            guessed = int(reconstruct(shares[:t - 1]))
            if guessed != secret:
                wrong_count += 1
        # At least some trials should fail (with high probability all 10 will)
        assert wrong_count > 0


# ---------------------------------------------------------------------------
# add_const
# ---------------------------------------------------------------------------

class TestAddConst:
    def test_add_positive_constant(self):
        secret, k, t, n = 20, 15, 2, 5
        shares = shamir_share(secret, t, n)
        result = _recon(add_const(shares, k), t)
        assert result == secret + k

    def test_add_zero(self):
        secret, t, n = 30, 2, 4
        shares = shamir_share(secret, t, n)
        assert _recon(add_const(shares, 0), t) == secret

    def test_add_const_returns_new_list(self):
        secret, t, n = 10, 2, 4
        shares = shamir_share(secret, t, n)
        result = add_const(shares, 5)
        assert result is not shares

    def test_add_large_constant(self):
        secret, k, t, n = 100, 3000, 3, 5
        shares = shamir_share(secret, t, n)
        expected = (secret + k) % field.order
        assert _recon(add_const(shares, k), t) == expected


# ---------------------------------------------------------------------------
# mult_const
# ---------------------------------------------------------------------------

class TestMultConst:
    def test_multiply_by_scalar(self):
        secret, k, t, n = 7, 6, 2, 5
        shares = shamir_share(secret, t, n)
        assert _recon(mult_const(shares, k), t) == secret * k

    def test_multiply_by_one(self):
        secret, t, n = 50, 2, 4
        shares = shamir_share(secret, t, n)
        assert _recon(mult_const(shares, 1), t) == secret

    def test_multiply_by_zero(self):
        secret, t, n = 99, 2, 4
        shares = shamir_share(secret, t, n)
        assert _recon(mult_const(shares, 0), t) == 0

    def test_multiply_does_not_modify_original(self):
        secret, t, n = 10, 2, 4
        shares = shamir_share(secret, t, n)
        original_vals = [int(v) for _, v in shares]
        mult_const(shares, 3)
        assert [int(v) for _, v in shares] == original_vals


# ---------------------------------------------------------------------------
# add_shares
# ---------------------------------------------------------------------------

class TestAddShares:
    def test_add_two_secrets(self):
        t, n = 2, 5
        s1 = shamir_share(10, t, n)
        s2 = shamir_share(20, t, n)
        assert _recon(add_shares(s1, s2), t) == 30

    def test_add_zero_secret(self):
        t, n = 2, 4
        s1 = shamir_share(17, t, n)
        s2 = shamir_share(0, t, n)
        assert _recon(add_shares(s1, s2), t) == 17

    def test_add_shares_is_commutative(self):
        t, n = 2, 5
        s1 = shamir_share(11, t, n)
        s2 = shamir_share(22, t, n)
        r1 = _recon(add_shares(s1, s2), t)
        r2 = _recon(add_shares(s2, s1), t)
        assert r1 == r2

    def test_x_coordinates_preserved(self):
        t, n = 2, 4
        s1 = shamir_share(5, t, n)
        s2 = shamir_share(3, t, n)
        result = add_shares(s1, s2)
        xs = [int(x) for x, _ in result]
        assert xs == list(range(1, n + 1))


# ---------------------------------------------------------------------------
# BGW / run_bgw
# ---------------------------------------------------------------------------

class TestBGW:
    def test_basic_multiplication(self):
        shares = run_bgw(2, 5, 6, 7)
        assert _recon(shares, 2) == 42

    def test_multiply_by_zero(self):
        shares = run_bgw(2, 5, 0, 99)
        assert _recon(shares, 2) == 0

    def test_multiply_by_one(self):
        shares = run_bgw(2, 5, 1, 55)
        assert _recon(shares, 2) == 55

    def test_multiply_two_ones(self):
        shares = run_bgw(2, 5, 1, 1)
        assert _recon(shares, 2) == 1

    def test_returns_n_shares(self):
        n = 5
        shares = run_bgw(2, n, 3, 4)
        assert len(shares) == n

    def test_output_share_x_values(self):
        n = 5
        shares = run_bgw(2, n, 3, 4)
        xs = [int(x) for x, _ in shares]
        assert xs == list(range(1, n + 1))

    def test_larger_t_and_n(self):
        # t=3, n=7: t <= n/2 = 3.5 ✓
        shares = run_bgw(3, 7, 8, 9)
        assert _recon(shares, 3) == 72

    def test_reconstruct_from_all_shares(self):
        t, n = 2, 5
        shares = run_bgw(t, n, 5, 6)
        assert _recon(shares, n) == 30

    def test_party_view_populated(self):
        from e3 import BGW
        parties = [BGW() for _ in range(5)]
        a_shares = shamir_share(3, 2, 5)
        b_shares = shamir_share(4, 2, 5)
        for p, a_s, b_s in zip(parties, a_shares, b_shares):
            p.round1(parties, a_s, b_s, 2)
        for p in parties:
            p.round2()
            view = p.get_view()
            assert 'input' in view
            assert 'output_share' in view
            assert 'received' in view

    def test_t_exceeds_half_n_raises(self):
        from e3 import BGW
        parties = [BGW() for _ in range(4)]
        a_shares = shamir_share(2, 3, 4)
        b_shares = shamir_share(3, 3, 4)
        with pytest.raises(AssertionError):
            parties[0].round1(parties, a_shares[0], b_shares[0], 3)
