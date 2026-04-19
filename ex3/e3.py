# Imports and definitions
import numpy as np
from collections import defaultdict
import galois
import pytest

field = galois.GF(2 ** 13 - 1)


def shamir_share(x, t, n):
    """Generate n shares of secret x using a (t,n) Shamir secret sharing scheme."""
    # Random polynomial of degree t-1: f(0) = x, coefficients in field
    coeffs = [field(int(x))] + [field(int(c)) for c in np.random.randint(0, field.order, t - 1)]
    # Evaluate polynomial at points 1..n
    shares = []
    for i in range(1, n + 1):
        xi = field(i)
        val = field(0)
        for j, c in enumerate(coeffs):
            val = val + c * xi ** j
        shares.append((field(i), val))
    return shares


def add_shares(shares1, shares2):
    """Add two sets of secret-shared values element-wise."""
    return [(x1, v1 + v2) for (x1, v1), (_, v2) in zip(shares1, shares2)]


def add_const(shares, k):
    """Add public constant k to a secret-shared value."""
    return [(x, v + field(int(k))) for x, v in shares]


def mult_const(shares, k):
    """Multiply a secret-shared value by public constant k."""
    return [(x, v * field(int(k))) for x, v in shares]


def reconstruct(shares):
    """Reconstruct the secret from at least t shares using Lagrange interpolation."""
    xs = field([int(x) for x, _ in shares])
    ys = field([int(y) for _, y in shares])
    poly = galois.lagrange_poly(xs, ys)
    return poly(field(0))


class Party:
    """A participant in a multiparty computation protocol."""
    def __init__(self):
        self.inbox = defaultdict(dict)  # inbox[round][sender_id] = msg
        self.view = {}

    def send(self, other, round, msg):
        other.inbox[round][id(self)] = msg

    def get_view(self):
        return self.view


class BGW(Party):
    def round1(self, parties, a_shr, b_shr, t):
        self.input = (a_shr, b_shr)
        self.parties = parties
        n = len(parties)
        assert t <= n / 2

        # Locally compute product of own shares (point on degree-2(t-1) polynomial h=f*g)
        _, ai = a_shr
        _, bi = b_shr
        hi = ai * bi

        # Re-share hi with a fresh degree-(t-1) polynomial so degree can be reduced
        hi_shares = shamir_share(int(hi), t, n)

        # Send the j-th share of hi to party j
        for party, (_, vj) in zip(parties, hi_shares):
            self.send(party, 1, vj)

    def round2(self):
        n = len(self.parties)

        # Lagrange coefficients λᵢ for evaluating at x=0 using nodes x=1..n
        # λᵢ = Πⱼ≠ᵢ (0 - j) / (i - j)
        xs = field([i for i in range(1, n + 1)])
        lambdas = []
        for i in range(n):
            num = field(1)
            den = field(1)
            for j in range(n):
                if j != i:
                    num = num * (field(0) - xs[j])
                    den = den * (xs[i] - xs[j])
            lambdas.append(num / den)

        # Collect one message per party (the share of hᵢ destined for this party)
        received = [self.inbox[1][id(party)] for party in self.parties]

        # Output share = Σᵢ λᵢ · received[i]  (linear combination reduces degree back to t-1)
        out = sum((lam * v for lam, v in zip(lambdas, received)), field(0))

        idx = self.parties.index(self)
        output_share = (field(idx + 1), out)
        self.view = {'input': self.input, 'received': received, 'output_share': output_share}
        return output_share


def run_bgw(t, n, a, b):
    """Execute the BGW protocol and return n output shares of a*b."""
    parties = [BGW() for _ in range(n)]

    a_shares = shamir_share(a, t, n)
    b_shares = shamir_share(b, t, n)

    for party, a_shr, b_shr in zip(parties, a_shares, b_shares):
        party.round1(parties, a_shr, b_shr, t)

    return [party.round2() for party in parties]


if __name__ == "__main__":
    # Task 1: Shamir secret sharing demo
    secret, t, n = 42, 3, 5
    shares = shamir_share(secret, t, n)
    print(f"Secret: {secret}")
    print(f"Reconstructed from {t} shares: {reconstruct(shares[:t])}")
    print(f"add_const(+10): {reconstruct(add_const(shares, 10)[:t])}")
    print(f"mult_const(*3):  {reconstruct(mult_const(shares, 3)[:t])}")

    # Task 2: BGW multiplication demo (requires t <= n/2, so t=2, n=5)
    t2, n2, a, b = 2, 5, 6, 7
    prod_shares = run_bgw(t2, n2, a, b)
    result = reconstruct(prod_shares[:t2])
    print(f"\nBGW: {a} * {b} = {a * b}, reconstructed: {result}")

