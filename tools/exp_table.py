"""The constants of include/numerical/elementary.hpp's exp, from exact decimal arithmetic, printed as the header holds
them: 128/ln 2; ln 2/128 as a head of 32 significant bits (truncated, so that its product by a reduction's integer
leaves an exact difference) and the double nearest the rest; ln 2^-1022, rounded toward zero, below which a result is
subnormal; and 2^(j/128) for j = 0…127, each the double nearest it and the double nearest what that one leaves out."""
import math
from decimal import Decimal, getcontext

getcontext().prec = 80


def truncated(value, bits):
    """`value` (positive) toward zero to `bits` significant bits."""
    power = 0
    while value >= 2 ** power:
        power += 1
    while value < 2 ** (power - 1):
        power -= 1
    return int(value * Decimal(2) ** (bits - power)) * 2.0 ** (power - bits)


def main():
    ln2 = Decimal(2).ln()
    high = truncated(ln2 / 128, 32)
    bound_exact = (Decimal(2) ** -1022).ln()
    bound = float(bound_exact)
    if Decimal(bound) < bound_exact:
        bound = math.nextafter(bound, 0.0)
    print("exp_inv_ln2_128 ", float(128 / ln2).hex())
    print("exp_ln2_128_high", high.hex())
    print("exp_ln2_128_low ", float(ln2 / 128 - Decimal(high)).hex())
    print("exp_normal_bound", bound.hex())
    for j in range(128):
        exact = Decimal(2) ** (Decimal(j) / 128)
        head = float(exact)
        print(f"{head.hex()}, {float(exact - Decimal(head)).hex()},")


if __name__ == "__main__":
    main()
