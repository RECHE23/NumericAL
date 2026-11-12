"""The constants of include/numerical/elementary.hpp's log1p, from exact decimal arithmetic, printed as the header holds
them: ln 2 to 42 significant bits toward zero and the double nearest what that leaves out, and for k = 90…180 the
double nearest 128/k, then −ln of that double to the nearest multiple of 2^−42 (so that its sum with an exponent's
multiple of the first is exact) and the double nearest what that leaves out."""
from decimal import ROUND_DOWN, Decimal, getcontext

getcontext().prec = 80


def pair(value):
    """`value` to the nearest multiple of 2^−42 and the double nearest what that leaves out."""
    head = float((value * 2**42).quantize(Decimal(1)) / 2**42)
    return head, float(value - Decimal(head))


def main():
    ln2 = Decimal(2).ln()
    # ln 2 in [1/2, 1): 42 significant bits are the multiples of 2^−42
    high = float((ln2 * 2**42).quantize(Decimal(1), rounding=ROUND_DOWN) / 2**42)
    print("log_ln2_high", high.hex())
    print("log_ln2_low ", float(ln2 - Decimal(high)).hex())
    for k in range(90, 181):
        inverse = 128 / k  # Python's division of two ints is the double nearest the quotient
        head, tail = pair(-Decimal(inverse).ln())
        print(f"{inverse.hex()}, {head.hex()}, {tail.hex()},")


if __name__ == "__main__":
    main()
