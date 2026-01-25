    gcd = np.gcd(numerator, denominator)
    result = (numerator//gcd, denominator//gcd)

    return result
