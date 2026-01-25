def solution(xs, n):
    e = np.empty_like(xs)
    if n >= 0:
        e[:,:n] = np.nan
        e[:,n:] = xs[:,:-n]
    else:
        e[:,n:] = np.nan
        e[:,:n] = xs[:,-n:]
    return e
result = solution(a, shift)
