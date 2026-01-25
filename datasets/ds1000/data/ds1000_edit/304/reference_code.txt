col = ( A.shape[0] // ncol) * ncol
B = A[len(A)-col:][::-1]
B = np.reshape(B, (-1, ncol))
