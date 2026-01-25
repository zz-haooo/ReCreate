result = np.correlate(a, np.hstack((b[1:], b)), mode='valid')
