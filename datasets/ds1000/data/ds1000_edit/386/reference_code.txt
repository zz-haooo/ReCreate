result = np.lib.stride_tricks.sliding_window_view(a, window_shape=(2,2)).reshape(-1, 2, 2)
