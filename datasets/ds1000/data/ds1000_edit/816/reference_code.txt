df = df[(np.abs(stats.zscore(df.select_dtypes(exclude='object'))) < 3).all(axis=1)]

