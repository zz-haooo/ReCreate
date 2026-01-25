def g(df):
    return pd.melt(df)

result = g(df.copy())
