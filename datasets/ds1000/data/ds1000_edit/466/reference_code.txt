df = pd.DataFrame({'lat': lat.ravel(), 'lon': lon.ravel(), 'val': val.ravel()})
df['maximum'] = df.max(axis=1)
