kdtree = scipy.spatial.cKDTree(points)
_, result = kdtree.query(extraPoints)

