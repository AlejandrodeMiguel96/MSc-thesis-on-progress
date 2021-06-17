import numpy as np
from feature import Feature
n_f = 6 + 8  # number of features to observe (6 faces + 8 corners)
corn_norm = 1 / np.sqrt(3)
face1 = Feature(np.array([1, 0, 0]))
face2 = Feature(np.array([-1, 0, 0]))
face3 = Feature(np.array([0, 1, 0]))
face4 = Feature(np.array([0, -1, 0]))
face5 = Feature(np.array([0, 0, 1]))
face6 = Feature(np.array([0, 0, -1]))
corn1 = Feature(np.array([corn_norm, corn_norm, -corn_norm]))
corn2 = Feature(np.array([-corn_norm, corn_norm, -corn_norm]))
corn3 = Feature(np.array([-corn_norm, corn_norm, corn_norm]))
corn4 = Feature(np.array([corn_norm, corn_norm, corn_norm]))
corn5 = Feature(np.array([corn_norm, -corn_norm, -corn_norm]))
corn6 = Feature(np.array([-corn_norm, -corn_norm, -corn_norm]))
corn7 = Feature(np.array([-corn_norm, -corn_norm, corn_norm]))
corn8 = Feature(np.array([corn_norm, -corn_norm, corn_norm]))
features = (face1, face2, face3, face4, face5, face6, corn1, corn2, corn3, corn4, corn5, corn6, corn7, corn8)  # all features
# features = (face1, face2, face3, face4, face5, face6)  # faces only
# features = (face1, face2, face5, face6)  # faces only without y-faces
# features = (corn1, corn2, corn3, corn4, corn5, corn6, corn7, corn8)  # corners only
# features = (face1, face2, face5, face6, corn1, corn2, corn3, corn4, corn5, corn6, corn7, corn8)  # not y-faces
