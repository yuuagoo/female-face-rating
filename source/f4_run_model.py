try:
    from sklearn.externals import joblib
except ImportError:
    import joblib
import argparse
import numpy as np
from sklearn import decomposition
from f1_facial_landmarks import find_facial_landmark
from f2_create_feature import generateAllFeatures

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument('-featuredim', type=int, default=20)
ap.add_argument('-inputfeatures', type=str, default='../data/features_ALL.txt')
args = vars(ap.parse_args())

lm_file = find_facial_landmark(args["image"])
landmarks = np.loadtxt(lm_file, delimiter=',', usecols=range(136))
my_features = generateAllFeatures(landmarks)
#use your own path
clf = joblib.load('../model/face_model.pkl')
features_train = np.loadtxt(args['inputfeatures'], delimiter=',')
pca = decomposition.PCA(n_components=args['featuredim'])
pca.fit(features_train)
my_features = pca.transform(my_features)
predictions = clf.predict(my_features)
print(predictions)
