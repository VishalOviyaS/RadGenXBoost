import numpy as np
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis

def extract_radiomics_features(img):

    img = img.astype(np.uint8)

    # -------- First order features --------
    mean = np.mean(img)
    std = np.std(img)
    skewness = skew(img.flatten())
    kurt = kurtosis(img.flatten())

    # -------- GLCM features --------
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0,0]
    energy = graycoprops(glcm, 'energy')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    correlation = graycoprops(glcm, 'correlation')[0,0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0,0]

    features = [
        mean,
        std,
        skewness,
        kurt,
        contrast,
        energy,
        homogeneity,
        correlation,
        dissimilarity,
        np.var(img)   # 10th feature
    ]

    return features