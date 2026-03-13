import pandas as pd
import numpy as np

def process_genomic_file(file):

    df = pd.read_csv(file)

    # Remove non-numeric columns if any
    df = df.select_dtypes(include=[np.number])

    # Take only first 100 genes (same size as model expects)
    df = df.iloc[:, :100]

    # Convert to numpy
    data = df.values

    return data