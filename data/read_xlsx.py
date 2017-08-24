import pandas as pd
import numpy as np

df = pd.read_excel('/home/joaomonteirof/Desktop/nmkt/VF/CorrelationsCorrected_overlaps.xlsx', 'Innovation_Data')

print(vars(df))

a=df['AFD']

print(type(a))

b = np.asarray(a)

print(type(b))

print(b.shape)

print(b[0])
