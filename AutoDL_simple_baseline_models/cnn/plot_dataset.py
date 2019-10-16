import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.io import wavfile
from librosa.feature import mfcc
from matplotlib import cm
from sklearn.preprocessing import normalize

_, a = wavfile.read("a.wav")

mean = np.mean(a)
std = np.std(a)
a = (a - mean) / std


mfcc_feature = mfcc(a.astype("float"), n_mfcc=13)
mfcc_mean = np.mean(mfcc_feature, axis=1)
mfcc_std = np.std(mfcc_feature, axis=1)

mfcc_feature = normalize(mfcc_feature, norm="l2")
print(mfcc_feature)
#mfcc_feature = (mfcc_feature - mfcc_mean.expand_dims(axis=1).repeat(349, axis=1)) / mfcc_std.expand_dims(axis=1).repeat(349, axis=1)

plt.plot(mfcc_feature)
plt.show()

#mfcc_feature = np.swapaxes(mfcc_feature, 0, 1)
fig, ax = plt.subplots()
cax = ax.imshow(mfcc_feature, interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect='auto')
ax.set_title('MFCC')
ax.set_xlabel("Features")
ax.set_ylabel("Time")
# Showing mfcc_data
plt.show()
# Showing mfcc_feat


