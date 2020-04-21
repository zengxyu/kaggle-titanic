import tkinter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('TkAgg')
x = np.random.normal(size=100)
sns.distplot(x)
plt.show()
