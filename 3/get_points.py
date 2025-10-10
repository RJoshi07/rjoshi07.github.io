import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('rug.png')  


plt.imshow(img)
plt.title("Click on points (close window when done)")

points = plt.ginput(n=10, timeout=0)  
plt.close()

import numpy as np
points = np.array(points)

print("Selected points:")
print(points)