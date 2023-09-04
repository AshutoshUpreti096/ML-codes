import numpy as np

speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]

speed.sort()

x = np.median(speed)

print("Median :->", x)