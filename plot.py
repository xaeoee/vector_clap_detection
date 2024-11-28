import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Updated landmark positions
wrist = np.array([0.5756480693817139, 0.8119476437568665, 0.029400156810879707])
one = np.array([0.5115987062454224, 0.7623839974403381, 0.029400156810879707])
st = np.array([0.6425312161445618, 0.5234341025352478, 0.029400156810879707])

# Calculate vectors
vector_wrist_to_one = one - wrist
vector_wrist_to_st = st - wrist

# Calculate cross product
cross_product = np.cross(vector_wrist_to_one, vector_wrist_to_st)

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot points
ax.scatter(*wrist, color='red', label='Wrist', s=50)
ax.scatter(*one, color='blue', label='One', s=50)
ax.scatter(*st, color='green', label='ST', s=50)

# Plot vectors
ax.quiver(*wrist, *vector_wrist_to_one, color='blue', label='Wrist to One')
ax.quiver(*wrist, *vector_wrist_to_st, color='green', label='Wrist to ST')

# Plot cross product vector
ax.quiver(*wrist, *cross_product, color='purple', label='Cross Product (Wrist to One x Wrist to ST)')

# Formatting the plot
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.legend()
ax.set_title("Vector Visualization and Cross Product (Updated Landmarks)")

plt.show()

# Print cross product vector
print("Cross Product Vector:", cross_product)
