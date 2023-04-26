import matplotlib.pyplot as plt

# Data
x = ['a', 'b', 'c', 'd']
y1 = [0.4, 0.4, 0.4, 0.4]
y2 = [0.660,0.328,0.418,0.741]

# Plotting
plt.plot(keyword, y1, label='y1')
plt.plot(keyword, y2, label='y2')

# Labels and Title
plt.xlabel('x_label')
plt.ylabel('y_label')
plt.title('Comparison of approaches')

# Legend
plt.legend()

# Show the graph
plt.show()
