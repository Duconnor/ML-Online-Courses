import autodiff as ad
import numpy as np
import matplotlib.pyplot as plt

# Prepare some sample data
mean_one = [0, 0]
scale_one = [1.5, 1]
mean_two = [5, 5]
scale_two = [1, 1.5]
num_data_per_class = 50
data_size = (num_data_per_class, 2)

data_one = np.random.normal(mean_one, scale_one, data_size)
data_two = np.random.normal(mean_two, scale_two, data_size)


# Construct the computation graph
''' 
TODO: What operations we need to add
1. __neg__ in Node -> Finished
2. __sub__ and __rsub__ in Node -> Finished
3. __truediv__ and __rtruediv__ in Node -> Finished
4. ExpOp -> Finished
5. LogOp -> Finished
6. ReduceSumOp -> Finished
'''

weight = ad.Variable(name='weight')
bias = ad.Variable(name='bias')
X = ad.Variable(name='X')
Y = ad.Variable(name='Y')

logits = ad.matmul_op(X, weight) + bias
output = 1 / (1 + ad.exp_op(-logits))
loss = -(ad.reduce_sum_op(Y * ad.log_op(output) + (1 - Y) * ad.log_op(1 - output)) / 100)

# Construct the gradient graph
grad_weight, grad_bias = ad.gradients(loss, [weight, bias])

# Execute the graph
executor = ad.Executor([output, loss, grad_weight, grad_bias])
X_val = np.concatenate((data_one, data_two), axis=0) # Shape 100 x 2
Y_val = np.concatenate((np.repeat(np.array([0]), num_data_per_class), np.repeat(np.array([1]), num_data_per_class))).reshape(-1, 1) # Shape 100 x 1
assert X_val.shape == (num_data_per_class * 2, 2)
assert Y_val.shape == (num_data_per_class * 2, 1)

# Initialize all parameters
weight_val = np.random.normal(size=(2, 1))
bias_val = np.random.normal(size=(1,))
# Hyperparameter settings
max_iter = 50
lr = 0.0001
# Launch the training loop
for iter_idx in range(max_iter):
    # Compute the loss
    output_val, loss_val, grad_weight_val, grad_bias_val = executor.run(feed_dict={weight: weight_val, bias: bias_val, X: X_val, Y: Y_val})

    # Update using gradient descent
    weight_val = weight_val - lr * grad_weight_val
    bias_val = bias_val - lr * np.sum(grad_bias_val)

    # Print the loss out to monitor the training process
    print('Iter %d, Loss %f' % (iter_idx + 1, loss_val))

    # Calculate the accuracy
    pred = np.zeros_like(output_val)
    pred[output_val >= 0.5] = 1
    pred[output_val < 0.5] = 0
    acc = np.mean(pred == Y_val)
    print('Iter %d, Accuracy %f' % (iter_idx + 1, acc))

# Visualize the decision boundary
plt.scatter(data_one[:, 0], data_one[:, 1])
plt.scatter(data_two[:, 0], data_two[:, 1])
x = np.linspace(-4, 10)
y = -(weight_val[0] * x + bias_val) / weight_val[1]
plt.plot(x, y, color='red')
plt.show()