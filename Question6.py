import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5806)


def guassian(n):
    return np.exp(-n ** 2)


def guassian_derivative(n):
    return -2 * n * guassian(n)


def linear(n):
    return n


def linear_derivative(n):
    return np.ones_like(n)


class RBF:
    def __init__(self):
        self.w1 = np.random.randn(2, 1)
        self.b1 = np.random.rand(2, 1)
        self.w2 = np.random.randn(1, 2)
        self.b2 = np.random.randn(1, 1)

        formatted_w1 = np.array2string(self.w1, formatter={'float_kind': lambda x: "%.2f" % x})
        print(f'The initial weights in the first layer are: \n {formatted_w1}')
        formatted_b1 = np.array2string(self.b1, formatter={'float_kind': lambda x: "%.2f" % x})
        print(f'The initial biases in the first layer are: \n {formatted_b1}')
        formatted_w2 = np.array2string(self.w2, formatter={'float_kind': lambda x: "%.2f" % x})
        print(f'The initial weights in the second layer are: \n {formatted_w2}')
        formatted_b2 = np.array2string(self.b2, formatter={'float_kind': lambda x: "%.2f" % x})
        print(f'The initial biases in the second layer are: \n {formatted_b2} \n')

    def forward(self, x):
        # x: a0 = p
        a_list = [x]
        n_list = []

        n11 = np.absolute(self.w1[0] - x) * self.b1[0]
        a11 = guassian(n11)
        n12 = np.absolute(self.w1[1] - x) * self.b1[1]
        a12 = guassian(n12)
        n1 = np.array([n11, n12]).reshape(2, 1)
        n_list.append(n1)
        a1 = np.array([a11, a12]).reshape(2, 1)
        a_list.append(a1)

        n2 = self.w2 @ a_list[-1] + self.b2
        n_list.append(n2)
        a_list.append(linear(n2))

        return a_list, n_list

    def backward(self, a_list, n_list, target):
        list_sensitivity = []
        sensitivity_last_layer = (-2) * linear_derivative(n_list[-1]) * (target - a_list[-1])
        list_sensitivity.append(sensitivity_last_layer)

        F = [[0, 0], [0, 0]]

        F[0][0] = guassian_derivative(n_list[0][0]).item()
        F[1][1] = guassian_derivative(n_list[0][1]).item()

        F_ndarray = np.array(F)
        sensitivity = F_ndarray @ self.w2.T @ list_sensitivity[0]
        list_sensitivity.insert(0, sensitivity)

        return list_sensitivity

    def update_params_1(self, list_sensitivity, a_list, learning_rate):
        list_delta_weights = []
        list_delta_biases = []

        delta_w11 = - learning_rate * list_sensitivity[0][0] * (self.b1[0] * (self.w1[0] - a_list[0][0]) /
                                                                np.absolute(a_list[0][0] - self.w1[0]))
        list_delta_weights.append(delta_w11)
        delta_w12 = - learning_rate * list_sensitivity[0][1] * (self.b1[1] * (self.w1[1] - a_list[0][0]) /
                                                                np.absolute(a_list[0][0] - self.w1[1]))
        list_delta_weights.append(delta_w12)
        delta_w2 = - learning_rate * list_sensitivity[1] @ a_list[1].T
        list_delta_weights.append(delta_w2)

        delta_b11 = - learning_rate * list_sensitivity[0][0] * np.absolute(a_list[0][0] - self.w1[0])
        list_delta_biases.append(delta_b11)
        delta_b12 = - learning_rate * list_sensitivity[0][1] * np.absolute(a_list[0][0] - self.w1[1])
        list_delta_biases.append(delta_b12)
        delta_b2 = - learning_rate * list_sensitivity[1]
        list_delta_biases.append(delta_b2)

        self.w2 -= learning_rate * list_sensitivity[1] @ a_list[1].T
        self.b2 -= learning_rate * list_sensitivity[1]

        self.w1[0] -= learning_rate * list_sensitivity[0][0] * (self.b1[0] * (self.w1[0] - a_list[0][0]) /
                                                                np.absolute(a_list[0][0] - self.w1[0]))
        self.w1[1] -= learning_rate * list_sensitivity[0][1] * (self.b1[1] * (self.w1[1] - a_list[0][0]) /
                                                                np.absolute(a_list[0][0] - self.w1[1]))
        self.b1[0] -= learning_rate * list_sensitivity[0][0] * np.absolute(a_list[0][0] - self.w1[0])
        self.b1[1] -= learning_rate * list_sensitivity[0][1] * np.absolute(a_list[0][0] - self.w1[1])

        return list_delta_weights, list_delta_biases

    def update_params(self, list_sensitivity, a_list, learning_rate, momentum_term, list_delta_weights,
                      list_delta_biases):
        new_list_delta_weights = []
        new_list_delta_biases = []

        delta_w11 = momentum_term * list_delta_weights[0] - (1 - momentum_term) * learning_rate * list_sensitivity[0][
            0] * (self.b1[0] * (self.w1[0] - a_list[0][0]) / np.absolute(a_list[0][0] - self.w1[0]))
        new_list_delta_weights.append(delta_w11)
        delta_w12 = momentum_term * list_delta_weights[1] - (1 - momentum_term) * learning_rate * list_sensitivity[0][
            1] * (self.b1[1] * (self.w1[1] - a_list[0][0]) / np.absolute(a_list[0][0] - self.w1[1]))
        new_list_delta_weights.append(delta_w12)
        delta_w2 = momentum_term * list_delta_weights[2] - (1 - momentum_term) * learning_rate * list_sensitivity[1] @ \
                   a_list[1].T
        new_list_delta_weights.append(delta_w2)

        delta_b11 = momentum_term * list_delta_biases[0] - (1 - momentum_term) * learning_rate * list_sensitivity[0][
            0] * np.absolute(a_list[0][0] - self.w1[0])
        new_list_delta_biases.append(delta_b11)
        delta_b12 = momentum_term * list_delta_biases[1] - (1 - momentum_term) * learning_rate * list_sensitivity[0][
            1] * np.absolute(a_list[0][0] - self.w1[1])
        new_list_delta_biases.append(delta_b12)
        delta_b2 = momentum_term * list_delta_biases[2] - (1 - momentum_term) * learning_rate * list_sensitivity[1]
        new_list_delta_biases.append(delta_b2)

        self.w2 += momentum_term * list_delta_weights[2] - (1 - momentum_term) * learning_rate * list_sensitivity[1] @ \
                   a_list[1].T
        self.b2 += momentum_term * list_delta_biases[2] - (1 - momentum_term) * learning_rate * list_sensitivity[1]

        self.w1[0] += momentum_term * list_delta_weights[0] - (1 - momentum_term) * learning_rate * list_sensitivity[0][
            0] * (self.b1[0] * (self.w1[0] - a_list[0][0]) / np.absolute(a_list[0][0] - self.w1[0]))
        self.w1[1] += momentum_term * list_delta_weights[1] - (1 - momentum_term) * learning_rate * list_sensitivity[0][
            1] * (self.b1[1] * (self.w1[1] - a_list[0][0]) / np.absolute(a_list[0][0] - self.w1[1]))
        self.b1[0] += momentum_term * list_delta_biases[0] - (1 - momentum_term) * learning_rate * list_sensitivity[0][
            0] * np.absolute(a_list[0][0] - self.w1[0])
        self.b1[1] += momentum_term * list_delta_biases[1] - (1 - momentum_term) * learning_rate * list_sensitivity[0][
            1] * np.absolute(a_list[0][0] - self.w1[1])

        return new_list_delta_weights, new_list_delta_biases


def train_rbf(rbf, inputs, targets, momentum_term, learning_rate, max_iterations, sse_cutoff):
    sse_history = []
    final_responses = []
    num_of_iterations = 0
    is_first_data = True
    list_delta_weights = []
    list_delta_biases = []
    for i in range(max_iterations):
        e_list = []
        responses = []
        for x, y in zip(inputs, targets):
            x = x.reshape(-1, 1)  # Ensure that x is a column vector
            y = y.reshape(-1, 1)  # Ensure that y is a column vector

            # Forward Propagation
            a_list, n_list = rbf.forward(x)

            e = y - a_list[-1]
            e_list.append(e)

            responses.append(a_list[-1])

            # Backward Propagation
            list_sensitivity = rbf.backward(a_list, n_list, y)

            # Weight and bias update
            if is_first_data:
                list_delta_weights, list_delta_biases = rbf.update_params_1(list_sensitivity, a_list, learning_rate)
                is_first_data = False
            else:
                list_delta_weights, list_delta_biases = rbf.update_params(list_sensitivity, a_list, learning_rate,
                                                                          momentum_term, list_delta_weights,
                                                                          list_delta_biases)

        # Calculate and store the sum of squared errors
        sse = sum([e ** 2 for e in e_list])
        sse_history.append(sse)

        final_responses = responses

        num_of_iterations += 1

        # Check for convergence
        if sse < sse_cutoff:
            print(f"Training stopped after {i + 1} iterations with SSE: {sse}")
            break

    # Plot the network response against the target function
    plt.scatter(inputs, targets, color='blue', label='Target')

    final_responses_ndarray = np.array(final_responses).squeeze()
    plt.plot(inputs, final_responses_ndarray, color='red', linestyle='-', label='Network after training')
    plt.xlabel('p')
    plt.ylabel('Magnitude')
    plt.title(f'Number of observations = {inputs.size} \n' +
              f'Number of iterations = {num_of_iterations} \n' +
              f'Learning ratio = {learning_rate} \n' +
              f'Momentum Term = {momentum_term} \n' +
              f'Number of neurons in the first layer = 2')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Plot the SSE vs number of iterations in log scale
    sse_history_formatted = np.array(sse_history).squeeze()
    plt.semilogx(range(len(sse_history)), sse_history_formatted, '-b')
    plt.semilogy(range(len(sse_history)), sse_history_formatted, '-b')
    plt.xlabel('Number of iterations')
    plt.ylabel('Magnitude')
    plt.title(f'sum of square of errors = {sse_history[-1][0][0]:.3f} \n' +
              f'Number of iterations = {num_of_iterations} \n' +
              f'Number of neurons in the first layer = 2 \n' +
              f'SSE error cut off = {sse_cutoff}')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
num_samples = 100

p_values = np.linspace(0, np.pi, num_samples)
targets = np.sin(p_values)
inputs = np.array([p_values]).T

# Define the network
rbf = RBF()

# Train the network
train_rbf(rbf, inputs, targets, momentum_term=0.1, learning_rate=0.01, max_iterations=2000, sse_cutoff=0.001)

# Please uncomment each of following lines to run the code for each momentum term
# train_rbf(rbf, inputs, targets, momentum_term=0.2, learning_rate=0.01, max_iterations=2000, sse_cutoff=0.001)
# train_rbf(rbf, inputs, targets, momentum_term=0.4, learning_rate=0.01, max_iterations=2000, sse_cutoff=0.001)
# train_rbf(rbf, inputs, targets, momentum_term=0.6, learning_rate=0.01, max_iterations=2000, sse_cutoff=0.001)
# train_rbf(rbf, inputs, targets, momentum_term=0.8, learning_rate=0.01, max_iterations=2000, sse_cutoff=0.001)
# train_rbf(rbf, inputs, targets, momentum_term=1, learning_rate=0.01, max_iterations=2000, sse_cutoff=0.001)

formatted_w1_final = np.array2string(rbf.w1, formatter={'float_kind': lambda x: "%.2f" % x})
print(f'The final trained weights in the first layer are: \n {formatted_w1_final}')
formatted_b1_final = np.array2string(rbf.b1, formatter={'float_kind': lambda x: "%.2f" % x})
print(f'The final trained biases in the first layer are: \n {formatted_b1_final}')
formatted_w2_final = np.array2string(rbf.w2, formatter={'float_kind': lambda x: "%.2f" % x})
print(f'The final trained weights in the second layer are: \n {formatted_w2_final}')
formatted_b2_final = np.array2string(rbf.b2, formatter={'float_kind': lambda x: "%.2f" % x})
print(f'The final trained biases in the second layer are: \n {formatted_b2_final}')
