import numpy as np
import utils
import typing
np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)
    bias = np.ones((X.shape[0],1))
    mean = np.mean(X)
    standard_variation = np.std(X)
    X = (X - mean) / standard_variation
    X = np.append(X, bias, axis = 1)
    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    # TODO: Implement this function (copy from last assignment)
    N = targets.shape[0]
    C_n = -targets*(np.log(outputs))
    loss = np.sum(C_n)/N
    return loss

def sigmoid(z):
    return 1/(1+np.exp(-z))

def diff_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z)) # Derivative of the sigmoid function

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z), axis = 1, keepdims=True)

def diff_cross_entropy(targets, outputs):
    return -(targets - outputs)

class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            #w = np.zeros(w_shape)
            w = np.random.uniform(1, -1, w_shape)
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        self.layer_outputs = []
        self.layer_sums = []
        
        z1 = np.dot(X,self.ws[0])
        a1 = sigmoid(z1)
        z2 = np.dot(a1,self.ws[1])
        a2 = softmax(z2)

        self.layer_outputs.append(a1)
        self.layer_outputs.append(a2)

        self.layer_sums.append(z1)
        self.layer_sums.append(z2)

        # HINT: For performing the backward pass, you can save intermediate activations in variables in the forward pass.
        # such as self.hidden_layer_output = ...
        return a2
    
    

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        
        self.grads = []
        delta = []

        delta_output = diff_cross_entropy(targets, outputs)  # delta_2 = grad C/a * f'(z2)
        # print(f"Shape of ws[1] : {self.ws[1].shape}")
        # print(f"Shape of d_output : {d_output.shape}")
        # print(f"Shape of ws[1].dot(d_output.T) : {self.ws[1].dot(d_output.T).shape}")
        # print(f"Shape of diff_sigmoid(z2) : {diff_sigmoid(self.layer_sums[1]).shape}")
        delta.append(self.ws[1].dot(delta_output.T) * diff_sigmoid(self.layer_sums[0]).T)  # delta_1 = weight_2^T delta_2 * f'(z1) 
        # delta[0] =  diff_cross_entropy(targets, outputs).T.dot(delta[1]) * np.dot(self.ws[1].T, delta[1]) #     
        delta.append(delta_output)
        # print(f"Shape of delta[1] : {delta[1].shape}")
        # print(f"Shape of weights[0] : {self.ws[0].shape}")
        # print(f"Shape of X : {X.shape}")
        self.grads.append(delta[0].dot(X).T/X.shape[0]) # Grad C with regards to w_ji
        self.grads.append(delta[1].T.dot(self.layer_outputs[0]).T/X.shape[0])   # Grad C with regards to w_kj
        # print(f"Shape of diff_cross_entropy : {diff_cross_entropy(targets, outputs).shape}")
        # print(f"Shape of grads[1] : {self.grads[1].shape}")

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    # TODO: Implement this function (copy from last assignment)
    Y = np.eye(num_classes)[Y]
    Y = np.squeeze(Y, axis=1)
    return Y


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
