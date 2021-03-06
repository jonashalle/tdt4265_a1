import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # TODO: Implement this function (task 3c)
    accuracy = 0.0
    correct_predictions = 0
    predictions = targets.shape[0]
    outputs = model.forward(X)
    #print("shape target: ",targets.shape,"   shape out: ", outputs.shape)
    outputs = model.forward(X)
    for idx, val in enumerate(outputs):
        target = targets[idx]
        if np.argmax(target)==np.argmax(val):
            correct_predictions += 1
    accuracy = correct_predictions/predictions
    
    return accuracy


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        # TODO: Implement this function (task 3b)
        outputs = self.model.forward(X_batch)                
        loss = cross_entropy_loss(Y_batch, outputs)
        self.model.backward(X_batch,outputs,Y_batch)
        self.model.w = self.model.w-learning_rate*self.model.grad
        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
            accuracy_train (float): Accuracy on train dataset
            accuracy_val (float): Accuracy on the validation dataset
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(Y_val, logits)

        accuracy_train = calculate_accuracy(
            X_train, Y_train, self.model)
        accuracy_val = calculate_accuracy(
            X_val, Y_val, self.model)
        return loss, accuracy_train, accuracy_val


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 2
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Intialize model
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    
    plt.ylim([0.2, .6])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([0.8, .96])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()
    
    # Train a model with L2 regularization (task 4b)

    model1 = SoftmaxModel(0.0)
    trainer1 = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg01, val_history_reg01 = trainer1.train(num_epochs)
    # You can finish the rest of task 4 below this point.
    #diff = model.w - model1.w
    #print(f"diff weights: {diff}")
    # Plotting of softmax weights (Task 4b)
    model.w = np.delete(model.w, 0, axis=0)
    model.w = model.w.reshape(28, 28, 10)
    temp_arr = model.w[:,:,0]
    for i in range(model.w.shape[2]-1):
        temp_arr = np.hstack((temp_arr, model.w[:,:,i+1]))
    model.w = temp_arr
    plt.imsave("task4b_softmax_weight_model_lambda_done.png", model.w, cmap="gray")

    
    model1.w = np.delete(model1.w, 0, axis=0)
    model1.w = model1.w.reshape(28, 28, 10)
    temp_arr = model1.w[:,:,0]
    for i in range(model1.w.shape[2]-1):
        temp_arr = np.hstack((temp_arr, model1.w[:,:,i+1]))
    model1.w = temp_arr
    plt.imsave("task4b_softmax_weight_model2.png", model1.w, cmap="gray")
    
    model_arr = np.vstack((model.w, model1.w))
    plt.imsave("task4b_softmax_weight_combine.png", model_arr, cmap="gray")
    
    # Plotting of accuracy for difference values of lambdas (task 4c)
    l2_lambdas = [2, .2, .02, .002]
        
    models = []
    train_hist = []
    val_hist =  []
    for idx, _lambda in enumerate(l2_lambdas):
        models[idx] = SoftmaxModel(_lambda)
        trainer = SoftmaxTrainer(
        models[idx], learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
        )
        train_hist[idx], val_hist[idx] = trainer.train(num_epochs)
        print("shape trainhist", train_hist[idx].shape)

    
    # plt.ylim([0.2, .6])
    # utils.plot_loss(train_history["loss"],
    #                 "Training Loss", npoints_to_average=10)
    # utils.plot_loss(val_history["loss"], "Validation Loss")
    # plt.legend()
    # plt.xlabel("Number of Training Steps")
    # plt.ylabel("Cross Entropy Loss - Average")
    # plt.savefig("task3b_softmax_train_loss.png")
    # plt.show()

    # # Plot accuracy
    # plt.ylim([0.8, .96])
    # utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    # utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    # plt.xlabel("Number of Training Steps")
    # plt.ylabel("Accuracy")
    # plt.legend()
    # plt.savefig("task3b_softmax_train_accuracy.png")
    # plt.show()

    # plt.savefig("task4c_l2_reg_accuracy.png")

    # # Task 4d - Plotting of the l2 norm for each weight

    # plt.savefig("task4d_l2_reg_norms.png")
    