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
   
    
    # Plotting of accuracy for difference values of lambdas (task 4c)
    l2_lambdas = [2, .2, .02, .002]
        
    models = []
    train_hists = []
    val_hists =  []
    for idx, _lambda in enumerate(l2_lambdas):
        models.append(SoftmaxModel(_lambda))
        trainer = SoftmaxTrainer(
        models[idx], learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
        )
        train_hist, val_hist = trainer.train(num_epochs)
        train_hists.append(train_hist)
        val_hists.append(val_hist)

    train_history = train_hists[0]
    val_history = val_hists[0]
    
    # plt.ylim([0.2, 0.6])
    # for i in range(len(val_hists)):
    #     utils.plot_loss(val_hists[i]["loss"], f"Validation Loss:{i}")
    # plt.legend()
    # plt.xlabel("Number of Training Steps")
    # plt.ylabel("Cross Entropy Loss - Average")
    # plt.savefig("task3b_softmax_train_loss_4c.png")
    # plt.show()

    # Plot accuracy
    plt.ylim([0.85, 0.96])
    for i in range(len(val_hists)):
        utils.plot_loss(val_hists[i]["accuracy"], f"Validation accuracy:lambda={l2_lambdas}")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy_4c.png")
    plt.show()

    # plt.savefig("task4c_l2_reg_accuracy.png")

    # # Task 4d - Plotting of the l2 norm for each weight

    # plt.savefig("task4d_l2_reg_norms.png")
    