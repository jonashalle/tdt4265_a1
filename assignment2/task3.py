import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model1 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer1 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model1, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history1, val_history1 = trainer1.train(num_epochs)

    use_improved_sigmoid = True
    use_improved_weight_init = False
    use_momentum = False

    model2 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer2 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model2, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history2, val_history2 = trainer2.train(num_epochs)

    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = False

    model3 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer3 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model3, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history3, val_history3 = trainer3.train(num_epochs)

    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True

    model4 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer4 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model4, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history4, val_history4 = trainer4.train(num_epochs)

    # Example created for comparing with and without shuffling.
    # For comparison, show all loss/accuracy curves in the same plot
    # YOU CAN DELETE EVERYTHING BELOW!
    
    plt.figure(figsize=(20,12))
    plt.subplot(2, 2, 1)
    plt.ylim([0, .4])
    utils.plot_loss(train_history1["loss"],
                    "Task 2", npoints_to_average=10)
    plt.legend()
    

    plt.subplot(2, 2, 2)
    plt.ylim([0, .4])
    utils.plot_loss(train_history1["loss"],
                    "Task 3a ", npoints_to_average=10)
    
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.ylim([0, .4])
    utils.plot_loss(train_history3["loss"],
                    "Task 3b", npoints_to_average=10)
    
    

    plt.subplot(2, 2, 4)
    plt.ylim([0, .4])
    utils.plot_loss(train_history4["loss"],
                    "Task 3c", npoints_to_average=10)

    plt.ylabel("Validation loss")
    plt.legend()

    plt.show()
