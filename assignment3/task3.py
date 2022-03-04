import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer
from task3_model1 import Model1
from task3_model2 import Model2
from task4a import Model4

def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    utils.plot_loss(trainer.test_history["loss"], label="Test loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    utils.plot_loss(trainer.train_history["accuracy"], label="Training Accuracy")
    utils.plot_loss(trainer.test_history["accuracy"], label="Test Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()
    
def compare_plots(trainer: Trainer, name: str,trainer2: Trainer):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss model 1")
    utils.plot_loss(trainer2.validation_history["loss"], label="Validation loss model 2")
    utils.plot_loss(trainer.test_history["loss"], label="Test loss model 1")
    utils.plot_loss(trainer2.test_history["loss"], label="Test loss model 2")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy model 1")
    utils.plot_loss(trainer2.validation_history["accuracy"], label="Validation Accuracy model 2")
    utils.plot_loss(trainer.test_history["accuracy"], label="Test Accuracy model 1")
    utils.plot_loss(trainer2.test_history["accuracy"], label="Test Accuracy model 2")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()

def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 32
    learning_rate = 5e-4
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    # model1 = Model1(image_channels=3, num_classes=10)
    # trainer1 = Trainer(
    #     batch_size,
    #     learning_rate,
    #     early_stop_count,
    #     epochs,
    #     model1,
    #     dataloaders
    # )
    # trainer1.train()
    #model2 = Model2(image_channels=3, num_classes=10)
    model2 = Model4()
    trainer2 = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model2,
        dataloaders
    )
    trainer2.train()
    #create_plots(trainer1, "task3_model1: ")
    create_plots(trainer2, "Task3_model2_plot")
    #compare_plots(trainer1,"Task3_comparing_models",trainer2)

if __name__ == "__main__":
    main()