from matplotlib import pyplot as plt
from itertools import cycle
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import numpy as np
from scipy import interp
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


class plotting():
    def __init__(self, y_test, y_pred, history, model_name, save_path):
        """

        """
        self.y_test = y_test
        self.y_pred = y_pred
        self.history = history
        self.save_path = save_path
        self.model_name = model_name


    def loss(self, show=False):
        """

        """
        plt.figure(1)
        plt.plot(self.history.history["loss"], label = "train_loss", color="darkorange", linewidth=2)
        plt.plot(self.history.history["val_loss"], label = "val_loss", color="navy", linewidth=2)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Loss versus epochs")
        plt.ylim([0 ,1])
        plt.legend(loc="upper right")

        plt.savefig(self.save_path + "/" + self.model_name + "_loss_plot.png")

        if show:
            plt.show()

    def rmse(self, show=False):
        """

        """
        plt.figure(2)
        plt.plot(self.history.history["rmse"], label = "rmse", color="darkorange", linewidth=2)
        plt.plot(self.history.history["val_rmse"], label = "val_rmse", color="navy", linewidth=2)
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.title("RMSE versus epochs")
        plt.ylim([0, 1])
        plt.legend(loc="lower right")
    
        plt.savefig(self.save_path + "/" + self.model_name + "_accuracy_plot.png")

        if show:
            plt.show()



    def accuracy(self, show=False):
        """

        """
        plt.figure(3)
        plt.plot(self.history.history["accuracy"], label = "accuracy", color="darkorange", linewidth=2)
        plt.plot(self.history.history["val_accuracy"], label = "val_accuracy", color="navy", linewidth=2)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy versus epochs")
        plt.ylim([0, 1])
        plt.legend(loc="lower right")
    
        plt.savefig(self.save_path + "/" + self.model_name + "_accuracy_plot.png")

        if show:
            plt.show()

        plt.figure(4)
        plt.plot(self.history.history["accuracy"], label = "accuracy", color="darkorange", linewidth=2)
        plt.plot(self.history.history["val_accuracy"], label = "val_accuracy", color="navy", linewidth=2)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy versus epochs")
        plt.ylim([0.5, 1])
        plt.legend(loc="lower right")
    
        plt.savefig(self.save_path + "/" + self.model_name + "_accuracy_plot_zoom.png")

        if show:
            plt.show()
 

    def roc(self, num_classes, show=False):
        """

        """
        # Plot linewidth.
        lw = 2
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(self.y_test[:, i], self.y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(self.y_test.ravel(), self.y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= num_classes 

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure(5)
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(num_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right", fontsize="small")
 
        plt.savefig(self.save_path + "/" + self.model_name + "_roc_plot.png")

        if show:
            plt.show()


        # Zoom in view of the upper left corner.
        plt.figure(6)
        plt.xlim(0, 0.2)
        plt.ylim(0.8, 1)
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(10), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right", fontsize="small")
  
        plt.savefig(self.save_path + "/" + self.model_name + "_roc_plot_zoom.png")

        if show:
            plt.show()

