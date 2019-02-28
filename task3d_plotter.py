import matplotlib.pyplot as plt
import pickle

#Plotting for task 3d

with open('task3_results_plots_model/results.pickle', 'rb') as handle:
    task3_result = pickle.load(handle)

with open('model1_data.p', 'rb') as handle:
    task2_result = pickle.load(handle)

train_loss_task_2 = task2_result[0]
validation_loss_task_2 = task2_result[1]


plt.plot(task3_result["train loss"], label="training loss, pretrained model")
plt.plot(task3_result["validation loss"], label="validation loss, pretrained model")
plt.plot(train_loss_task_2, label="training loss, our model")
plt.plot(validation_loss_task_2, label="validation loss, our model")
plt.legend()
plt.show()