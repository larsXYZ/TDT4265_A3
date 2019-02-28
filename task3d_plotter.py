import matplotlib.pyplot as plt
import pickle

with open('task3_results_plots_model/results.pickle', 'rb') as handle:
    task3_result = pickle.load(handle)

print(task3_result)

plt.plot(task3_result["train loss"], label="training loss, pretrained model")
plt.plot(task3_result["validation loss"], label="validation loss, pretrained model")
plt.legend()
plt.show()