import pickle
import matplotlib.pyplot as plt

f = open("pretrained_models/fullmap_2400_1000epochs/progress.pkl", "rb")
data = pickle.load(f)

train_loss = data['trainLoss']
val_loss = data['valLoss']
epochs = range(0, len(train_loss))

plt.plot(epochs, train_loss, label='Train loss')
plt.plot(epochs, val_loss, label='Val loss')

plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.legend(loc="upper left")
plt.savefig('fullMapPerformance.png')