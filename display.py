import numpy as np
import matplotlib.pyplot as plt

trainX = np.load('trainX.npy')

random_index = np.random.randint(0, len(trainX))
values = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

random_image = trainX[random_index]

plt.imshow(random_image.squeeze(), cmap='gray')  
plt.title(f"Random Image {random_index}")
plt.axis('off') 

plt.savefig('random_image.png')
plt.close()

trainy = np.load('trainy.npy')

random_label = trainy[random_index]

print("Random image saved as 'random_image.png'")
print(f"Label of random image: {random_label}")
print(f"Value of label from random image: {values[random_label]}")