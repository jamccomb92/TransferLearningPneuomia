import pandas as pd
import matplotlib.pyplot as plt
import csv

headers = ['Epoch','Acc','Loss','Val_Acc','Val_Loss']
df = pd.read_csv("resnet.csv")

Acc = df['acc']
Epoch = df['epoch']
Loss = df['loss']
Val_Acc = df['val_acc']
Val_Loss = df['val_loss']

#df1 = pd.read_csv("vgg16_noWeights.csv")

#Acc1 = df1['acc']
#Epoch1 = df1['epoch']
#Loss1 = df1['loss']
#Val_Acc1 = df1['val_acc']
#Val_Loss1 = df1['val_loss']

print(df)
#print(df1)
plt.plot(Epoch,Acc,label='Acc')
plt.plot(Epoch,Loss, label='Loss')
plt.plot(Epoch, Val_Acc,label='Val_Acc')
plt.plot(Epoch, Val_Loss, label='Val_Loss')    
plt.legend(loc='best')
plt.title('ResNet50 Model with ImageNet Weights')
plt.axis([-1,30,0,1])
plt.show()

#plt.plot(Epoch,Acc1,label='Acc')
#plt.plot(Epoch,Loss1, label='Loss')
#plt.plot(Epoch, Val_Acc1,label='Val_Acc')
#plt.plot(Epoch, Val_Loss1, label='Val_Loss')
#plt.legend(loc='best')
#plt.title('VGG16 Model with No Weights')
#plt.axis([-1,20,0,1])
#plt.show()

