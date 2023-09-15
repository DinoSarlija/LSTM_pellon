import torch
from torch import nn

import pandas as pd
import numpy as np

from data_visualization import plot_graphs, save_model_info
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")


df = pd.read_csv('real_for_all_NS_2000_2021.csv')
data = df.drop(['LOK','DAT'], axis=1)

X_ambrozija = pd.concat([data.iloc[:,:16], data['PRAM']], axis = 1)
X_breza = pd.concat([data.iloc[:,:16], data['PRBR']], axis = 1)
X_trava = pd.concat([data.iloc[:,:16], data['PRTR']], axis = 1)


X_ambrozija_filtered = X_ambrozija.loc[(X_ambrozija['MSC'] >= 6.0) & (X_ambrozija['MSC'] <= 11.0)]

mm = MinMaxScaler()

### If we are using filtered data, then the line below should be uncommented.
#X_ambrozija_mm = mm.fit_transform(X_ambrozija_filtered)

### If we are using filtered data, then the line below should be commented out.
X_ambrozija_mm = mm.fit_transform(X_ambrozija)
X_breza_mm = mm.fit_transform(X_breza)
X_trava_mm = mm.fit_transform(X_trava)

number_of_days = 5
percentage = 0.7
sample = int(len(X_ambrozija_mm) * percentage)

X_list=[]
y_list=[]
for i in range(len(X_ambrozija_mm) - number_of_days):

    X_list.append(X_ambrozija_mm[i:i+number_of_days])
    y_list.append(X_ambrozija_mm[i+number_of_days][16])

X = np.array(X_list)
y = np.array(y_list)


X_train_numpy = X[:sample]
X_test_numpy = X[sample:]

X_train = torch.from_numpy(X_train_numpy).to(device)
X_test = torch.from_numpy(X_test_numpy).to(device)

y_train_numpy = y[:sample]
y_test_numpy = y[sample:]

y_train = torch.from_numpy(y_train_numpy).to(device)
y_test = torch.from_numpy(y_test_numpy).to(device)


input_size = X_train.shape[2]

hidden_size = 512

model = nn.LSTM(input_size, hidden_size).to(device)

### If we do not need dropout, then the line below should be commented out.
dropout = nn.Dropout(p=0.25)
transform = nn.Linear(hidden_size,1).to(device)

learning_rate = 0.01
epochs = 1000

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
criterion = torch.nn.MSELoss()


model.train()
model.float()

for epoch in tqdm(range(1,epochs+1)):

    for interval in range(X_train.shape[0]):

        output, (hn, cn) = model(X_train[interval].float())

        dropout_out = dropout(hn)

        fin_out = transform(dropout_out)

        optimizer.zero_grad()

        loss = criterion(fin_out[0][0], y_train[interval].float())
        loss.backward()

        optimizer.step()


        if epoch % (round(epochs * 0.1)) == 0 and interval == (X_train.shape[0]-1):
            print("Epoch:", epoch, ",  loss:", loss.item())



model.eval()

all_predictions = []

with torch.no_grad():
    for x in X_test:
        output, (hn, cn) = model(x.float())

        predict = transform(hn).detach().item()

        all_predictions.append(predict)

y_predict = np.array(all_predictions)


#save_model_info(number_of_days, mm, percentage, X_train, y_train, X_test, y_test, hidden_size, transform, learning_rate, epochs, optimizer, criterion, y_test_numpy, y_predict)

#is_filtered = False
#plot_graphs(y_predict, y_test_numpy, sample, number_of_days, epochs, is_filtered)

