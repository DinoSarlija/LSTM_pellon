import pandas as pd
import numpy as np
import datetime as dt

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.metrics import explained_variance_score

def plot_graph(x, y, file_name, interval_by_user, color, filter=False):
    fig, ax = plt.subplots()
    ax.set(xlabel='Datumi', ylabel='Koncentracija',
        title='Skalirana koncentracija peludi u zraku kroz 6 godina')
    ax.grid()

    if filter == False:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=interval_by_user))

    plt.xticks(rotation=60)
    plt.plot(x, y, color)

    if filter == True:
        plt.savefig(file_name + '_filtrirano.png', bbox_inches = 'tight')
    else:
        plt.savefig(file_name + '.png', bbox_inches = 'tight')

    plt.show()


def plot_two_graphs(x, y1, y2, file_name, interval_by_user, filter=False, segment=False):

    fig, ax = plt.subplots()


    ax.grid()
    if filter == False:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))

    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval = interval_by_user))

    plt.xticks(rotation=60)
    plt.plot(x, y1, color='r')
    plt.plot(x, y2, color='b')
    if segment == True:
        ax.set(xlabel='Datumi', ylabel='Koncentracija', title = 'Skalirana koncentracija peludi u zraku kroz 6 mjeseci')
        if filter == True:
            plt.savefig(file_name + '_zajedno_segment_filtrirano' + '.png', bbox_inches = 'tight')
        else:
            plt.savefig(file_name + '_zajedno_segment' + '.png', bbox_inches = 'tight')
    else:
        ax.set(xlabel='Datumi', ylabel='Koncentracija', title = 'Skalirana koncentracija peludi u zraku kroz 6 godina')
        if filter == True:
            plt.savefig(file_name + '_zajedno_filtrirano' + '.png', bbox_inches = 'tight')
        else:
            plt.savefig(file_name + '_zajedno' + '.png', bbox_inches = 'tight')
    plt.show()

def plot_graphs(y_predict, y_test_numpy, sample, number_of_days, epochs, is_filtered):

    if is_filtered == False:
        dates = pd.read_csv("real_for_all_NS_2000_2021.csv")['DAT'].to_numpy()[sample + number_of_days:]
        x_dates = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in dates]

        file_name = "epochs_" + str(epochs) + "_sequence_" + str(number_of_days)

        # predikcija
        plot_graph(x_dates, y_predict, file_name + '_prediction', 200, color='b')

        # originalni podatci
        plot_graph(x_dates, y_test_numpy, file_name + '_original', 200, color='r')

        # zajedno
        plot_two_graphs(x_dates, y_test_numpy, y_predict, file_name, 150)

        # zajedno ali samo par mjeseci
        start_date = x_dates.index(dt.date.fromisoformat('2021-06-01'))
        end_date = x_dates.index(dt.date.fromisoformat('2021-12-01'))
        x_segment = np.array(x_dates)[start_date:end_date]

        plot_two_graphs(x_segment, y_test_numpy[start_date:end_date], y_predict[start_date:end_date], file_name, 15, segment=True)

    else:

        dates = pd.read_csv("real_for_all_NS_2000_2021.csv")[['DAT', 'MSC']]
        dates = dates.loc[(dates['MSC'] >= 6.0) & (dates['MSC'] <= 11.0)]['DAT'].to_numpy()

        dates_list = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in dates]

        dates_str_list = []

        for date in dates_list:
            dates_str_list.append(str(date))
        x_dates = np.array(dates_str_list)[sample + number_of_days:]

        file_name = "epochs_" + str(epochs) + "_sequence_" + str(number_of_days)

        # predikcija
        plot_graph(x_dates, y_predict, file_name + '_prediction', 100, color='b', filter=True)

        # originalni podatci
        plot_graph(x_dates, y_test_numpy, file_name + '_original', 100, color='r', filter=True)

        # zajedno
        plot_two_graphs(x_dates, y_test_numpy, y_predict, file_name, 100, filter=True)

        # zajedno ali samo par mjeseci
        start_date = np.where(x_dates == '2021-06-01')[0][0]
        end_date = np.where(x_dates == '2021-11-30')[0][0]

        plot_two_graphs(x_dates[start_date:end_date], y_test_numpy[start_date:end_date], y_predict[start_date:end_date], file_name, 10, filter=True, segment=True)

def save_model_info(number_of_days, scalar, percentage, X_train, y_train, X_test, y_test, hidden_size, transform, learning_rate, epochs, optimizer, criterion, y_test_numpy, y_predict, dropout=None):
    with open('model_info.txt', 'w') as f:
        f.write("Ambrozija\n \n")
        f.write("Sequence (broj dana) je " + str(number_of_days) + "\n \n")
        f.write("Za skaliranje podataka je korišten " + str(scalar) + "\n \n")
        f.write("Podjela podataka:\n")
        f.write("- treniranje: " + str(percentage) +", X = " + str(X_train.shape) +", y = "+ str(y_train.shape) + "\n")
        f.write("- testiranje: " + str(round(1 - percentage,2)) +", X = " + str(X_test.shape) +", y = "+ str(y_test.shape) + "\n \n")
        f.write("Podatci za LSTM model:\n")
        f.write("- input size = " + str(X_train.shape[2]) + "\n")
        f.write("- hidden size = " + str(hidden_size) + "\n")
        f.write("- transform = " + str(transform) + "\n")
        f.write("- learing rate = " + str(learning_rate) + "\n")
        f.write("- epochs = " + str(epochs) + "\n")
        f.write("- optimizer = " + str(optimizer) + "\n")
        f.write("- criterion = " + str(criterion) + "\n \n")
        f.write("Usporedba predikcije i stvarnih rezultata: \n")
        f.write("- explained variance regression score function: " + str(explained_variance_score(y_test_numpy, y_predict)) + "\n")
        if dropout != None:
            f.write("Dropout - " + str(dropout) + "\n")