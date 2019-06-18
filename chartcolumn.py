import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd


def chart_column(train, test, train_0, train_1, test_0, test_1, name):
    sns.set(style="darkgrid")
    percent_train_0 = train_0 * 100 / train
    percent_train_1 = 100 - percent_train_0
    percent_test_0 = test_0 * 100 / test
    percent_test_1 = 100 - percent_test_0
    data = [percent_train_0, percent_train_1, percent_test_0, percent_test_1]

    data = pd.DataFrame({'Lable': ['X train', 'X train', 'X test', 'X test'],
                         'Percent': np.array(data),
                         'X': ['X_0', 'X_1', 'X_0', 'X_1']})
    print(data)
    ax = sns.barplot(x=data['Lable'], y=data['Percent'],
                     data=data, hue=data['X'])
    ax.set(xlabel="X")
    figure = ax.get_figure()
    figure.savefig('Results/Image/' + name + '/chart_column_' + name + '.png', dpi=72)
    figure.clf()


def chart_line_AUC_and_Time(name, svm, dt, rf, nb, dataname, method, algorithm, AUCorTime):
    sns.set(style="darkgrid")
    data = pd.read_csv(name)
    label = ['data_index', 'input_dim', 'balance_rate', 'auc_svm', 'auc_dt', 'auc_rf',\
            'auc_nb', 'time_svm', 'time_dt', 'time_rf', 'time_nb']
    data_non_learning_feature = pd.read_csv("Results/RF_AUC_DIF/" + dataname + "/AUC_Input.csv", names=label)
    time_non_learning_feature = data_non_learning_feature.tail(1)
    time_svm = 1000 * sum(data[svm]) / len(data[svm])
    time_dt = 1000 * sum(data[dt]) / len(data[dt])
    time_rf = 1000 * sum(data[rf]) / len(data[rf])
    time_nb = 1000 * sum(data[nb]) / len(data[nb])
    # data['algorithm'] = np.repeat(np.array(['svm','dt','rf']), int(len(data)/3+2))[:len(data)]
    if AUCorTime == 'AUC':
        if algorithm == 1:
            ax = sns.lineplot(x=data['# epoch'], y=data[rf], data=data)
            ax.set(xlabel='Epoch', ylabel='AUC')

            figure = ax.get_figure()
            figure.savefig('Results/Image/' + dataname + '/chart_line_AUC_' + dataname + "_" + method + '_RF' + '.png',
                           dpi=72)
            figure.clf()
        if algorithm == 2:
            ax = sns.lineplot(x=data['# epoch'], y=data[svm], data=data)  # '''hue=algorithm''')
            ax.set(xlabel='Epoch', ylabel='AUC')

            figure = ax.get_figure()
            figure.savefig('Results/Image/' + dataname + '/chart_line_AUC_' + dataname + "_" + method + '_SVM' + '.png',
                           dpi=72)
            figure.clf()
        if algorithm == 3:
            ax = sns.lineplot(x=data['# epoch'], y=data[dt], data=data)
            ax.set(xlabel='Epoch', ylabel='AUC')

            figure = ax.get_figure()
            figure.savefig('Results/Image/' + dataname + '/chart_line_AUC_' + dataname + "_" + method + '_DT' + '.png',
                           dpi=72)
            figure.clf()
        if algorithm == 4:
            ax = sns.lineplot(x=data['# epoch'], y=data[nb], data=data)
            ax.set(xlabel='Epoch', ylabel='AUC')

            figure = ax.get_figure()
            figure.savefig('Results/Image/' + dataname + '/chart_line_AUC_' + dataname + "_" + method + '_NB' + '.png',
                           dpi=72)
            figure.clf()
        if algorithm == 0:
            plt.style.use('ggplot')
            line_svm, = plt.plot('# epoch', svm, data=data, label='SVM', color='green')
            line_rf, = plt.plot('# epoch', rf, data=data, label='RF', color='red')
            line_dt, = plt.plot('# epoch', dt, data=data, label='DT', color='blue')
            line_nb, = plt.plot('# epoch', nb, data=data, label='NB', color='yellow')
            plt.legend(handles=[line_svm, line_rf, line_dt, line_nb])
            plt.ylabel('AUC')
            plt.xlabel('Epoch')
            plt.savefig('Results/Image/' + dataname + '/chart_line_AUC_' + dataname + "_" + method + '.png', dpi=72)
            plt.clf()
    else:
        if algorithm == 1:
            plt.style.use('ggplot')
            plt.bar('RF_Learning_Feautre', time_rf, label='RF', color='red')
            plt.bar('RF_Normal', time_non_learning_feature['time_rf'], label='RF', color='red', width=0.5)
            plt.ylabel('Time (ms)')
            plt.xlabel('Deep learning network')
            plt.savefig('Results/Image/' + dataname + '/chart_line_Time_' + dataname + "_" + method + '.png', dpi=72)
            plt.clf()
        if algorithm == 2:
            ax = sns.lineplot(x=data['# epoch'], y=data[svm], data=data)  # '''hue=algorithm''')
            ax.set(xlabel='SVM', ylabel='Time (ms)')
            plt.xticks([])
            figure = ax.get_figure()
            figure.savefig(
                'Results/Image/' + dataname + '/chart_line_Time_' + dataname + "_" + method + '_SVM' + '.png', dpi=72)
            figure.clf()
        if algorithm == 3:
            ax = sns.lineplot(x=data['# epoch'], y=data[dt], data=data)
            ax.set(xlabel='Decision Tree', ylabel='Time (ms)')
            plt.xticks([])
            figure = ax.get_figure()
            figure.savefig('Results/Image/' + dataname + '/chart_line_Time_' + dataname + "_" + method + '_DT' + '.png',
                           dpi=72)
            figure.clf()
        if algorithm == 4:
            ax = sns.lineplot(x=data['# epoch'], y=data[nb], data=data)
            ax.set(xlabel='Naive Baves', ylabel='Time (ms)')
            plt.xticks([])
            figure = ax.get_figure()
            figure.savefig('Results/Image/' + dataname + '/chart_line_Time_' + dataname + "_" + method + '_NB' + '.png',
                           dpi=72)
            figure.clf()
        if algorithm == 0:

            '''
            data = [time_svm, time_dt, time_rf, time_nb]

            data = pd.DataFrame({'Lable': ['SVM', 'DT', 'RF', 'NB'],
                                 'SVM' : time_svm,
                                 'DT': time_dt,
                                 'RF': time_rf,
                                 'NB': time_nb
                                })

            barplot_svm = sns.barplot(x='SVM', y=time_svm, label='SVM', data=data)
            barplot_dt = sns.barplot(x='DT', y=time_dt,label='DT', data=data)
            barplot_rf = sns.barplot(x='RF', y=time_rf,label='RF', data=data)
            barplot_nb = sns.barplot(x='NB', y=time_nb,label='NB', data=data)
            barplot_svm.set(xlabel="Algorithm", ylabel='Time (ms)')
            barplot_svm.legend()
            figure = ax.get_figure()
            figure.savefig('Results/Image/'+dataname+'/chart_line_Time_' + dataname + "_" + method + '.png', dpi=72)
            figure.clf()
            data['algorithm'] = np.repeat(np.array(['svm', 'dt', 'rf']), int(len(data) / 3 + 2))[:len(data)]
            '''
            plt.style.use('ggplot')
            plt.bar('SVM', time_svm, label='SVM', color='green')
            plt.bar('RF', time_rf, label='RF', color='red')
            plt.bar('DT', time_dt, label='DT', color='blue')
            plt.bar('NB', time_nb, label='NB', color='yellow')
            plt.legend(['SVM', 'DT', 'RF', 'NB'])
            plt.ylabel('Time (ms)')
            plt.xlabel('Algorithm')
            plt.savefig('Results/Image/' + dataname + '/chart_line_Time_' + dataname + "_" + method + '.png', dpi=72)
            plt.clf()


context = []


def take_display_traning(str):
    context.append(str)


def take_display_traning(str):
    context.append(str)
