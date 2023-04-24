import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

if __name__ == '__main__':
    # -------------------------数据读取----------------------------------- #
    data_path = r'C:\Users\zc846\Desktop\咸鱼\闲鱼\code\variant.xlsx'
    dataset = pd.read_excel(data_path, header=0)
    Y = dataset['Critic_rating']
    X = dataset.drop(columns=['Critic_rating'], axis=1)

    Y = np.array(Y)
    X = np.array(X)

    # -------------------------标准化------------------------------------- #
    ss_X = StandardScaler()
    ss_y = StandardScaler()
    X = ss_X.fit_transform(X.reshape(494, 18))
    Y = ss_y.fit_transform(Y.reshape(-1, 1))

    # ---------------------划分测试集与验证集-------------------------------- #, random_state=2
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.1, random_state=1)

    # ------------------------参数寻优------------------------------------- #
    param_grid = [
        {'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
         'gamma':[i for i in range(1, 100, 10)],
         'C':[i for i in range(1, 100, 10)]},
    ]
    svmr = SVR()
    gs = GridSearchCV(svmr, param_grid, cv=4, verbose=2, n_jobs=2, scoring='neg_mean_squared_error')
    search = gs.fit(X_train, y_train)
    svmr_opt = search.best_estimator_

    # ------------------------训练模型------------------------------------- #
    svmr_opt.fit(X_train, y_train)
    svm_y_train = svmr_opt.predict(X_train)
    svm_y_test = svmr_opt.predict(X_test)

    # -------------------------逆标准化------------------------------------ #
    y_train = ss_y.inverse_transform(y_train)
    y_test = ss_y.inverse_transform(y_test)
    rf_y_train = ss_y.inverse_transform(svm_y_train.reshape(-1, 1))
    rf_y_test = ss_y.inverse_transform(svm_y_test.reshape(-1, 1))

    # -----------------------精度评价-------------------------------------- #
    svm_r2_train = r2_score(y_train, svm_y_train)
    svm_r2_test = r2_score(y_test, svm_y_test)
    svm_rmse_train = mean_squared_error(y_train, svm_y_train)
    svm_rmse_test = mean_squared_error(y_test, svm_y_test)
    svm_mae_train = mean_absolute_error(y_train, svm_y_train)
    svm_mae_test = mean_absolute_error(y_test, svm_y_test)
    print('R-squared of training set of random forest model:', svm_r2_train)
    print('R-squared of testing set of random forest model:', svm_r2_test)
    print('RMSE of training set of random forest model:', svm_rmse_train)
    print('RMSE of testing set of random forest model:', svm_rmse_test)
    print('MAE of training set of random forest model:', svm_mae_train)
    print('MAE of testing set of random forest model:', svm_mae_test)
    # y_test = y_test.flatten()

    # --------------------------可视化------------------------------------- #
    plt.figure(1, figsize=(25, 10), dpi=300)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.subplot(121)
    plt.scatter(y_train, rf_y_train)
    plt.xlabel("Measurement value", fontsize=20)
    plt.ylabel("Predictive value", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tick_params(which='major',width=2 , length=8)
    plt.xlim(6, 10)
    plt.ylim(6, 10)
    plt.text(6.2, 9.7, 'R-squared :' + str(round(svm_r2_train, 2)), fontsize=20)
    plt.text(6.2, 9.4, 'RMSE :' + str(round(svm_rmse_train, 2)), fontsize=20)
    plt.text(6.2, 9.1, 'MAE :' + str(round(svm_mae_train, 2)), fontsize=20)
    plt.title('Train set', fontsize=30)

    plt.subplot(122)
    plt.scatter(y_test, rf_y_test)
    plt.xlabel("Measurement value", fontsize=20)
    plt.ylabel("Predictive value", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tick_params(which='major', width=2, length=8)
    plt.xlim(6, 10)
    plt.ylim(6, 10)
    plt.title('Test set', fontsize=30)
    plt.text(6.2, 9.7, 'R-squared :' + str(round(svm_r2_test, 2)), fontsize=20)
    plt.text(6.2, 9.4, 'RMSE :' + str(round(svm_rmse_test, 2)), fontsize=20)
    plt.text(6.2, 9.1, 'MAE :' + str(round(svm_mae_test, 2)), fontsize=20)
    plt.show()