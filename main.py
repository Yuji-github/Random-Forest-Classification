import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# splitting dataset
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import StandardScaler

# classifier
from sklearn.ensemble import RandomForestClassifier # for random forest

# confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix

# plotting
from matplotlib.colors import ListedColormap # for plot


def RFC():
    # import data
    dataset = pd.read_csv('Social_Network_Ads.csv')

    # independent variables
    independent = dataset.iloc[:, :-1].values

    # dependent variable
    dependent = dataset.iloc[:, -1].values

    # print(independent, dependent)

    # plotting with the variables
    # plt.scatter(independent[:, 0], dependent, color='red')
    # plt.title('Before training')
    # plt.xlabel('Age')
    # plt.ylabel('Estimated Salary')
    # plt.legend()
    # plt.show()

    # splitting dataset into 4 parts 300 customers go to train
    x_train, x_test, y_train, y_test = train_test_split(independent, dependent, train_size=0.75, random_state=0)

    # feature scaling
    # this is not necessary to do and teh results won't be changed
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)

    # classifiler
    # criterion='entropy' is for gathering info
    classifier = RandomForestClassifier(n_estimators=500, criterion='entropy', random_state=0)
    classifier.fit(x_train, y_train)

    # prediction
    print('The prediction is %.2f' % classifier.predict(sc.transform([[30, 87000]])))
    if int(classifier.predict(sc.transform([[30, 87000]])) == 1):
        print("This person will buy the product")
    else:
        print("This person will NOT buy the product")

    # confusion matrix
    cm = confusion_matrix(y_true=y_test, y_pred=classifier.predict(x_test))
    print(cm)

    print('\nThe correctness is %.2f percent' % accuracy_score(y_true=y_test, y_pred=classifier.predict(x_test)))

    # taking several min to plot
    # because a lot of calculations behind the code

    # Train set visualization
    x_set, y_set = sc.inverse_transform(x_train), y_train
    X1, X2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 10, stop=x_set[:, 0].max() + 10, step=1),
                         np.arange(start=x_set[:, 1].min() - 1000, stop=x_set[:, 1].max() + 1000, step=1))
    plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
    plt.title('Random Forest with Entropy (Training set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    RFC()
