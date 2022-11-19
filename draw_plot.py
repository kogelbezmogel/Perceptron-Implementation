import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    csv_Cdata = pd.read_csv('./data_files/data1.csv')


    file = open('./data_files/ClassPerc.4h')
    weights = [ float(w) for w in file.readline().split() ]
    bias = float( file.readline() )
    file.close()

    a1 = weights[0]
    a2 = weights[1]
    fun = lambda x: (-a1/a2)*x + (0.5 - bias)/a2
    x = [ x/100 for x in range(500, 1200)]
    y = [ fun(x/100) for x in range(500, 1200) ]

    # vs real model
    X = csv_Cdata[['X', 'Y']]
    Y = csv_Cdata['class']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    
    model = Sequential()
    model.add( Dense(1, activation='sigmoid') )
    model.compile( loss='binary_crossentropy', optimizer='adam' )
    model.fit(x_train, y_train, epochs=1000, batch_size=32)
    weights = model.get_weights()    
    print( f"weights: {weights}")

    a1 = weights[0][0]
    a2 = weights[0][1]
    bias = weights[1]
    fun = lambda x: (-a1/a2)*x + (0.5 - bias)/a2
    x_2 = [ x/100 for x in range(500, 1200)]
    y_2 = [ fun(x/100) for x in range(500, 1200) ]

    sns.scatterplot(data=csv_Cdata, x='X', y='Y', hue='class', alpha=0.5, size=1, palette='RdBu')
    plt.plot(x, y, color='red')
    plt.plot(x_2, y_2, color='green')

    plt.show()
    