import reg
import numpy as np
import random
import matplotlib.pyplot as plt

def lin_test(graph = False):

    data_x = [[x+1] for x in range(10)]
    data_y = []
    for n in range(len(data_x)):
        data_y.append([(random.random() - 0.5) + 3*data_x[n][0] + 7])

    # Testing Init
    layer1 = reg.Lin_reg_layer(1, 1, bias_node=False)
    layer2 = reg.Lin_reg_layer(21, 34)
    layer3 = reg.Lin_reg_layer(1, 1)

    # Testing new_coef and _add_bias
    assert np.shape(layer1.coef_ret()) == (1, 1)
    assert np.shape(layer2.coef_ret()) == (21+1, 34)

    # Testing feed_for
    assert np.shape(layer3.feed_for(data_x)) == np.shape(data_y)

    # Testing grad_descent and cost
    cost = []
    for n in range(10000):
        layer3.grad_descent(data_x, data_y, 0.01)
        cost.append(layer3.cost(data_x, data_y))
    if graph == True:
        plt.plot(cost)
        plt.ylabel("cost"), plt.xlabel("iterations")
        plt.show()

    #According to how the elements in data_y was created, the coeficients should be close to 3 and 7
    assert np.matmul(np.transpose(layer3.coef_ret()-[[7], [3]]),
                     layer3.coef_ret()-[[7], [3]]) < 0.2

def log_test(graph = False):

    data_x = [[x+1] for x in range(10)]
    data_y = [[0], [0], [0], [0], [1], [1], [0], [1], [1], [1]] #Larger value of x, the more likely y=1

    # Testing Init
    layer1 = reg.Log_reg_layer(1, 1, bias_node=False)
    layer2 = reg.Log_reg_layer(21, 34)
    layer3 = reg.Log_reg_layer(1, 1)

    # Testing new_coef and _add_bias
    assert np.shape(layer1.coef_ret()) == (1, 1)
    assert np.shape(layer2.coef_ret()) == (21+1, 34)

    # Testing sigmoid
    assert layer1.sigmoid(0) == 0.5
    assert layer1.sigmoid(2) - 0.8807970779778824440597 < 0.0000001
    assert layer1.sigmoid(-1) - 0.2689414213699951207488 < 0.0000001



    # Testing feed_for
    assert np.shape(layer3.feed_for(data_x)) == np.shape(data_y)

    # Testing grad_descent and cost
    cost = []
    for n in range(10000):
        layer3.grad_descent(data_x, data_y, 0.01)
        cost.append(layer3.cost(data_x, data_y))

    if graph == True:
        plt.plot(cost)
        plt.ylabel("cost"), plt.xlabel("iterations")
        plt.show()

    assert np.mean(layer3.feed_for([[1],[2],[3],[4],[5]])) < 0.5
    assert np.mean(layer3.feed_for([[6],[7],[8],[9],[10]])) > 0.5


def main():
    lin_test(True)
    log_test(True)


if __name__ == "__main__":
    main()
