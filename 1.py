import numpy as np

import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.uniform(0, 1, (n_inputs, n_neurons))*0.1
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

class Activation_Step:
    def forward(self, inputs):
        return np.where(inputs > 0, 1, 0)

class Activation_Linear:
    def forward(self, inputs):
        return inputs  # Linear activation is just the identity function

class Activation_Sigmoid:
    def forward(self, inputs):
        return 1 / (1 + np.exp(-inputs))

class Activation_ReLU:
    def forward(self, inputs):
        return np.maximum(0, inputs)

# 데이터 생성
# X, y = spiral_data(samples=2, classes=3)
x = np.linspace(0,2*np.pi,100).reshape(-1,1)
y = np.sin(x)
# 각 활성화 함수 적용
step_activation = Activation_Step()
linear_activation = Activation_Linear()
sigmoid_activation = Activation_Sigmoid()
relu_activation = Activation_ReLU()

# Dense Layer 생성
dense1 = Layer_Dense(1, 8)
dense2 = Layer_Dense(8, 8)
dense3 = Layer_Dense(8, 1)

dense1.weights = np.asarray([[1.0],[2.0],[1.0],[2.0],[1.0],[2.0],[1.0],[2.0]]).T
dense1.biases = np.asarray([1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0])

output = dense1.forward(x)
sigmoid_output = sigmoid_activation.forward(output)
output = dense2.forward(sigmoid_output)
relu_output = relu_activation.forward(output)
output = dense3.forward(relu_output)



#
# # 활성화 함수 적용
# step_output = step_activation.forward(output)
# linear_output = linear_activation.forward(output)


# 결과 출력
# print("Layer Output:\n", layer_output)
# print("Step Activation Output:\n", step_output)
# print("Linear Activation Output:\n", linear_output)
# print("Sigmoid Activation Output:\n", sigmoid_output)
# print("ReLU Activation Output:\n", relu_output)


plt.plot(x,y,color='blue')
plt.plot(x,output,color='red')
plt.show()