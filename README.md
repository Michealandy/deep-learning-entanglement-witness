# deep-learning-entanglement-witness

I took the version of deep learning algorithm for image recognition and adapted to recognize a property of quantum systems called  "Entanglement". 

Using the Python library QuTIP, density matrices describing quantum states were randomly generated in order to create the necessary dataset. The class of states analized were 2-qubit states as well as qubit-qutrit states.

For the 2-qubit case, each density operator is a 4x4 complex matrix. Decoupling real and imaginary parts of each matrix element in two different columns, we are left with a set of 4x8 matrices. Therefore, the 2-qubit case has 32 features as input and the output is a binary function (in terms of one-hot encode: [1,0] is connected to the quantum state being entangled; [0,1] is connected to the quantum state being not entangled, or "separable").

For the qubit-qutrit case, the original matrix is a 6x6 complex matrix. Each decoupled matrix then is a 6x12 matrix, leading to 72 features generating the same binary response in the output.

In the deep learning regime using deep neural networks (DNN), one hidden layer do the trick. The DNN was written also using the Python library TensorFlow. Our whole dataset has 5000 states, where 4000 states (80%) were randomly selected to the train set and the 1000 remaining states (20%) were designated as test set.

For the 2-qubit case, a total of 32 hidden neurons, 1000 epochs and a batch size of 4 gives an accuracy usually above 98%, depending of the random initial conditions for the weights and biases. Also it was used Gradient Descent Optmizer, Lambda regularization, and simple sum distace as the loss function. For the qubit-qutrit case, a total of 50 hidden neurons, and 2000 epochs give an accuracy usually above 99%. For both cases the total loss in the end is still very high, despite decresing during the whole learning process.
