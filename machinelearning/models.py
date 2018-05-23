import numpy as np

import backend
import nn


class Model(object):
    """Base model class for the different applications"""

    def __init__(self):
        self.get_data_and_monitor = None
        self.learning_rate = 0.0

    def run(self, x, y=None):
        raise NotImplementedError("Model.run must be overriden by subclasses")

    def train(self):
        """
        Train the model.

        `get_data_and_monitor` will yield data points one at a time. In between
        yielding data points, it will also monitor performance, draw graphics,
        and assist with automated grading. The model (self) is passed as an
        argument to `get_data_and_monitor`, which allows the monitoring code to
        evaluate the model on examples from the validation set.
        """
        for x, y in self.get_data_and_monitor(self):
            graph = self.run(x, y)
            graph.backprop()
            graph.step(self.learning_rate)


class RegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        self.learning_rate = 0.022
        self.hidden_size = 200
        self.input_dim = 1
        self.m1 = nn.Variable(self.input_dim, self.hidden_size)
        self.b1 = nn.Variable(1, self.hidden_size)
        self.m2 = nn.Variable(self.hidden_size, 1)
        self.b2 = nn.Variable(1, 1)


    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """

        graph = nn.Graph([self.m1, self.b1, self.m2, self.b2])
        input_x = nn.Input(graph, x)

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.

            input_y = nn.Input(graph, y)
            xm = nn.MatrixMultiply(graph, input_x, self.m1)
            xm_plus_b = nn.MatrixVectorAdd(graph, xm, self.b1)
            loss_1 = nn.ReLU(graph, xm_plus_b)
            loss_1m = nn.MatrixMultiply(graph, loss_1, self.m2)
            loss_1m_plus_b = nn.MatrixVectorAdd(graph, loss_1m, self.b2)
            loss = nn.SquareLoss(graph, loss_1m_plus_b, input_y)
            return graph

        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array

            xm = nn.MatrixMultiply(graph, input_x, self.m1)
            xm_plus_b = nn.MatrixVectorAdd(graph, xm, self.b1)
            loss_1 = nn.ReLU(graph, xm_plus_b)
            loss_1m = nn.MatrixMultiply(graph, loss_1, self.m2)
            loss_1m_plus_b = nn.MatrixVectorAdd(graph, loss_1m, self.b2)
            return graph.get_output(loss_1m_plus_b)


class OddRegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers.

    Unlike RegressionModel, the OddRegressionModel must be structurally
    constrained to represent an odd function, i.e. it must always satisfy the
    property f(x) = -f(-x) at all points during training.
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        self.learning_rate = 0.04
        self.hidden_size = 200
        self.input_dim = 1
        self.m1 = nn.Variable(self.input_dim, self.hidden_size)
        self.b1 = nn.Variable(1, self.hidden_size)
        self.m2 = nn.Variable(self.hidden_size, 1)
        self.b2 = nn.Variable(1, 1)

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        graph = nn.Graph([self.m1, self.b1, self.m2, self.b2])
        input_x_pos = nn.Input(graph, x)
        input_x_neg = nn.Input(graph, x * [-1])
        negative_one = nn.Input(graph, np.array([[-1.0]]))

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            input_y = nn.Input(graph, y)

            xm_pos_pos = nn.MatrixMultiply(graph, input_x_pos, self.m1)
            xm_pos_neg = nn.MatrixMultiply(graph, input_x_neg, self.m1)

            xm_plus_b_pos = nn.MatrixVectorAdd(graph, xm_pos_pos, self.b1)
            xm_plus_b_neg = nn.MatrixVectorAdd(graph, xm_pos_neg, self.b1)

            loss_1_pos = nn.ReLU(graph, xm_plus_b_pos)
            loss_1_neg = nn.ReLU(graph, xm_plus_b_neg)

            loss_1m_pos = nn.MatrixMultiply(graph, loss_1_pos, self.m2)
            loss_1m_neg = nn.MatrixMultiply(graph, loss_1_neg, self.m2)

            loss_1m_plus_b_pos = nn.MatrixVectorAdd(graph, loss_1m_pos, self.b2)
            loss_1m_plus_b_neg = nn.MatrixVectorAdd(graph, loss_1m_neg, self.b2)

            negate = nn.MatrixMultiply(graph, loss_1m_plus_b_neg, negative_one)
            final = nn.Add(graph, loss_1m_plus_b_pos, negate)

            loss = nn.SquareLoss(graph, final, input_y)
            return graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            xm_pos_pos = nn.MatrixMultiply(graph, input_x_pos, self.m1)
            xm_pos_neg = nn.MatrixMultiply(graph, input_x_neg, self.m1)

            xm_plus_b_pos = nn.MatrixVectorAdd(graph, xm_pos_pos, self.b1)
            xm_plus_b_neg = nn.MatrixVectorAdd(graph, xm_pos_neg, self.b1)

            loss_1_pos = nn.ReLU(graph, xm_plus_b_pos)
            loss_1_neg = nn.ReLU(graph, xm_plus_b_neg)

            loss_1m_pos = nn.MatrixMultiply(graph, loss_1_pos, self.m2)
            loss_1m_neg = nn.MatrixMultiply(graph, loss_1_neg, self.m2)

            loss_1m_plus_b_pos = nn.MatrixVectorAdd(graph, loss_1m_pos, self.b2)
            loss_1m_plus_b_neg = nn.MatrixVectorAdd(graph, loss_1m_neg, self.b2)

            negate = nn.MatrixMultiply(graph, loss_1m_plus_b_neg, negative_one)
            final = nn.Add(graph, loss_1m_plus_b_pos, negate)
            return graph.get_output(final)



class DigitClassificationModel(Model):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_digit_classification

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        self.learning_rate = 0.3
        self.hidden_size = 200
        self.xinput_dim = 784
        self.yinput_dim = 10
        self.m1 = nn.Variable(self.xinput_dim, self.hidden_size)
        self.b1 = nn.Variable(1, self.hidden_size)
        self.m2 = nn.Variable(self.hidden_size, self.yinput_dim)
        self.b2 = nn.Variable(1, self.yinput_dim)

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 10) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should predict a (batch_size x 10) numpy array of scores,
        where higher scores correspond to greater probability of the image
        belonging to a particular class. You should use `nn.SoftmaxLoss` as your
        training loss.

        Inputs:
            x: a (batch_size x 784) numpy array
            y: a (batch_size x 10) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 10) numpy array of scores (aka logits)
        """
        graph = nn.Graph([self.m1, self.b1, self.m2, self.b2])
        input_x = nn.Input(graph, x)

        if y is not None:
            input_y = nn.Input(graph, y)
            xm = nn.MatrixMultiply(graph, input_x, self.m1)
            xm_plus_b = nn.MatrixVectorAdd(graph, xm, self.b1)
            loss_1 = nn.ReLU(graph, xm_plus_b)
            loss_1m = nn.MatrixMultiply(graph, loss_1, self.m2)
            loss_1m_plus_b = nn.MatrixVectorAdd(graph, loss_1m, self.b2)
            nn.SoftmaxLoss(graph, loss_1m_plus_b, input_y)
            return graph
        else:
            xm = nn.MatrixMultiply(graph, input_x, self.m1)
            xm_plus_b = nn.MatrixVectorAdd(graph, xm, self.b1)
            loss_1 = nn.ReLU(graph, xm_plus_b)
            loss_1m = nn.MatrixMultiply(graph, loss_1, self.m2)
            loss_1m_plus_b = nn.MatrixVectorAdd(graph, loss_1m, self.b2)
            return graph.get_output(loss_1m_plus_b)


class DeepQModel(Model):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.

    (We recommend that you implement the RegressionModel before working on this
    part of the project.)
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_rl

        self.num_actions = 2
        self.state_size = 4

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        self.learning_rate = 0.01
        self.hidden_size = 100
        self.xinput_dim = self.state_size
        self.yinput_dim = self.num_actions
        self.m1 = nn.Variable(self.xinput_dim, self.hidden_size)
        self.b1 = nn.Variable(1, self.hidden_size)
        self.m2 = nn.Variable(self.hidden_size, self.yinput_dim)
        self.b2 = nn.Variable(1, self.yinput_dim)

    def run(self, states, Q_target=None):
        """
        Runs the DQN for a batch of states.

        The DQN takes the state and computes Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]

        When Q_target == None, return the matrix of Q-values currently computed
        by the network for the input states.

        When Q_target is passed, it will contain the Q-values which the network
        should be producing for the current states. You must return a nn.Graph
        which computes the training loss between your current Q-value
        predictions and these target values, using nn.SquareLoss.

        Inputs:
            states: a (batch_size x 4) numpy array
            Q_target: a (batch_size x 2) numpy array, or None
        Output:
            (if Q_target is not None) A nn.Graph instance, where the last added
                node is the loss
            (if Q_target is None) A (batch_size x 2) numpy array of Q-value
                scores, for the two actions
        """
        graph = nn.Graph([self.m1, self.b1, self.m2, self.b2])
        input_x = nn.Input(graph, states)

        if Q_target is not None:
            input_y = nn.Input(graph, Q_target)
            xm = nn.MatrixMultiply(graph, input_x, self.m1)
            xm_plus_b = nn.MatrixVectorAdd(graph, xm, self.b1)
            loss_1 = nn.ReLU(graph, xm_plus_b)
            loss_1m = nn.MatrixMultiply(graph, loss_1, self.m2)
            loss_1m_plus_b = nn.MatrixVectorAdd(graph, loss_1m, self.b2)
            nn.SquareLoss(graph, loss_1m_plus_b, input_y)
            return graph
        else:
            xm = nn.MatrixMultiply(graph, input_x, self.m1)
            xm_plus_b = nn.MatrixVectorAdd(graph, xm, self.b1)
            loss_1 = nn.ReLU(graph, xm_plus_b)
            loss_1m = nn.MatrixMultiply(graph, loss_1, self.m2)
            loss_1m_plus_b = nn.MatrixVectorAdd(graph, loss_1m, self.b2)
            return graph.get_output(loss_1m_plus_b)

    def get_action(self, state, eps):
        """
        Select an action for a single state using epsilon-greedy.

        Inputs:
            state: a (1 x 4) numpy array
            eps: a float, epsilon to use in epsilon greedy
        Output:
            the index of the action to take (either 0 or 1, for 2 actions)
        """
        if np.random.rand() < eps:
            return np.random.choice(self.num_actions)
        else:
            scores = self.run(state)
            return int(np.argmax(scores))


class LanguageIDModel(Model):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_lang_id

        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        self.learning_rate = 0.3
        self.hidden_size = 200

        self.xinput_dim = self.num_chars
        self.yinput_dim = len(self.languages)

        self.C_training = nn.Variable(self.xinput_dim, self.hidden_size)
        self.H_traing = nn.Variable(self.hidden_size, self.hidden_size)

        self.m = nn.Variable(self.hidden_size, self.yinput_dim)
        self.b = nn.Variable(1, self.yinput_dim)

    def run(self, xs, y=None):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        (batch_size x self.num_chars) numpy array, where every row in the array
        is a one-hot vector encoding of a character. For example, if we have a
        batch of 8 three-letter words where the last word is "cat", we will have
        xs[1][7,0] == 1. Here the index 0 reflects the fact that the letter "a"
        is the inital (0th) letter of our combined alphabet for this task.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 5) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node that represents a (batch_size x hidden_size)
        array, for your choice of hidden_size. It should then calculate a
        (batch_size x 5) numpy array of scores, where higher scores correspond
        to greater probability of the word originating from a particular
        language. You should use `nn.SoftmaxLoss` as your training loss.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a (batch_size x self.num_chars) numpy array
            y: a (batch_size x 5) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 5) numpy array of scores (aka logits)

        Hint: you may use the batch_size variable in your code
        """
        batch_size = xs[0].shape[0]
        graph = nn.Graph([self.C_training, self.H_traing, self.m, self.b])

        H = np.zeros((batch_size, self.hidden_size))
        inputH = nn.Input(graph, H)
        for X in xs:
            inputX = nn.Input(graph, X)
            CWx = nn.MatrixMultiply(graph, inputX, self.C_training)
            HWh = nn.MatrixMultiply(graph, inputH, self.H_traing)
            inputH = nn.ReLU(graph, nn.Add(graph, CWx, HWh))

        xm = nn.MatrixMultiply(graph, inputH, self.m)
        xm_plus_b = nn.MatrixVectorAdd(graph, xm, self.b)

        if y is not None:
            input_y = nn.Input(graph, y)
            nn.SquareLoss(graph, xm_plus_b, input_y)
            return graph
        else:
            return graph.get_output(xm_plus_b)
