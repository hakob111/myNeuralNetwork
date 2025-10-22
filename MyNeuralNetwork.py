import torch
import torch.nn.init as init
import sklearn


class MyNeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hidnodes = hiddennodes
        self.outnodes = outputnodes
        self.lr = learningrate
        self.activation_function = torch.sigmoid

        # Weights with random values

        def init_weights(rows, columns, nod_count):
            """Function for init values of weights to avoid gradient saturation 📉"""
            st_dev = nod_count ** (-0.5)  # square root from the count of input nodes

            weights = torch.empty(rows, columns)
            init.normal_(weights, mean=0.0, std=st_dev)
            return weights

        self.we_in_to_hid = init_weights(self.hidnodes, self.inodes, self.hidnodes)
        self.we_hid_to_out = torch.rand(self.outnodes, self.hidnodes, self.outnodes)
        pass

    def train(self):
        pass

    def query(self, input_list):
        inputs = torch.tensor(input_list, dtype=torch.float32).reshape(-1, 1)
        hidden_inputs = torch.matmul(self.we_in_to_hid, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = torch.matmul(self.we_hid_to_out, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
