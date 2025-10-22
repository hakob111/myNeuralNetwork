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
        self.we_hid_to_out = init_weights(self.outnodes, self.hidnodes, self.outnodes)

    def train(self, inputs_list, targets_list):
        """This is a train function for training model
            Here the error is back-propagated through the weight values.
            using differential equations """
        inputs = torch.tensor(inputs_list, dtype=torch.float32).unsqueeze(0).T
        targets = torch.tensor(targets_list, dtype=torch.float32).unsqueeze(0).T
        hidden_inputs = torch.matmul(self.we_in_to_hid, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = torch.matmul(self.we_hid_to_out, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs
        hidden_errors = torch.matmul(self.we_hid_to_out.T, output_errors)
        # sigmoid * (1 -sigmoid) error propagation gradient
        self.we_hid_to_out += self.lr * torch.matmul((output_errors * final_outputs * (1.0 - final_outputs)),
                                                    torch.transpose(hidden_outputs))
        self.we_in_to_hid += self.lr * torch.matmul((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                                    torch.transpose(inputs))

    def query(self, input_list):
        """ it is a prediction function """
        inputs = torch.tensor(input_list, dtype=torch.float32).reshape(-1, 1)
        hidden_inputs = torch.matmul(self.we_in_to_hid, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = torch.matmul(self.we_hid_to_out, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
