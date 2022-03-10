# BMI 203 Project 7: Neural Network

# Import necessary dependencies here


# TODO: Write your test functions and associated docstrings below.

def test_forward():
    #ensure that dimensions are correct, that the cache and output are the appropriate length/dimensions (respectively)
    #ensure the output equals what you expect (array equal protocol)
    
    #create a neural network object and dummy data
    nn_arch = [{'input_dim':2,'output_dim':3,'activation':'relu'},
               {'input_dim':3,'output_dim':2,'activation':'sigmoid'}]
    nn = NeuralNetwork(nn_arch,
                      lr=0.01,
                      seed=42,
                      batch_size=15,
                      epochs=100)
    dummy = np.array([[1,2],
                    [3,4],
                    [5,6],
                    [7,8]])
    #check if the output is what is expected
    output,cache = nn.forward(dummy)
    
    #insert expected output for the forward pass (prior to training)
    expected = np.array([[0.48322134, 0.45455472],
                         [0.47916976, 0.43447566],
                         [0.47512091, 0.41460935],
                         [0.47107534, 0.39501693]])
    
    comparison = output==expected
    arrays_are_equal = comparison.all() #compares all array values to make sure they are equal at the same indices
    assert arrays_are_equal==True
    
    #also check that the dimensions are what one would expect. In this case, the output should be a 4x2 array
    assert output.shape[0] == 4
    assert output.shape[1] == 2
    
    #check that A0,A1,A2,Z1,and Z2 are in the cache (i.e., the cache has been updated)
    assert ('A0' in cache)==True
    assert ('A1' in cache)==True
    assert ('A2' in cache)==True
    assert ('Z1' in cache)==True
    assert ('Z2' in cache)==True
    assert ('A5' in cahce)==False


def test_single_forward():
    
    #create a neural network object and dummy data
    nn_arch = [{'input_dim':2,'output_dim':3,'activation':'relu'},
               {'input_dim':3,'output_dim':2,'activation':'sigmoid'}]
    nn = NeuralNetwork(nn_arch,
                      lr=0.01,
                      seed=42,
                      batch_size=15,
                      epochs=100)
    dummy = np.array([[1,2],
                    [3,4],
                    [5,6],
                    [7,8]])
    
    #Get the inputs necessary for _single_forward
    layer_idx = 1
    W1 = param_dict['W' + str(layer_idx)]
    b1 = param_dict['b' + str(layer_idx)]
    activation = "relu"
    
    #check if A1 and Z1 are calculated correctly based on the output of _single_forward
    A1,Z1 = nn._single_forward(W1, b1, dummy, activation)
    
    expectedA1 = np.array() #insert expected A1 and Z1
    expectedZ1 = np.array()
    
    comparison1 = expectedA1==A1
    comparison2 = expectedA2==A2
    
    arrays_are_equal1 = comparison1.all()
    arrays_are_equal2 = comparison2.all()
    
    assert arrays_are_equal1==True
    assert arrays_are_equal2==True


def test_single_backprop():

    pass


def test_predict():
    #create a dummy neural network and dummy data
    nn_arch = [{'input_dim':2,'output_dim':3,'activation':'relu'},
               {'input_dim':3,'output_dim':2,'activation':'sigmoid'}]
    nn = NeuralNetwork(nn_arch,
                      lr=0.01,
                      seed=42,
                      batch_size=15,
                      epochs=100)
    dummy = np.array([[1,2],
                    [3,4],
                    [5,6],
                    [7,8]])
    
    #train the model on the dummy data
    per_epoch_loss_train, per_epoch_loss_val = nn.fit(X_train, y_train, X_test, y_test)
    #now that the weights have been updated, predict based on a held out set
    y_hat,cache = nn.predict(held_out)
    
    #assert that the observed and expected y_hat arrays are equal
    expected_yhat = np.array()
    comparison = expected_yhat==y_hat
    arrays_are_equal=comparison.all()
    assert arrays_are_equal==True
    
    #assert that the accuracy is within a certain range
    assert np.mean(y-y_hat)<###
    assert np.mean(y-y_hat)>###


def test_binary_cross_entropy():
    #check that is gives you a scalar
    pass


def test_binary_cross_entropy_backprop():
    #check that it gives you a vector
    pass


def test_mean_squared_error():
    #check that it gives you a scalar
    pass


def test_mean_squared_error_backprop():
    #check that it gives you a vector
    pass


def test_one_hot_encode():
    #create a dummy seq_arr
    seq_arr = ["ACTGATGCAT","AGAT", "TCGAGTC"]
    #get the one hot encodings
    encodings = one_hot_encode_seq(seq_arr)
    #assert that the encoding is 4x as long as the original sequence
    assert len(encodings[0])==4*len(seq_arr[0])
    #assert that the third sequence was properly encoded, to spot-check
    expected = [] #insert expected encoding
    assert expected = encodings[2]


def test_sample_seqs():
    #test that after balancing, the labels vector should have the same number of trues and falses
    #use list.count to see if the output labels vector of sample_seqs has equal numbers of true and false
    #check that it works in both the class0>class1 and class1>class0 case
    new_labels.count(True)
    new_labels.count(False)
    pass
