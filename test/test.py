# BMI 203 Project 7: Neural Network

# Import necessary dependencies here


# TODO: Write your test functions and associated docstrings below.

def test_forward():
    """
    Unit test for the full forward propagation method
    """
    
    #create a neural network object and dummy data
    nn_arch = [{'input_dim':2,'output_dim':3,'activation':'relu'},
               {'input_dim':3,'output_dim':2,'activation':'sigmoid'}]
    nn = NeuralNetwork(nn_arch,
                      lr=0.01,
                      seed=42,
                      batch_size=15,
                      epochs=100,
                      loss_function = "mean squared error")
    dummy = np.array([[1,2],
                    [3,4],
                    [5,6],
                    [7,8]])
    
    
    #check if the output is what is expected
    output,cache = nn.forward(dummy)
    expected = np.array([[0.48322134, 0.45455472],
                         [0.47916976, 0.43447566],
                         [0.47512091, 0.41460935],
                         [0.47107534, 0.39501693]])
    assert np.allclose(output, expected)==True #numpy.allclose function checks if arrays have the same shape and all values are equal(within a reasonable error amount)
    
    
    #also check that the dimensions are what one would expect. In this case, the output should be a 4x2 array
    assert output.shape[0] == 4
    assert output.shape[1] == 2
    
    #check that A0,A1,A2,Z1,and Z2 are in the cache (i.e., check that the cache has been updated)
    assert ('A0' in cache)==True
    assert ('A1' in cache)==True
    assert ('A2' in cache)==True
    assert ('Z1' in cache)==True
    assert ('Z2' in cache)==True
    assert ('A5' in cache)==False


def test_single_forward():
    """
    Unit test for a single forward pass through one layer of a neural network
    """
    #create a neural network object and dummy data
    nn_arch = [{'input_dim':2,'output_dim':3,'activation':'relu'},
               {'input_dim':3,'output_dim':2,'activation':'sigmoid'}]
    nn = NeuralNetwork(nn_arch,
                      lr=0.01,
                      seed=42,
                      batch_size=15,
                      epochs=100,
                      loss_function = "mean squared error")
    dummy = np.array([[1,2],
                    [3,4],
                    [5,6],
                    [7,8]])
    
    #Get the inputs necessary for _single_forward
    layer_idx = 1
    W1 = nn._param_dict['W' + str(layer_idx)]
    b1 = nn._param_dict['b' + str(layer_idx)]
    activation = "relu"
    
    #check if A1 and Z1 are calculated correctly based on the output of _single_forward
    A1,Z1 = nn._single_forward(W1, b1, dummy, activation)
    
    #check if A1 and Z1 are the right shapes and are filled with the right values using np.allclose
    expectedA1 = np.array([[0.17993984, 0.4461183 , 0.        ],
                           [0.25162981, 0.88026198, 0.        ],
                           [0.32331978, 1.31440566, 0.        ],
                           [0.39500975, 1.74854933, 0.        ]])
    expectedZ1 = np.array([[ 0.17993984,  0.4461183 , -0.11719017],
                           [ 0.25162981,  0.88026198, -0.21084823],
                           [ 0.32331978,  1.31440566, -0.3045063 ],
                           [ 0.39500975,  1.74854933, -0.39816437]])
    
    assert np.allclose(expectedA1, A1)==True
    assert np.allclose(expectedZ1, Z1)==True


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
                      epochs=100,
                      loss_function = "mean squared error")
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
    """
    Unit test for sampling scheme
    """
    #create an example list of seqs and labels
    seqs = ["ACTG", "CTAG", "TGAC", "TAAC", "TGGA"]
    labels = [True, True, True, True, False]

    original_true_count = new_labels.count(True)
    original_false_count = new_labels.count(False)

    new_seqs, new_labels = sample_seqs(seqs,labels)

    #test that the amount of True and False sequences is equal
    assert new_labels.count(True)==new_labels.count(False)
    #test that sampling with replacement happened correctly for the original minority class
    assert new_seqs.count("TGGA")==4
    #check that the final lists are the same length as the initial majority class times 2
    assert len(new_seqs)==original_true_count*2
    assert len(new_labels)==original_true_count*2


    #test that sampling can work the other way around too, no matter which class is imbalanced
    seqs = ["ACTG", "CTAG", "TGAC", "TAAC", "TGGA"]
    labels = [True, False, False, False, False]
    original_true_count = new_labels.count(True)
    original_false_count = new_labels.count(False)
    new_seqs, new_labels = sample_seqs(seqs,labels)

    assert new_labels.count(True)==new_labels.count(False)
    assert new_seqs.count("ACTG")==4
    assert len(new_seqs)==original_false_count*2
    assert len(new_labels)==original_false_count*2
