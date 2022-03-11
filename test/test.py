# BMI 203 Project 7: Neural Network

# Import necessary dependencies here
from sklearn.datasets import load_digits
import numpy as np

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
    """
    Unit test for backpropagation through a single layer of a neural network
    """
    
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

    y_hat, cache = nn.forward(dummy)

    dW1_expected = np.array([[16., 20.],
                             [16., 20.],
                             [ 0.,  0.]])
    db1_expected = np.array([[4.],
                             [4.],
                             [0.]])

    last_layer = 0
    curr_layer = last_layer+1
    param_dict = nn._param_dict
    layer = {'input_dim':2,'output_dim':3,'activation':'relu'}

    dA_prev = nn._mean_squared_error_backprop(dummy, y_hat)
    dA_curr = dA_prev

    #get the inputs necessary for _single_backprop method
    curr_activation = layer["activation"] #activation function of the current layer
    A_prev = cache["A"+str(last_layer)] #inputs of the previous layer
    Z_curr = cache["Z"+str(curr_layer)] #transformed inputs of the current layer
    W_curr = param_dict["W"+str(curr_layer)] #weights of the current layer
    b_curr = param_dict["b"+str(curr_layer)] #bias terms of the current layer

    #single backprop through current layer
    dA_prev, dW_curr, db_curr = nn._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, curr_activation)

    #check that derivatives equal what they should
    assert np.allclose(db1_expected, db_curr)
    assert np.allclose(dW_curr, dW1_expected)

    #check that dA_prev, dA_curr, and A_prev are all the same dimensions
    assert dA_prev.shape==dA_curr.shape
    assert dA_curr.shape==A_prev.shape
    assert dA_curr.shape==dA_prev.shape

    #same for dW_curr, W_curr and db_curr, b_curr
    assert dW_curr.shape==W_curr.shape
    assert db_curr.shape==b_curr.shape
    
    
    

def test_predict():
    """
    Unit test for the predict function
    """
    #use autoencoder to get some outputs to check
    digits = load_digits().data
    
    nn_arch = [{'input_dim': 64, 'output_dim': 16, 'activation': 'relu'},
               {'input_dim': 16, 'output_dim': 64, 'activation': 'relu'}]
    
    ae = NeuralNetwork(nn_arch,
                   lr = 1e-7,#0.0000001,
                   seed=42,
                   batch_size = 10,
                   epochs=3000,
                   loss_function = "mean squared error")
    #create a held-out set for future accuracy calculation
    train_val, held_out = train_test_split(digits, test_size=0.1, random_state=42)
    #use a 70/30 train/test split of the remaining data
    X_train, X_val = train_test_split(train_val, test_size=0.3, random_state=42)
    
    #train the model
    per_epoch_loss_train, per_epoch_loss_val = ae.fit(X_train, X_train, X_val, X_val)
    
    
    #now that the weights have been updated, predict based on the held out set
    y_hat = ae.predict(held_out)
    
    #spot check y_hat to make sure it was caculated correctly
    y5_expected = np.array([ 2.6156432 ,  4.23995345,  4.87994946,  0.        ,  5.38035388,
        8.06807737,  0.        ,  8.64782441,  0.16946276,  5.67471117,
        3.04991667,  8.52799572,  3.96517504,  1.57155148,  4.22811623,
        3.15698586,  5.45570063,  0.        ,  6.59736653,  0.        ,
        0.        ,  8.70898651,  3.95407468,  5.96662173,  5.01333359,
        6.87866524,  3.20624718,  0.        , 10.93008802,  7.13986826,
        0.        ,  0.91371885,  6.11734777,  0.77879442,  0.60782059,
        7.39628404,  7.46901678,  4.37257994,  4.57852723,  7.11095388,
        2.57955955,  4.5859205 ,  6.53739168,  5.54173975,  6.86636608,
        0.        ,  0.        ,  0.        ,  0.        ,  4.78575943,
        4.41724675,  0.        ,  2.77665245,  3.57300446,  0.        ,
        1.74362348,  6.08310471,  4.63253425,  0.        ,  2.50386765,
        0.        ,  0.        ,  7.23852294,  0.        ])
    
    y29_expected = np.array([ 3.84997801,  3.19548952,  1.97788815,  0.        ,  6.85300661,
       10.1092117 ,  0.        , 13.15127903,  1.39502083,  7.42629281,
        6.94884123,  7.74237572,  4.10536826,  0.97183991,  3.37217776,
        0.336039  ,  5.99783348,  0.        ,  7.67898466,  0.        ,
        0.        ,  6.90031994,  3.88116713,  6.33716956,  4.88545874,
        5.43475982,  3.70293359,  0.        , 10.1781509 ,  8.48304368,
        0.        ,  0.30650027,  5.54485423,  2.86410505,  3.92793687,
        9.73157899,  7.28636658,  4.46775585,  5.1328326 ,  7.59249288,
        3.7407753 ,  2.52011064,  6.13836064,  4.06839628,  7.10802398,
        0.        ,  0.        ,  0.        ,  0.        ,  4.19940942,
        2.88414292,  0.        ,  6.62987895,  6.0187832 ,  0.        ,
        5.03092313,  7.20403264,  3.19994329,  0.        ,  1.08682258,
        0.        ,  0.        ,  8.39485858,  0.        ])

    assert np.allclose(y_hat[5],y5_expected)==True
    assert np.allclose(y_hat[29],y29_expected)==True
    
    #make sure that y_hat is the same size as the input, since this is an autoencoder case
    assert held_out.shape==y_hat.shape
    
    #check that the predict output is equal to what the forward function would give you after training had occurred
    forward_output, cache = ae.forward(held_out)
    predict_output = ae.predict(held_out)
    assert np.allclose(forward_output, predict_output)


def test_binary_cross_entropy():
    """
    Unit test for binary cross entropy function
    """
    
    pass


def test_binary_cross_entropy_backprop():
    #check that it gives you a vector
    pass


def test_mean_squared_error():
    digits = load_digits().data
    nn_arch = [{'input_dim': 64, 'output_dim': 16, 'activation': 'relu'},
               {'input_dim': 16, 'output_dim': 64, 'activation': 'relu'}]
    ae = NeuralNetwork(nn_arch,
                   lr = 1e-7,#0.0000001,
                   seed=42,
                   batch_size = 10,
                   epochs=3000,
                   loss_function = "mean squared error")
    #create a held-out set
    train_val, held_out = train_test_split(digits, test_size=0.1, random_state=42)
    #use a 70/30 train/test split of the remaining data
    X_train, X_val = train_test_split(train_val, test_size=0.3, random_state=42)
    #train the model
    per_epoch_loss_train, per_epoch_loss_val = ae.fit(X_train, X_train, X_val, X_val)
    
    #now that the weights have been updated, predict based on the held out set
    y_hat = ae.predict(held_out)
    
    #check that mse function gives you a scalar even for two multi-dimensional inputs, and that the scalar is close to what is expected
    assert type(ae._mean_squared_error(held_out, y_hat))==np.float64
    assert np.isclose(ae._mean_squared_error(held_out, y_hat),21.899199329351575)
    
    


def test_mean_squared_error_backprop():
    #check that it gives you a vector
    pass


def test_one_hot_encode():
    """
    Unit test for function that generates one hot encodings
    """
    #create a dummy seq_arr
    seq_arr = ["ACTGATGCAT","AGAT", "TCGAGTC"]
    #get the one hot encodings
    encodings = one_hot_encode_seqs(seq_arr)
    #assert that the encoding is 4x as long as the original sequence
    assert len(encodings[0])==4*len(seq_arr[0])

    seq_arr = ["AGA"]
    encoded = one_hot_encode_seqs(seq_arr)
    expected = [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    assert expected == list(encoded[0])


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
