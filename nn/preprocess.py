# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike


# Defining preprocessing functions
def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one hot encoding of a list of nucleic acid sequences
    for use as input into a fully connected neural net.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence --- check this in test file!
            length due to the one hot encoding scheme for nucleic acids.

            For example, if we encode 
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    """
    #create a dictionary for each letter
    d = {'A':np.array([1, 0, 0, 0]), 'T':np.array([0, 1, 0, 0]), 'C':np.array([0, 0, 1, 0]), 'G':np.array([0, 0, 0, 1])} 
    encodings = np.empty((len(seq_arr),), dtype=object) #create an array to be by each sequence's encoded array
    seq_counter=0

    #iterate through list of sequences
    for seq in seq_arr:
        #create an empty array for each seq to be filled with each seq letter's encoding
        seq_encoded = np.ones((len(seq),4))

        #go letter by letter storing the current letter's 1x4 array in the larger container array, then flatten that array
        letter_counter = 0
        for letter in seq:
            current_encoding = d[letter] #get the encoding for the current letter
            seq_encoded[letter_counter] = current_encoding #add current encoding to array
            letter_counter+=1 #increment the index counter for the next letter

        flattened_seq = seq_encoded.flatten() #flatten the sequence's encodings into a single array entry
        #convert the flattened seq to a list
        encodings[seq_counter] = list(flattened_seq) #add the sequence's encoded list to the encodings container array
        #increment sequence counter
        seq_counter+=1
    
    return encodings


def sample_seqs(seqs: List[str],labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample your sequences to account for class imbalance. 
    Consider this as a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels #i.e. classification labels where one is the minority label, and one is the majority

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    #get list of indices where labels = False
    indices_0 = [i for i in range(len(seqs)) if labels[i]==False]
    #get list of indices where labels = True
    indices_1 = [i for i in range(len(seqs)) if labels[i]==True]
    
    num_class0 = len(indices_0) #get the number of labels in class 0
    num_class1 = len(seqs) - num_class0 #get the number of labels in class 1
    
    class0 = []
    class1 = []
    oversample_class1 = []
    oversample_class0 = []
    
    #get all the observations that belong to class 0
    for i in range(len(seqs)):
        if labels[i]==False:
            class0.append(seqs[i])
        else:
            class1.append(seqs[i])
    
    #if class0 has many more observations than class 1, oversample the sequences in class 1
    if num_class0>num_class1:
        #sample class1 seqs with replacement
        oversample_class1 = np.random.choice(np.array(class1), size=num_class0, replace=True)
        #join the oversampled seq list and the other class seq list to get a final balanced list of seqs
        sampled_seqs = list(oversample_class1) + class0
        sampled_labels = [True for i in range(len(oversample_class1))] + [False for i in range(len(class0))]
    
    #if class1 has many more observations than class 0, oversample the sequences in class 0
    elif num_class1>num_class0:
        #sample class0 with replacement
        oversample_class0 = np.random.choice(np.array(class0), size=num_class1, replace=True)
        #join the oversampled seq list and the other class seq list to get a final balanced list of seqs
        sampled_seqs = list(oversample_class0) + class1
        sampled_labels = [False for i in range(len(oversample_class0))] + [True for i in range(len(class1))]
        
    elif num_class0==num_class1:
        sampled_seqs = seqs
        sampled_labels = labels
    
    return sampled_seqs, sampled_labels
