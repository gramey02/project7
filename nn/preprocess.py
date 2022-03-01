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
    d = {'A':np.array([1, 0, 0, 0]), 'T':np.array([0, 1, 0, 0]), 'C':np.array([0, 0, 1, 0]), 'G':np.array([0, 0, 0, 1])} #create a dictionary for each letter
    encodings = np.empty((len(seq_arr),1), dtype=object) #create an array to be by each sequence's encoded array
    seq_counter=0

    #iterate through list of sequences
    for seq in seq_arr:
        #create an empty array for each seq to be filled with each seq letter's encoding
        seq_encoded = np.ones((len(seq),4), dtype=np.int8)

        #go letter by letter storing the current letter's 1x4 array in the larger container array, then flatten that array
        letter_counter = 0
        for letter in seq:
            current_encoding = d[letter] #get the encoding for the current letter
            seq_encoded[letter_counter] = current_encoding #add current encoding to array
            letter_counter+=1 #increment the index counter for the next letter

        flattened_seq = seq_encoded.flatten() #flatten the sequence's encodings into a single array entry
        encodings[seq_counter,0] = flattened_seq #add the sequence's encoded array to the encodings container array
        #increment sequence counter
        seq_counter+=1
    
    return encodings


def sample_seqs(
        seqs: List[str]
        labels: List[bool]) -> Tuple[List[seq], List[bool]]:
    """
    This function should sample your sequences to account for class imbalance. 
    Consider this as a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    pass
