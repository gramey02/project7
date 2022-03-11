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
    d = {'A':[1, 0, 0, 0], 'T':[0, 1, 0, 0], 'C':[0, 0, 1, 0], 'G':[0, 0, 0, 1]} 
    encodings = [] #list to be filled with the encoding of each sequence

    #iterate through list of sequences
    for seq in seq_arr:
        #create an empty list for each seq that will be filled with all of the letter encodings
        seq_encoded = np.empty((len(seq),4)) #empty numpy array []
        
        letter_counter = 0
        #go letter by letter storing the current letter's 1x4 list in the larger container list for the seq
        for letter in seq:
            current_encoding = d[letter] #get the encoding for the current letter
            #add to existing seq_encoded list with the newest letter's encoding
            seq_encoded[letter_counter] = current_encoding
            letter_counter+=1
        
        seq_flattened = seq_encoded.flatten() #flatten the encoded letters into a single array
            
        encodings.append(seq_flattened) #append current seq's encoding to larger encodings list
    
    return np.array(encodings)


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


def correct_length(pos_seqs:List[str], neg_seqs:List[str]) -> List[str]:
    """
    This function reads in the much longer negative sequences and splits them up into chunks as big as the positive sequences
    
    Args:
        pos_seqs: List[str]
            List of all rap1 binding sites
        neg_seqs: List[str]
            List of sites that aren't bound by rap1

    Returns:
        negatives_split: List[str]
            List of negative sequences split up into appropriate lengths
    """
    
    correct_seq_length = len(pos_seqs[0]) #get desired length from first entry of positive sequences
    negatives_split = []
    
    for seq in neg_seqs:
        seq_split = []
        for chunk in range(int(len(seq)/correct_seq_length)):
            seq_split.append(seq[(chunk*correct_seq_length):((chunk+1)*correct_seq_length)])
            
        negatives_split = negatives_split + seq_split
    
    return negatives_split


#
# DO NOT MODIFY ANY OF THESE FUNCTIONS THEY ARE ALREADY COMPLETE!
#


# Defining I/O functions
def read_text_file(filename: str) -> List[str]:
    """
    This function reads in a text file into a list of sequences.

    Args:
        filename: str
            Filename, should end in .txt.

    Returns:
        arr: List[str]
            List of sequences.
    """
    with open(filename, "r") as f:
        seq_list = [line.strip() for line in f.readlines()]
    return seq_list


def read_fasta_file(filename: str) -> List[str]:
    """
    This function reads in a fasta file into a numpy array of sequence strings.

    Args:
        filename: str
            File path and name of file, filename should end in .fa or .fasta.

    Returns:
        seqs: List[str]
            List of sequences.
    """
    with open(filename, "r") as f:
        seqs = []
        seq = ""
        for line in f:
            if line.startswith(">"):
                seqs.append(seq)
                seq = ""
            else:
                seq += line.strip()
        seqs = seqs[1:]
        return seqs