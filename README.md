N-gram based Language-Modelling using Python
============================================

Implementation of Language Modelling based N-gram probability prediction  


Problem 
 Description       : This program implements the N-gram model using Maximum
                     Likelihood Estimate Method (MLE) and Simple Good Turing 
                     (SGT) method.   
                     The N-gram model in NLP is based upon 
                     predicting the next word in a word sequence based upon
                     last few previous words in the sequence. The number of
                     previous words considered in this prediction denotes N
                     in N-gram model. For this program, the values considered
                     for N are 1 (which single word or unigram) and 2 (which
                     means pair of words or bigram). Using probabilistic models
                     like MLE and SGT, this program tries to find out the 
                     Probabilities of different words in a set of text 
                     (corpus) based upon a set of words (vocabulary). 
                     The SGT method implementation follows 
                     approach mentioned in Gale-Sampson 1995 Paper titled as
                     "Good Turing Frequency Estimation Without Tears".
                     (Section 6: THE PROCEDURE STEP BY STEP)
                     The SGT estimates for the probabilities are later smoothed
                     using the linear regression method as mentioned in 
                     the Gale-Sampson Paper.

 Usage             : This program takes following inputs:
                     1) -N = number of top probabilities results to be shown
                             in output
                     2) -P = the precision level to be shown in fractional
                             part of Probabilities
                     3) -w = the name of word file used to build vocabulary
                     4) -i = the names of data files constituting corpus
                     
                     e.g. to run this program for N = 5 , P =6 , 
                     for a word file "linux.w" and data files "pg2600.txt" &
                     "pg2554.txt", use following command
