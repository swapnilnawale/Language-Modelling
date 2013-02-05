##############################################################################
# Problem 
# Description       : This program implements the N-gram model using Maximum
#                     Likelihood Estimate Method (MLE) and Simple Good Turing 
#                     (SGT) method.   
#                     The N-gram model in NLP is based upon 
#                     predicting the next word in a word sequence based upon
#                     last few previous words in the sequence. The number of
#                     previous words considered in this prediction denotes N
#                     in N-gram model. For this program, the values considered
#                     for N are 1 (which single word or unigram) and 2 (which
#                     means pair of words or bigram). Using probabilistic models
#                     like MLE and SGT, this program tries to find out the 
#                     Probabilities of different words in a set of text 
#                     (corpus) based upon a set of words (vocabulary). 
#                     The SGT method implementation follows 
#                     approach mentioned in Gale-Sampson 1995 Paper titled as
#                     "Good Turing Frequency Estimation Without Tears".
#                     (Section 6: THE PROCEDURE STEP BY STEP)
#                     The SGT estimates for the probabilities are later smoothed
#                     using the linear regression method as mentioned in 
#                     the Gale-Sampson Paper.
#
# Usage             : This program takes following inputs:
#                     1) -N = number of top probabilities results to be shown
#                             in output
#                     2) -P = the precision level to be shown in fractional
#                             part of Probabilities
#                     3) -w = the name of word file used to build vocabulary
#                     4) -i = the names of data files constituting corpus
#                     
#                     e.g. to run this program for N = 5 , P =6 , 
#                     for a word file "linux.w" and data files "pg2600.txt" &
#                     "pg2554.txt", use following command
#
#       python Ngram_Modelling.py -N 5 -P 6 -w linux.w -i pg2600.txt pg2554.txt
#                     
#                     Please note that sequence of the inputs SHOULD be same as
#                     shown above. i.e. -N <N value> -P <P value> -w 
#                     <single word file> -i <list of data files 
#                     separated by space>
#                     
#                     This program will produce first the total number of 
#                     tokens (all words) present in data and word files. Then
#                     it will show total number of types (distinct words) from
#                     data and word files. Then it will show top N unigram 
#                     probabilities calculated by MLE and SGT method. It will
#                     then show total probability of observed and observed 
#                     events.
#                     The remaining output will be top N bigram probabilities
#                     calculates by MLE and SGT method. and total probability
#                     of observed and unobserved events.
#
# Algorithm         : 1) This program first reads the input data files and
#                     word file and make their copies to retain original 
#                     files intact.
#                     2) It then cleans all copy files by converting all
#                     characters in the file to lowercase and replacing
#                     all non-alphabetic characters to a space.
#                     3) It then combines word copy and all input copy files to
#                     create a single file for vocabulary. And it also
#                     combines all data copy files to get a single data file.
#                     4) This program counts the total number of tokens and
#                     total number of types to display them to user.
#                     5) It then calculates the MLE probabilities for unigrams
#                     in single data file and shows top N probabilities to user
#                     6) After that it calculates the GT prob. for unigrams 
#                     and displays top N probabilities to the user along with 
#                     total observed and unobserved prob.
#                     7) It then repeats step 5 and 6 for bigrams. 
#
# Author            : Swapnil Nawale 
#
# Date              : 09/13/2012
#
# Version           : 1.0
#
# Prog. Language    : Programming Language used for this program is Python
#                     (Version 2.7.3). I have used Python for coding this
#                     program for following two reasons: 
#                     1) It has many features for text-processing and 
#                     file operations, equally powerful as perl. 
#                     2) I am more comfortable with coding in Python than
#                     in Perl.I found Python easy to code syntaxwise.
#                   
#                     The basic Python code, used in this program, is
#                     learnt from the book "Think Python - How to Think Like 
#                     a Computer Scientist (by Allen B. Downey)" and from
#                     Google's Python Class , present online at 
#                     http://code.google.com/edu/languages/google-python-class/
#                       
#                     I have tried to follow Python coding conventions
#                     mentioned in PEP 8 -- Style Guide for Python Code 
#                     (present at 
#                     http://www.python.org/dev/peps/pep-0008/#programming
#                     -recommendations) as much closely as possible. 
#                     Few of the important coding conventions followed are 
#                     mentioned below as well.
#                       
#                     1) Single line comments in Python starts with # and
#                     multiline comments start and end with ''' (three 
#                     single quotes). 
#                       
#                     2) Variable names and functions names are lowercase
#                     and separated by underscore.Constants are denoted 
#                     using all capital letters separated by underscore.
#
#                     3) Python uses indentation levels to specify functions
#                     , loops, if-constructs boundries. There are no curly 
#                     braces to specify these boundries. I have used 
#                     tabspacing of 4 characters for indenting the code.
#                       
#                     4) I have tried to maintain 79 lines characters per line
#                     to improve code readability, especially while using 
#                     hardcopy of code.
#                       
# Text-Editor used  : vim editor on Linux Platform
#
# Notes             : 1) This program starts execution from typical python 
#                     boilerplate syntax present at the last 12 lines of 
#                     this file and then proceeds to main method.
#                     2) The output shows that total observed and unobserved
#                     probabilities by SGT do not sum to 1. I tried to a lot
#                     to find the reason for this by checking calculations in 
#                     in the code many times. Still could not find the reason.
#                     The smoothed freq counts (r*/c*) appear correctly in the
#                     the results but total probablities are not.
#                     
#					  This could be because of two reasons:
#					  1) One reason for this could be switching of x and y 
#					  values while calculating r* using linear regression.
#					  or
#					  2) One more possible reason behind this, could be that 
#					  the Gale-Sampson paper says that while calculating 
#					  probablit of unseen events, we need to divide the 
#					  N_1 by the total number of events but EXCLUDING count 
#					  of unonberved events. While the "fishing in lake" example 
#					  given in book considers the count of unobeserved events 
#					  while doing division. I have used the approach given in
#				      in Gale-Sampson's Paper.
###############################################################################

#!/usr/bin/python

'''
import statements to include Python's in-bulit module functionalities in the
program
'''
# sys module is used to access command line argument, exit function etc.
import sys

# re module is used to access regular expression related facilities
import re

# os module is used to acess file manipulation features
import os

# operator module is used for sorting datastructures
import operator

# itertools module is used for forming bigrams using unigrams
import itertools

# collections module is used for creating ordered hash tables / dicts 
import collections

# math module is used to calculate log values and powers
import math

'''
Set the value of debug flag. debug flag is used to decide whether to print
debug information in the output or not. This flag will be a global variable.
'''
debug = False

'''
import datetime module to measure execution times of few some code segments
(for debugging purpose only). Used to check if optimization is needed
'''

if debug:
    import datetime

###############################################################################
# Function      : create_copy(original_file_name)
# Description   : This function creates a copy of the file, which is passed as
#                 parameter to the function. It opens the file, reads all line
#                 from it and writes them into a copy file. Copy file name will
#                 be same as original file name appended with '.copy' 
#                 extension.This function will return the name of copy file 
#                 created.
# Arguments     : original_file_name- Name of the file for which copy is to be
#                 created
# Returns       : The name of copy file created.
###############################################################################
def create_copy(original_file_name):
    
    '''
    Open the original file using in built open function in read mode.
    Open function takes two parameters: 
    1) Path of the file to be opened
    2) Mode in which file should be opened. Valid modes are
       r  - read 
       w  - write
       a  - append
       r+ - read and write
       b  - binary
    open function returns a file object (refered hereafter as file handle).
    We can use this file handle to read, write or append a string to
    file.
    '''
    original_file_handle = open(original_file_name, 'r')

    '''
    Create a copy file using open function in write mode. Name of the copy
    file will be original copy file appened with '.copy'
    '''
    copy_file_name = original_file_name + ".copy"

    copy_file_handle = open(copy_file_name, 'w')

    '''
    Read all lines from the original file using its file handle.
    For this, use in-built readlines function. readlines function reads 
    lines from the file specified by file handle and returns a Python list
    containing each line as one element of list. This list will also have a 
    trailing newline character to each element.
    e.g. If the contents of file are:

    Neo
    Trinity
    Morpheus

    then, readlines will return a list with the contents like:
    ['Neo\n','Trinity\n','Morpheus\n']

    '''
    # read all lines from original file and store them into a list
    original_file_list = original_file_handle.readlines()

    ''' 
    Iterate over the original_file_list so that it's content can be 
    written into copy file. For iteration over the list, For loop can be used.
    Such for loops reads each element of list and store them in a counter 
    variable automatically. This counter variable is modified to new element
    on each iteration.
    e.g. If the elements of a list sample_list are as follows :

    ['Neo\n','Trinity\n','Morpheus\n']

    and if we write for loop for above list like below

    for sample_list_element in sample_list:
        print sample_list_element

    then value of sample_list_element for first iteration will be Neo\n.
    Similarly values of this variable for second and third iteration will be
    Trinity\n and Morpheus\n resp.
    '''

    '''
    iterate over original_file_list and write the contents of list to copy file
    using in-built write function. write function writes a string passed to it
    into a file, specified by file handle on which write function is called.
    '''
    for original_file_line in original_file_list:
        copy_file_handle.write(original_file_line)

    '''
    Close original and copy file using inbuilt close function called on file
    handle.
    '''

    original_file_handle.close()
    copy_file_handle.close()

    return copy_file_name 

###############################################################################
# End of create_copy Function
###############################################################################

###############################################################################
# Function      : clean_file(file_name)
# Description   : This function cleans the file passed as parameter to it.
#                 Cleaning is done in two steps. First, all characters from
#                 file are converted into lowercase by calling 
#                 'convert_to_lowercase' function. And then, all non-alphabetic
#                 from file are converted to a space by calling 
#                 'convert_non_alpha_chars' function.
# Arguments     : file_name - Name of the file to be cleaned.
# Returns       : None.
###############################################################################
def clean_file(file_name):
    
    convert_to_lowercase(file_name)
    convert_non_alpha_chars(file_name)

###############################################################################
# End of clean_file Function
###############################################################################

###############################################################################
# Function      : convert_to_lowercase(file_name)
# Description   : This function converts all characters from a file to 
#                 lowercase. And stores all converted characters into the 
#                 same file by overwriting it.
# Arguments     : file_name - Name of the file to be converted into all 
#                             lowercase characters.
# Returns       : None.
###############################################################################
def convert_to_lowercase(file_name):
    
    # open the file in read mode
    file_handle = open(file_name, 'r') 

    '''
    Get all lines from this file using readlines() function and store into
    a list.
    '''
    file_lines_list = file_handle.readlines()

    '''
    Close the file. This is required as I want to overwrite the same file.
    Overwriting will avoid creation of multiple intermediate files.
    '''
    file_handle.close()

    '''
    Now open the file in write mode so as to write lowercase converted 
    character into it. Using write mode for a file in Python erases 
    the existing file with the same. And this will facilitate overwriting
    for the program.
    '''
    file_handle = open(file_name, 'w')

    '''
    Iterate over the file_lines_list and convert all uppercase characters 
    from each element line of list to lowercase. For this, inbuilt lower() 
    function will be used.
    Write the converted lines into the file specified by file_handle.
    '''
    for file_line in file_lines_list:
        file_handle.write(file_line.lower())

    # close the file before exiting function
    file_handle.close()

###############################################################################
# End of convert_to_lowercase Function
###############################################################################

###############################################################################
# Function      : convert_non_alpha_chars(file_name)
# Description   : This function converts all non-alpha characters from a file 
#                 to space. The contents of the file, along with 
#                 space-converted file is stored in the same file by 
#                 overwriting 
# Arguments     : file_name - Name of the file for which conversion from non- 
#                              alpha chars to space is intended.
# Returns       : None.
###############################################################################

def convert_non_alpha_chars(file_name):

    # open the file in read mode
    file_handle = open(file_name, 'r') 

    '''
    Convert all non-alpha characters to space using regular exprssions.
    Regular expressions in Python use inbuilt 're' module. This module has
    a useful function called as 'sub'. This function stands for substitution.
    sub function takes three parameters as described below:
    1) A regular expression pattern to be searched within a given string,
    2) A substitution string that replaces all occurrences pattern 
    (mentioned in 1 above) within the given string,
    3) The string for which substitution is required.

    The third string parameter can also be read directly from a file.
    In-built 'read' function can be used to read all contents of a file and 
    store them into a string, which can be passed a third parameter to 
    'sub' function. read function works upon file handle for a file.
    
    sub function returns a new string after substitution is applied to
    input string.

    Regular expressions patterns in Python are usually preceded by a 'r'
    , which indicates that it is a 'raw; pattern. A raw pattern does not 
    require extra backslashes to escape backslashes in patterns. 
    '''

    # substitute non-alpha chars to space from file
    converted_file_line = re.sub(r'[^a-z]', ' ', file_handle.read())
    
    ''' 
    Close the file. This is required as I want to overwrite the same file.
    Overwriting will avoid creation of multiple intermediate files.
    '''
    file_handle.close()

    ''' 
    Now open the file in write mode so as to write characters, converted 
    from non-alpha to space, into it. Using write mode for a file in Python 
    erases the existing file with the same. And this will facilitate 
    overwriting the file for the program.
    '''
    file_handle = open(file_name, 'w')

    # write the converted contets of file
    file_handle.write(converted_file_line)

    # close the file again before exiting function
    file_handle.close()

###############################################################################
# End of convert_non_alpha_chars Function
###############################################################################

###############################################################################
# Function      : unigram_prob_mle_method(data_file,types_list,p_param)
# Description   : This function calculates the unigram probabilities for input 
#                 data files using MLE method. It uses vocabulary specified
#                 by another parameter types_list. This function uses relative
#                 simple relative frequencies to calculate unigram probability.
# Arguments     : data_file_name - A file representing sample corpus for which 
#                                  probabilities are to be decided
#                 type_list - A list comparising of all word types
#                 p_param - Precision level to be used while showing 
#                           probabilities
# Returns       : A dictionary object consisting each unigram as key and its 
#                 frequency as value for that key
#                 A dictionary object consisting each unigram as key and its
#                 MLE probability as value for that key
#                 A list containing all words in the data file to be used in
#                 calculating bigram probabilities in further processing
###############################################################################

def unigram_prob_mle_method(data_file_name, types_list, p_param):

    ''' 
    Get the total number of tokens from the data file passed as parameter. 
    This count will be used in calculating unigram probabilities. 
    This count will denote the sample size for our N-Gram Model.
    '''

    data_file_handle = open(data_file_name, 'r')

    data_tokens_list = data_file_handle.read().split()

    data_file_token_count = len(data_tokens_list)

    if debug:
        print data_file_token_count
    
    # close data file
    data_file_handle.close()

    '''
    Define two dictionary object (referred as dict in Python) to hold 
    frequencies and probabilities for each unigram.A dict object in Python is a
    datastructure similar to Maps. It stores unique set of keys and 
    values for each key. 
   
    Both dict objects here will have unigrams as their keys. One dict object
    will have frequency counts as value for keys and other dict object will 
    have probability of unigrams as value.
    
    '''
    ''' 
    Create first dict object to hold frequency counts for each unigram 
    and initialize it 
    '''
    unigram_freq_mle_dict = {}
    
    ''' 
    Create seconf dict object to hold mle prob. for each unigram 
    and initialize it 
    '''
    unigram_prob_mle_dict = {}

    '''
    Open the data file again for checking frequency counts of unigrams from the
    data file. This file was closed above after counting number of tokens from 
    it. This kind of repeatative file closing and opening was required as I was
    getting some weird results without doing this. I was not able to read from
    the file if file is not closed earlier. So, I used this workaround to deal
    with the issue. 
    '''

    data_file_handle = open(data_file_name, 'r')

    # read the contents of data file into a string
    data_file_contents =  data_file_handle.read()
    

    if debug :
        start_t =  datetime.datetime.now()
    
    '''
   
    Start counting the freq of each unigram from the data file.
    For this, first all multispace characters in the data file are
    converted into a single space character. Then using regex '\w+'
    separate out all words in the file. All these words are stored
    in the a word list with name "words". Next, a Counter object will
    be used to count the frequencies of each unigram from this list.
    Counter objects in python returns mapping of each word with 
    their frequencies.

    
    Using the frequencies calculated above, MLE probabilities will be
    calculated by dividing each frequencie by the total token count in
    data file.

    To calculate the unigram frequencies, I tried several approaches.

    One approach was to use count of <space><unigram><space> occurrences 
    from the data file. But it failed for large data files as it was 
    not efficient. This approach was not giving correct counts as well.

    Then, I checked the possibility of using regex '\b'+unigram+r'\b'.
    It worked correctly but was too slow for large data files again.
    The code tried for this approach is given below in commented format. 
    
    # iterate over the types_list to retrieve each type / unigram 
    for unigram in types_list:
        
        ''
        Calculate the freq. of each unigram from the data file. For this,
        in-built count function will be used. Count function will search for
        ocurrences using regex with word boundary anchors \b for the 
        counting frequency of unigrams. 
        ''

        unigram_count = len(re.findall(r'\b'+unigram+r'\b',data_file_contents))
        unigram_freq_mle_dict[unigram] = unigram_count

        ''
        Calculate the MLE probabilities for each unigram. For this,
        divide unigram count calculated above by the total token count from
        data file.  
        Add unigram and probabilities to unigram_prob_mle_dict as 
        key - value pair. All probabilities will have a precision level specied
        for their fractional parts. I tried to look for python's in-built 
        functionalties for this. I found some modules like decimal but 
        unfortunately could not get them to working for program. So I resorted
        to write a user-defined function called precision_level. This
        function will be called to get probabilities with specific precision
        level.
        ''
        
        unigram_prob = float(unigram_count) / float(data_file_token_count)
        unigram_prob =  precision_level(unigram_prob, p_param) 
        unigram_prob_mle_dict[unigram] = unigram_prob
        
    Finally all approaches failed, I thought separating all the words from 
    the data files in one go. The counter object proved useful for this.
    The example usage of counter object was reffered from Python 
    Documentation present in the link:

    http://docs.python.org/library/collections.html
    '''
 
    data_file_contents1 = " ".join(data_file_contents.split())
    words = re.findall('\w+', data_file_contents1)
    unigram_freq_counter_obj =  collections.Counter(words)
    
    for unigram in types_list:
        unigram_freq_mle_dict[unigram] = 0
        unigram_prob_mle_dict[unigram] = precision_level(float(0),p_param)

    for unigram_str_freq in unigram_freq_counter_obj.most_common():
        unigram_freq_mle_dict[unigram_str_freq[0]] = unigram_str_freq[1]
        
        unigram_prob = float(unigram_str_freq[1]) / \
                       float(data_file_token_count)
        unigram_prob =  precision_level(unigram_prob, p_param) 
        unigram_prob_mle_dict[unigram_str_freq[0]] = unigram_prob

    if debug:
        print unigram_freq_mle_dict
        print unigram_prob_mle_dict

    if debug:
        end_t = datetime.datetime.now()
        print (end_t - start_t).microseconds

    
    '''
    Return the word list "words" from this function. It will be used in 
    calculating bigram probabilies again. This will avoid reading the data 
    files again to generate the word list. Reading data files to separate
    out words involve regex, which makes the execution of program very slow.
    So, it is useful to send the words list out from this function again.
    As python uses pass by reference method for function parameters,
    I don't think passing large objects will create a lot of performance
    overhead. 
    '''  

    return unigram_freq_mle_dict, unigram_prob_mle_dict, words

###############################################################################
# End of unigram_prob_mle_method Function
###############################################################################

###############################################################################
# Function      : precision_level(fractional_num, precision_level)
# Description   : This function converts a float number to have a specific
#                 precision level.
# Arguments     : fractional_num - The float number for which precision is 
#                                  required
#                 precision_level - Level of precision required
# Returns       : The new float number with precision specied.
###############################################################################

def precision_level(fractional_num, precision_level):

    '''
    To make a float number to have a specific level of precision (say P), 
    first convert that number to string. Split this string at decimal point to
    get integer and fractional parts. Check the digit at (P+1)th position.If
    that digit is greater than or equal to 5 then add 1 to digit at 
    Pth Position. If the digit at (P+1)th position is 9 then make digit at 
    Pth Position as zero and add 1 to digit at (P-1)th Position. 
    Finally, take substring of fractional part of length P. 
    Combine integer part and this substring with decimal point between them. 
    Convert it to float and return it as a float with specific precision level
    '''
   
    ''' 
    Force each floats to have 18 digits after decimal point. 
    This will convert the floats like 0.0, 0.5 to have trailing zeros
    like 0.000000000000000000. This was required for proper working
    of precision_level function. The factor 18 here was chosen after observing
    maximum number of digits after decimal point generated for a float by 
    Python.
    '''
    
    fractional_num =  "%.18f "%fractional_num
    
    fractional_num_str = str(fractional_num)

    fractional_num_part_list =  fractional_num_str.split('.')    
    
    int_part = fractional_num_part_list[0]
    fraction_part = fractional_num_part_list[1]

    pl = int(precision_level)

    p_plus_one_position_char = fraction_part[pl:pl+1]
    
    if int(p_plus_one_position_char) >= 5:
       
        if int(p_plus_one_position_char) != 9:
            
            conv_str_1 = fraction_part[0:pl-1]
            conv_str_2 = str(int(fraction_part[pl-1:pl])+1)
            conv_str_3 = fraction_part[pl:]
        
        else:

            conv_str_1 = fraction_part[0:pl-2]
            conv_str_2 = str(int(fraction_part[pl-2:pl-1])+1)
            conv_str_3 = "0" + fraction_part[pl+1:]

        conv_str = int_part + "." + conv_str_1 + conv_str_2 + conv_str_3

    else:
        
        conv_str = fractional_num_str
        
    if debug:
        print conv_str

    return float(conv_str[0:len(int_part) + 1 + pl])

###############################################################################
# End of precision_level Function
###############################################################################

###############################################################################
# Function      : bigram_prob_mle_method(data_file_name, types_list,  
#                                        unigram_freq_mle_dict, bigrams_list,
#                                        p_param,words)
# Description   : This function calculates the bigram probabilities for input 
#                 data files using MLE method. It uses vocabulary specified
#                 by another parameter types_list. This function uses relative
#                 simple relative frequencies to calculate bigram probability.
# Arguments     : data_file_name - A file representing sample corpus for which 
#                                  probabilities are to be decided
#                 type_list - A list comparising of all word types
#                 unigram_freq_mle_dict - A dict object mapping unigrams to
#                 their frequency in data file
#                 bigrams_list - A list of all possible bigrams 
#                 p_param - Precision level to be used while showing 
#                           probabilities
#                 words - A list of all words in the data file
# Returns       : A dictionary object consisting each bigram as key and its 
#                 frequency as value for that key
#                 A dictionary object consisting each bigram as key and its
#                 MLE probability as value for that key
###############################################################################

def bigram_prob_mle_method(data_file_name, types_list, unigram_freq_mle_dict,\
                           bigrams_list, p_param, words):

    ''' 
    Create a dict object to hold frequency counts for each bigram
    and initialize it 
    '''
    bigram_freq_mle_dict = {}
    
    ''' 
    Create another dict object to hold MLE prob. for each bigram
    and initialize it 
    '''
    bigram_prob_mle_dict = {}

    '''
    Open the data file for checking frequency counts of bigrams from the
    it.
    '''

    data_file_handle = open(data_file_name, 'r')

    # read the contents of data file into a string
    data_file_contents =  data_file_handle.read()
    
    data_tokens_list = data_file_contents.split()

    data_file_token_count = len(data_tokens_list)

    if debug :
        start_t =  datetime.datetime.now()

    '''
    Start counting the occurrences of bigrams from the data file.
    For this, the list of words passed as parameter will be used.
    The count of bigrams from the word list will be done using
    Counter object, islice and izip functions from itertools module.

    The sample code for counting the bigrams was posted as a question
    asked by me on www.stackoverflow.com. The discussion of this
    question is present in the link:

    http://stackoverflow.com/questions/12488722/
    counting-bigrams-pair-of-two-words-in-a-file-using-python 

    I have the code suggested in the answer by stackoverflow user
    Abhinav Sarkar.

    Before turning to stackoverflow.com, I tried to use some approaches
    to count the bigrams. All of them worked fine but failed badly
    on performance front as the number of bigrams was huge.
    The code for one approach of using regex  \b<unigram 1>\b\s+\b<unigram 2>
    is given below on commented format.

    # iterate over the bigrams_list to retrieve each bigram 
    for bigram in bigrams_list:
        
        #''
        #Calculate the freq. of each bigram from the data file. For this,
        #regular expression \b<unigram 1>\b\s+\b<unigram 2> will be used.
        #Add bigram and its freq. count to bigram_freq_mle_dict as 
        #key - value pair.
        #''
        if debug:
            print bigram
            print re.findall(r'\b' + bigram[0] + r'\b\s+'\
                           + r'\b' + bigram[1] + r'\b',\
                             data_file_contents)
 
        
        bigram_count = len(re.findall(r'\b' + bigram[0] + r'\b\s+'\
                                    + r'\b' + bigram[1] + r'\b',\
                                    data_file_contents))
        
        if debug:
            print bigram
            print bigram_count
        
        bigram_freq_mle_dict[bigram] = bigram_count

        #''
        #Calculate the MLE probabilities for each bigram. For this,
        #divide bigram count calculated above by the freq. count of first word
        #of the bigram.
 
        #Add bigram and probabilities to bigram_prob_mle_dict as 
        #key - value pair. All probabilities will have a precision level specied
        #for their fractional parts calculated by calling precision_level 
        #function
        #''
   
        if debug:
            print bigram[0]
            print unigram_freq_mle_dict[bigram[0]] 

        try:
            
            # Get joint - prob for bigram
            unigram_count = unigram_freq_mle_dict[bigram[0]]
            bigram_prob = float(bigram_count) / \
                      float(unigram_count)
            
            bigram_prob = bigram_prob * (float(unigram_count) / \
                          float(data_file_token_count) )


        except ZeroDivisionError:
            #'' 
            #If above division results in divide by zero,
            #make the prob as zero.
            #''
            bigram_prob = 0


        bigram_prob =  precision_level(bigram_prob, p_param) 
        bigram_prob_mle_dict[bigram] = bigram_prob
   
    
    if not debug:
        print bigram_freq_mle_dict
        #print bigram_prob_mle_dict

    '''

    
    bigram_freq_counter_obj =  collections.Counter(itertools.izip(words,\
                               itertools.islice(words, 1, None)))
    '''
    This object bigram_freq_counter_obj will have bigram and their freq 
    mappings only for those bigrams which are actually present in the 
    data file. Bigrams with zero count are not the part of above counter
    object. To make a dict of all bigrams (zero and non-zero freq) and
    their freq, I thought of first initializing dict object 
    bigram_freq_mle_dict with default frequencies as 0. And just replace
    frequencies of non-zero freq bigrams with the freq found in 
    bigram_freq_counter_obj.

    For this, I needed to iterate over bigrams list, which was of really huge
    size. And the code for initializing dict bigram_freq_mle_dict with 0 freq
    values could not proceed and I was getting Memory Error as the dict 
    operations for such a large objects consume a lot of memory.

    The code that gave me Memory Error is given below.
    
    for bigram in bigrams_list:
        bigram_freq_mle_dict[bigram] = 0
        bigram_prob_mle_dict[bigram] = precision_level(float(0),p_param)
    
    So, instead of storing freq. of all bigrams in bigram_freq_mle_dict, I 
    resorted to store only non-zero freq bigrams in bigram_freq_mle_dict.
    Similarly only non-zero prob will be stored in bigram_prob_mle_dict. 
    '''

    for bigram_str_freq in bigram_freq_counter_obj.most_common():
        bigram_freq_mle_dict[bigram_str_freq[0]] = bigram_str_freq[1]
        
        try:
            # Get joint - prob for bigram
            unigram_count = unigram_freq_mle_dict[bigram_str_freq[0][0]]
            bigram_prob = float(bigram_str_freq[1]) / \
                      float(unigram_count)
            
            bigram_prob = bigram_prob * (float(unigram_count) / \
                          float(data_file_token_count) )

        except ZeroDivisionError:
            '''
            If above division results in divide by zero,
            make the prob as zero.
            '''
            bigram_prob = 0

        bigram_prob =  precision_level(bigram_prob, p_param) 
        bigram_prob_mle_dict[bigram_str_freq[0]] = bigram_prob


    if debug:
        end_t = datetime.datetime.now()
        print (end_t - start_t).microseconds

    return bigram_freq_mle_dict, bigram_prob_mle_dict

###############################################################################
# End of bigram_prob_mle_method Function
###############################################################################

###############################################################################
# Function      : unigram_prob_sgt_method(unigram_freq_mle_dict,data_file_name,
#                                          p_param)
# Description   : This function calculates the unigram probabilities for input 
#                 data files using SGT method. The SGT method used here is 
#                 further smoothed using Linear Good Smoothing specified in
#                 paper by Gale and Sampson(1995).
# Arguments     : unigram_freq_mle_dict - A dict object mapping unigrams to
#                 to their frequency counts
#                 data_file_name - A file representing sample corpus for which 
#                                  probabilities are to be decided
#                 p_param - Precision level to be used while showing 
#                           probabilities
# Returns       : A dict object storing mapping of freq values to bigrams   
#                 A dict object storing mapping of freq values to prob
#                 A dict object storing mapping of freq values to smoothed freq
#                 Total observed prob 
###############################################################################

def unigram_prob_sgt_method(unigram_freq_mle_dict, data_file_name, p_param):

    ''' 
    The approach used for calculating SGT unigram probabilities is taken
    from Gale and Sampson's Paper. I am sticking to variable naming 
    conventions used in it. The variable names used in the program 
    for the varibles mentioned in Section 6 "THE PROCEDURE STEP BY STEP"
    of the paper are as follows:
    
    ----------------------------------------------------------------------
    |    Variable name used in paper | Corresponding variable name used  |
    |                                |     in this program               |
    ----------------------------------------------------------------------
    | P_0                            |  prob_0                           |
    ----------------------------------------------------------------------
    | N'                             |  N_prime                          |
    ----------------------------------------------------------------------
    | a                              |  a_value                          |
    ----------------------------------------------------------------------
    | b                              |  b_value                          |
    ----------------------------------------------------------------------
    | N                              |  N_value                          |
    ----------------------------------------------------------------------
    | r                              |  r_value                          |
    ----------------------------------------------------------------------
    | n                              |  n_value                          |
    ----------------------------------------------------------------------
    | Z                              |  Z_value                          |
    ----------------------------------------------------------------------
    | log r                          |  log_r                            |
    ----------------------------------------------------------------------
    | log Z                          |  Z_value                          |
    ----------------------------------------------------------------------
    | r*                             |  smoothed_r                       |
    ----------------------------------------------------------------------
    | p                              |  smoothed_prob                    |
    ----------------------------------------------------------------------

    For the sake of simplicity, I am not going to create a single table
    that stores the mapping of r to n, Z, log Z, log r, r*, p as mentioned
    in paper. Instead, I am going to maintain separate dict objects for each 
    mapping. 
    e.g.a dict object mapping r to n , another dict object mapping r to Z.

    A master dict object mapping r to the list of unigrams with r freq. will
    also be maintained. This will be used to lookup unigrams with  freq.

    A sample master dict object will look like this:
    
    -----------------------------------------------------------------------
    | freq          |   list of unigrams with that freq                   |
    -----------------------------------------------------------------------
    | 4             |   [Achilles, Hector, Cassandra]                     |
    -----------------------------------------------------------------------
    | 1             |   [Agamemnon, Prium]                                |
    -----------------------------------------------------------------------

    This means Agmemnon and Prium unigrams appears once in the corpus while
    other unigrams appear 4 times.
    '''
    
    '''
    First we need to calculate the different frequencies that appear for
    all unigrams. For this, a list freq_list will be maintained. It will just
    enlist  all different frequencies (r values) existing in 
    unigram_freq_mle_dict. 
    '''
    # Create and initialize freq_list to an empty list
    
    freq_list = []
     
    '''
    Iterate over unigram_freq_mle_dict to get diffrent freq and add them to
    freq_list    
    '''

    if debug:
        print "\nunigram to freq map"
        print unigram_freq_mle_dict
            
    '''
    Here the in-built function fetches both key and value together
    from dict object at the same time over each iteration.
    '''

    for unigram_str, unigram_freq in unigram_freq_mle_dict.iteritems():
        
        if debug:
            print unigram_str
            print unigram_freq
        
        # Add unigram frequency to freq_list if it is not already present
        
        if unigram_freq not in freq_list:
            freq_list.append(unigram_freq)
        
   
    if debug:
        print freq_list

    # Sort the freq_list in ascending order
    freq_list = sorted(freq_list)

    if debug:
        print "\nfreq_list"
        print freq_list

    '''
    Create the master dict mapping r values to the list of unigrams.
    For this iterate over freq_list and get individual frequencies r.
    Then for each of the individual freq r, find all unigrams with freq r
    from unigram_freq_mle_dict. Add the r and a list of unigrams with r
    freq in master dict as key value pair. A sample master dict is shown 
    above.

    In python normal dict object is unordered and here we need to keep
    master dict sorted in ascending order of r. So, instead of normal
    dict, an OrderDict object will be used. OrderedDict class belongs
    to Python module collections. The usage of ordered dicts was learned
    from the article by Doug Hellman, present at the link:

    http://www.doughellmann.com/PyMOTW/collections/ordereddict.html
    '''

    master_dict = collections.OrderedDict()

    for r_value in freq_list:
        
        unigrams_list_for_freq = []
        
        for unigram_str, unigram_freq in unigram_freq_mle_dict.iteritems():
                if unigram_freq == r_value:
                    unigrams_list_for_freq.append(unigram_str)

        if debug:
            print unigrams_list_for_freq

        master_dict[r_value] = unigrams_list_for_freq
    
    if debug:
        print "\nmaster dict"
        print master_dict
    
    '''
    Create a dict storing mapping of r values to frequency of r frequencies
    in our corpus. This mapping is basically from r to n_r.
    To calculate n_r for each r, get the length of list of unigrams stored in
    master dict for each r freq.
    '''
    r_to_n_mapping_dict = collections.OrderedDict()

    for r_value in freq_list:

        if debug:
            print master_dict[r_value]

        n_value = len(master_dict[r_value])

        r_to_n_mapping_dict[r_value] = n_value
    
    if debug:
        print "\nr to n_r mapping dict"
        print r_to_n_mapping_dict

    '''
    Calculate N value or sample size. This will be the sum of products of
    r and n_r values from r to n mapping dict. We won't conside the values for
    r = 0 i.e. count of unseen species, while calculating N value.
    '''
    
    N_value = 0

    for r, n in r_to_n_mapping_dict.items():
        
        if r == 0:
            continue

        N_value = N_value + (r * n)

    if debug:
        print "\nN value"
        print N_value

    
    '''
    Calculate Probability of unseen unigrams i.e. P_0. For this, divide the 
    value of frequency of unigrams which appear once (n_1) by sample size.
    Make this P_0 probability to have precision specified by p_param.
    '''
    #****** Note to myself : Think about approach when N_1 itself is zero.

    if debug:
        print r_to_n_mapping_dict
    
    P_0 = float(r_to_n_mapping_dict[1]) / float(N_value)
    P_0 =  precision_level(P_0, p_param)
    
    if debug:
        print P_0

    '''
    Calculate Z values for each frequency r. Z values for each freq. will be 
    calculated based upon the freq. count immediately previous r and 
    immediately next r values in r_to_n_mapping_dict. For, the first freq in
    the r_to_n_mapping_dict (excluding r freq. 0), the value of previous r will
    be taken as zero. And for last freq in r_to_n_mapping_dict, the value of
    next freq will be (2 * r - immediately previous freq.). For other freq.,
    Z value will be calculated as (2 * n_r / (immediately next freq - 
    immediately prev freq). The mappings of r values to Z values will be stored
    in an ordered dict r_to_Z_mapping
    '''
    
    # iterate over r_to_n_mapping_dict to get z values
    r_to_Z_mapping_dict = collections.OrderedDict()

    '''
    Get the length of freq_list to know whether we have arrived
    at last freq while calculating Z values. A counter will also
    be maintained to facilitate this.
    '''
    total_freq = len(freq_list)
 
    freq_counter = 0

    for r, n in r_to_n_mapping_dict.items():
    
        '''
        Insert dummy Z value -1 when r = 0 in r_to_Z_mapping_dict as 
        it is not important while calculating Z values.
        '''

        if r == 0:
            r_to_Z_mapping_dict[0] = -1
            freq_counter = freq_counter + 1
            continue

        # process first freq.
        if freq_counter == 1:
            if debug:
                print "\nfirst freq: " + str(r)

            prev_freq = 0
            next_freq = freq_list[freq_counter + 1]
            

        # process last freq.
        elif freq_counter == (total_freq - 1):
            if debug:
                print "\nlast freq: " + str(r)
            
            prev_freq = freq_list[freq_counter - 1]
            next_freq = (2 * r) - prev_freq 

        # process intermediate freq.
        else:
            if debug:
                print "\nintermidiate freq: " + str(r)
        
            prev_freq = freq_list[freq_counter - 1]
            next_freq = freq_list[freq_counter + 1]

        
       
        # calculate z values and insert them in r_to_Z_mapping_dict
        Z_value = float(2 * n) / float(next_freq - prev_freq)
        r_to_Z_mapping_dict[r] = Z_value
        
        if debug:
            print "counter value"
            print freq_counter
            print "prev freq"
            print prev_freq
            print "next freq"
            print next_freq
            print "Z_value"
            print Z_value

        freq_counter = freq_counter + 1

    if debug:
        print "\nr to z_r mapping dict"
        print r_to_Z_mapping_dict
    
    '''
        Create two ordered dicts to store log of freq r and log z and fill in
        them.
    '''

    r_to_log_r_mapping_dict = collections.OrderedDict()
    r_to_log_Z_mapping_dict = collections.OrderedDict()
    
    for r in freq_list:
        
        '''
        Fill dummy values of log r and log Z when r = 0
        '''
        if r == 0:
            r_to_log_r_mapping_dict[0] = -1
            r_to_log_Z_mapping_dict[0] = -1
            continue
        
        r_to_log_r_mapping_dict[r] = math.log10(r)
        r_to_log_Z_mapping_dict[r] = math.log10(r_to_Z_mapping_dict[r])

    
    if debug:
        print "\nr to log r mapping dict"
        print r_to_log_r_mapping_dict
        print "\nr to log Z mapping dict"
        print r_to_log_Z_mapping_dict

    '''
    Use the linear regression analysis to find line of best fit to log r
    and log z values. This will give us a and b param values.
    For doing regression analysis, call reg_analysis method and pass the
    log r , log z values, present in r_to_log_r_mapping_dict and 
    r_to_log_Z_mapping_dict, to it.
    '''
    
    a_value, b_value = reg_analysis(r_to_log_r_mapping_dict ,\
                                    r_to_log_Z_mapping_dict)
    
    if debug:
        print"\na_value, b_value"
        print a_value 
        print b_value

    '''
    Start calculating smoothed frequencies r* for each r. The value of r*
    is decided based upon either by n_r values estimates 
    or smoothed z_r values estimates.
    
    As per the Gale,Sampson's Paper, we need to switch between above two 
    estimates. If the value of r* based upon n_r values estimates 
    is x and that based upon smoothed z_r values estimates is y. 
    Then we decide the switch depending upon absolute difference of x and y 
    compared with 1.96 times the standard deviation of n_r values estimates.
    
    The values of smoothed r* freq. will be stored in a ordered dict object
    mapping r values to their r* values. 
    '''

    r_to_smoothed_r_mapping_dict = collections.OrderedDict()

    # create a flag to decide switching between x and y values
    switch_flag = False

    # iterate over freq_list to calculate smoothed freq.
    for counter in range(0,total_freq):
       
        r_value = freq_list[counter]

        # if switch_flag is false then calculate both x and y estimates
        
        if switch_flag == False:

            next_r_value = freq_list[counter + 1]
            n_r = r_to_n_mapping_dict[r_value]
            n_r_plus_1 = r_to_n_mapping_dict[next_r_value]

            if debug:

                print "\nr+1 in list"
                print next_r_value
                print "\nn_r"
                print n_r
                print "\nn_r_plus_1"
                print n_r_plus_1
            
            x_value = float(r_value + 1) * (float(n_r_plus_1) / float(n_r)) 
            
            '''
            Assigned r* value for r == 0 based upon the estimate of N_r values.
            '''

            if r_value == 0:
                r_to_smoothed_r_mapping_dict[0] = x_value 
                continue
 
            if debug:
                print "\n x value"
                print x_value
            
            if debug:
                print "\n smoothed z_r + 1"
                print smoothed_z_r_calc(a_value , b_value, next_r_value)
                print "\n smoothed z_r"
                print smoothed_z_r_calc(a_value , b_value, r_value)

            y_value = float(r_value + 1) * \
                      (smoothed_z_r_calc(a_value , b_value, r_value + 1) / \
                       smoothed_z_r_calc(a_value , b_value, r_value))
            
            if debug:
                print "\ny_value"
                print y_value

            x_y_diff = math.fabs(x_value - y_value)

            if debug:
                print "x_y_diff"
                print x_y_diff
            
            if debug:
                print "\nr+1 in list"
                print next_r_value
                print "\nn_r"
                print n_r
                print "\nn_r_plus_1"
                print n_r_plus_1

                
            # calculate variance of n_r values estimates
            variance1 = math.pow(r_value+1, 2)
            variance2 = (float(n_r_plus_1) / float(n_r * n_r))
            variance3 = (1 + (float(n_r_plus_1) / float (n_r)))
            
            variance = math.sqrt(variance1 * variance2 * variance3)
           
            if debug:
                print "\n variance"
                print variance
            
            if x_y_diff > (1.96 * variance):
                r_to_smoothed_r_mapping_dict[r_value] = x_value
            else:
                r_to_smoothed_r_mapping_dict[r_value] = y_value
                switch_flag = True
        
        else:
                '''
                Here the switch to y values will happen. So we need to
                use z_r values based y estimates. 
                '''
                y_value = float(r_value + 1) * \
                        (smoothed_z_r_calc(a_value , b_value, r_value + 1) / \
                        smoothed_z_r_calc(a_value , b_value, r_value))
                
                r_to_smoothed_r_mapping_dict[r_value] = y_value

    if  debug:
        print "r_to_smoothed_r_mapping_dict"
        print r_to_smoothed_r_mapping_dict
            
    '''
    Calculate the value of N_prime
    '''
    N_prime = 0
    
    for r in freq_list:
        
        if r == 0:
            continue
        
        N_prime = N_prime + (r_to_n_mapping_dict[r] * \
                             r_to_smoothed_r_mapping_dict[r])
    
    if debug:
        print "n prime"
        print N_prime

    '''
    Calculate the SGT probability estimate p_r for the each freq.These estimates
    will be stored in a ordered dict storing r to p_r mappings
    '''

    r_to_p_r_mapping_dict = collections.OrderedDict()

    for r in freq_list:
        
        if r == 0:
            r_to_p_r_mapping_dict[0] = P_0
            continue
        
        r_to_p_r_mapping_dict[r] = precision_level((1 - P_0) * \
                        (float(r_to_smoothed_r_mapping_dict[r]) / N_prime),\
                        p_param)

    if debug:
        print "r_to_p_r_mapping_dict"
        print r_to_p_r_mapping_dict

    '''
    Calculate total observed probabllity from the r_to_p_r_mapping_dict.
    For this take sum of all probabilities except the probability when r=0.
    '''
    
    total_observed_prob = 0 

    for r, p in r_to_p_r_mapping_dict.items():
        if r != 0:
            total_observed_prob =  total_observed_prob + p

    if debug:
        print total_observed_prob
    
    return master_dict, r_to_p_r_mapping_dict, r_to_smoothed_r_mapping_dict,\
    total_observed_prob
    
###############################################################################
# End of unigram_prob_sgt_method Function
###############################################################################

###############################################################################
# Function      : reg_analysis(r_to_log_r_mapping_dict , 
#                              r_to_log_Z_mapping_dict)
# Description   : This function calculates a and b values by  regression
#                 analysis of log r and log z values set passed to function.
#                 The method of regression analysis is learnt from the 
#                 tutorial present at:
#                 http://www.schoolworkout.co.uk/documents/s1/Regression.doc
# Arguments     : r_to_log_r_mapping_dict- A dict with set of log r values
#                 r_to_log_Z_mapping_dict- A dict with set of log Z values
# Returns       : a and b values of line of best fit.
###############################################################################

def reg_analysis(r_to_log_r_mapping_dict , r_to_log_Z_mapping_dict):

    '''
    Get the values of log r and log z into separate lists from the dicts 
    passed. Neglect the values when r = 0 as it is not used for regression 
    analysis. The line of best fit will be calculated by plotting log r values 
    at x axis and log z values at y axis. So, I am referring log r values by
    x and log z values as y in the variable names used in this function.
    '''
    x_list = []
    y_list = []

    for r, log_r in r_to_log_r_mapping_dict.iteritems():
        if r != 0:
            x_list.append(log_r)

    for r, log_z in r_to_log_Z_mapping_dict.iteritems():
        if r != 0:
            y_list.append(log_z)

    if debug:
        print x_list
        print y_list
    
    '''
    Get the length of x_list, used a total count of sample points of 
    reg. analysis
    '''
    number_of_sample_points = len(x_list)

    # calculate means of x values and y values
    sum_x = 0
    sum_y = 0
    mean_x = 0
    mean_y = 0

    for counter in range(0,number_of_sample_points):
        sum_x = sum_x + x_list[counter]
        sum_y = sum_y + y_list[counter]

    if debug:
        print "\nsum x and y"
        print sum_x
        print sum_y
        print "\ntotal sample size"
        print number_of_sample_points

    mean_x = float(sum_x) / float(number_of_sample_points)
    mean_y = float(sum_y) / float(number_of_sample_points)

    if debug:
        print "\nmean x and y"
        print mean_x
        print mean_y

    # calculate sum of products of x and y values
    product_sum = 0

    for counter in range(0,number_of_sample_points):
        product_sum = product_sum + (x_list[counter] * y_list[counter])

    if debug:
        print "\nproduct sum"
        print product_sum

    # calculate sum of squares of x values
    square_sum = 0
    
    for counter in range(0,number_of_sample_points):
        square_sum = square_sum + (x_list[counter] * x_list[counter])

    if debug:
        print "\nsquare sum"
        print square_sum

    s_xy_value =  float(product_sum) - (float(sum_x * sum_y) / \
                                        float(number_of_sample_points)) 
    if debug:
        print "\nsxy"
        print s_xy_value

    s_xx_value = float(square_sum) - (float(sum_x * sum_x) / \
                                      float(number_of_sample_points))

    if debug:
        print "\nsxx"
        print s_xx_value

    b_value = float(s_xy_value) / float(s_xx_value)
    a_value = float(mean_y) - (float(b_value) * float(mean_x))

    if debug:
        print a_value
        print b_value

    return a_value, b_value


###############################################################################
# End of reg_analysis Function
###############################################################################

###############################################################################
# Function      : smoothed_z_r_calc(a_param , b_param, freq) 
# Description   : This function calculates the smoother freq z_r for a freq. 
# Arguments     : a_param- a value of line of best fit
#                 b_param- b value of line of best fit
#                 freq - freq to be smoothed
# Returns       : smoothed freq.
###############################################################################

def smoothed_z_r_calc(a_param , b_param, freq):

    '''
    Return the smoothed freq calculated by formula antilog(a + b * log r)
    As base 10 logs are used in this program, this antilog value will be 
    10^(a + b * log r)
    '''
    return math.pow(10, (a_param + (b_param * math.log10(freq))))

###############################################################################
# End of smoothed_z_r_calc Function
###############################################################################

###############################################################################
# Function      : bigram_prob_sgt_method(bigram_freq_mle_dict,
#                 p_param, zero_freq_bigram_count)
# Description   : This function calculates bigram probabilities for input 
#                 data files using SGT method. The SGT method used here is 
#                 further smoothed using Linear Good Smoothing specified in
#                 paper by Gale and Sampson(1995).
# Arguments     : bigram_freq_mle_dict - A dict object mapping unigrams to
#                 to their frequency counts
#                 p_param - Precision level to be used while showing 
#                           probabilities
#                 zero_freq_bigram_count - Freq. of unobserved events
# Returns       : A dict object storing mapping of freq values to bigrams   
#                 A dict object storing mapping of freq values to prob
#                 A dict object storing mapping of freq values to smoothed freq
#                 Total observed prob 
#                 Total unobserved freq
#                 Total unobserved freq
###############################################################################
def bigram_prob_sgt_method(bigram_freq_mle_dict,\
                       p_param, zero_freq_bigram_count):

    ''' 
    The approach used for calculating SGT bigram probabilities is taken
    from Gale and Sampson's Paper.
    '''
    
    '''
    First we need to decide the different frequencies that appear for
    all non-zero freq bigrams. For this, a list freq_list will be maintained. 
    It will just enlist  all different frequencies (r values) existing in 
    bigram_freq_mle_dict. 
    '''
    # Create and initialize freq_list to an empty list
    
    freq_list = []
    
    '''
    Iterate over unigram_freq_mle_dict to get diffrent freq and add them to
    freq_list    
    '''

    if  debug:
        print "\nbigram to freq map"
        print bigram_freq_mle_dict
            
    '''
    Here the in-built function iteritems() fetches both key and value together
    from dict object at the same time over each iteration.
    ''' 

    for bigram_str, bigram_freq in bigram_freq_mle_dict.iteritems():
        if debug:
            print "individual str and freq"
            print bigram_str
            print bigram_freq
        
        # Add bigram frequency to freq_list if it is not already present
        
        if bigram_freq not in freq_list:
            freq_list.append(bigram_freq)
        
    
    # Sort the freq_list in ascending order
    freq_list = sorted(freq_list)

    if debug:
        print "\nfreq_list"
        print freq_list

    '''
    Create the master dict mapping r values to the list of bigrams.
    For this iterate over freq_list and get individual frequencies r.
    Then for each of the individual freq r, find all bigrams with freq r
    from bigram_freq_mle_dict. Add the r and a list of bigrams with r
    freq in master dict as key value pair.
    '''

    master_dict = collections.OrderedDict()

    for r_value in freq_list:
        
        bigrams_list_for_freq = []
        
        for bigram_str, bigram_freq in bigram_freq_mle_dict.iteritems():
                if bigram_freq == r_value:
                    bigrams_list_for_freq.append(bigram_str)

        master_dict[r_value] = bigrams_list_for_freq
    
    if debug:
        print "\nmaster dict"
        print master_dict
 
    '''
    Create a dict storing mapping of r values to frequency of r frequencies
    in our corpus. This mapping is basically from r to n_r.
    To calculate n_r for each r, get the length of list of bigrams stored in
    master dict for each r freq.
    '''
    r_to_n_mapping_dict = collections.OrderedDict()

    for r_value in freq_list:

        if debug:
            print master_dict[r_value]

        n_value = len(master_dict[r_value])

        r_to_n_mapping_dict[r_value] = n_value
    
    if debug:
        print "\nr to n_r mapping dict"
        print r_to_n_mapping_dict

    '''
    Calculate N value or sample size. This will be the sum of products of
    r and n_r values from r to n mapping dict. 
    '''
    
    N_value = 0

    for r, n in r_to_n_mapping_dict.items():

        N_value = N_value + (r * n)

    if debug:
        print "\nN value"
        print N_value
 
    '''
    Calculate Probability of unseen unigrams i.e. P_0. For this, divide the 
    value of frequency of unigrams which appear once (n_1) by sample size.
    Make this P_0 probability to have precision specified by p_param.
    '''
    P_0 = float(r_to_n_mapping_dict[1]) / float(N_value)
    P_0 =  precision_level(P_0, p_param)
    
    if debug:
        print P_0

    '''
    Calculate Z values for each frequency r. Z values for each freq. will be 
    calculated based upon the freq. count of immediately previous r and 
    immediately next r values in r_to_n_mapping_dict. For, the first freq in
    the r_to_n_mapping_dict, the value of previous r will
    be taken as zero. And for last freq in r_to_n_mapping_dict, the value of
    next freq will be (2 * r - immediately previous freq.). For other freq.,
    Z value will be calculated as (2 * n_r / (immediately next freq - 
    immediately prev freq). The mappings of r values to Z values will be stored
    in an ordered dict r_to_Z_mapping
    '''
    r_to_Z_mapping_dict = collections.OrderedDict()

    '''
    Get the length of freq_list to know whether we have arrived
    at last freq while calculating Z values. A counter will also
    be maintained to facilitate this.
    '''
    total_freq = len(freq_list)
 
    freq_counter = 0

    for r, n in r_to_n_mapping_dict.items():
    
        # process first freq.
        if freq_counter == 0:
            if debug:
                print "\nfirst freq: " + str(r)

            prev_freq = 0
            next_freq = freq_list[freq_counter + 1]
            

        # process last freq.
        elif freq_counter == (total_freq - 1):
            if debug:
                print "\nlast freq: " + str(r)
            
            prev_freq = freq_list[freq_counter - 1]
            next_freq = (2 * r) - prev_freq 

        # process intermediate freq.
        else:
            if debug:
                print "\nintermidiate freq: " + str(r)
        
            prev_freq = freq_list[freq_counter - 1]
            next_freq = freq_list[freq_counter + 1]

        
       
        # calculate z values and insert them in r_to_Z_mapping_dict
        Z_value = float(2 * n) / float(next_freq - prev_freq)
        r_to_Z_mapping_dict[r] = Z_value
        
        if debug:
            print "counter value"
            print freq_counter
            print "prev freq"
            print prev_freq
            print "next freq"
            print next_freq
            print "Z_value"
            print Z_value

        freq_counter = freq_counter + 1

    if debug:
        print "\nr to z_r mapping dict"
        print r_to_Z_mapping_dict
 
    '''
    Create two ordered dicts to store log of freq r and log z and fill in
    them.
    '''

    r_to_log_r_mapping_dict = collections.OrderedDict()
    r_to_log_Z_mapping_dict = collections.OrderedDict()
    
    for r in freq_list:
        
        r_to_log_r_mapping_dict[r] = math.log10(r)
        r_to_log_Z_mapping_dict[r] = math.log10(r_to_Z_mapping_dict[r])

    
    if debug:
        print "\nr to log r mapping dict"
        print r_to_log_r_mapping_dict
        print "\nr to log Z mapping dict"
        print r_to_log_Z_mapping_dict
    
    '''
    Use the linear regression analysis to find line of best fit to log r
    and log z values. This will give us a and b param values.
    For doing regression analysis, call reg_analysis method and pass the
    log r , log z values, present in r_to_log_r_mapping_dict and 
    r_to_log_Z_mapping_dict, to it.
    '''
    
    a_value, b_value = reg_analysis(r_to_log_r_mapping_dict ,\
                                    r_to_log_Z_mapping_dict)
    
    if debug:
        print"\na_value, b_value"
        print a_value 
        print b_value

    '''
    Start calculating smoothed frequencies r* for each r. The value of r*
    is decided based upon either by n_r values estimates 
    or smoothed z_r values estimates.
    
    As per the Gale,Sampson's Paper, we need to switch between above two 
    estimates. If the value of r* based upon n_r values estimates 
    is x and that based upon smoothed z_r values estimates is y. 
    Then we decide the switch depending upon absolute difference of x and y 
    compared with 1.96 times the standard deviation of n_r values estimates.
    
    The values of smoothed r* freq. will be stored in a ordered dict object
    mapping r values to their r* values. 
    '''

    r_to_smoothed_r_mapping_dict = collections.OrderedDict()

    # create a flag to decide switching between x and y values
    switch_flag = False

    # iterate over freq_list to calculate smoothed freq.
    for counter in range(0,total_freq):
       
        r_value = freq_list[counter]

        # if switch_flag is false then calculate both x and y estimates
        
        if switch_flag == False:

            next_r_value = freq_list[counter + 1]
            n_r = r_to_n_mapping_dict[r_value]
            n_r_plus_1 = r_to_n_mapping_dict[next_r_value]

            if debug:

                print "\nr+1 in list"
                print next_r_value
                print "\nn_r"
                print n_r
                print "\nn_r_plus_1"
                print n_r_plus_1
            
            x_value = float(r_value + 1) * (float(n_r_plus_1) / float(n_r)) 
            
            if debug:
                print "\n x value"
                print x_value
            
            if debug:
                print "\n smoothed z_r + 1"
                print smoothed_z_r_calc(a_value , b_value, next_r_value)
                print "\n smoothed z_r"
                print smoothed_z_r_calc(a_value , b_value, r_value)

            y_value = float(r_value + 1) * \
                      (smoothed_z_r_calc(a_value , b_value, r_value + 1) / \
                       smoothed_z_r_calc(a_value , b_value, r_value))
            
            if debug:
                print "\ny_value"
                print y_value

            x_y_diff = math.fabs(x_value - y_value)

            if debug:
                print "x_y_diff"
                print x_y_diff
            
            if debug:
                print "\nr+1 in list"
                print next_r_value
                print "\nn_r"
                print n_r
                print "\nn_r_plus_1"
                print n_r_plus_1

                
            # calculate variance of n_r values estimates
            variance1 = math.pow(r_value+1, 2)
            variance2 = (float(n_r_plus_1) / float(n_r * n_r))
            variance3 = (1 + (float(n_r_plus_1) / float (n_r)))
            
            variance = math.sqrt(variance1 * variance2 * variance3)
           
            if debug:
                print "\n variance"
                print variance
            
            if x_y_diff > (1.96 * variance):
                r_to_smoothed_r_mapping_dict[r_value] = x_value
            else:
                r_to_smoothed_r_mapping_dict[r_value] = y_value
                switch_flag = True
        
        else:
                '''
                Here the switch to y values will happen. So we need to
                use z_r values based y estimates.
                '''

                y_value = float(r_value + 1) * \
                        (smoothed_z_r_calc(a_value , b_value, r_value + 1) / \
                        smoothed_z_r_calc(a_value , b_value, r_value))
                
                r_to_smoothed_r_mapping_dict[r_value] = y_value

    if  debug:
        print "r_to_smoothed_r_mapping_dict"
        print r_to_smoothed_r_mapping_dict

    '''
    Calculate the value of N_prime
    '''
    N_prime = 0
    
    for r in freq_list:
        
        N_prime = N_prime + (r_to_n_mapping_dict[r] * \
                             r_to_smoothed_r_mapping_dict[r])
    
    if not debug:
        print "n prime"
        print N_prime
 
    
    '''
    Calculate the SGT probability estimate p_r for the each freq.These estimates
    will be stored in a ordered dict storing r to p_r mappings
    '''

    r_to_p_r_mapping_dict = collections.OrderedDict()

    for r in freq_list:
        
        r_to_p_r_mapping_dict[r] = precision_level((1 - P_0) * \
                        (float(r_to_smoothed_r_mapping_dict[r]) / N_prime),\
                        p_param)

    if debug:
        print "r_to_p_r_mapping_dict"
        print r_to_p_r_mapping_dict

    '''
    Calculate total observed probabllity from the r_to_p_r_mapping_dict.
    For this take sum of all probabilities.
    '''
    
    total_observed_prob = 0 

    for r, p in r_to_p_r_mapping_dict.items():
        total_observed_prob =  total_observed_prob + p

    if debug:
        print total_observed_prob
    
    
    '''
    Calculate freq for unobserved events.
    '''

    unobserved_freq = float(r_to_n_mapping_dict[1]) / \
                      float(zero_freq_bigram_count)

    return master_dict, r_to_p_r_mapping_dict, r_to_smoothed_r_mapping_dict,\
    total_observed_prob, unobserved_freq , P_0
     
 
###############################################################################
# End of bigram_prob_sgt_method Function
###############################################################################

###############################################################################
# Function      : main()
# Description   : Entry point for the project.
# Arguments     : None. Command Line Arguments in Python are retrieved from
#                 sys.argv variable of sys module.
# Returns       : None.
###############################################################################
def main():
    
    ''' 
        Info. on processing the command line arguments given to the program.
        e.g. Sample usage of program is :
        
        python Ngram_Modelling.py -N 2 -P 4 -w linux.w -i 28054.txt pg2554.txt

        The desciption of the command line arguments intended for this 
        program is as follows:
        
        1) -N : Number of top ranked N- Gram probabilities to be displayed in
        output 
        2) -P : Precision level used in displaying N-Gram probabilities
        3) -w : word list file used to denote the vocabulary for the N-Gram 
        model.
        4) -i : input data files. Multiple data files can be provided to this 
        switch separated by space.
        
        These command line arguments are accessed in Python using sys.argv 
        variable. This variable is a Python list. Individual command line 
        arguments can be accessed using indices over this list.
        e.g. For the sample usage mentioned above, the value of sys.argv is:
        
        ['Ngram_Modelling.py', '-N', '2', '-P', '4', '-w', 'linux.w', '-i', 
        '28054.txt', 'pg2554.txt', 'pg2600.txt']

        Value for -N argument can be accessed by sys.argv[2]. Value for 
        -P arguments can be accessed by sys.argv[4]

        The count of all command line argument can be calculated by measuring
        size of sys.argv list. For this, in-built 'len' function is used.  
        Thus len(sys.argv) gives the total count of command line arguments.
        
    '''
    
    '''
    Check if any command line argument is passed to program. If not 
    throw error showing proper sample usage. 'Print' statement is used for
    printing to std output in Python.
    '''

    if (len(sys.argv) > 1):
        if debug:
            print "At least one parameter passed to program !"

        '''
        Get the values for N, P and word file from sys.argv and store them 
        into different variables.
        '''

        n_param = sys.argv[2]
        p_param = sys.argv[4]
        word_file_name = sys.argv[6]
        
        if debug:
            print n_param
            print p_param
            print word_file_name

        '''
        As multiple input data files can be passed to the program, we need 
        to store the name of these files in a Python list. For this, first
        separate out the names of input files from sys.argv list. The names
        of input data files start from 8th element of sys.argv list (starting 
        indices with zero) and span till the end of it. So, we can slice 
        the subset of sys.argv from 8th element to the end using 
        slice notation of Python. Slice notation is used to get subset of a 
        list specifying start and stop indices boundries. 
        e.g.

        If we do sys.argv[8:11], then we will get another list containing 
        8th, 9th and 10th element of the sys.argv.

        If we do sys.argv[8:] (not specifying stop boundry), we will get 
        another list containing elements starting from 8th position till end
        of sys.argv list.
        '''

        # Get list of input files from sys.argv list and store it another list
        input_files_list = sys.argv[8:]

        if debug:
            print input_files_list
    
        '''
        First make copy of word file. This copy will be cleaned up and used for
        any processing subsequently. This way original contents of word file 
        will be retained and its copy will only be modified.
        For copying the word file, call create_copy function. The name of 
        copy file returned from create_copy function will be stored in a 
        separate variable word_copy_file.
        '''
        
        word_copy_file = create_copy(word_file_name)

        if debug:
            print  word_copy_file

        '''
        Similarly make copies of all input data files. For this, iterate over 
        the input_files_list using for loop and call create_copy to create 
        copies of individual data file from the input_files_list.
        The names of each copy file will be stored into another list 
        named as 'input_copy_files_list' for futher usage in program.
        So, this variable 'input_copy_files_list' will be initialized before
        creating copies.
        '''

        # initialize copy list for input data files to an empty list
        input_copy_files_list = []
        
        for input_file in input_files_list:
            
            '''
            Create copies of input files and store the names of copy files 
            into 'input_copy_files_list' using in-built append function.
            append function adds the an element, which is passed as parameter 
            to it, into a list.
            '''
            input_copy_files_list.append(create_copy(input_file))
        
        if debug:
            print input_copy_files_list
        
        # Start cleaning all copy files
    
        '''
        Clean the word copy file. Cleaning will convert all uppercase 
        characters from the copy file and replace all non-alphabetic 
        characters into a space. Cleaning will be done by calling clean_file 
        function on word copy file.
        '''
        clean_file(word_copy_file)

        '''
        Similarly clean all copies of input data files. For this, iterate
        over the input_copy_files_list and call clean_file function to
        clean each element of that list.
        '''

        for input_copy_file in input_copy_files_list:
            clean_file(input_copy_file)

        
        '''
        Create a file by combining contents of word list copy file and  
        all input data files. This file is referred to here as vocab file,
        as it will be used for creating vocabulary for our N-Gram Model.

        The vocab file is named vocab_file.txt. It will have contents of word  
        list copy file appended to contents of all cleaned input files. 

        open function with append mode will create the vocab file. 
        Append mode can be used to append the contents all input file to this
        single vocab file.
        '''

        vocab_file_name = 'vocab_file.txt'
        
        # remove vocab  file if it is present from previous runs
            
        if os.path.exists(vocab_file_name):
            os.remove(vocab_file_name)


        vocab_file_handle = open(vocab_file_name, 'a')

        # open, read and copy the contents of word copy file into vocab file
        word_copy_file_handle = open(word_copy_file, 'r')

        vocab_file_handle.write(word_copy_file_handle.read())

        # close word copy file
        word_copy_file_handle.close()

        ''' 
        iterate over all input copy files and append their content to 
        vocab file. For each iteration, individual copy file needs
        to be opened, read and closed.
        '''

        for input_copy_file in input_copy_files_list:
            input_copy_file_handle = open(input_copy_file, 'r')
            vocab_file_handle.write(input_copy_file_handle.read())
            input_copy_file_handle.close()

        # close vocab file finally
        vocab_file_handle.close()

        
        '''
        Create a single data file by appending contents of each cleaned
        input data files only. This single data file will contain all word 
        tokens from all input data file. This data file does NOT contain 
        words tokens from word list file. We will use this single data file 
        to calculate different frequency counts further in program.
        
        The name single data file will be 'data_file.txt'.
        '''
    
        data_file_name = 'data_file.txt'

        # remove single data file if it is present from previous runs
        
        if os.path.exists(data_file_name):
            os.remove(data_file_name)

        data_file_handle = open(data_file_name, 'a')

        '''
        iterate over all input copy files and append their content to 
        single data file. For each iteration, individual copy file needs
        to be opened, read and closed.
        '''

        for input_copy_file in input_copy_files_list:
            input_copy_file_handle = open(input_copy_file, 'r')
            data_file_handle.write(input_copy_file_handle.read())
            input_copy_file_handle.close()

        # close single data file
        data_file_handle.close()

        '''
        Get total number of tokens and types from the vocab file created 
        earlier. 
        '''

        '''
        To calculate the number of tokens, read vocab file into a string and 
        split that string by whitespaces. To split the string, in-built 
        split function is used. It takes a parameter specifying delimiter
        by which string should be split. If no parameter is passed, then
        string will be split by whitespaces by default.

        split function returns a list containing all substrings of split
        strings. The length of this list will give us total number of tokens
        in vocab file
        '''

        vocab_file_handle = open(vocab_file_name, 'r')

        tokens_list = vocab_file_handle.read().split()
        
        if debug:
            print len(tokens_list)

        '''
        Print the total number of tokens. Format the output with comma-
        grouping using in-built format function. The usage of format function
        was referred from a question asked on stackoverflow.com online forum.

        The usage of format function was shown in the reply by stackoverflow 
        user 'martineau'. This complete discussion related to comma-grouping
        in output is present online at :

        http://stackoverflow.com/questions/3909457/
        whats-the-easiest-way-to-add-commas-to-an-integer-in-python
        '''

        print "\ntotal tokens\t=\t" + format(len(tokens_list), ",d")

        '''
        To calculate the total number of types, find out the distinct elements
        of the tokens_list. For this, inbuilt 'set' function can be used. set
        function takes a list and converts it to a set eliminating duplicate 
        elements of the list. This set is required to be converted into list
        again for creating list out of this set. Usage of set function to 
        find distinct elements from a list was borrowed from a blog entry 
        present online at :
        
        http://mattdickenson.com/2011/12/31/find-unique-values-in-list-python/
        
        The length of new list, created with distict elements from token_list 
        will give us the total number of types in vocabulary.
        '''

        '''
        I had tried to design my own approach for finding distinct element 
        below. It worked fine but it took a long time to execute as we have
        a very large number of tokens. So I commented out my code and used
        the set approach mentioned in the blog given above.

        types_list = []

        for token in tokens_list:
            if token not in types_list:
                types_list.append(token)
                print token
        '''

        types_list = list(set(tokens_list))

        if debug:
            print len(types_list)
        
        # Print total number of types with formatted output
        print "total types\t=\t" + format(len(types_list), ",d")


        '''
        Start calculating MLE probabilities for unigrams. For this, 
        a function unigram_prob_mle_method will be called. Pass the data file
        (which represents our sample) , types list (which represent the
        vocabulary of the list) and precision level (specified by command 
        line argument P) to be attained in probabilities.
        Get the two dicts, one containing mapping of unigrams to their freq. 
        and another containing mapping of unigrams to their MLE prob from this
        function. Also get the list of words in data file.
        '''

        unigram_freq_mle_dict, unigram_prob_mle_dict , words \
                                                = unigram_prob_mle_method\
                                                       (data_file_name, \
                                                         types_list,  \
                                                         p_param)
        if debug:
            print "output freq list"
            print unigram_freq_mle_dict

        '''
        Sort the dict object having mappings of unigrams to their MLE prob
        in descending order of MLE prob. This is required to fetch top N 
        probabilities from this map. Here, the probabilities are value of the
        dict objects. The code used for this, has been borrowed from the
        an answer to a question on 'sorting dict by values' posted at 
        www.stackoverflow.com forum by stackoverflow user "Devin Jeanpierre". 
        This question can be found at this link:

        www.stackoverflow.com/questions/613183/
        python-sort-a-dictionary-by-value
        '''
        
        sorted_dict = sorted(unigram_prob_mle_dict.iteritems(), \
                             key=operator.itemgetter(1), reverse=True)
        
        if debug:
            print sorted_dict
        
        '''
        Print top N unigram probabilities and their freq. to output. 
        Here N refers to the -N command line argument's value. 
        To print top N probabilities (which can involves more than N result
        due to ties), iterate over the sorted dict while maintaining a 
        separate counter. This counter will be incremented whenever a 
        new probability is encountered. 
        
        When counter equals to N iterating process will stop 
        giving only top N unigram probabilities.Fetch corresponding unigram 
        frequencies from unigram_freq_mle_dict returned by 
        unigram_prob_mle_method above.
        '''
        print "\nTop " + n_param + " unigrams (MLE)\n" 
        
        # iterate over the sorted dict to get top N unigram probabilities

        current_probability = sorted_dict[0][1]

        prob_counter = 1

        for counter in range (0,len(types_list)):

            if prob_counter ==  int(n_param):
                if debug:
                    print "inside if"
                break

            unigram_str = sorted_dict[counter][0]
            unigram_prob = sorted_dict[counter][1]
            unigram_freq = unigram_freq_mle_dict[unigram_str]

            '''
            For table like pretty printing, format specifier : is used 
            as described on Python 2.7.3 documentation present at the link:
            
            http://docs.python.org/tutorial/inputoutput.html
            
            The sample code referred from above link was
            print '{0:10} ==> {1:10d}'.format(name, phone)
            '''
            
            
            print '{0:40}      {1:40}     '.format(unigram_str, \
                                                   str(unigram_prob)) + \
                                                format(unigram_freq, ",d")

            if current_probability != unigram_prob:
                current_probability = unigram_prob
                prob_counter = prob_counter + 1

        print "\n"

        '''
        Start calculating unigram probabilities using Simple Good Turing
        method (SGT method). The SGT method used in this program uses
        linear regression smoothing as mentioned in the paper "Good-Turing
        Frequency Estimation Without Tears" by Gale and Sampson (1995).
        For calculating, a function unigram_prob_sgt_method will be called.
        This function takes following parameters:

        1)  A dict object storing mapping of unigrams to their frequency
            count (This dict object was returned by unigram_prob_mle_method
            function above). 
        2)  data file name
        3)  precision level to be attained while calculating probabilities
        
        '''

        master_dict, r_to_p_r_mapping_dict, r_to_smoothed_r_mapping_dict,\
        total_observed_prob = unigram_prob_sgt_method(unigram_freq_mle_dict,\
                                            data_file_name,\
                                            p_param)

        '''
        Sort r_to_p_r_mapping_dict in descending order of prob.
        '''
        sorted_dict_GT = sorted(r_to_p_r_mapping_dict.iteritems(), \
                             key=operator.itemgetter(1), reverse=True)
  
        if debug:
            print master_dict
            print sorted_dict_GT
            print r_to_smoothed_r_mapping_dict
 
        '''  
        Print top N unigram GT probabilities and their freq. to output. 
        '''
        
        print "\nTop " + n_param + " unigrams (GT)\n" 

        for counter in range(0,int(n_param)):
            
            if counter == len(sorted_dict_GT):
                break
            
            unigram_freq = sorted_dict_GT[counter][0]
            unigram_prob = sorted_dict_GT[counter][1]
            unigram_smoothed_freq = r_to_smoothed_r_mapping_dict[unigram_freq]

            unigram_str_list = master_dict[unigram_freq]
            
            for unigram in unigram_str_list:
                print '{0:40}      {1:40}     '.format(unigram, \
                                                str(unigram_prob)) + \
                                                str(precision_level(\
                                                    unigram_smoothed_freq,\
                                                    p_param))
 
        
        '''
        Print total observed and unobserved prob.
        '''

        print "\ntotal observed probability\t:\t" +  str(total_observed_prob)

        print "total unobserved probability\t:\t" + str(r_to_p_r_mapping_dict[0])


        '''
        Start forming bigrams from the vocabulary. Vocabulary is represented 
        by types_list in this program. Total number of bigrams in vocab will
        be number of types * number of types. We want to form bigrams by all
        possible pairwise combinations of types. For this, we basically need
        permutation with repeatation allowed. This functionality can be 
        achieved by taking cartesian product of list of types with itself.
        
        Python provides product() function from itertools module for taking 
        cartesian product. I have learned about it from Python 2.7.3 
        documentation, while dealing with a problem related to 
        permutations earlier for some other purpose.

        The usage of product function is described at the Python 2.7.3
        documentation present at the link :

        http://docs.python.org/library/itertools.html#itertools.product

        This function returned the tuples of unigrams.
        e.g. If we take product of a list [ceasar, brutus] with
        itself, we will get bigrams like this:
        (ceasar,ceasar), (ceasar,brutus), (brutus, ceasar), (brutus, brutus) 
        '''

        bigrams_list = itertools.product(types_list, repeat = 2)
       

        ''' 
        Start calculating MLE probabilities for bigrams. For this, 
        a function bigram_prob_mle_method will be called. Pass the data file
        (which represents our sample) , types list (which represent the
        vocabulary of the list), unigram_freq_mle_dict (to get freq of 
        unigrams), bigrams_list and precision level (specified by command line 
        argument P) to be attained in probabilities.
        
        Get the two dicts, one containing mapping of bigrams to their freq. 
        and another containing mapping of bigrams to their MLE prob from this
        function.
        '''

        bigram_freq_mle_dict, bigram_prob_mle_dict = bigram_prob_mle_method\
                                                       (data_file_name, \
                                                        types_list,  \
                                                        unigram_freq_mle_dict,\
                                                        bigrams_list,\
                                                        p_param, words)

        if debug:
            print "op freq list"
            print bigram_freq_mle_dict
        
        '''
        Sort the dict object having mappings of bigrams to their MLE prob
        in descending order of MLE prob. This is required to fetch top N 
        probabilities from this map.
        '''
        
        sorted_dict_bigram = sorted(bigram_prob_mle_dict.iteritems(), \
                             key=operator.itemgetter(1), reverse=True)
        
        if debug:
            print sorted_dict_bigram
            print "Complete element of bigram list:"
            print sorted_dict_bigram[0]
            print "Bigram pair:"
            print sorted_dict_bigram[0][0]
            print "Bigram probability:"
            print sorted_dict_bigram[0][1]
            print "Individual member of bigram:"
            print sorted_dict_bigram[0][0][0]
            print len(sorted_dict_bigram) 

        '''
        Print top N bigram probabilities and their freq. to output. 
     
        To print top N probabilities (which can involves more than N result
        due to ties), iterate over the sorted bigram dict while maintaining a 
        separate counter. This counter will be incremented whenever a 
        new probability is encountered. 
        
        When counter equals to N iterating process will stop 
        giving only top N bigram probabilities.Fetch corresponding bigram 
        frequencies from bigram_freq_mle_dict returned by 
        bigram_prob_mle_method above.
        '''
        print "\nTop " + n_param + " bigrams (MLE)\n"
        
       # iterate over the sorted dict to get top N unigram probabilities

        current_probability = sorted_dict_bigram[0][1]  
       
        prob_counter = 1
        
        for counter in range (0,len(sorted_dict_bigram)):

            if prob_counter ==  int(n_param):
                if debug:
                    print "inside if"
                break

            bigram_pair = sorted_dict_bigram[counter][0]
            bigram_str = bigram_pair[0] + " " + bigram_pair[1] 
            bigram_prob = sorted_dict_bigram[counter][1]
            bigram_freq = bigram_freq_mle_dict[bigram_pair]

            print '{0:40}      {1:40}     '.format(bigram_str, \
                                                   str(bigram_prob)) + \
                                                format(bigram_freq, ",d")

            if current_probability != bigram_prob:
                current_probability = bigram_prob
                prob_counter = prob_counter + 1

        print "\n"
    
        '''
        Start calculating the SGT probabilities for bigrams.
        
        For calculatin prob, a function bigram_prob_sgt_method will be called.
        This function takes following parameters:

        1)  A dict object storing mapping of bigrams to their frequency
            count (This dict object was returned by bigram_prob_mle_method
            function above). 
        2)  precision level to be attained while calculating probabilities
        3)  the count of zero freq bigrams.
        '''

        '''
        First, get the count of zero-frequency bigrams. 
        This number can be calculated by substracting count od non-zero freq 
        count (given by the size of bigram_freq_mle_dict) from the square 
        of number of total types.
        '''

        zero_freq_bigram_count = (len(types_list) * len(types_list)) - \
                                  len(bigram_freq_mle_dict)
        
        if debug:
            print zero_freq_bigram_count
        
        master_dict, r_to_p_r_mapping_dict, r_to_smoothed_r_mapping_dict,\
        total_observed_prob, unobserved_freq,P_0 = \
                                bigram_prob_sgt_method(bigram_freq_mle_dict,\
                                            p_param, zero_freq_bigram_count)
        
        '''
        Sort r_to_p_r_mapping_dict in descending order of prob.
        '''
        sorted_dict_GT = sorted(r_to_p_r_mapping_dict.iteritems(), \
                             key=operator.itemgetter(1), reverse=True)
  
        if debug:
            print sorted_dict_GT
 
        '''  
        Print top N bigram GT probabilities and their freq. to output. 
        '''
        
        print "\nTop " + n_param + " bnigrams (GT)\n" 

        for counter in range(0,int(n_param)):
            
            if counter == len(sorted_dict_GT):
                break
            
            bigram_freq = sorted_dict_GT[counter][0]
            bigram_prob = sorted_dict_GT[counter][1]
            bigram_smoothed_freq = r_to_smoothed_r_mapping_dict[bigram_freq]

            bigram_str_list = master_dict[bigram_freq]
            
            for bigram in bigram_str_list:
                print '{0:40}      {1:40}     '.format(bigram[0] + " " + \
                                                bigram[1], \
                                                str(bigram_prob)) + \
                                                str(precision_level(\
                                                    bigram_smoothed_freq,\
                                                    p_param))
 
        
        '''
        Print total observed and unobserved prob.
        '''

        print "\ntotal observed probability\t:\t" +  str(total_observed_prob)

        print "total unobserved probability\t:\t" + str(P_0)


    
    else:
        if debug:
            print "No parameter passed to the program !"

        print "\n\tPlease provide proper inputs to the program !"
        print "\tSample usage: "
        print "\tpython Ngram_Modelling.py -N 2 -P 4 -w linux.w -i 28054.txt\n"


##############################################################################
# End of main function
##############################################################################

'''
Boilerplate syntax to specify that main() method is the entry point for 
this program. Python allows any user-defined method to be entry point for a
program. Following syntax is used for this. 
To keep up with std. conventions, I have used function with name main 
as entry point.
'''

if __name__ == '__main__':
 
    main()

##############################################################################
# End of Ngram_Modelling.py program
##############################################################################
