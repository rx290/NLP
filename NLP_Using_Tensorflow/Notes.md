# NLP Using Tensorflow

## Tokenization

        The process of representing words in a way that a computer can process them with a view to later training a neural network that can understand their meaning is called tokenization.
        How is it done you may ask?
        It is done by using an encoding scheme, which means that the words are represented by some numbers, some famous encoding techniques are ASCII, Unicode etc.

Sample code:

    # importing tensorflow
    import tensorflow as tf
    
    # importing keras
    from tensorflow import keras
    
    # importing Tokenizer
    from tensorflow.keras.preprocessing.text import Tokenizer as tknz

    # Corpus/Sentences that we want to encode
    sentences = [
        "I love my dog",
        "I love my cat"
    ]

    # creating an instance of the tokenizer

    tokenizer = tknz(num_words = 100)
    
    """
    The parameter passed here is telling the tokenizer to keep n number of unique words in its memory
    """
    
    # we will now be letting the tokenizer to go through our corpus/sentences and encode them
    tokenizer.fit_on_texts(sentences)
    
    # we can then see the indexed/encoded words using the attribute word_index
    word_index = tokenizer.word_index
    
    print(word_index)
    # it will print a dictionary with all the encoded data presented in key value pair.

_ _ _

## Sequencing - Turning sentences/corpus into data

     The process of finding similarities between tokenized collection i.e. sentences and creating a map for the arrangement is called sequencing.

     for example: 
     1. I love my dog
     2. I love my cat

    can be encoded like this:

    1. 1 2 3 4
    2. 1 2 3 5

    Then imagine we have a corpus like this:
    1. I love my dog
    2. I love my cat
    3. You love my cat!
    4. Do you think my cat is amazing?

    which can be encoded like this

    word index ={'amazing':10,'dog':3, 'you':5, 'cat': 6,
    'think':8, 'i': 4, 'is': 9, 'my':1, 'do':7,
    'love':2}

    Sequenced sentences would be like this:
     [[4,2,1,3],[4,2,1,6],[5,2,1,6],[5,8,1,6,9,10]]
    or in a more formatted way:

    1. 4 2 1 3
    2. 4 2 1 6
    3. 5 2 1 6
    4. 5 8 1 6 9 10

    That is what sequencing is all about and now this data is ready for the neural network training.

Sample code:

    # importing tensorflow
    import tensorflow as tf
    
    # importing keras
    from tensorflow import keras
    
    # importing Tokenizer
    from tensorflow.keras.preprocessing.text import Tokenizer as tknz

    # Corpus/Sentences that we want to encode
    sentences = [
        "I love my dog",
        "I love my cat",
        "You love my cat!',
        'Do you think my cat is amazing?'
    ]

    # creating an instance of the tokenizer

    tokenizer = tknz(num_words = 100)
    
    """
    The parameter passed here is telling the tokenizer to keep n number of unique words in its memory
    """
    
    # we will now be letting the tokenizer to go through our corpus/sentences and encode them
    tokenizer.fit_on_texts(sentences)
    
    # we can then see the indexed/encoded words using the attribute word_index
    word_index = tokenizer.word_index

    # we will now generate the sequences for the corpus/sentences that are their in our data frame using the attribute texts_to_sequences
    sequences = tokenizer.texts_to_sequences(sentences)
    
    print(word_index)
    # it will print a dictionary with all the encoded data presented in key value pair.