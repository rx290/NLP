# NLP Using Tensorflow

    Tokenization:
        The process of representing words in a way that a computer can process them with a view to later training a neural network that can understand their meaning is called tokenization.
        How is it done you may ask?
        It is done by using an encoding scheme, which means that the words are represented by some numbers, some famous encoding techniques are ASCII, Unicode etc.

Sample code:
   `` # importing tensorflow
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
``