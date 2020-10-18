#Setup

#Import TensorFlow and other libraries

import tensorflow as tf

import numpy as np
import os
import time

#Download the Shakespeare dataset

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

#Read the data

# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print('Length of text: {} characters'.format(len(text)))

#Length of text: 1115394 characters

# Take a look at the first 250 characters in text
print(text[:250])

#First Citizen:
#Before we proceed any further, hear me speak.

#All:
#Speak, speak.

#First Citizen:
#You are all resolved rather to die than to famish?

#All:
#Resolved. resolved.

#First Citizen:
#First, you know Caius Marcius is chief enemy to the people.

# The unique characters in the file
vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))

#65 unique characters

#Process the text

#Vectorize the text

#Before training, we need to map strings to a numerical representation.
#Create two lookup tables: one mapping characters to numbers, and another for numbers to characters.

# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

#Now we have an integer representation for each character.
# Notice that you mapped the character as indexes from 0 to len(unique).

print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')

#{
#  '\n':   0,
#  ' ' :   1,
#  '!' :   2,
#  '$' :   3,
#  '&' :   4,
#  "'" :   5,
#  ',' :   6,
#  '-' :   7,
#  '.' :   8,
#  '3' :   9,
#  ':' :  10,
#  ';' :  11,
#  '?' :  12,
#  'A' :  13,
#  'B' :  14,
#  'C' :  15,
#  'D' :  16,
#  'E' :  17,
#  'F' :  18,
#  'G' :  19,
# ...
#}

# Show how the first 13 characters from the text are mapped to integers
print('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

#Create training examples and targets

# Next divide the text into example sequences.
# Each input sequence will contain seq_length characters from the text.
# For each input sequence, the corresponding targets contain the same length of text, except shifted one character to the right.
#So break the text into chunks of seq_length+1.
# For example, say seq_length is 4 and our text is "Hello". The input sequence would be "Hell", and the target sequence "ello".
#To do this first use the tf.data.Dataset.from_tensor_slices function to convert the text vector into a stream of character indices.

# The maximum length sentence you want for a single input in characters
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
    print(idx2char[i.numpy()])

#The batch method lets us convert these individual characters to sequences of the desired size.

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))

#For each sequence, duplicate and shift it to form the input and target text by using the map method to apply a simple function to each batch.

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

#Print the first example input and target values:

for input_example, target_example in  dataset.take(1):
    print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    print('Target data:', repr(''.join(idx2char[target_example.numpy()])))

#Input data:  'First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou'
#Target data: 'irst Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou '

#Each index of these vectors is processed as a one time step.
# For the input at time step 0, the model receives the index for "F" and tries to predict the index for "i" as the next character.
# At the next timestep, it does the same thing but the `RNN` considers the previous step context in addition to the current input character.

for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

#Step    0
#  input: 18 ('F')
#  expected output: 47 ('i')
#Step    1
#  input: 47 ('i')
#  expected output: 56 ('r')
#Step    2
#  input: 56 ('r')
#  expected output: 57 ('s')
#Step    3
#  input: 57 ('s')
#  expected output: 58 ('t')
#Step    4
#  input: 58 ('t')
#  expected output: 1 (' ')

#Create training batches

#You used tf.data to split the text into manageable sequences.
# But before feeding this data into the model, you need to shuffle the data and pack it into batches.

# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Build The Model

#Use tf.keras.Sequential to define the model. For this simple example three layers are used to define our model:

#tf.keras.layers.Embedding: The input layer. A trainable lookup table that will map the numbers of each character to a vector with embedding_dim dimensions;
#f.keras.layers.GRU: A type of RNN with size units=rnn_units (You can also use an LSTM layer here.)
#tf.keras.layers.Dense: The output layer, with vocab_size outputs.

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

#For each character the model looks up the embedding, runs the GRU one timestep with the embedding as input, and applies the dense layer to generate logits predicting the log-likelihood of the next character.

#Try the model

#First check the shape of the output:

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

(64, 100, 65) # (batch_size, sequence_length, vocab_size)

model.summary()

#Model: "sequential"
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #
#=================================================================
#embedding (Embedding)        (64, None, 256)           16640
#_________________________________________________________________
#gru (GRU)                    (64, None, 1024)          3938304
#_________________________________________________________________
#dense (Dense)                (64, None, 65)            66625
#=================================================================
#Total params: 4,021,569
#Trainable params: 4,021,569
#Non-trainable params: 0
#_________________________________________________________________

#To get actual predictions from the model you need to sample from the output distribution, to get actual character indices.
# This distribution is defined by the logits over the character vocabulary.

#Try it for the first example in the batch:

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

#This gives us, at each timestep, a prediction of the next character index:

#array([41, 60,  3, 31, 47, 21, 61,  6, 56, 42, 39, 40, 52, 60, 37, 37, 27,
#       11,  6, 56, 64, 62, 43, 42,  6, 34,  1, 30, 16, 45, 46, 11, 17,  8,
#       26,  8,  1, 46, 37, 21, 37, 53, 34, 49,  5, 58, 11,  9, 42, 62, 14,
#       56, 56, 30, 31, 32, 63, 53, 10, 23, 35,  5, 19, 19, 46,  3, 23, 63,
#       61, 11, 57,  0, 35, 48, 32,  4, 37,  7, 48, 23, 39, 30, 20, 26,  1,
#       52, 57, 23, 46, 56, 11, 22,  7, 47, 16, 27, 38, 51, 55, 28])

#Decode these to see the text predicted by this untrained model:

print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))

#Input:
# "dness! Make not impossible\nThat which but seems unlike: 'tis not impossible\nBut one, the wicked'st c"

#Next Char Predictions:
# "cv$SiIw,rdabnvYYO;,rzxed,V RDgh;E.N. hYIYoVk't;3dxBrrRSTyo:KW'GGh$Kyw;s\nWjT&Y-jKaRHN nsKhr;J-iDOZmqP"

#Train the model

#At this point the problem can be treated as a standard classification problem.
# Given the previous RNN state, and the input this time step, predict the class of the next character.

#Attach an optimizer, and a loss function

#The standard tf.keras.losses.sparse_categorical_crossentropy loss function works in this case because it is applied across the last dimension of the predictions.

#Because your model returns logits, you need to set the from_logits flag.

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())

#Prediction shape:  (64, 100, 65)  # (batch_size, sequence_length, vocab_size)
#scalar_loss:       4.174373

#Configure the training procedure using the tf.keras.Model.compile method.
# Use tf.keras.optimizers.Adam with default arguments and the loss function.

model.compile(optimizer='adam', loss=loss)

#Configure checkpoints

#Use a tf.keras.callbacks.ModelCheckpoint to ensure that checkpoints are saved during training.

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

#Execute the training

EPOCHS = 10

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

#Epoch 1/10
#172/172 [==============================] - 5s 27ms/step - loss: 2.6807
#Epoch 2/10
#172/172 [==============================] - 5s 27ms/step - loss: 1.9748
#Epoch 3/10
#172/172 [==============================] - 5s 26ms/step - loss: 1.7063
#Epoch 4/10
#172/172 [==============================] - 5s 26ms/step - loss: 1.5543
#Epoch 5/10
#172/172 [==============================] - 5s 27ms/step - loss: 1.4633
#Epoch 6/10
#172/172 [==============================] - 5s 26ms/step - loss: 1.4028
#Epoch 7/10
#172/172 [==============================] - 5s 26ms/step - loss: 1.3568
#Epoch 8/10
#172/172 [==============================] - 5s 26ms/step - loss: 1.3187
#Epoch 9/10
#172/172 [==============================] - 5s 26ms/step - loss: 1.2845
#Epoch 10/10
#172/172 [==============================] - 5s 26ms/step - loss: 1.2528

#Generate text


#Restore the latest checkpoint
#To keep this prediction step simple, use a batch size of 1.

#Because of the way the RNN state is passed from timestep to timestep, the model only accepts a fixed batch size once built.

#To run the model with a different batch_size, you need to rebuild the model and restore the weights from the checkpoint.

tf.train.latest_checkpoint(checkpoint_dir)

'./training_checkpoints/ckpt_10'

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()

#Model: "sequential_1"
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #
#=================================================================
#embedding_1 (Embedding)      (1, None, 256)            16640
#_________________________________________________________________
#gru_1 (GRU)                  (1, None, 1024)           3938304
#_________________________________________________________________
#dense_1 (Dense)              (1, None, 65)             66625
#=================================================================
#Total params: 4,021,569
#Trainable params: 4,021,569
#Non-trainable params: 0
#_________________________________________________________________

#The prediction loop
#The following code block generates the text:

#Begin by choosing a start string, initializing the RNN state and setting the number of characters to generate.

#Get the prediction distribution of the next character using the start string and the RNN state.

#Then, use a categorical distribution to calculate the index of the predicted character.
# Use this predicted character as our next input to the model.

#The RNN state returned by the model is fed back into the model so that it now has more context, instead of only one character.
# After predicting the next character, the modified RNN states are again fed back into the model, which is how it learns as it gets more context from the previously predicted characters.

#Looking at the generated text, you'll see the model knows when to capitalize, make paragraphs and imitates a Shakespeare-like writing vocabulary.
# With the small number of training epochs, it has not yet learned to form coherent sentences.

def generate_text(model, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 1000

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperature results in more predictable text.
    # Higher temperature results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # Pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

print(generate_text(model, start_string=u"ROMEO: "))



#ROMEO: ghast I cut go,
#Know the normander and the wrong:
#To our Morsuis misdress are behiod;
#And after as if no other husion.

#VALERIS:
#Your father and of worms?

#LADY GREY:
#Your hot can dost.

#WARWICK:
#Then, atient the bade, truckle aid,
#Dearve your tongue should be cred to our face,
#Bear trouble my father valiant,' in the company.

#SICINIUS:
#O God!'Sir afeard?

#MIRANDA:
#Come, good med,---or whom by the duke?

#DUKE VINCENTIO:
#Yes, that are bore indocation!

#IO:
#None not, my lord's sons.

#MIRANDA:
#Of some King?'
#And, if thou was, a partanot young to thee.

#JULIET:
#O, tell; then I'll see them again? There's not so reder
#no mother, and my three here to us. You might shall not speak, these this
#same this within; what armpy I might
#but though some way.

#ROMEO:
#Our daughter of the fool, that great come.
#So, not the sun summer so all the sends,
#Your ludgers made before the souls of years, and thereby there. Lady, father, were well the sold, pass, remeate.

#Second King Richard's daughter,
#Which chee

#The easiest thing you can do to improve the results is to train it for longer (try EPOCHS = 30).

#You can also experiment with a different start string, try adding another RNN layer to improve the model's accuracy, or adjust the temperature parameter to generate more or less random predictions.

#Customized Training

#The procedure works as follows:

#First, reset the RNN state. You do this by calling the tf.keras.Model.reset_states method.

#Next, iterate over the dataset (batch by batch) and calculate the predictions associated with each.

#Open a tf.GradientTape, and calculate the predictions and loss in that context.

#Calculate the gradients of the loss with respect to the model variables using the tf.GradientTape.grads method.

#Finally, take a step downwards by using the optimizer's tf.train.Optimizer.apply_gradients method.

model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

#WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer
#WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.iter
#WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.beta_1
#WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.beta_2
#WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.decay
#WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.learning_rate
#WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer's state 'm' for (root).layer_with_weights-0.embeddings
#WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer's state 'm' for (root).layer_with_weights-2.kernel
#WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer's state 'm' for (root).layer_with_weights-2.bias
#WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer's state 'm' for (root).layer_with_weights-1.cell.kernel
#WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer's state 'm' for (root).layer_with_weights-1.cell.recurrent_kernel
#WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer's state 'm' for (root).layer_with_weights-1.cell.bias
#WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer's state 'v' for (root).layer_with_weights-0.embeddings
#WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer's state 'v' for (root).layer_with_weights-2.kernel
#WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer's state 'v' for (root).layer_with_weights-2.bias
#WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer's state 'v' for (root).layer_with_weights-1.cell.kernel
#WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer's state 'v' for (root).layer_with_weights-1.cell.recurrent_kernel
#WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer's state 'v' for (root).layer_with_weights-1.cell.bias
#WARNING:tensorflow:A checkpoint was restored (e.g. tf.train.Checkpoint.restore or tf.keras.Model.load_weights) but not all checkpointed values were used. See above for specific issues. Use expect_partial() on the load status object, e.g. tf.train.Checkpoint.restore(...).expect_partial(), to silence these warnings, or use assert_consumed() to make the check explicit. See https://www.tensorflow.org/guide/checkpoint#loading_mechanics for details.


optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(inp, target):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                target, predictions, from_logits=True))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss

# Training step
EPOCHS = 10

for epoch in range(EPOCHS):
    start = time.time()

    # resetting the hidden state at the start of every epoch
    model.reset_states()

    for (batch_n, (inp, target)) in enumerate(dataset):
        loss = train_step(inp, target)

        if batch_n % 100 == 0:
            template = 'Epoch {} Batch {} Loss {}'
            print(template.format(epoch + 1, batch_n, loss))

    # saving (checkpoint) the model every 5 epochs
    if (epoch + 1) % 5 == 0:
        model.save_weights(checkpoint_prefix.format(epoch=epoch))

    print('Epoch {} Loss {:.4f}'.format(epoch + 1, loss))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

model.save_weights(checkpoint_prefix.format(epoch=epoch))

#Epoch 1 Batch 0 Loss 4.174976348876953
#Epoch 1 Batch 100 Loss 2.351067304611206
#Epoch 1 Loss 2.1421
#Time taken for 1 epoch 6.3171796798706055 sec

#Epoch 2 Batch 0 Loss 2.166642665863037
#Epoch 2 Batch 100 Loss 1.9492360353469849
#Epoch 2 Loss 1.7901
#Time taken for 1 epoch 5.3413612842559814 sec

#Epoch 3 Batch 0 Loss 1.804692029953003
#Epoch 3 Batch 100 Loss 1.6545528173446655
#Epoch 3 Loss 1.6328
#Time taken for 1 epoch 5.337632179260254 sec

#Epoch 4 Batch 0 Loss 1.6188888549804688
#Epoch 4 Batch 100 Loss 1.5314372777938843
#Epoch 4 Loss 1.5319
#Time taken for 1 epoch 5.2844321727752686 sec

#Epoch 5 Batch 0 Loss 1.470827579498291
#Epoch 5 Batch 100 Loss 1.4400928020477295
#Epoch 5 Loss 1.4442
#Time taken for 1 epoch 5.46646785736084 sec

#Epoch 6 Batch 0 Loss 1.4113285541534424
#Epoch 6 Batch 100 Loss 1.387071132659912
#Epoch 6 Loss 1.3713
#Time taken for 1 epoch 5.243147373199463 sec

#Epoch 7 Batch 0 Loss 1.3486154079437256
#Epoch 7 Batch 100 Loss 1.353363037109375
#Epoch 7 Loss 1.3270
#Time taken for 1 epoch 5.295132160186768 sec

#Epoch 8 Batch 0 Loss 1.2960264682769775
#Epoch 8 Batch 100 Loss 1.3038402795791626
#Epoch 8 Loss 1.3556
#Time taken for 1 epoch 5.228798151016235 sec

#Epoch 9 Batch 0 Loss 1.2495232820510864
#Epoch 9 Batch 100 Loss 1.30863618850708
#Epoch 9 Loss 1.2699
#Time taken for 1 epoch 5.33559775352478 sec

#Epoch 10 Batch 0 Loss 1.2161246538162231
#Epoch 10 Batch 100 Loss 1.2242770195007324
#Epoch 10 Loss 1.2360
#Time taken for 1 epoch 5.377742528915405 sec