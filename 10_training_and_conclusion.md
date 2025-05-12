# Part 10: Putting It All Together: Training on a Toy Dataset and Conclusion

Welcome to the grand finale, Part 10 of our "Transformer from Scratch with NumPy" series! We have successfully built all the individual components and assembled the full Transformer architecture in Part 9. Now, we'll discuss how one would go about training such a model, demonstrate a forward pass on a toy dataset, calculate a loss, and conclude our journey.

**Disclaimer:** Implementing a full, efficient training loop with backpropagation for a complex model like the Transformer *entirely from scratch in NumPy* is a monumental task, primarily undertaken for deep educational purposes or in resource-constrained environments. Modern deep learning frameworks (PyTorch, TensorFlow) automate gradient calculation and offer optimized operations, making them the practical choice for real-world training. This part will focus on the conceptual understanding.

## 10.1 The Training Process: An Overview

Training a sequence-to-sequence model like the Transformer typically involves the following steps:

1.  **Dataset Preparation:**
    *   Collect a large corpus of paired sequences (e.g., English sentences and their French translations).
    *   Tokenize the sequences: Convert text into sequences of numerical IDs based on a vocabulary.
    *   Create input and target pairs: For a translation task, the source sentence is the input, and the target sentence is what the model should predict.
    *   Handle padding: Ensure all sequences in a batch have the same length by adding special `<pad>` tokens.
    *   Create masks: Generate padding masks and look-ahead masks.

2.  **Model Initialization:**
    *   Instantiate the Transformer model with chosen hyperparameters (number of layers, `d_model`, `num_heads`, `d_ff`, vocab sizes, dropout rate).
    *   Initialize the model's learnable parameters (weights and biases in embedding layers, linear projections in MHA and PFF, LayerNorm parameters, final linear layer). This was done with small random values in our dummy implementations.

3.  **Training Loop:** Iterate for a chosen number of epochs:
    a.  **Batching:** Divide the dataset into mini-batches.
    b.  For each batch:
        i.  **Forward Pass:** Feed the source and target sequences (and their masks) into the model to get output logits.
        ii. **Loss Calculation:** Compare the model's output logits with the actual target sequences to compute a loss. Cross-entropy loss is standard for classification tasks like predicting the next token.
        iii. **Backward Pass (Backpropagation):** Calculate the gradients of the loss with respect to all learnable parameters in the model. This is the most complex part to implement manually.
        iv. **Parameter Update:** Adjust the model's parameters using an optimizer (e.g., Adam, SGD) and the calculated gradients to minimize the loss.

4.  **Evaluation:** Periodically evaluate the model on a separate validation set to monitor performance and prevent overfitting.

5.  **Inference/Prediction:** Once trained, use the model to generate output sequences for new, unseen input sequences.

## 10.2 A Toy Dataset Example: Reversing a Sequence of Numbers

Let's define a very simple task: teaching the Transformer to reverse a sequence of numbers. For example, if the input is `[1, 2, 3]`, the output should be `[3, 2, 1]`.

*   **Vocabulary:** `0` (pad), `1` (SOS - start of sequence), `2` (EOS - end of sequence), `3, 4, 5, 6, 7` (our numbers).
    *   Source Vocab Size: 8
    *   Target Vocab Size: 8
*   **Example Pair:**
    *   Source: `[1, 3, 4, 5, 2]` (SOS, 1, 2, 3, EOS)
    *   Target (for training input to decoder - shifted right): `[1, 5, 4, 3, 2]` (SOS, 3, 2, 1, EOS)
    *   Target (for loss calculation): `[5, 4, 3, 2, 2]` (3, 2, 1, EOS, EOS - we predict EOS after the sequence)

```python
import numpy as np

# --- Re-include necessary components from Part 9 for a runnable snippet ---
# Softmax, Positional Encoding, LayerNorm, Dropout, Dummy MHA, Dummy PFF,
# EncoderBlock, TransformerEncoder, DecoderBlock, TransformerDecoder,
# EmbeddingLayer, FinalLinearLayer, Transformer (full model)

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def get_positional_encoding(seq_len, d_model):
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe[np.newaxis, :, :]

class LayerNormalization:
    def __init__(self, d_model, eps=1e-6):
        self.d_model = d_model; self.eps = eps
        self.gamma = np.ones(d_model); self.beta = np.zeros(d_model)
    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x_normalized = (x - mean) / np.sqrt(variance + self.eps)
        return self.gamma * x_normalized + self.beta

def dropout_layer(x, rate, training=True):
    if not training or rate == 0: return x
    mask = np.random.binomial(1, 1 - rate, size=x.shape) / (1 - rate)
    return x * mask

class DummyMultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model; self.num_heads = num_heads
        # Dummy weights for illustration (not actually used in this simplified forward)
        self.W_q = [np.random.randn(d_model, d_model // num_heads) for _ in range(num_heads)]
        self.W_k = [np.random.randn(d_model, d_model // num_heads) for _ in range(num_heads)]
        self.W_v = [np.random.randn(d_model, d_model // num_heads) for _ in range(num_heads)]
        self.W_o = np.random.randn(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len_q, _ = query.shape
        output = np.random.rand(batch_size, seq_len_q, self.d_model) 
        seq_len_k = key.shape[1]
        attention_weights = np.random.rand(batch_size, self.num_heads, seq_len_q, seq_len_k)
        return output, attention_weights

class DummyPositionwiseFeedForward:
    def __init__(self, d_model, d_ff):
        self.d_model = d_model; self.d_ff = d_ff
        # Dummy weights
        self.W1 = np.random.randn(d_model, d_ff)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model)
        self.b2 = np.zeros(d_model)
    def forward(self, x):
        # Simplified: just return random of correct shape
        return np.random.rand(x.shape[0], x.shape[1], self.d_model)

class EncoderBlock:
    def __init__(self, d_model, num_heads, d_ff, dropout_rate):
        self.mha = DummyMultiHeadAttention(d_model, num_heads)
        self.pff = DummyPositionwiseFeedForward(d_model, d_ff)
        self.norm1 = LayerNormalization(d_model); self.norm2 = LayerNormalization(d_model)
        self.dropout_rate = dropout_rate
    def forward(self, x, mask, training=True):
        attn_output, _ = self.mha.forward(x, x, x, mask)
        attn_output = dropout_layer(attn_output, self.dropout_rate, training)
        out1 = self.norm1.forward(x + attn_output)
        pff_output = self.pff.forward(out1)
        pff_output = dropout_layer(pff_output, self.dropout_rate, training)
        out2 = self.norm2.forward(out1 + pff_output)
        return out2

class TransformerEncoder:
    def __init__(self, num_blocks, d_model, num_heads, d_ff, dropout_rate):
        self.encoder_blocks = [EncoderBlock(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_blocks)]
        self.dropout_rate = dropout_rate
    def forward(self, x_emb, mask, training=True):
        x = dropout_layer(x_emb, self.dropout_rate, training)
        for block in self.encoder_blocks: x = block.forward(x, mask, training)
        return x

class DecoderBlock:
    def __init__(self, d_model, num_heads, d_ff, dropout_rate):
        self.masked_mha = DummyMultiHeadAttention(d_model, num_heads)
        self.encoder_decoder_mha = DummyMultiHeadAttention(d_model, num_heads)
        self.pff = DummyPositionwiseFeedForward(d_model, d_ff)
        self.norm1 = LayerNormalization(d_model); self.norm2 = LayerNormalization(d_model); self.norm3 = LayerNormalization(d_model)
        self.dropout_rate = dropout_rate
    def forward(self, target_x, encoder_output, look_ahead_mask, padding_mask, training=True):
        attn1_out, _ = self.masked_mha.forward(target_x, target_x, target_x, look_ahead_mask)
        attn1_out = dropout_layer(attn1_out, self.dropout_rate, training)
        out1 = self.norm1.forward(target_x + attn1_out)
        attn2_out, _ = self.encoder_decoder_mha.forward(out1, encoder_output, encoder_output, padding_mask)
        attn2_out = dropout_layer(attn2_out, self.dropout_rate, training)
        out2 = self.norm2.forward(out1 + attn2_out)
        pff_output = self.pff.forward(out2)
        pff_output = dropout_layer(pff_output, self.dropout_rate, training)
        out3 = self.norm3.forward(out2 + pff_output)
        return out3

class TransformerDecoder:
    def __init__(self, num_blocks, d_model, num_heads, d_ff, dropout_rate):
        self.decoder_blocks = [DecoderBlock(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_blocks)]
        self.dropout_rate = dropout_rate
    def forward(self, target_x_emb, encoder_output, look_ahead_mask, padding_mask, training=True):
        x = dropout_layer(target_x_emb, self.dropout_rate, training)
        for block in self.decoder_blocks: x = block.forward(x, encoder_output, look_ahead_mask, padding_mask, training)
        return x

class EmbeddingLayer:
    def __init__(self, vocab_size, d_model):
        self.embeddings = np.random.randn(vocab_size, d_model) * 0.01
    def forward(self, token_ids):
        return self.embeddings[token_ids]

class FinalLinearLayer:
    def __init__(self, d_model, target_vocab_size):
        self.weights = np.random.randn(d_model, target_vocab_size) * 0.01
        self.bias = np.zeros(target_vocab_size)
    def forward(self, x):
        return np.dot(x, self.weights) + self.bias

class Transformer:
    def __init__(self, num_encoder_blocks, num_decoder_blocks, d_model, num_heads, d_ff, 
                 source_vocab_size, target_vocab_size, max_seq_len, dropout_rate=0.1):
        self.d_model = d_model; self.max_seq_len = max_seq_len; self.dropout_rate = dropout_rate
        self.source_embedding = EmbeddingLayer(source_vocab_size, d_model)
        self.target_embedding = EmbeddingLayer(target_vocab_size, d_model)
        self.pos_encoding_source = get_positional_encoding(max_seq_len, d_model)
        self.pos_encoding_target = get_positional_encoding(max_seq_len, d_model)
        self.encoder = TransformerEncoder(num_encoder_blocks, d_model, num_heads, d_ff, dropout_rate)
        self.decoder = TransformerDecoder(num_decoder_blocks, d_model, num_heads, d_ff, dropout_rate)
        self.final_linear_layer = FinalLinearLayer(d_model, target_vocab_size)

    def forward(self, source_tokens, target_tokens, source_padding_mask, target_padding_mask, look_ahead_mask, training=True):
        source_seq_len = source_tokens.shape[1]; target_seq_len = target_tokens.shape[1]
        source_emb = self.source_embedding.forward(source_tokens) * np.sqrt(self.d_model)
        source_emb_pe = dropout_layer(source_emb + self.pos_encoding_source[:, :source_seq_len, :], self.dropout_rate, training)
        target_emb = self.target_embedding.forward(target_tokens) * np.sqrt(self.d_model)
        target_emb_pe = dropout_layer(target_emb + self.pos_encoding_target[:, :target_seq_len, :], self.dropout_rate, training)
        encoder_output = self.encoder.forward(source_emb_pe, source_padding_mask, training)
        decoder_output = self.decoder.forward(target_emb_pe, encoder_output, look_ahead_mask, target_padding_mask, training)
        return self.final_linear_layer.forward(decoder_output)

# --- Toy Dataset & Parameters ---
PAD_ID = 0
SOS_ID = 1 # Start Of Sequence
EOS_ID = 2 # End Of Sequence
NUM_START_ID = 3 # Actual numbers start from ID 3

# Vocabulary: 0:PAD, 1:SOS, 2:EOS, 3:"0", 4:"1", ..., 12:"9"
# For simplicity, let's use numbers 0-6 (IDs 3 to 9)
VOCAB_SIZE = NUM_START_ID + 7 # 3 + 7 = 10 (0-6 are 7 numbers)

# Hyperparameters for the toy model
d_model_toy = 32
num_heads_toy = 2
d_ff_toy = 64
num_enc_blocks_toy = 1
num_dec_blocks_toy = 1
max_seq_len_toy = 10 # Max length of sequence like [SOS, n1, n2, n3, EOS]
dropout_toy = 0.0 # No dropout for this simple example

# Instantiate the Transformer for the toy task
toy_transformer = Transformer(
    num_encoder_blocks=num_enc_blocks_toy,
    num_decoder_blocks=num_dec_blocks_toy,
    d_model=d_model_toy,
    num_heads=num_heads_toy,
    d_ff=d_ff_toy,
    source_vocab_size=VOCAB_SIZE,
    target_vocab_size=VOCAB_SIZE,
    max_seq_len=max_seq_len_toy,
    dropout_rate=dropout_toy
)

# --- Prepare a single data sample ---
# Source: [SOS, 5, 3, 6, EOS]  (IDs: [1, 5+NUM_START_ID, 3+NUM_START_ID, 6+NUM_START_ID, 2])
# Target (input to decoder): [SOS, 6, 3, 5, EOS] (IDs: [1, 6+NUM_START_ID, 3+NUM_START_ID, 5+NUM_START_ID, 2])
# Target (for loss):         [6, 3, 5, EOS, PAD] (IDs: [6+NUM_START_ID, 3+NUM_START_ID, 5+NUM_START_ID, 2, 0])

source_seq_toy = np.array([[SOS_ID, 5+NUM_START_ID, 3+NUM_START_ID, 6+NUM_START_ID, EOS_ID, PAD_ID, PAD_ID]]) # Batch size 1, len 7
decoder_input_toy = np.array([[SOS_ID, 6+NUM_START_ID, 3+NUM_START_ID, 5+NUM_START_ID, EOS_ID, PAD_ID, PAD_ID]])
target_labels_toy = np.array([[6+NUM_START_ID, 3+NUM_START_ID, 5+NUM_START_ID, EOS_ID, PAD_ID, PAD_ID, PAD_ID]])

actual_src_len = 5
actual_tgt_len = 5 # SOS + reversed_seq + EOS

# --- Create Masks ---
# 1. Source Padding Mask (for Encoder Self-Attention)
src_padding_mask_toy = (source_seq_toy == PAD_ID)[:, np.newaxis, np.newaxis, :].astype(int)

# 2. Target Look-Ahead Mask (for Decoder Self-Attention)
tgt_seq_len_toy = decoder_input_toy.shape[1]
tgt_look_ahead_mask_triu = np.triu(np.ones((tgt_seq_len_toy, tgt_seq_len_toy)), k=1).astype(int)
tgt_padding_component = (decoder_input_toy == PAD_ID)[:, np.newaxis, :].astype(int)
tgt_look_ahead_mask_toy = np.maximum(tgt_look_ahead_mask_triu, tgt_padding_component)[:, np.newaxis, :, :]

# 3. Encoder Output Padding Mask (for Decoder Encoder-Decoder Attention)
enc_out_padding_mask_toy = (source_seq_toy == PAD_ID)[:, np.newaxis, np.newaxis, :].astype(int)

print("--- Toy Example: Forward Pass & Loss ---")
print("Source Sequence (IDs):", source_seq_toy)
print("Decoder Input (IDs):", decoder_input_toy)
print("Target Labels (IDs for loss):", target_labels_toy)

# Forward pass
output_logits_toy = toy_transformer.forward(
    source_seq_toy,
    decoder_input_toy,
    src_padding_mask_toy,
    enc_out_padding_mask_toy,
    tgt_look_ahead_mask_toy,
    training=False # Or True if we had actual training
)

print("Output Logits shape:", output_logits_toy.shape) # (batch, target_seq_len, vocab_size)

# --- Loss Calculation: Cross-Entropy Loss ---
def cross_entropy_loss(logits, labels, pad_id):
    """
    Calculates cross-entropy loss, ignoring padding.
    Args:
        logits (np.ndarray): Output logits from the model (batch, seq_len, vocab_size).
        labels (np.ndarray): True token IDs (batch, seq_len).
        pad_id (int): Token ID for padding, to be ignored in loss.
    Returns:
        float: Average cross-entropy loss per non-padded token.
    """
    probs = softmax(logits, axis=-1)
    batch_size, seq_len, vocab_size = probs.shape
    
    # Select the probabilities corresponding to the true labels
    # This uses advanced indexing
    true_label_probs = probs[np.arange(batch_size)[:, np.newaxis], 
                             np.arange(seq_len)[np.newaxis, :], 
                             labels]
    
    # Avoid log(0) - clip probabilities
    true_label_probs = np.clip(true_label_probs, 1e-9, 1.0)
    log_probs = -np.log(true_label_probs)
    
    # Create a mask to ignore padding tokens in the loss
    non_pad_mask = (labels != pad_id).astype(float)
    
    # Apply mask and calculate sum of loss
    masked_log_probs = log_probs * non_pad_mask
    total_loss = np.sum(masked_log_probs)
    num_non_pad_tokens = np.sum(non_pad_mask)
    
    if num_non_pad_tokens == 0:
        return 0.0 # Avoid division by zero if all tokens are padding
        
    return total_loss / num_non_pad_tokens

loss = cross_entropy_loss(output_logits_toy, target_labels_toy, PAD_ID)
print(f"Calculated Cross-Entropy Loss: {loss:.4f}")

# In a real training loop, you would now:
# 1. Calculate gradients of 'loss' w.r.t. all learnable parameters in 'toy_transformer'.
#    (e.g., toy_transformer.source_embedding.embeddings, toy_transformer.final_linear_layer.weights, etc.,
#     and all weights/biases within MHA, PFF, LayerNorm if they were fully implemented with learnable params).
# 2. Update these parameters using an optimizer (e.g., Adam: params -= learning_rate * gradients).
```

## 10.3 Backpropagation and Optimization (Conceptual)

*   **Backpropagation:** This is the algorithm used to compute gradients. It involves applying the chain rule recursively, starting from the loss function and going backward through each layer of the network. For a Transformer, this means calculating gradients for:
    *   The final linear layer.
    *   Each decoder block (FFN, encoder-decoder MHA, masked self-MHA, LayerNorms).
    *   Each encoder block (FFN, self-MHA, LayerNorms).
    *   Embedding layers.
    Manually deriving and implementing these for every matrix multiplication, addition, softmax, normalization, etc., is extremely complex and error-prone in NumPy.

*   **Optimizers:** Once gradients are obtained, optimizers update the model parameters. Common optimizers include:
    *   **SGD (Stochastic Gradient Descent):** `param = param - learning_rate * gradient`
    *   **Adam:** A more sophisticated optimizer that adapts learning rates for each parameter, often leading to faster convergence.
    Implementing these also requires storing and updating additional variables (like momentum for Adam).

## 10.4 Inference (Prediction)

Once the model is trained, generating an output sequence for a new source sequence involves:
1.  **Encode Source:** Pass the tokenized source sequence through the encoder to get `encoder_output`.
2.  **Start Decoder Input:** Begin with a target sequence containing only the `SOS_ID`.
3.  **Iterative Decoding:** Loop until `EOS_ID` is generated or a maximum length is reached:
    a.  Create appropriate masks for the current target sequence.
    b.  Pass the current `target_sequence`, `encoder_output`, and masks through the decoder and the final linear layer to get logits for the next token.
    c.  Apply softmax to get probabilities.
    d.  Select the next token (e.g., by taking the `argmax` of probabilities - greedy decoding, or by sampling - beam search is common for better quality).
    e.  Append the predicted token to the `target_sequence`.
    f.  If the predicted token is `EOS_ID`, stop.

## 10.5 Conclusion of the Series

Congratulations on reaching the end of this 10-part journey into building a Transformer from scratch with NumPy! We have covered an immense amount of ground:

*   **Part 1: Introduction:** Set the stage for Transformers and NumPy.
*   **Part 2: Scaled Dot-Product Attention:** Implemented the core attention mechanism.
*   **Part 3: Multi-Head Attention:** Built a more powerful attention by combining multiple attention heads.
*   **Part 4: Position-wise Feed-Forward Networks:** Added another key layer component.
*   **Part 5: Positional Encoding:** Addressed the Transformer's lack of inherent sequence awareness.
*   **Part 6: Encoder Block:** Assembled the first major building block of the Transformer.
*   **Part 7: Full Transformer Encoder:** Stacked Encoder Blocks to create the complete encoder.
*   **Part 8: Transformer Decoder Block:** Built the more complex decoder block with its two attention mechanisms.
*   **Part 9: Complete Transformer Architecture:** Integrated the encoder, decoder, embeddings, and final output layer.
*   **Part 10: Training and Conclusion:** Discussed the training process, loss functions, and the conceptual steps for making the model learn.

While our NumPy implementation focused on the forward pass and the architectural understanding, it provides a deep insight into the inner workings of one of the most influential neural network architectures today. The hands-on approach, even without a full training loop, demystifies many of the complex interactions within the model.

**Key Takeaways from the Series:**
*   Transformers rely heavily on **self-attention** and **multi-head attention** to capture contextual relationships in sequences.
*   **Positional encodings** are vital for injecting sequence order information.
*   The **encoder-decoder architecture** is powerful for sequence-to-sequence tasks.
*   **Residual connections** and **layer normalization** are crucial for training deep networks.
*   **Masking** is essential for handling padding and ensuring auto-regressive behavior in the decoder.

From here, you are well-equipped to dive deeper into advanced Transformer variants, explore efficient implementations in frameworks like PyTorch or TensorFlow, and apply these concepts to various NLP tasks and beyond.

Thank you for following along, and happy coding!
