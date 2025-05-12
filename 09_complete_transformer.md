# Part 9: Building the Complete Transformer Architecture

Welcome to Part 9 of our "Transformer from Scratch with NumPy" series! We've meticulously constructed all the core components: attention mechanisms, feed-forward networks, positional encodings, encoder blocks, and decoder blocks. Now, it's time to assemble them into the full end-to-end Transformer model as described in the original "Attention Is All You Need" paper.

## 9.1 Overview of the Full Transformer Architecture

The Transformer model consists of two main parts: an **Encoder** and a **Decoder**.

*   **Encoder:** Processes the input sequence (e.g., a sentence in the source language) and generates a sequence of continuous representations (contextual embeddings). It's made up of a stack of identical Encoder Blocks.
*   **Decoder:** Takes the encoder's output and the target sequence (e.g., the sentence in the target language, shifted right during training) to generate the output sequence token by token in an auto-regressive manner. It's made up of a stack of identical Decoder Blocks.

The overall data flow is as follows:
1.  **Input Embeddings:** Source and target sequences are converted into dense vector embeddings.
2.  **Positional Encoding:** Positional information is added to these embeddings.
3.  **Encoder Pass:** The source embeddings (with positional encoding) are passed through the encoder stack.
4.  **Decoder Pass:** The target embeddings (with positional encoding and appropriate masking) and the encoder's output are passed through the decoder stack.
5.  **Final Linear Layer & Softmax:** The decoder's output is passed through a final linear layer to project it to the size of the target vocabulary, followed by a softmax function to obtain probability distributions over the vocabulary for each output token.

## 9.2 Core Components Recap

We will be using:
*   **Embedding Layer:** To convert input token IDs into dense vectors. (We'll define a simple one).
*   **Positional Encoding:** `get_positional_encoding` (from Part 5).
*   **TransformerEncoder:** (from Part 7) - a stack of `EncoderBlock`s.
*   **TransformerDecoder:** (We'll define this) - a stack of `DecoderBlock`s.
*   **EncoderBlock:** (from Part 6) - contains Multi-Head Attention and Position-wise FFN.
*   **DecoderBlock:** (from Part 8) - contains Masked Multi-Head Self-Attention, Encoder-Decoder Attention, and Position-wise FFN.
*   **MultiHeadAttention:** (from Part 3).
*   **PositionwiseFeedForward:** (from Part 4).
*   **LayerNormalization:** (from Part 6).
*   **Final Output Layer:** A linear transformation followed by softmax.

For the NumPy implementation within this markdown, we'll use dummy/simplified versions of `MultiHeadAttention` and `PositionwiseFeedForward` within the `EncoderBlock` and `DecoderBlock` definitions to keep the example focused and concise. The text will assume you'd use the full versions from previous parts in a complete implementation.

## 9.3 NumPy Implementation

Let's start by bringing in and defining the necessary components.

```python
import numpy as np

# --- Softmax (from Part 2) ---
def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# --- Positional Encoding (from Part 5) ---
def get_positional_encoding(seq_len, d_model):
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe[np.newaxis, :, :]

# --- LayerNormalization (from Part 6) ---
class LayerNormalization:
    def __init__(self, d_model, eps=1e-6):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model) # Learnable scale
        self.beta = np.zeros(d_model)  # Learnable shift

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x_normalized = (x - mean) / np.sqrt(variance + self.eps)
        return self.gamma * x_normalized + self.beta

# --- Dropout Layer (from Part 6) ---
def dropout_layer(x, rate, training=True):
    if not training or rate == 0:
        return x
    mask = np.random.binomial(1, 1 - rate, size=x.shape) / (1 - rate)
    return x * mask

# --- Dummy MultiHeadAttention & PositionwiseFeedForward (for brevity in this example) ---
class DummyMultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        # In a real MHA, W_q, W_k, W_v, W_o matrices (learnable) would be here.
        # For simplicity, forward will return random data of correct shape.

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len_q, _ = query.shape
        output = np.random.rand(batch_size, seq_len_q, self.d_model) # (batch, seq_len_q, d_model)
        seq_len_k = key.shape[1]
        attention_weights = np.random.rand(batch_size, self.num_heads, seq_len_q, seq_len_k)
        return output, attention_weights

class DummyPositionwiseFeedForward:
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff
        # Real PFF would have two linear layers with weights and biases (learnable).

    def forward(self, x):
        return np.random.rand(x.shape[0], x.shape[1], self.d_model)

# --- EncoderBlock (from Part 6, using Dummies) ---
class EncoderBlock:
    def __init__(self, d_model, num_heads, d_ff, dropout_rate):
        self.mha = DummyMultiHeadAttention(d_model, num_heads)
        self.pff = DummyPositionwiseFeedForward(d_model, d_ff)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.dropout_rate = dropout_rate

    def forward(self, x, mask, training=True):
        attn_output, _ = self.mha.forward(x, x, x, mask)
        attn_output = dropout_layer(attn_output, self.dropout_rate, training)
        out1 = self.norm1.forward(x + attn_output)
        
        pff_output = self.pff.forward(out1)
        pff_output = dropout_layer(pff_output, self.dropout_rate, training)
        out2 = self.norm2.forward(out1 + pff_output)
        return out2

# --- TransformerEncoder (from Part 7) ---
class TransformerEncoder:
    def __init__(self, num_blocks, d_model, num_heads, d_ff, dropout_rate):
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.encoder_blocks = [EncoderBlock(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_blocks)]
        self.dropout_rate = dropout_rate # For dropout on embeddings+PE

    def forward(self, x_emb, mask, training=True):
        # x_emb is already (batch_size, seq_len, d_model) including positional encoding
        x = dropout_layer(x_emb, self.dropout_rate, training)
        for i in range(self.num_blocks):
            x = self.encoder_blocks[i].forward(x, mask, training)
        return x

# --- DecoderBlock (from Part 8, using Dummies) ---
class DecoderBlock:
    def __init__(self, d_model, num_heads, d_ff, dropout_rate):
        self.masked_mha = DummyMultiHeadAttention(d_model, num_heads)
        self.encoder_decoder_mha = DummyMultiHeadAttention(d_model, num_heads)
        self.pff = DummyPositionwiseFeedForward(d_model, d_ff)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.norm3 = LayerNormalization(d_model)
        self.dropout_rate = dropout_rate

    def forward(self, target_x, encoder_output, look_ahead_mask, padding_mask, training=True):
        # Masked Self-Attention
        attn1_out, _ = self.masked_mha.forward(target_x, target_x, target_x, look_ahead_mask)
        attn1_out = dropout_layer(attn1_out, self.dropout_rate, training)
        out1 = self.norm1.forward(target_x + attn1_out)
        
        # Encoder-Decoder Attention
        attn2_out, _ = self.encoder_decoder_mha.forward(out1, encoder_output, encoder_output, padding_mask)
        attn2_out = dropout_layer(attn2_out, self.dropout_rate, training)
        out2 = self.norm2.forward(out1 + attn2_out)
        
        # Position-wise Feed-Forward
        pff_output = self.pff.forward(out2)
        pff_output = dropout_layer(pff_output, self.dropout_rate, training)
        out3 = self.norm3.forward(out2 + pff_output)
        return out3

# --- TransformerDecoder (New: Stacks DecoderBlocks) ---
class TransformerDecoder:
    def __init__(self, num_blocks, d_model, num_heads, d_ff, dropout_rate):
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.decoder_blocks = [DecoderBlock(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_blocks)]
        self.dropout_rate = dropout_rate # For dropout on embeddings+PE

    def forward(self, target_x_emb, encoder_output, look_ahead_mask, padding_mask, training=True):
        # target_x_emb is already (batch_size, target_seq_len, d_model) including positional encoding
        x = dropout_layer(target_x_emb, self.dropout_rate, training)
        for i in range(self.num_blocks):
            x = self.decoder_blocks[i].forward(x, encoder_output, look_ahead_mask, padding_mask, training)
        return x

# --- Simple Embedding Layer (New) ---
class EmbeddingLayer:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Initialize embedding matrix with small random values
        # This would be learned during actual training.
        self.embeddings = np.random.randn(vocab_size, d_model) * 0.01 

    def forward(self, token_ids):
        # token_ids: (batch_size, seq_len)
        # Output: (batch_size, seq_len, d_model)
        return self.embeddings[token_ids]

# --- Final Linear Layer (New) ---
class FinalLinearLayer:
    def __init__(self, d_model, target_vocab_size):
        self.d_model = d_model
        self.target_vocab_size = target_vocab_size
        # Initialize weights and bias (learnable)
        self.weights = np.random.randn(d_model, target_vocab_size) * 0.01
        self.bias = np.zeros(target_vocab_size)

    def forward(self, x):
        # x: (batch_size, target_seq_len, d_model)
        # Output: (batch_size, target_seq_len, target_vocab_size)
        return np.dot(x, self.weights) + self.bias

# --- The Full Transformer Model (New) ---
class Transformer:
    def __init__(self, num_encoder_blocks, num_decoder_blocks, 
                 d_model, num_heads, d_ff, 
                 source_vocab_size, target_vocab_size, 
                 max_seq_len, dropout_rate=0.1):
        
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.source_embedding = EmbeddingLayer(source_vocab_size, d_model)
        self.target_embedding = EmbeddingLayer(target_vocab_size, d_model)
        
        self.pos_encoding_source = get_positional_encoding(max_seq_len, d_model)
        self.pos_encoding_target = get_positional_encoding(max_seq_len, d_model)
        
        self.encoder = TransformerEncoder(num_encoder_blocks, d_model, num_heads, d_ff, dropout_rate)
        self.decoder = TransformerDecoder(num_decoder_blocks, d_model, num_heads, d_ff, dropout_rate)
        
        self.final_linear_layer = FinalLinearLayer(d_model, target_vocab_size)
        self.dropout_rate = dropout_rate # General dropout for embeddings

    def forward(self, source_tokens, target_tokens, source_padding_mask, target_padding_mask, look_ahead_mask, training=True):
        """
        Forward pass for the full Transformer model.
        Args:
            source_tokens (np.ndarray): Source sequence token IDs, shape (batch_size, source_seq_len).
            target_tokens (np.ndarray): Target sequence token IDs, shape (batch_size, target_seq_len).
            source_padding_mask (np.ndarray): Mask for padding in source sequence for encoder MHA.
                                            Shape (batch_size, 1, source_seq_len, source_seq_len).
            target_padding_mask (np.ndarray): Mask for padding in target sequence for decoder's encoder-decoder MHA.
                                            Shape (batch_size, 1, target_seq_len, source_seq_len).
            look_ahead_mask (np.ndarray): Mask for decoder's self-attention.
                                          Shape (batch_size, 1, target_seq_len, target_seq_len).
            training (bool): If True, applies dropout.
        Returns:
            np.ndarray: Output logits, shape (batch_size, target_seq_len, target_vocab_size).
        """
        source_seq_len = source_tokens.shape[1]
        target_seq_len = target_tokens.shape[1]

        # 1. Source Embeddings + Positional Encoding
        source_emb = self.source_embedding.forward(source_tokens)
        # Scale embeddings as per original paper (optional but common)
        source_emb *= np.sqrt(self.d_model)
        source_emb_pe = source_emb + self.pos_encoding_source[:, :source_seq_len, :]
        source_emb_pe = dropout_layer(source_emb_pe, self.dropout_rate, training)

        # 2. Target Embeddings + Positional Encoding
        target_emb = self.target_embedding.forward(target_tokens)
        target_emb *= np.sqrt(self.d_model)
        target_emb_pe = target_emb + self.pos_encoding_target[:, :target_seq_len, :]
        target_emb_pe = dropout_layer(target_emb_pe, self.dropout_rate, training)

        # 3. Encoder Pass
        encoder_output = self.encoder.forward(source_emb_pe, source_padding_mask, training)

        # 4. Decoder Pass
        # The padding_mask for decoder's MHA2 should be based on source sequence padding.
        decoder_output = self.decoder.forward(target_emb_pe, encoder_output, look_ahead_mask, target_padding_mask, training)

        # 5. Final Linear Layer and Softmax (Softmax is usually applied outside, with the loss function)
        output_logits = self.final_linear_layer.forward(decoder_output)
        
        return output_logits

```

## 9.4 Mask Creation for the Full Model

When using the Transformer, creating the correct masks is crucial:

1.  **Source Padding Mask (`source_padding_mask`):**
    *   Used in the Encoder's self-attention layers.
    *   Masks out `<pad>` tokens in the source sequence.
    *   Shape: `(batch_size, 1, 1, source_seq_len)` or `(batch_size, 1, source_seq_len, source_seq_len)` if it's a combined mask for all queries attending to keys.
    *   If `token_id == padding_token_id`, then mask value is 1 (prevent attention), else 0.

2.  **Target Look-Ahead Mask (`look_ahead_mask`):**
    *   Used in the Decoder's first (masked) self-attention layer.
    *   Prevents positions from attending to subsequent positions in the target sequence.
    *   Combines a standard look-ahead mask (upper triangle) with the target padding mask (if any).
    *   Shape: `(batch_size, 1, target_seq_len, target_seq_len)`.

3.  **Target Padding Mask (for Encoder-Decoder Attention - `target_padding_mask` in `forward` args):**
    *   Used in the Decoder's second (encoder-decoder) attention layer.
    *   Masks out `<pad>` tokens from the *source sequence* (i.e., `encoder_output`).
    *   Shape: `(batch_size, 1, 1, source_seq_len)` or `(batch_size, 1, target_seq_len, source_seq_len)`.
    *   This ensures the decoder doesn't attend to padded parts of the encoder's output.

Helper function to create masks (simplified for example):

```python
def create_padding_mask_for_encoder_self_attn(sequence_padded, pad_token_id):
    # sequence_padded: (batch_size, seq_len)
    mask = (sequence_padded == pad_token_id).astype(int)
    return mask[:, np.newaxis, np.newaxis, :] # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask_for_decoder_self_attn(target_seq_len):
    # Creates the upper triangular mask for subsequent positions
    look_ahead = np.triu(np.ones((target_seq_len, target_seq_len)), k=1).astype(int)
    return look_ahead[np.newaxis, np.newaxis, :, :] # (1, 1, target_seq_len, target_seq_len), broadcasts to batch

def create_combined_decoder_mask(target_sequence_padded, pad_token_id):
    # target_sequence_padded: (batch_size, target_seq_len)
    target_seq_len = target_sequence_padded.shape[1]
    
    # 1. Look-ahead part (prevents seeing future tokens)
    look_ahead = np.triu(np.ones((target_seq_len, target_seq_len)), k=1).astype(int)
    # (target_seq_len, target_seq_len)
    
    # 2. Padding part for target sequence (prevents seeing padding in target itself)
    target_pad = (target_sequence_padded == pad_token_id).astype(int)[:, np.newaxis, :]
    # (batch_size, 1, target_seq_len)
    
    # Combine: if either look_ahead is 1 OR target_pad is 1 for the key, then mask.
    # The mask should be (batch_size, 1, target_seq_len, target_seq_len)
    # A position (q, k) is masked if look_ahead[q, k] == 1 OR target_pad[batch, k] == 1
    combined_mask = np.maximum(look_ahead, target_pad)
    return combined_mask[:, np.newaxis, :, :]

```
Note: The `scaled_dot_product_attention` function from Part 2 expects mask values of `0` to attend and non-zero (e.g., `1`) to prevent attention. The mask creation functions above follow this convention.

## 9.5 Simple Input/Output Example

Let's instantiate and run our full Transformer model.

```python
# --- Example Usage ---
np.random.seed(1337)

# Model Hyperparameters
num_enc_blocks = 2
num_dec_blocks = 2
d_model_hyper = 64 # Must be divisible by num_heads
num_heads_hyper = 4
d_ff_hyper = 128
source_vocab_size_hyper = 1000 # Example source vocabulary size
target_vocab_size_hyper = 1200 # Example target vocabulary size
max_seq_len_hyper = 50       # Max sequence length for positional encoding
dropout_rate_hyper = 0.1

# Instantiate the Transformer
transformer_model = Transformer(
    num_encoder_blocks=num_enc_blocks,
    num_decoder_blocks=num_dec_blocks,
    d_model=d_model_hyper,
    num_heads=num_heads_hyper,
    d_ff=d_ff_hyper,
    source_vocab_size=source_vocab_size_hyper,
    target_vocab_size=target_vocab_size_hyper,
    max_seq_len=max_seq_len_hyper,
    dropout_rate=dropout_rate_hyper
)

# Dummy Input Data
batch_size_ex = 2
source_seq_len_ex = 10
target_seq_len_ex = 12 # For training, target usually shifted input
pad_token_id_ex = 0

# Source and Target token sequences (random integers representing token IDs)
# Replace with actual tokenized data in a real scenario.
# Ensure token IDs are < vocab_size
source_tokens_ex = np.random.randint(1, source_vocab_size_hyper, size=(batch_size_ex, source_seq_len_ex))
source_tokens_ex[0, -2:] = pad_token_id_ex # Add some padding to the first batch item

target_tokens_ex = np.random.randint(1, target_vocab_size_hyper, size=(batch_size_ex, target_seq_len_ex))
target_tokens_ex[1, -3:] = pad_token_id_ex # Add some padding to the second batch item

# Create Masks (Simplified for this example)
# 1. Source Padding Mask (for Encoder Self-Attention)
#    Masks <pad> tokens in the source. Shape (batch, 1, 1, src_len)
src_padding_mask_ex = (source_tokens_ex == pad_token_id_ex)[:, np.newaxis, np.newaxis, :].astype(int)

# 2. Target Look-Ahead Mask (for Decoder Self-Attention)
#    Prevents attending to future tokens and <pad> tokens in the target.
#    Shape (batch, 1, tgt_len, tgt_len)
tgt_look_ahead_mask_triu = np.triu(np.ones((target_seq_len_ex, target_seq_len_ex)), k=1).astype(int)
tgt_padding_component = (target_tokens_ex == pad_token_id_ex)[:, np.newaxis, :].astype(int) # (batch, 1, tgt_len)
tgt_look_ahead_mask_ex = np.maximum(tgt_look_ahead_mask_triu, tgt_padding_component)[:, np.newaxis, :, :]

# 3. Encoder Output Padding Mask (for Decoder Encoder-Decoder Attention)
#    Masks <pad> tokens in the source (encoder output) when decoder attends to it.
#    Shape (batch, 1, 1, src_len) - this will broadcast over target_seq_len queries.
#    Or (batch, 1, tgt_len, src_len) if you want to be explicit.
enc_out_padding_mask_ex = (source_tokens_ex == pad_token_id_ex)[:, np.newaxis, np.newaxis, :].astype(int)


print("--- Full Transformer Example ---")
print("Source Tokens (shape {}):
".format(source_tokens_ex.shape), source_tokens_ex)
print("Target Tokens (shape {}):
".format(target_tokens_ex.shape), target_tokens_ex)
print("Source Padding Mask (shape {}):
".format(src_padding_mask_ex.shape), src_padding_mask_ex[0,:,:,:5]) # Print snippet
print("Target Look-Ahead Mask (shape {}):
".format(tgt_look_ahead_mask_ex.shape), tgt_look_ahead_mask_ex[0,:,:3,:3]) # Print snippet
print("Encoder Output Padding Mask (shape {}):
".format(enc_out_padding_mask_ex.shape), enc_out_padding_mask_ex[0,:,:,:5]) # Print snippet

# Forward pass (training=False for consistent dropout behavior in example)
output_logits_ex = transformer_model.forward(
    source_tokens_ex,
    target_tokens_ex,
    src_padding_mask_ex,      # For encoder's self-attention
    enc_out_padding_mask_ex,  # For decoder's enc-dec attention (masking encoder output)
    tgt_look_ahead_mask_ex,   # For decoder's self-attention
    training=False
)

print("
Output Logits shape:", output_logits_ex.shape)
# Expected: (batch_size_ex, target_seq_len_ex, target_vocab_size_hyper)
# (2, 12, 1200)

# Apply softmax to get probabilities (usually done by loss function)
output_probs_ex = softmax(output_logits_ex, axis=-1)
print("Output Probabilities shape:", output_probs_ex.shape)
print("Sample Probabilities (sum over vocab for 1st token, 1st batch item):", np.sum(output_probs_ex[0, 0, :]))

if output_logits_ex.shape == (batch_size_ex, target_seq_len_ex, target_vocab_size_hyper):
    print("
Output logits shape is correct!")
else:
    print("
Output logits shape is INCORRECT!")

```

**Running the Example:**
Executing this code will initialize the full Transformer model with dummy weights and pass some sample data through it. You'll see the shapes of inputs, masks, and the final output logits. The output logits will have the shape `(batch_size, target_seq_len, target_vocab_size)`, ready to be used with a cross-entropy loss function during training.

## 9.6 Key Takeaways

*   The full Transformer model elegantly combines an encoder and a decoder stack.
*   Input and target sequences are first embedded and augmented with positional encodings.
*   The encoder processes the source sequence to produce contextual representations.
*   The decoder uses these representations and the (shifted) target sequence to auto-regressively generate output tokens.
*   A final linear layer followed by softmax converts decoder outputs into probability distributions over the target vocabulary.
*   Correctly implementing and applying masks (padding and look-ahead) is absolutely critical for the Transformer to function as intended.

## 9.7 What's Next?

We have now built the entire Transformer architecture from scratch using NumPy! This is a significant achievement. The final step in our series, **Part 10: Putting It All Together: Training on a Toy Dataset**, will discuss the conceptual steps involved in training this model. While a full NumPy-based training loop with backpropagation for all these components is beyond the scope of a single tutorial part (it's a very deep network!), we'll outline the process, define a simple loss function, and discuss how one might approach gradient updates if implementing the backward pass manually. This will provide a conceptual understanding of how such a model learns.

Stay tuned for the grand finale!
