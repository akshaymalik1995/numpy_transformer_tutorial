# Part 7: Implementing the Full Transformer Encoder

Welcome to Part 7 of our "Transformer from Scratch with NumPy" series. In Part 6, we constructed a single `EncoderBlock`. Now, we'll take that building block and stack multiple instances of it to create the complete Transformer Encoder. We will also integrate the input embeddings (conceptually) and the positional encodings (from Part 5) that prepare the input sequence for the encoder stack.

## 7.1 Introduction to the Full Encoder

The encoder side of the Transformer is responsible for processing an input sequence of tokens and transforming it into a sequence of continuous representations (contextual embeddings). These representations ideally capture the meaning and context of each token in relation to the entire input sequence.

The Transformer encoder achieves this by using a stack of `N` identical `EncoderBlock`s. The output of one block becomes the input to the next. This layered approach allows the model to learn increasingly complex features and relationships within the data.

**Overall Architecture:**
1.  **Input Embeddings:** The input sequence of tokens (e.g., words) is first converted into dense vector representations (embeddings).
2.  **Positional Encoding:** Since the Transformer architecture itself doesn't inherently process sequential order (due to the self-attention mechanism operating on sets of vectors), positional information is added to the embeddings.
3.  **Encoder Stack:** The sum of embeddings and positional encodings is then fed into the stack of `N` `EncoderBlock`s.

## 7.2 Input to the Encoder

### 7.2.1 Input Embeddings

In a typical NLP pipeline, raw text is first tokenized (e.g., split into words or sub-words). Each token is then mapped to a high-dimensional vector using an embedding layer. These embeddings are usually learned during the training process.

For this tutorial series, we are focusing on the Transformer architecture itself and will assume that the input to our encoder is already a sequence of numerical embeddings. For example, if our input sentence has `seq_len` tokens and our model uses an embedding dimension of `d_model`, the input to the encoder (before positional encoding) will be a NumPy array of shape `(batch_size, seq_len, d_model)`.

### 7.2.2 Positional Encoding

As discussed in Part 5, positional encodings are crucial for providing the model with information about the relative or absolute position of tokens in the sequence. We implemented the `get_positional_encoding` function using sinusoidal functions.

The positional encodings have the same dimension `d_model` as the token embeddings. They are added element-wise to the token embeddings before being fed into the first encoder block.

Let's recall the `get_positional_encoding` function from Part 5:

```python
import numpy as np

def get_positional_encoding(seq_len, d_model):
    """
    Generates sinusoidal positional encodings.
    Args:
        seq_len (int): Length of the sequence.
        d_model (int): Dimension of the model (embedding dimension).
    Returns:
        np.ndarray: Positional encoding matrix of shape (1, seq_len, d_model).
                    The first dimension is 1 to allow broadcasting with batch_size.
    """
    position = np.arange(seq_len)[:, np.newaxis] # (seq_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)) # (d_model/2)
    
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe[np.newaxis, :, :] # Shape: (1, seq_len, d_model)
```

## 7.3 The `TransformerEncoder` Class

We will now define a `TransformerEncoder` class that encapsulates the stack of `N` `EncoderBlock`s. This class will:

1.  Initialize `N` instances of `EncoderBlock`.
2.  In its `forward` method:
    a.  Take the input embeddings.
    b.  Add positional encodings to these embeddings.
    c.  Pass the result sequentially through each `EncoderBlock` in the stack.

For the `EncoderBlock` itself, we'll use the implementation from Part 6. For simplicity in this file, we'll redefine the `EncoderBlock` and its necessary components (`LayerNormalization`, `dropout_layer`, and dummy versions of `MultiHeadAttention` and `PositionwiseFeedForward`) so the example is self-contained.

## 7.4 NumPy Implementation of `TransformerEncoder`

First, let's bring in the necessary components. We'll use the `LayerNormalization` and `dropout_layer` from Part 6, and simplified (dummy) versions of `MultiHeadAttention` and `PositionwiseFeedForward` for the `EncoderBlock` example to keep it concise. In a real scenario, you'd import these from their respective part files.

```python
import numpy as np

# --- Positional Encoding (from Part 5) ---
def get_positional_encoding(seq_len, d_model):
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe[np.newaxis, :, :]

# --- LayerNormalization class (from Part 6) ---
class LayerNormalization:
    def __init__(self, d_model, eps=1e-6):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x_normalized = (x - mean) / np.sqrt(variance + self.eps)
        output = self.gamma * x_normalized + self.beta
        return output

# --- dropout_layer function (from Part 6) ---
def dropout_layer(x, rate, training=True):
    if not training or rate == 0:
        return x
    mask = np.random.binomial(1, 1 - rate, size=x.shape) / (1 - rate)
    return x * mask

# --- Dummy MultiHeadAttention (Placeholder for Part 3's implementation) ---
class DummyMultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len_q, _ = query.shape
        # Simulate MHA output (just returns random data of correct shape)
        output = np.random.rand(batch_size, seq_len_q, self.d_model)
        # Dummy attention weights
        seq_len_k = key.shape[1]
        attention_weights = np.random.rand(batch_size, self.num_heads, seq_len_q, seq_len_k)
        return output, attention_weights

# --- Dummy PositionwiseFeedForward (Placeholder for Part 4's implementation) ---
class DummyPositionwiseFeedForward:
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff

    def forward(self, x):
        # Simulate PFF output (just returns random data of correct shape)
        return np.random.rand(x.shape[0], x.shape[1], self.d_model)

# --- EncoderBlock class (from Part 6) ---
class EncoderBlock:
    def __init__(self, multi_head_attention, positionwise_feed_forward, d_model, dropout_rate):
        self.multi_head_attention = multi_head_attention
        self.positionwise_feed_forward = positionwise_feed_forward
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.dropout_rate = dropout_rate

    def forward(self, x, mask, training=True):
        attn_output, _ = self.multi_head_attention.forward(query=x, key=x, value=x, mask=mask)
        attn_output_dropout = dropout_layer(attn_output, self.dropout_rate, training)
        norm1_input = x + attn_output_dropout
        norm1_output = self.norm1.forward(norm1_input)
        
        ffn_output = self.positionwise_feed_forward.forward(norm1_output)
        ffn_output_dropout = dropout_layer(ffn_output, self.dropout_rate, training)
        norm2_input = norm1_output + ffn_output_dropout
        output = self.norm2.forward(norm2_input)
        return output

# --- Full Transformer Encoder ---
class TransformerEncoder:
    def __init__(self, num_blocks, d_model, num_heads, d_ff, dropout_rate, max_seq_len):
        """
        Initializes the Transformer Encoder.
        Args:
            num_blocks (int): Number of EncoderBlocks to stack.
            d_model (int): Dimension of the model (embedding dimension).
            num_heads (int): Number of attention heads for MultiHeadAttention.
            d_ff (int): Dimension of the hidden layer in PositionwiseFeedForward.
            dropout_rate (float): Dropout rate.
            max_seq_len (int): Maximum sequence length for positional encoding.
        """
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        
        # Positional encoding - precompute for max_seq_len
        self.pos_encoding = get_positional_encoding(max_seq_len, d_model)
        
        # Create a list of EncoderBlocks
        # In a real implementation, MHA and PFF would be properly initialized with weights.
        # Here, we use the dummy versions for simplicity of the example.
        self.encoder_blocks = []
        for _ in range(num_blocks):
            mha_instance = DummyMultiHeadAttention(d_model, num_heads)
            pff_instance = DummyPositionwiseFeedForward(d_model, d_ff)
            encoder_block = EncoderBlock(mha_instance, pff_instance, d_model, dropout_rate)
            self.encoder_blocks.append(encoder_block)

    def forward(self, x, mask, training=True):
        """
        Forward pass for the Transformer Encoder.
        Args:
            x (np.ndarray): Input tensor (embeddings), shape (batch_size, seq_len, d_model).
            mask (np.ndarray): Mask for the Multi-Head Attention layers in EncoderBlocks.
                               Shape should be compatible (e.g., (batch_size, 1, seq_len, seq_len)).
            training (bool): Whether the model is in training mode (for dropout).
        Returns:
            np.ndarray: Output tensor from the encoder, shape (batch_size, seq_len, d_model).
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Add positional encoding
        # x is (batch_size, seq_len, d_model)
        # self.pos_encoding is (1, max_seq_len, d_model)
        # We need to slice pos_encoding to match current seq_len and add to x.
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Apply dropout to the embeddings + positional encoding
        x = dropout_layer(x, self.dropout_rate, training)
        
        # 2. Pass through the stack of EncoderBlocks
        for i in range(self.num_blocks):
            x = self.encoder_blocks[i].forward(x, mask, training)
            
        return x

```

## 7.5 Simple Input/Output Example

Let's test our `TransformerEncoder` with some dummy data.

```python
# --- Example Usage ---
np.random.seed(123) # For reproducibility

# Parameters
batch_size = 2
seq_len = 15       # Length of the input sequence
d_model = 128      # Embedding dimension
num_heads = 8      # Number of attention heads
d_ff = 256         # Dimension of FFN hidden layer
dropout_rate = 0.1
num_encoder_blocks = 3 # Number of EncoderBlocks in the stack
max_seq_len_for_pe = 50 # Max sequence length for precomputed positional encoding

# Instantiate the Transformer Encoder
transformer_encoder = TransformerEncoder(
    num_blocks=num_encoder_blocks,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    dropout_rate=dropout_rate,
    max_seq_len=max_seq_len_for_pe
)

# Create dummy input tensor (e.g., token embeddings)
# Shape: (batch_size, seq_len, d_model)
input_embeddings = np.random.rand(batch_size, seq_len, d_model)

# Create a dummy mask (e.g., no padding, so all tokens attend to each other)
# Shape: (batch_size, 1, seq_len_q, seq_len_k)
# For self-attention in encoder, seq_len_q = seq_len_k = seq_len
encoder_attention_mask = np.zeros((batch_size, 1, seq_len, seq_len), dtype=int)

# Pass input through the Transformer Encoder
# Set training=False for this example to make dropout a pass-through for consistent output
encoder_output = transformer_encoder.forward(input_embeddings, mask=encoder_attention_mask, training=False)

print("--- Transformer Encoder Example ---")
print("Input embeddings shape:", input_embeddings.shape)
print("Encoder attention mask shape:", encoder_attention_mask.shape)
print("Positional encoding shape (sliced for current seq_len):".format(seq_len), transformer_encoder.pos_encoding[:, :seq_len, :].shape)
print("Output tensor shape:", encoder_output.shape)
print("
Sample of Output tensor (first item in batch, first token, first 5 features):
", encoder_output[0, 0, :5])

# Verify output shape
if encoder_output.shape == (batch_size, seq_len, d_model):
    print("
Output shape is correct!")
else:
    print("
Output shape is INCORRECT!")

```

**Running the Example:**
If you combine all the Python code blocks (Positional Encoding, LayerNorm, Dropout, Dummy MHA, Dummy PFF, EncoderBlock, TransformerEncoder, and the example usage) into a single script and run it, you will see the output. The `encoder_output` will have the same shape as the input embeddings `(batch_size, seq_len, d_model)`, which is `(2, 15, 128)` in this example. This demonstrates that the entire encoder stack processes the sequence and produces representations of the same dimensionality.

## 7.6 Key Takeaways

*   The Transformer Encoder is composed of a stack of `N` identical `EncoderBlock`s.
*   Input to the encoder first involves converting tokens to embeddings and then adding positional encodings.
*   The `TransformerEncoder` class manages this stack and the initial addition of positional encodings.
*   Each `EncoderBlock` within the stack applies self-attention and feed-forward operations, refining the representations at each step.

## 7.7 What's Next?

With the encoder fully implemented, we are ready to move to the other major component of the original Transformer architecture: the Decoder. In **Part 8: Implementing the Transformer Decoder Block**, we will build the `DecoderBlock`, which has a slightly different structure than the `EncoderBlock` due to its need to attend to the encoder's output and handle masked self-attention for auto-regressive generation. Stay tuned!
