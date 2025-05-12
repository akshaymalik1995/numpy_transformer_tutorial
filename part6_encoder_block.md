# Part 6: Constructing a Transformer Encoder Block

Welcome to Part 6! We've diligently built all the necessary components:

1.  **Scaled Dot-Product Attention** (Part 2)
2.  **Multi-Head Attention** (Part 3)
3.  **Position-wise Feed-Forward Network** (Part 4)
4.  **Positional Encoding** (Part 5)

Now, it's time to assemble these into a complete **Transformer Encoder Block**. The encoder part of the original Transformer is a stack of these identical blocks.

## 6.1 Architecture of an Encoder Block

A single Transformer Encoder Block consists of two main sub-layers:

1.  **Multi-Head Self-Attention Mechanism:** This allows the model to weigh the importance of different words in the input sequence when encoding a particular word.
2.  **Position-wise Fully Connected Feed-Forward Network:** This processes the output of the attention mechanism at each position independently.

Around each of these two sub-layers, a **residual connection** is employed, followed by **layer normalization**. This is crucial for training deep networks by preventing vanishing/exploding gradients and stabilizing the learning process.

So, the operations within an encoder block for an input `x` are:

1.  **Sub-layer 1 (Multi-Head Attention):**
    *   `attention_output = MultiHeadAttention(Q=x, K=x, V=x)` (Self-attention, so Q, K, and V are all derived from the same input `x`)
    *   `x = LayerNorm(x + attention_output)` (Add & Norm)

2.  **Sub-layer 2 (Feed-Forward Network):**
    *   `ffn_output = PositionwiseFeedForward(x)`
    *   `x = LayerNorm(x + ffn_output)` (Add & Norm)

The output of these operations becomes the input to the next encoder block (if any) or the final output of the encoder stack.

**Dropout:** The original Transformer paper also applies dropout to the output of each sub-layer *before* it is added to the sub-layer input (the residual connection) and normalized. For simplicity in our NumPy-only implementation, we will omit dropout for now, but it's an important regularization technique in practice.

## 6.2 Layer Normalization

Before we build the encoder block, let's quickly implement Layer Normalization. Unlike Batch Normalization, Layer Normalization normalizes the inputs across the features (i.e., along the `d_model` dimension) for each data sample (each token in each sequence) independently.

**Formula:**

For a vector `x` (e.g., the `d_model`-dimensional representation of a token):

`LayerNorm(x) = γ * ( (x - μ) / sqrt(σ^2 + ε) ) + β`

Where:
*   `μ` is the mean of the elements in `x`.
*   `σ^2` is the variance of the elements in `x`.
*   `γ` (gamma) is a learnable scaling parameter (vector of size `d_model`).
*   `β` (beta) is a learnable shifting parameter (vector of size `d_model`).
*   `ε` (epsilon) is a small constant added for numerical stability (e.g., 1e-5).

In our NumPy implementation, we'll initialize `γ` to ones and `β` to zeros, effectively making them non-operational initially. In a full training setup, these would be learned.

```python
import numpy as np

class LayerNormalization:
    def __init__(self, d_model, epsilon=1e-5):
        """
        Initialize the Layer Normalization layer.

        Args:
            d_model (int): Dimension of the model (features dimension).
            epsilon (float): Small constant for numerical stability.
        """
        self.d_model = d_model
        self.epsilon = epsilon
        # Learnable parameters (initialized as if they don't change the input)
        self.gamma = np.ones(d_model)  # Scale
        self.beta = np.zeros(d_model)   # Shift

    def forward(self, x):
        """
        Apply Layer Normalization.

        Args:
            x (np.ndarray): Input tensor, shape (batch_size, seq_len, d_model).

        Returns:
            np.ndarray: Normalized tensor, shape (batch_size, seq_len, d_model).
        """
        # Calculate mean and variance along the last dimension (d_model)
        # keepdims=True is important for broadcasting later
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)

        # Normalize
        x_normalized = (x - mean) / np.sqrt(variance + self.epsilon)

        # Scale and shift
        # self.gamma and self.beta are (d_model,)
        # They will be broadcasted to (batch_size, seq_len, d_model)
        output = self.gamma * x_normalized + self.beta

        return output

```

## 6.3 NumPy Implementation of the Encoder Block

Now, let's combine Multi-Head Attention, Position-wise Feed-Forward Network, and Layer Normalization to create the `EncoderBlock` class. We'll need to import or define the classes from previous parts.

For brevity, I'll assume the necessary classes (`MultiHeadAttention`, `PositionwiseFeedForward`) are defined in the same scope or imported. Let's paste their definitions here for completeness within this part's context, along with the `scaled_dot_product_attention` and `softmax` helpers.

```python
# --- Prerequisite code from previous parts --- #
def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.swapaxes(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask == 0, scores, -1e9)
    attention_weights = softmax(scores, axis=-1)
    output = np.matmul(attention_weights, V)
    return output, attention_weights

class MultiHeadAttention:
    def __init__(self, d_model, num_heads, seed=None):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        if seed is not None: np.random.seed(seed)
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)
        self.W_o = np.random.randn(d_model, d_model)

    def split_heads(self, x, batch_size):
        seq_len = x.shape[1]
        x = x.reshape(batch_size, seq_len, self.num_heads, -1)
        return x.transpose(0, 2, 1, 3)

    def forward(self, Q_in, K_in, V_in, mask=None):
        batch_size, seq_len_q = Q_in.shape[0], Q_in.shape[1]
        Q_proj = np.matmul(Q_in, self.W_q)
        K_proj = np.matmul(K_in, self.W_k)
        V_proj = np.matmul(V_in, self.W_v)
        Q_split = self.split_heads(Q_proj, batch_size)
        K_split = self.split_heads(K_proj, batch_size)
        V_split = self.split_heads(V_proj, batch_size)
        if mask is not None:
            mask_expanded = np.expand_dims(mask, axis=1)
        else:
            mask_expanded = None
        attention_output, attention_weights = scaled_dot_product_attention(Q_split, K_split, V_split, mask=mask_expanded)
        attention_output_transposed = attention_output.transpose(0, 2, 1, 3)
        concat_attention = attention_output_transposed.reshape(batch_size, seq_len_q, self.d_model)
        output = np.matmul(concat_attention, self.W_o)
        return output, attention_weights

class PositionwiseFeedForward:
    def __init__(self, d_model, d_ff, seed=None):
        self.d_model = d_model
        self.d_ff = d_ff
        if seed is not None: np.random.seed(seed)
        limit_w1 = np.sqrt(6.0 / (d_model + d_ff))
        self.W_1 = np.random.uniform(-limit_w1, limit_w1, (d_model, d_ff))
        self.b_1 = np.zeros(d_ff)
        limit_w2 = np.sqrt(6.0 / (d_ff + d_model))
        self.W_2 = np.random.uniform(-limit_w2, limit_w2, (d_ff, d_model))
        self.b_2 = np.zeros(d_model)

    def relu(self, x): return np.maximum(0, x)

    def forward(self, x):
        hidden_output = np.matmul(x, self.W_1) + self.b_1
        activated_output = self.relu(hidden_output)
        output = np.matmul(activated_output, self.W_2) + self.b_2
        return output
# --- End of prerequisite code --- #

class EncoderBlock:
    def __init__(self, d_model, num_heads, d_ff, seed=None):
        """
        Initialize a Transformer Encoder Block.

        Args:
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimension of the inner layer in PositionwiseFeedForward.
            seed (int, optional): Random seed for reproducibility of weights.
        """
        if seed is not None: # Manage seed for sub-components
            # Simple way to get different seeds for sub-components from a master seed
            mha_seed = seed
            pffn_seed = seed + 1 if seed is not None else None
            # LayerNorm doesn't have random init in this simple version, but could if gamma/beta were random
        else:
            mha_seed = None
            pffn_seed = None

        self.mha = MultiHeadAttention(d_model, num_heads, seed=mha_seed)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, seed=pffn_seed)

        self.layernorm1 = LayerNormalization(d_model)
        self.layernorm2 = LayerNormalization(d_model)
        
        # Dropout would be initialized here too, e.g., self.dropout_rate = dropout_rate

    def forward(self, x, mask=None):
        """
        Perform the forward pass for the Encoder Block.

        Args:
            x (np.ndarray): Input to the encoder block, shape (batch_size, seq_len, d_model).
            mask (np.ndarray, optional): Mask for the self-attention mechanism.
                                         Shape (batch_size, seq_len_q, seq_len_k).
                                         For self-attention, seq_len_q = seq_len_k = seq_len.

        Returns:
            np.ndarray: Output of the encoder block, shape (batch_size, seq_len, d_model).
        """
        # 1. Multi-Head Self-Attention sub-layer
        # Q=x, K=x, V=x for self-attention
        # The mask here is typically a padding mask for the input sequence.
        attention_output, _ = self.mha.forward(Q_in=x, K_in=x, V_in=x, mask=mask)
        # (Dropout would be applied to attention_output here)
        # Add & Norm
        x_after_mha = self.layernorm1.forward(x + attention_output)

        # 2. Position-wise Feed-Forward sub-layer
        ffn_output = self.ffn.forward(x_after_mha)
        # (Dropout would be applied to ffn_output here)
        # Add & Norm
        output = self.layernorm2.forward(x_after_mha + ffn_output)

        return output

```

**Explanation of the Code:**

1.  **`EncoderBlock.__init__(...)`:**
    *   Initializes an instance of `MultiHeadAttention` (`self.mha`).
    *   Initializes an instance of `PositionwiseFeedForward` (`self.ffn`).
    *   Initializes two instances of `LayerNormalization` (`self.layernorm1` and `self.layernorm2`), one for after the attention sub-layer and one for after the FFN sub-layer.
    *   A simple seed management is added to allow reproducible weight initialization in sub-components if a master seed is provided.

2.  **`EncoderBlock.forward(self, x, mask=None)`:**
    *   Takes the input `x` (shape `(batch_size, seq_len, d_model)`) and an optional `mask`.
    *   **Multi-Head Attention Sub-layer:**
        *   `attention_output, _ = self.mha.forward(Q_in=x, K_in=x, V_in=x, mask=mask)`: Performs self-attention. The input `x` is used for Queries, Keys, and Values. The `mask` (if provided) is passed to the MHA layer. We only need the output, not the attention weights, for the block's forward pass.
        *   `x_after_mha = self.layernorm1.forward(x + attention_output)`: Implements the residual connection (`x + attention_output`) followed by layer normalization.
    *   **Position-wise Feed-Forward Sub-layer:**
        *   `ffn_output = self.ffn.forward(x_after_mha)`: Passes the output of the first sub-layer through the FFN.
        *   `output = self.layernorm2.forward(x_after_mha + ffn_output)`: Implements the second residual connection (`x_after_mha + ffn_output`) followed by layer normalization.
    *   Returns the final `output` of the encoder block, which has the same shape as the input `x`.

## 6.4 Simple Input/Output Examples

Let's test our `EncoderBlock`.

```python
# Example Usage of EncoderBlock

# Parameters
batch_size_eb = 2
seq_len_eb = 5      # Input sequence length
d_model_eb = 16     # Model dimension (must be divisible by num_heads)
num_heads_eb = 4    # Number of attention heads
d_ff_eb = 32        # Feed-forward inner dimension (e.g., 2 * d_model or 4 * d_model)

# Initialize EncoderBlock
# Using a seed for reproducible weight initialization in MHA and PFFN
encoder_block = EncoderBlock(d_model=d_model_eb, 
                             num_heads=num_heads_eb, 
                             d_ff=d_ff_eb, 
                             seed=123)

# Dummy input data (e.g., token embeddings + positional encodings)
np.random.seed(456)
input_to_encoder_block = np.random.rand(batch_size_eb, seq_len_eb, d_model_eb)

# Dummy padding mask (optional)
# Mask: 0 = attend, 1 = do not attend
# Example: in the first batch item, the last token is padding.
# In the second batch item, the last two tokens are padding.
# Mask shape should be (batch_size, seq_len_q, seq_len_k)
# For self-attention, seq_len_q = seq_len_k = seq_len_eb
padding_mask = np.zeros((batch_size_eb, seq_len_eb, seq_len_eb), dtype=int)

# Mask for first batch item: last token is padding
# This means the last key is masked for all queries in the first batch item.
padding_mask[0, :, -1] = 1

# Mask for second batch item: last two tokens are padding
padding_mask[1, :, -2:] = 1

# Note: A more typical padding mask might be simpler, e.g., (batch_size, 1, seq_len_k)
# if it only depends on keys. Our MHA expects (batch_size, seq_len_q, seq_len_k)
# or for it to be broadcastable. The current mask structure is fine for self-attention.

print("--- Transformer Encoder Block Example ---")
print(f"Input tensor shape: {input_to_encoder_block.shape}")
if padding_mask is not None:
    print(f"Padding mask shape: {padding_mask.shape}")

# Forward pass through the encoder block
output_encoder_block = encoder_block.forward(input_to_encoder_block, mask=padding_mask)
# output_encoder_block = encoder_block.forward(input_to_encoder_block, mask=None) # Test without mask

print(f"\nOutput of Encoder Block shape: {output_encoder_block.shape}")

# Verify output shape: (batch_size, seq_len, d_model)
assert output_encoder_block.shape == (batch_size_eb, seq_len_eb, d_model_eb)

print("\nSample of Input (first batch, first token, first 4 features):\n", input_to_encoder_block[0, 0, :4])
print("\nSample of Output (first batch, first token, first 4 features):\n", output_encoder_block[0, 0, :4])

# The internal attention weights could also be inspected if mha.forward returned them
# and EncoderBlock also returned them, e.g., for visualization.
```

**Running the Example:**

Save all the class definitions (`LayerNormalization`, `MultiHeadAttention`, `PositionwiseFeedForward`, `EncoderBlock`, and helpers) and the example usage into a Python file. Running it will demonstrate a forward pass through a single encoder block, showing that the output shape matches the input shape (`d_model` is preserved).

## 6.5 Key Takeaways

*   A Transformer Encoder Block is composed of a Multi-Head Self-Attention sub-layer and a Position-wise Feed-Forward Network sub-layer.
*   Residual connections and Layer Normalization are applied around each sub-layer, which are critical for training deep models.
*   The output of an encoder block has the same dimensionality (`d_model`) as its input, allowing multiple blocks to be stacked.
*   Dropout is typically used for regularization but was omitted here for simplicity.

## 6.6 What's Next?

With a single `EncoderBlock` implemented, the next step (Part 7) is to create the **Full Transformer Encoder**, which involves stacking multiple `EncoderBlock` instances. We will also discuss how the initial input embeddings and positional encodings are fed into this stack.

Stay tuned!
