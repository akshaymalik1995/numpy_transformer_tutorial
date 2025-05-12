# Transformer from Scratch with NumPy - Tutorial Series

This repository contains a 10-part tutorial series on implementing the Transformer neural network architecture from scratch using only NumPy. The goal is to provide a clear, step-by-step guide with a focus on mathematical intuition and fundamental implementation details, avoiding high-level deep learning libraries.

## Tutorial Parts

The series progressively builds the Transformer model, with each part focusing on specific components:

1.  **[Part 1: Introduction to Transformers and NumPy Setup](./01_introduction.md)**
    *   Overview of the Transformer architecture.
    *   Rationale for using NumPy.
    *   Environment setup and basic NumPy operations.

2.  **[Part 2: Implementing Scaled Dot-Product Attention](./02_scaled_dot_product_attention.md)**
    *   Explanation of the attention mechanism.
    *   Detailed breakdown and NumPy implementation of Scaled Dot-Product Attention.
    *   Masking.

3.  **[Part 3: Building Multi-Head Attention from Scratch](./03_multi_head_attention.md)**
    *   Concept of Multi-Head Attention.
    *   Implementation of the `MultiHeadAttention` class, combining multiple scaled dot-product attention heads.

4.  **[Part 4: Creating the Position-wise Feedforward Layer](./04_positionwise_feedforward.md)**
    *   Description of the Position-wise Feed-Forward Network (FFN).
    *   NumPy implementation of the `PositionwiseFeedForward` class.

5.  **[Part 5: Adding Positional Encoding](./05_positional_encoding.md)**
    *   Necessity of positional information in Transformers.
    *   Implementation of sinusoidal positional encoding.

6.  **[Part 6: Constructing a Transformer Encoder Block](./06_encoder_block.md)**
    *   Architecture of a single Encoder Block.
    *   Implementation of Layer Normalization, Dropout (conceptual), and the `EncoderBlock` class, integrating Multi-Head Attention and FFN with residual connections.

7.  **[Part 7: Implementing the Full Transformer Encoder](./07_full_transformer_encoder.md)**
    *   Stacking multiple Encoder Blocks.
    *   Integrating input embeddings and positional encoding.
    *   Implementation of the `TransformerEncoder` class.

8.  **[Part 8: Implementing the Transformer Decoder Block](./08_transformer_decoder_block.md)**
    *   Architecture of a single Decoder Block (Masked Multi-Head Self-Attention, Encoder-Decoder Attention, FFN).
    *   Implementation of the `DecoderBlock` class and look-ahead mask creation.

9.  **[Part 9: Building the Complete Transformer Architecture](./09_complete_transformer.md)**
    *   Assembling the full Transformer model: Encoder, Decoder, Embedding Layer, and Final Linear Layer.
    *   Comprehensive mask creation.
    *   Implementation of the full `Transformer` class.

10. **[Part 10: Putting It All Together: Training on a Toy Dataset and Conclusion](./10_training_and_conclusion.md)**
    *   Conceptual discussion of the training loop, loss calculation (cross-entropy), backpropagation, and optimization.
    *   Forward pass example on a toy dataset (e.g., sequence reversal).
    *   Series conclusion and further learning.

## How to Use

Each part is a self-contained Markdown file (`.md`) that includes explanations, mathematical formulas, and NumPy code snippets. You can follow the parts sequentially to build up your understanding and implementation. The code examples are designed to be run in a Python environment with NumPy installed.

Enjoy learning about Transformers!
