# Seed corpus examples

This folder contains different corpus examples for training the Seed model. 
Each corpus is a collection of text files that can be used to train the model 
on specific tasks or domains. The corpora are designed to be simple and focused, 
allowing for effective training without overwhelming the model with unnecessary complexity.

## toddler_reasoning_30000.txt

This corpus emphasizes simple reasoning rather than just memorizing conversations.
It will cover counting, comparisons, spatial relationships, cause and effect, sequencing,
object properties, everyday problem-solving, and basic logic.

### Recommended settings for training with toddler_reasoning_30000.txt

**Embedding width: 256**

Determines how much information each token can represent internally. A width of 256 provides a 
reasonable balance between learning capacity, memory consumption, and training speed for a small 
language model.

**Layers: 4**

Each transformer layer processes information and learns increasingly complex patterns. 
Four layers provide enough depth for experimenting with basic language understanding, 
comparisons, and simple reasoning without making the model unnecessarily large.

**Attention heads: 8**

Attention heads allow the model to focus on different relationships between tokens. 
With an embedding width of 256 and 8 heads, each head operates on 32 dimensions.

**Batch size: 8**

Controls how many training windows are processed per update. A batch size of 8 generally 
produces more stable gradient estimates than a batch size of 2, although it requires more memory. 
Reduce it to 4 if memory becomes an issue.

**Training steps: 30,000**

Because the model starts with random weights, it needs substantial training to learn language patterns. 
A budget of 30,000 steps provides a useful starting experiment for the 30,000-example corpus.

More steps do not automatically mean better results. Monitor training and evaluation loss to detect 
overfitting.

**Learning rate: 0.0003**

Controls how much the model's parameters change during each training update. A learning rate of 0.0003 
is a conservative starting point that may provide more stable training than 0.001, particularly with 
an Adam-like optimizer.

**Context window: 256 tokens**

Determines how much text the model can process at once. Because this model uses UTF-8 bytes as 
tokens, 256 tokens represent approximately 256 English ASCII characters.

A 64-token window may be too short to contain a complete question, answer, and explanation. 
Increasing it to 256 allows the model to learn from longer reasoning examples.

### Expected training workload

With the recommended configuration:

- Training steps: 30,000
- Batch size: 8
- Context window: 256 tokens
- Tokens processed per step: 2,048
- Total tokens processed: 61,440,000

This assumes each training step processes 8 complete windows of 256 tokens.

The 61.44 million tokens represent repeated exposure to the training corpus, 
not 61.44 million unique tokens.

### Monitoring training

Evaluate the model at approximately:

- 1,000 steps
- 5,000 steps
- 10,000 steps
- 20,000 steps
- 30,000 steps

Monitor both training loss and evaluation loss.

A decreasing training loss indicates that the model is learning to predict its training data. 
However, this does not necessarily mean it is developing general reasoning abilities.

If training loss continues decreasing while evaluation loss increases, the model may be overfitting.

**Goal:** Train a small language model from random initialization to recognize basic English 
patterns, follow simple conversational structures, and demonstrate elementary reasoning on 
previously unseen examples.

**Important:** These settings are experimental starting values, not guaranteed optimal settings. 
Actual results depend on the training implementation, corpus quality, hardware, and 
optimization algorithm.
