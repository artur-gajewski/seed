# How language models work, and what Seed makes visible

A language model learns to predict what comes next in text. Given a sequence such as
“The sky is”, it assigns probabilities to possible continuations. Generating a sentence means
choosing a continuation, adding it to the sequence, and making another prediction. Repeating
this process can produce a paragraph, a conversation, or a program.

That simple objective is the starting point for understanding large language models, or LLMs.
Predicting text well can require learning grammar, relationships between concepts, common
facts, and patterns of explanation. The objective does not explicitly demand truthfulness or
understanding, however. A plausible continuation can still be false, inconsistent, or unrelated
to what the user intended.

Seed makes this process small enough to inspect. Its model engine is implemented directly in
TypeScript, including the Transformer, gradient calculations, and optimizer. New models start
without pretrained language knowledge. The **Training** section teaches a model from a corpus;
**New Life** teaches a separate model through short messages and supplied answers. Neither
section calls an external language-model service to provide its generated responses.

## Text becomes tokens

Neural networks operate on numbers, so text first passes through a tokenizer. A token can
represent a word, part of a word, a byte, or a special boundary. Many LLMs use vocabularies of
word fragments. Seed uses a particularly direct representation: UTF-8 bytes.

Seed's vocabulary contains 258 tokens: the 256 possible byte values, a beginning-of-text token,
and an end-of-response token. An ordinary ASCII letter takes one byte token. Other characters,
including many accented letters and emoji, take several. Seed can therefore accept new
characters without building a new vocabulary, but a token is not necessarily a complete
character or word. Early output may contain replacement characters such as `�` because an
untrained model can generate invalid byte sequences.

This also explains Seed's context limits. A 256-token window holds 256 byte tokens, including
any special tokens in that window, rather than 256 words. Role labels, spaces, and punctuation
consume space too. A long prompt can lose its beginning before the model produces an answer.

The tokenizer also establishes useful boundaries. In training text, Seed inserts an
end-of-response token after assistant answers written in its dialogue format. The explicit
story marker `<|endofstory|>` becomes the same stop token. The model must learn when that
token belongs; stopping at a natural boundary is part of the prediction task.

## From tokens to predictions

A token's numeric identifier is only an index. Seed looks it up in an **embedding table** to
obtain a vector: a list of learned numbers. It adds a learned position vector so that the
network can distinguish where tokens occur. Those vectors then pass through Transformer
blocks.

The central operation is **causal self-attention**. At each position, the network computes
queries, keys, and values from the current vectors. Query–key scores determine how strongly
that position combines information from available positions. A causal mask prevents a token
from attending to later tokens. During training, the answer may be present elsewhere in the
input sequence, but a prediction cannot look ahead to see it.

Multiple attention heads perform these combinations in parallel with different learned
projections. This gives the network several ways to relate positions. A head is not assigned
a fixed human role such as “grammar” or “facts”; its behavior emerges through training.

Each block also contains a feed-forward network that transforms the representation at each
position. Seed expands the vector to four times its width, applies a nonlinear GELU function,
and projects it back. Residual connections add block outputs to their inputs, while layer
normalization helps keep the numerical behavior manageable. Stacking blocks allows later
layers to transform information already combined by earlier layers.

Finally, an output projection produces one score, or **logit**, for each vocabulary token.
Softmax converts scores into probabilities. The result is a distribution over possible next
tokens, rather than a retrieved sentence or a direct lookup of a stored answer.

Seed implements a decoder-only Transformer in [server/model.ts](../server/model.ts). The
underlying attention, matrix operations, and derivatives live in
[server/tensor.ts](../server/tensor.ts).

## Learning means changing weights

The network's learned numbers are its **parameters**, often called weights. Initially, its
main weight matrices contain small random values; biases and normalization parameters use
simple initial values. At this stage, predictions carry no learned knowledge of English.

Training supplies examples with known next tokens. Conceptually, a sequence like `hello`
produces aligned input and target sequences:

```text
Input:   h e l l
Target:  e l l o
```

The model predicts a distribution at every input position. **Cross-entropy loss** measures
how much probability it assigned to the actual next tokens. For one target token, the loss
is `-log(p)`, where `p` is its predicted probability. A confident correct prediction has low
loss; assigning little probability to the target produces high loss.

Backpropagation calculates how each parameter contributed to that loss. These derivatives,
called gradients, tell the optimizer how to adjust the parameters. Seed uses AdamW, which
tracks moving averages of gradients and their squares, and applies weight decay to matrices.
It also clips the overall gradient norm to limit unusually large updates.

A training step therefore runs a forward prediction, measures the error, propagates
gradients backward, and updates weights. A batch combines gradients from multiple examples
before one update. Repeating this cycle changes the probability distributions the model will
produce in future contexts.

The examples are not simply copied into an answer table. Their influence is distributed
across shared parameters. This makes generalization possible: a model may answer a phrasing
it has never seen. It also makes interference possible: teaching one pattern can weaken
another, and a model can memorize examples without learning a useful general rule.

## Training: learning from a prepared corpus

Seed's Training section exposes the ingredients of this process. You provide text and choose
the architecture and training settings. Width controls the size of token representations;
layers control how many Transformer blocks transform them; heads divide attention into
separate projections; context controls how much preceding text is available. Width must be
divisible by the number of heads. Batch size, update count, and learning rate control how
training proceeds.

The default Training model has width 256, four layers, eight heads, and a 256-token context,
with roughly 3.36 million parameters. It shares important mechanisms with LLMs while being
far smaller than models normally described as large. Increasing its size raises capacity and
computational cost; it does not supply missing knowledge or better examples.

The training engine in [server/studio.ts](../server/studio.ts) encodes the corpus and reserves
its final 10% of tokens for validation. Training samples windows from the remaining portion
and predicts the token following each input token. Its learning rate warms up, then follows
a cosine decay toward a lower rate. The default run uses batches of eight for 30,000 updates
with a peak learning rate of 0.0003. An update is not a complete pass through the corpus.

The loss chart helps distinguish fitting from generalization. Training loss measures
prediction on sampled training text. Validation loss measures prediction on held-out text
without updating the weights. Falling training loss alongside rising validation loss can
indicate overfitting. Seed evaluates up to three fixed validation windows at the first
update, every 100 updates, and the final update, so the displayed validation value is not a
new measurement at every step.

Even a good validation curve needs interpretation. Repeated examples across the contiguous
split can make validation easier than truly unfamiliar questions. Low byte-prediction loss
also does not establish factual accuracy or reliable reasoning. A separate collection of
unseen prompts is useful for checking whether the model learned the behavior you wanted.

The corpus determines that behavior. Stories provide examples of narrative continuation.
`User:` / `Assistant:` exchanges provide examples of answering questions in that format.
Those labels are ordinary text conventions the model learns, rather than built-in knowledge
of what an assistant should do. Varied, accurate examples give it more opportunity to learn
relationships beyond a single memorized wording.

Starting a new model resets learned weights. Continuing an existing model retains them and
the optimizer state, then trains on the supplied corpus with a restarted learning-rate
schedule. Studio checkpoints preserve model configuration, weights, optimizer state, update
count, and chart history. They do not include the source corpus, which must be kept separately.
The studio saves automatically every 100 total updates and also offers an explicit save.

## Generation uses learned probabilities

During generation, Seed runs the forward computation without backpropagation or optimizer
updates. It predicts one token, samples it, appends it to the context, and repeats until the
output limit or a predicted stop token. As the sequence grows, only the most recent tokens
within the configured context window remain visible to the model.

Temperature changes the sampling distribution: lower values concentrate probability on
high-scoring candidates, while higher values spread it more broadly. Top-k restricts sampling
to the highest-scoring `k` candidates when enabled. These controls change selection from the
model's predictions; they do not teach new information or repair an incorrect learned fact.

This distinguishes **context** from **learning**. Ordinary Seed Chat formats recent turns
as input, allowing them to influence the next reply within the context limit. It does not
update the model's weights. A fact supplied in that conversation can affect an answer without
becoming persistent learned knowledge. New Life explicitly adds a training process to
interaction.

## New Life: learning through interaction

New Life uses its own model and saved lessons, independently of the Training model. Each
parameter preset has its own learning. Its implementation in
[server/new-life.ts](../server/new-life.ts) uses the same Transformer and optimizer, but
organizes data around short lessons instead of a large imported corpus.

An ordinary message becomes a lesson containing `User: <message>`. Seed performs 12 training
updates, saves the learning, and then generates a reply from
`User: <message>\nAssistant:`. A message alone supplies examples of language, but it does not
specify the correct answer to its question. Repeatedly asking a question cannot provide an
answer the model has never been taught.

Teaching supplies that missing target. For example, first send:

```text
What color is the sky?
```

Then teach the latest question with:

```text
!The sky is blue.
```

The resulting lesson contains both the user question and the desired assistant answer. The
model trains on next-token prediction across that formatted sequence, including the answer
and its stop token. It does not use a separate answer-only loss or a reward for agreeing with
the user. This is supervised learning from a supplied example.

The interface also recognizes explicit correction phrases such as `No, the sky is blue.`
through text patterns in [src/lib/newLifeInput.ts](../src/lib/newLifeInput.ts). The prefix is
removed and the remaining answer is sent through the same teaching action. This recognition
does not depend on the untrained model understanding criticism. Bare feedback such as `No.`
does not supply a target answer. The `!` form makes teaching explicit.

After teaching, the interface displays an acknowledgment selected from fixed phrases. That
acknowledgment is interface feedback, not evidence that the model can now produce the taught
answer. Ask the question again to inspect its generated behavior. Several lessons or repeated
practice may be needed, and a correct answer to one wording does not prove generalization.

Each of New Life's 12 updates includes the newest lesson plus randomly sampled saved lessons
to fill the preset's batch. This **replay** gives earlier examples continued influence and
helps reduce forgetting. It does not eliminate forgetting or guarantee that all saved
lessons will be reproduced correctly. New Life has no separate held-out validation split.

Lessons are keyed by their exact message text. Teaching a new answer to the same message
replaces its saved answer; asking that message again preserves the answer already taught.
Replacing a lesson changes future training data rather than instantly erasing the previous
answer's influence on the weights. Generated replies are never automatically used as lessons,
which avoids treating the model's own guesses as correct answers.

Messages and answers are each limited to 100 UTF-8 bytes on one line, allowing a complete
formatted lesson to fit within the 256-token context. New Life retains up to 1,000 distinct
lessons per model size. It saves weights, optimizer state, and replay lessons after every
successful learning action. Unlike the studio checkpoint, these saved lessons include the
text used for learning. The visible conversation is temporary, and reply generation uses
the current message rather than the full conversation transcript.

## What Seed teaches about larger models

Seed makes a distinction that polished chat interfaces can hide: a tokenizer represents
text, a network predicts tokens, an optimizer changes its parameters, and a surrounding
application decides what to train on, what context to provide, and what to display. Training
and New Life differ chiefly in those surrounding decisions, rather than in the basic
mathematics of their models.

Many larger assistants combine broad pretraining with later instruction training and
preference-based optimization. Their applications may also provide document retrieval,
tools, or persistent storage. These are additional mechanisms; they do not follow
automatically from having a Transformer or a chat box. Seed's New Life teaching loop is a
small supervised-learning experiment, not a recreation of every stage used to build a
production assistant.

Seed lets you observe the foundations directly: random predictions becoming structured
text, lower loss sometimes accompanying memorization, and new lessons competing with old
ones. Its Training section makes corpus design and evaluation visible. New Life makes the
difference between asking, teaching, remembering examples, and generating an answer visible.
Both show how learned numerical relationships can produce useful language—and why useful
language must still be tested against the task it is meant to serve.
