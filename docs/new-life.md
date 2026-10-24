# New life: teach through chat

[Documentation](README.md) · [Seed](../README.md)

Open **New life** below Chat to grow a separate model from random weights. Send a
short message, then type a correction in the same chat input to supply the response
you wanted. For example, send `What is 1 + 1?` and teach `!1 plus 1 is 2.` Early replies
will be gibberish; a handful of lessons does not produce a fluent assistant.

### Try a first lesson

1. Send `What color is the sky?` in **Say something simple…**.
2. Wait for Seed's reply.
3. Enter `!The sky is blue.` in **Say something simple…** and press Enter to teach and save.
4. Ask the question again, then introduce other short questions and answers.

Enter submits the chat input. After sending a message or correction, focus returns to the message
input, and the conversation follows the latest row automatically. The counters show
remembered lessons, taught answers, and total learning updates.

Use **Up** and **Down** in the input to recall successfully submitted messages and
corrections. Down past the latest entry restores the draft you were writing. Recalled
text can be edited before sending; browsing history does not train the model. Input
history lasts until the page is refreshed.

### Teach with `!` or a written correction

You can also teach directly in **Say something simple…**: prefix your preferred answer
with `!`, for example `!The sky is blue.`. This teaches an answer to the latest message
you sent in the visible conversation, rather than sending a new question. The `!` is
removed before learning and does not count toward the 100-byte limit. Send a message
first; the shortcut needs a previous reply to correct. Both Enter and the send button
support this shortcut, and the input clears and regains focus after saving successfully.

For example, after asking `What color is the sky?`, type `!The sky is blue.` and press
Enter. Seed learns and saves that question-and-answer pair, then acknowledges it in
the chat with a randomly chosen phrase such as “Ok, I will try to learn this.” These
acknowledgments are interface messages, not model-generated answers or training data.
Ask the question again when you want to check its progress. Further corrections still
teach the latest question, rather than the acknowledgment. To teach an earlier
question, ask it again first.

New life also recognizes explicit written corrections, ignoring capitalization:

- `No. The sky is blue.` or `No, the sky is blue.`
- `Nope, the sky is blue.` or `Not quite. The sky is blue.`
- `That's wrong. The sky is blue.` or `Incorrect: The sky is blue.`
- `Actually, the sky is blue.` or `Correction: The sky is blue.`
- `The correct answer is The sky is blue.` or `The right answer: The sky is blue.`
- `It should be The sky is blue.`, `You should say The sky is blue.`, or `I meant The sky is blue.`

These forms use the same teaching action as `!`: the correction prefix is removed,
the supplied answer is paired with your latest message, and learning is saved. The
byte counter measures the extracted answer. The composer shows **Teaching latest
answer** when it recognizes a correction. Combined cues such as `No, the correct
answer is 4.` also work.

Recognition uses explicit text patterns, not the model's understanding of arbitrary
feedback. Bare feedback such as `No.` and questions such as `Actually, what color is
the sky?` remain ordinary messages. Use `!` for an answer that
is itself a question or wording the shortcut does not recognize.

### How learning works

Each message or taught answer triggers 12 small training updates. Training combines
the latest lesson with randomly selected previous lessons to reduce forgetting.
User messages teach language patterns; question-and-answer lessons teach responses.
Generated replies are never automatically added to training. Teaching another answer
to the same question replaces its previous lesson; asking again preserves the taught
answer. Message limits follow the selected context window, from 208 bytes for the smallest
preset to 2000 bytes for 153M. When teaching, the counter shows the answer space
remaining after the original message and role labels, up to the message limit. Up to 1,000 distinct questions
are retained; older lessons are not silently removed when this limit is reached.

Every ordinary message trains before generating a reply, including repeated questions. Replies
also use random sampling (temperature 0.8), so the same question can produce different
answers even when saved learning is restored correctly. Refreshing the page does not
itself train the model. Early output may contain `�` replacement characters because
the byte-level model has not yet learned to generate valid UTF-8 consistently.

### Continuing later

New Life offers eight model presets, each with its own learning. The 153M preset is
the default when no previous selection is saved. The dropdown shows
parameter counts. Width, layers, attention heads, batch size, and context window update
below the dropdown when you select a preset.

| Parameters | Width | Layers | Attention heads | Batch size | Context tokens |
| ---------- | ----- | ------ | --------------- | ---------- | -------------- |
| 38K        | 32    | 1      | 4               | 2          | 256            |
| 108K       | 64    | 1      | 4               | 3          | 384            |
| 529K       | 128   | 2      | 4               | 4          | 512            |
| 1.6M       | 192   | 3      | 6               | 5          | 768            |
| 2.8M       | 256   | 3      | 8               | 6          | 1024           |
| 7.8M       | 384   | 4      | 8               | 6          | 1280           |
| 58M        | 768   | 8      | 12              | 8          | 1536           |
| 153M       | 1024  | 12     | 16              | 8          | 2048           |

The largest preset has 153,782,530 trainable parameters.
Weights, optimizer state, and replay lessons are saved atomically after each learning
action to `data/models/new-life-checkpoint-<parameters>.pt`, independently
of the studio's `data/models/seed_default.pt`. Existing Transformer New life JSON/V8
checkpoints are converted to `.pt` when their preset is first loaded, preserving
lessons, weights, and optimizer moments. Legacy files are retained. An existing `.pt`
checkpoint takes precedence. Saved learning reloads automatically
after a server restart.
Conversation display is kept only for the current browser session; saved lessons
include your messages and taught answers. Replies use the current message as context.
This experimental mode can memorize lessons and can still forget earlier skills.
