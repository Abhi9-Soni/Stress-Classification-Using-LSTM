# Assignment explanation: stressed vs. normal text classification using LSTM

## Aim

Build a supervised natural-language-processing model that takes a text post and
predicts one of the supplied dataset labels: **normal (0)** or **stressed (1)**.
The project covers data preparation, a sequence model, a conventional baseline,
evaluation on unseen examples and reusable inference. Its predictions concern the
dataset labels; they do not diagnose a person's mental health.

## 1. What the dataset contains

The uploaded file has 1,800 examples and two columns. `post` is the model input;
`label` is the target. There are initially 900 examples per class. The first 900
records are stressed and the last 900 are normal. A row-order split would therefore
produce misleading class distributions. The file decodes using cp1252 rather than
UTF-8, which explains why a plain `pd.read_csv()` call may fail.

There are no missing or empty posts. After deterministic text normalization,
one pair of identical inputs has conflicting labels; both rows are removed.
Two further pairs repeat the same label; one copy of each is removed. The final
dataset has **1,796 posts: 897 normal and 899 stressed**. We do not assume that
these are 1,796 distinct people because the CSV contains no author identifiers.

## 2. Preprocessing and preventing leakage

The `clean_text()` function normalizes Unicode, lowercases text, replaces URLs,
emails and user mentions with placeholders, removes most punctuation, and
collapses extra whitespace. It preserves apostrophes, numbers and stop words.
Words such as “not” matter: “I am worried” and “I am not worried” should retain
different token sequences. Stemming and lemmatization are not required here.

The duplicate rule is applied before splitting so that the same normalized input
cannot be in training and testing. This deterministic cleaning does not estimate
statistics. Learned transformations, including vocabulary selection and TF-IDF
weights, are fitted only on training data. Fitting them on the complete dataset
would expose the model-building process to information from held-out examples.

The seed-42 stratified split is:

| Split | Normal | Stressed | Total | Role |
|---|---:|---:|---:|---|
| Training | 628 | 629 | 1,257 | Learn weights and vocabulary |
| Validation | 134 | 135 | 269 | Select the stopping epoch |
| Test | 135 | 135 | 270 | Estimate final performance |

“Stratified” means preserving approximately the same class proportions in each
split. Assertions check that normalized texts do not overlap. This protects against
exact normalized duplicates, not all paraphrases or hidden author/thread overlap.

## 3. How words become model inputs

Neural networks cannot process raw word strings as numerical features directly.
`TextVectorization` assigns integer IDs to training-vocabulary tokens. ID 0 is
padding; ID 1 represents unseen words. The vocabulary is capped at 10,000 IDs;
this run used 9,992 including reserved IDs.

Each input has 256 positions. This is the smaller of the training 95th-percentile
post length and a predeclared cap of 256. Short sequences are padded on the right;
long ones retain only their first 256 tokens. About 14.96% of training posts are
truncated. This keeps computation manageable but can remove informative endings.
Tokenization and truncation are identical during training and prediction.

## 4. What the LSTM learns

An **embedding** layer maps each token ID to a learned vector of 64 numbers. These
vectors are randomly initialized and trained on this dataset; they are not pretrained
GloVe or Word2Vec embeddings. A batch of 32 posts has shape `(32, 256)` before
embedding and `(32, 256, 64)` afterward.

An **LSTM**, or Long Short-Term Memory network, reads the vectors in order. Its
64-unit hidden state summarizes what it has processed, while a separate cell state
supports longer-term memory. Its gates control what information is forgotten,
written and exposed. With sigmoid activation σ and elementwise product ⊙:

\[
f_t=\sigma(W_f[h_{t-1},x_t]+b_f),\quad
i_t=\sigma(W_i[h_{t-1},x_t]+b_i)
\]
\[
\tilde c_t=\tanh(W_c[h_{t-1},x_t]+b_c),\quad
c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t
\]
\[
o_t=\sigma(W_o[h_{t-1},x_t]+b_o),\quad
h_t=o_t\odot\tanh(c_t)
\]

Here x_t is the current word embedding. The forget gate f_t scales old memory,
the input gate i_t scales candidate memory, and the output gate o_t controls the
hidden state. These operations can help preserve useful information across words;
they do not guarantee semantic understanding.

`mask_zero=True` passes a padding mask from the embedding to the LSTM, so padded
positions do not act as meaningful words. Spatial dropout drops embedding features
consistently across timesteps during training. The final LSTM hidden state enters
a 32-unit ReLU layer, dropout and one sigmoid output. Inference disables dropout.

| Trainable layer | Output shape per example | Parameters in recorded run |
|---|---|---:|
| Embedding | 256 × 64 | 639,488 |
| LSTM | 64 | 33,024 |
| Dense ReLU | 32 | 2,080 |
| Dense sigmoid | 1 | 33 |
| **Total** | | **674,625** |

For input dimension d and hidden size h, Keras LSTM parameters total
`4h(d + h + 1)`: four gate/candidate transformations with input weights,
recurrent weights and biases. With d = h = 64, this is 33,024.

## 5. Training objective

The sigmoid output p lies between 0 and 1. Binary cross-entropy penalizes incorrect
predictions, especially confident ones:

\[
L=-\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right].
\]

Adam optimizes the network at learning rate 0.001 with batches of 32. Clip norm 1.0
limits gradient magnitude. Spatial dropout 0.2 and dense dropout 0.5 regularize the
model. Training allows up to 20 epochs but stops after four consecutive epochs
without a validation-loss improvement. The recorded run stopped after epoch 8 and
restored epoch 4. The test set played no role in that selection.

The decision rule is fixed in advance: score ≥ 0.5 means stressed; otherwise normal.
The score is a model output, not an established clinical probability. No probability
calibration or clinical validation was performed.

## 6. Baseline and measured results

The TF-IDF baseline represents informative words and two-word phrases, then trains
logistic regression. It uses the same cleaned inputs and split but does not truncate
posts to 256 tokens. This is a practical baseline comparison, not a controlled study
isolating only the classifier architecture.

| Model | Test accuracy | Stressed precision | Stressed recall | Stressed F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| TF-IDF + logistic regression | 93.33% | 0.9398 | 0.9259 | 0.9328 | 0.9832 |
| LSTM | 92.59% | 0.9137 | 0.9407 | 0.9270 | 0.9747 |

The LSTM correctly classified 250 of 270 posts. Its confusion matrix is:

| Actual label | Predicted normal | Predicted stressed |
|---|---:|---:|
| Normal | 123 | 12 |
| Stressed | 8 | 127 |

Precision = 127/(127 + 12): among predicted stressed posts, how many carry that label?
Recall = 127/(127 + 8): among actually stressed-labeled posts, how many are detected?
F1 is the harmonic mean of precision and recall. Accuracy = (123 + 127)/270.
ROC-AUC evaluates score ranking across thresholds; it is not the same as accuracy.

The baseline made two fewer test errors. The LSTM had higher stressed recall but
lower precision. These results do not establish that one method is significantly
better overall. A simpler lexical model can perform strongly on a small dataset,
especially when labels correlate with topic vocabulary. Do not claim LSTM superiority
just because it is a deep-learning model.

## 7. Saving, reuse and validation

`stress_text_model.keras` stores the vectorizer and trained network. `config.json`
records label mapping, threshold, sequence length, data hash, split sizes and package
versions. `preprocessing.py` preserves the deterministic cleaning function.
`predict.py` loads these files and classifies raw input text. Saving weights alone
would omit crucial preprocessing information.

All 12 Python cells were executed sequentially on the actual dataset in a local CPU
environment. The notebook contains those real outputs and plots. Notebook schema
and cell syntax were checked. Split overlap assertions and a save/reload prediction
comparison passed; standalone prediction was also exercised. The Colab upload widget
and GPU execution were not run in this environment. Results may vary on Colab.

## 8. Discussion and conclusion

The assignment demonstrates an end-to-end LSTM text classifier with 92.59% held-out
accuracy and transparent baseline reporting. Its strongest methodological features
are resolving duplicate ambiguity, isolating train-fitted preprocessing, using a
validation set for early stopping, and saving everything needed for prediction.

Limitations include the small sample, one split and one seed, unknown annotation
quality, missing author/source metadata, long-text truncation, and possible topic
shortcuts. The source and license of the CSV were not provided. Future work should
first document the data and evaluate stronger splits, then investigate repeated-seed
results, pretrained embeddings or bidirectional LSTMs using validation data. A public
GitHub project should report these limitations alongside the measured results.
