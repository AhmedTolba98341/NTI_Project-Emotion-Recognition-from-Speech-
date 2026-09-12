# NTI_Project-Emotion-Recognition-from-Speech


Overview

This project develops a deep learning system for recognizing human
emotions from speech audio. The model uses the CREMA-D dataset and
classifies each audio recording into one of six emotions:

-Angry

-Disgust

-Fear

-Happy

-Neutral

-Sad

The main approach converts speech signals into log-Mel spectrograms,
which are then used as image-like inputs to a custom Convolutional
Neural Network (CNN).

The preprocessing pipeline was designed to improve the quality and
consistency of the audio representation by applying:

1. Audio trimming

2. Noise reduction

3. Fixed-length padding/truncation

4. Log-Mel spectrogram extraction

5. Per-clip normalization

6. SpecAugment-style training augmentation

7. Explicit class-based sample weighting

An important part of the project is the use of an actor-independent
split, which prevents the same speaker from appearing in different
dataset splits.

Dataset

The project uses the CREMA-D (Crowd-sourced Emotional Multimodal
Actors Dataset).

Each audio filename contains information about:

-Actor ID

-Sentence

-Emotion

-Intensity

The six emotion classes are mapped as follows:

Code   Emotion

ANG    Angry
DIS    Disgust
FEA    Fear
HAP    Happy
NEU    Neutral
SAD    Sad

Dataset Split

The data was divided using GroupShuffleSplit, with the actor ID used
as the grouping variable.

Split            Samples

Training           5,152
Validation         1,142
Test               1,148
Total      7,442

The actor leakage check produced empty intersections:

-Train ∩ Validation = empty

-Train ∩ Test = empty

-Validation ∩ Test = empty

This means that no actor is shared between the training, validation, and
test sets.

The emotion distributions are also approximately balanced. Neutral
contains fewer samples than the other classes, so class weights are
calculated and applied during training.

Audio Preprocessing

Sampling Rate

All audio files are loaded as mono audio at:

16,000 Hz

Fixed Duration

Each recording is converted to a fixed length of:

3.5 seconds = 56,000 samples

Short recordings are zero-padded, while longer recordings are truncated.

Trimming

Silence is removed using Librosa's trimming function with:

-top_db = 25

Noise Reduction

Noise reduction is applied using the noisereduce package with
non-stationary noise reduction.

The pipeline also checks the output for invalid values. If noise
reduction fails or produces invalid values, the trimmed audio is
retained instead.

A runtime warning from noisereduce may appear for some recordings, but
the preprocessing code is designed to fall back to the undenoised signal
when necessary.

Log-Mel Spectrogram

The processed audio is converted into a log-Mel spectrogram.

Main parameters:

Parameter                   Value

Sample rate             16,000 Hz
Number of Mel bands           128
FFT size                    1,024
Hop length                    512
Number of frames              110

The final input shape is:

(128, 110, 1)

The spectrogram is converted to log power using:

librosa.power_to_db()

Each spectrogram is normalized using its own mean and standard
deviation.

The spectrograms are then cached as NumPy arrays:

/content/spec_features/
├── X_train.npy
├── X_val.npy
├── X_test.npy
├── y_train.npy
├── y_val.npy
└── y_test.npy

Data Augmentation

SpecAugment-style augmentation is applied only to the training data.

Two types of masking are used:

-Frequency masking: up to 15% of Mel bands

-Time masking: up to 15% of time frames

The validation and test sets are not augmented.

This helps the CNN become less dependent on specific frequency or time
regions in the training spectrograms.

Class Weighting

Class weights are calculated using Scikit-learn's compute_class_weight
with balanced weighting.

The resulting weights are:

Angry    : 0.9758
Disgust  : 0.9758
Fear     : 0.9758
Happy    : 0.9758
Neutral  : 1.1418
Sad      : 0.9758

Because Neutral has fewer training samples, it receives a higher weight.

The weights are converted into explicit sample weights and passed
through the training dataset.

CNN Architecture

The model is a custom CNN designed to operate directly on the log-Mel
spectrograms.

Architecture:

Input: (128, 110, 1)

Conv2D 32 filters
Batch Normalization
Max Pooling
Dropout 0.25

Conv2D 64 filters
Batch Normalization
Max Pooling
Dropout 0.25

Conv2D 128 filters
Batch Normalization
Max Pooling
Dropout 0.30

Conv2D 128 filters
Batch Normalization

Global Average Pooling

Dropout 0.40
Dense 128, ReLU
Dropout 0.40

Dense 6, Softmax

Number of Parameters

The network contains approximately:

258,950 total parameters

Trainable parameters: 258,246

Non-trainable parameters: 704

Training

The CNN is trained using:

Optimizer: Adam

Initial learning rate: 0.001

Loss: Categorical Cross-Entropy

Batch size: 32

Maximum epochs: 60

Early stopping: enabled

Learning-rate reduction: enabled

Callbacks

Two callbacks are used:

EarlyStopping - Monitors validation accuracy - Patience: 10 epochs -
Restores the best model weights

ReduceLROnPlateau - Monitors validation loss - Reduction factor:
0.5 - Patience: 4 epochs - Minimum learning rate: 1e-6

Training stopped after epoch 44 because the early-stopping condition was
reached.

The best validation accuracy observed during training was approximately:

59.28%

Test Results

The final model was evaluated on the completely held-out test set of
1,148 recordings.

Metric             Score

Accuracy      0.5601
Precision     0.5787
Recall        0.5601
F1-score      0.5633

Classification Report

Emotion                  Precision     Recall   F1-score     Support

Angry                         0.66       0.65       0.66         196
Disgust                       0.55       0.54       0.54         196
Fear                          0.49       0.47       0.48         196
Happy                         0.44       0.60       0.51         196
Neutral                       0.78       0.52       0.62         168
Sad                           0.57       0.58       0.58         196
Macro Average         0.58   0.56   0.56   1,148
Weighted Average      0.58   0.56   0.56   1,148

Confusion Matrix Analysis

The confusion matrix shows that the model performs best on Angry,
while Fear is one of the more difficult emotions to distinguish.

The main observations are:

Angry has relatively strong recognition, with 127 correctly
classified samples.

Disgust has 105 correct predictions.

Fear has 93 correct predictions and is frequently confused with
Happy and Sad.

Happy has 117 correct predictions and relatively high recall.

Neutral has high precision but lower recall, meaning predictions
classified as Neutral are often correct, but many actual Neutral
samples are assigned to other emotions.

Sad has 114 correct predictions and is also confused with
Disgust and Fear.

Overall, the errors are concentrated among emotions with similar
acoustic characteristics, particularly Fear, Happy, and Sad.

Training Behavior

The training accuracy steadily improved from approximately 34% in the
first epoch to more than 64% by the later epochs.

Validation accuracy was more variable but gradually improved, reaching a
best value of approximately 59.28%.

This difference between training and validation performance suggests
that the CNN learns useful emotion-related patterns but still has some
generalization difficulty on unseen actors.

The learning-rate scheduler progressively reduced the learning rate
during training, allowing the model to continue refining its parameters
after the initial learning phase.

Project Pipeline

The complete workflow can be summarized as:

CREMA-D Audio
      │
      ▼
Actor-Independent Split
      │
      ├── Train
      ├── Validation
      └── Test
      │
      ▼
Load Audio at 16 kHz
      │
      ▼
Trim Silence
      │
      ▼
Noise Reduction
      │
      ▼
Fix Length to 3.5 Seconds
      │
      ▼
Log-Mel Spectrogram
      │
      ▼
Per-Clip Normalization
      │
      ├── Training → SpecAugment + Sample Weighting
      ├── Validation
      └── Test
      │
      ▼
Custom CNN
      │
      ▼
6 Emotion Classes
      │
      ▼
Evaluation
      ├── Accuracy
      ├── Precision
      ├── Recall
      ├── F1-score
      └── Confusion Matrix

Technologies Used

Python

TensorFlow / Keras

Librosa

NumPy

Pandas

Scikit-learn

noisereduce

Matplotlib

tqdm

Google Colab

Kaggle Dataset API

Main Strengths

Uses a large speech emotion dataset with 7,442 audio files.

Uses actor-independent splitting to avoid speaker leakage.

Applies consistent audio preprocessing.

Converts speech into informative log-Mel spectrogram
representations.

Uses training-only SpecAugment.

Uses explicit sample weighting for class imbalance.

Uses early stopping and adaptive learning-rate reduction.

Reports multiple evaluation metrics instead of accuracy alone.

Includes a confusion matrix for detailed error analysis.

Limitations

The final test accuracy is 56.01%, so the model is not yet accurate
enough for reliable real-world emotion recognition.

The main limitations observed from the results are:

Several emotions have overlapping acoustic characteristics.

Fear has relatively low precision, recall, and F1-score.

Happy has relatively low precision despite achieving 60% recall.

Validation accuracy fluctuates considerably during training.

The model shows a noticeable gap between training and validation
performance.

The custom CNN is relatively lightweight, which limits its
representational capacity compared with larger pretrained audio
models.

Possible Future Improvements

Future experiments could investigate:

Transfer learning using pretrained audio models.

Larger CNN architectures such as EfficientNet-style spectrogram
models.

Transformer-based audio models.

More advanced SpecAugment settings.

Mixup or other audio augmentation methods.

Hyperparameter tuning.

Class-specific error analysis.

Ensemble models.

Audio features that combine log-Mel spectrograms with MFCC or
prosodic features.

More systematic handling of difficult emotion pairs such as Fear
vs. Sad and Fear vs. Happy.

Result Summary

Model: CNN + Corrected Preprocessing + Explicit Sample Weighting

Test Accuracy: 56.01%

Test F1-score: 56.33%

The model demonstrates that a custom CNN operating on normalized log-Mel
spectrograms can learn meaningful emotion-related patterns from speech.
The actor-independent evaluation provides a more realistic measure of
generalization to unseen speakers, while the confusion matrix highlights
which emotions remain difficult to distinguish.

<img width="675" height="455" alt="image" src="https://github.com/user-attachments/assets/dc394294-a03c-4a2d-8207-fa728eea9541" />


