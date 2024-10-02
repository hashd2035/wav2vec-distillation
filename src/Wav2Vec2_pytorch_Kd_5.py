import os
import librosa
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Config
from torch.utils.data import Dataset, DataLoader

dataset_path = "../dataset"

audio_paths = []
transcriptions = []

# Walk through the dataset directory
for root, dirs, files in os.walk(dataset_path):
    # Look for transcription files
    for file in files:
        if file.endswith(".trans.txt"):
            transcription_file = os.path.join(root, file)
            # Read the transcription file
            with open(transcription_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    # Each line contains "file_id transcription"
                    file_id, transcription = line.strip().split(" ", 1)
                    flac_file = os.path.join(root, file_id + ".flac")
                    if os.path.exists(flac_file):
                        audio_paths.append(flac_file)
                        transcriptions.append(transcription.lower())

# Create a DataFrame
data = pd.DataFrame({
    'audio_path': audio_paths,
    'transcription': transcriptions
})

# Shuffle the DataFrame
data = data.sample(frac=1).reset_index(drop=True)

print("Total samples:", len(data))
data.head()

# Split into training and temp (validation + test)
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)

# Split temp into validation and test
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(valid_data)}")
print(f"Test samples: {len(test_data)}")


# Load pre-trained teacher model and processor
teacher_model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(teacher_model_name)
teacher_model = Wav2Vec2ForCTC.from_pretrained(teacher_model_name)
teacher_model.eval()


# Create a smaller configuration for the student model
student_config = Wav2Vec2Config(
    vocab_size=processor.tokenizer.vocab_size,
    hidden_size=256,             # Reduced from 768
    num_hidden_layers=6,         # Reduced from 12
    num_attention_heads=4,       # Reduced from 12
    intermediate_size=1024,      # Reduced from 3072
    hidden_act="gelu",
    layerdrop=0.1,
    activation_dropout=0.1,
    attention_dropout=0.1,
    feat_proj_dropout=0.1,
    hidden_dropout=0.1,
    initializer_range=0.02,
    mask_time_prob=0.05,
    mask_feature_prob=0.0,
)

# Initialize the student model
student_model = Wav2Vec2ForCTC(student_config)


def distillation_loss(student_logits, teacher_logits, labels, input_lengths, label_lengths, alpha=0.5, temperature=2.0):
    # Compute CTC loss
    ctc_loss_fn = torch.nn.CTCLoss(blank=processor.tokenizer.pad_token_id, zero_infinity=True)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    ctc_loss = ctc_loss_fn(
        student_log_probs.transpose(0, 1),  # (T, N, C)
        labels,
        input_lengths,
        label_lengths
    )

    # Compute distillation loss (KL divergence)
    T = temperature
    distill_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction='batchmean'
    ) * (T ** 2)

    # Total loss
    loss = alpha * ctc_loss + (1 - alpha) * distill_loss
    return loss


MAX_AUDIO_DURATION = 15.0  # in seconds
MAX_INPUT_LENGTH = int(MAX_AUDIO_DURATION * 16000)  # Number of samples


class LibriSpeechDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.data.iloc[idx]['audio_path']
        transcription = self.data.iloc[idx]['transcription']

        # Load audio
        speech_array, sampling_rate = librosa.load(audio_path, sr=16000)

        # Truncate audio if longer than MAX_INPUT_LENGTH
        if len(speech_array) > MAX_INPUT_LENGTH:
            speech_array = speech_array[:MAX_INPUT_LENGTH]

        # Process input values
        input_values = self.processor(
            speech_array, sampling_rate=16000, return_tensors="pt"
        ).input_values.squeeze(0)
        input_length = input_values.shape[0]

        # Encode labels (updated method)
        labels = self.processor(
            text=transcription, return_tensors="pt"
        ).input_ids.squeeze(0)
        label_length = labels.shape[0]

        assert input_length <= MAX_INPUT_LENGTH, f"Input length {input_length} exceeds maximum allowed length."

        return {
            "input_values": input_values,
            "labels": labels,
            "input_length": input_length,
            "label_length": label_length
        }

def collate_fn(batch):
    input_values = [item['input_values'] for item in batch]
    labels = [item['labels'] for item in batch]
    input_lengths = torch.tensor([item['input_length'] for item in batch], dtype=torch.long)
    label_lengths = torch.tensor([item['label_length'] for item in batch], dtype=torch.long)

    # Pad input values
    input_values_padded = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)

    # Pad labels
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=processor.tokenizer.pad_token_id
    )

    return {
        "input_values": input_values_padded,
        "labels": labels_padded,
        "input_lengths": input_lengths,
        "label_lengths": label_lengths
    }


batch_size = 4  # Adjust based on your GPU memory

# Create datasets
train_dataset = LibriSpeechDataset(train_data, processor)
valid_dataset = LibriSpeechDataset(valid_data, processor)
test_dataset = LibriSpeechDataset(test_data, processor)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Move models to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
student_model.to(device)
teacher_model.to(device)
teacher_model.eval()  # Teacher model in evaluation mode

# Optimizer
optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)


num_epochs = 3  # Adjust based on your needs
alpha = 0.5     # Weight for CTC loss vs. distillation loss
temperature = 2.0

for epoch in range(num_epochs):
    student_model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)
        input_lengths = batch["input_lengths"].to(device) # Move input_lengths to device
        label_lengths = batch["label_lengths"].to(device) # Move label_lengths to device

        # Ensure that input_lengths are not larger than the student model's input sequence length
        input_lengths = torch.clamp(input_lengths, max=student_model.wav2vec2.config.max_length)

        # Forward pass through student
        student_outputs = student_model(input_values, attention_mask=input_values.ne(0))
        student_logits = student_outputs.logits  # (batch_size, sequence_length, vocab_size)

        # Forward pass through teacher (no gradient computation)
        with torch.no_grad():
            teacher_outputs = teacher_model(input_values, attention_mask=input_values.ne(0))
            teacher_logits = teacher_outputs.logits  # (batch_size, sequence_length, vocab_size)

        # Compute loss
        loss = distillation_loss(
            student_logits, teacher_logits, labels, input_lengths, label_lengths, alpha, temperature
        )
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

student_model.eval()

# Let's test on a few samples
num_samples = 5

for i in range(num_samples):
    sample = test_dataset[i]
    input_values = sample["input_values"].unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = student_model(input_values).logits

    # Predicted token IDs
    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode the IDs to text
    transcription = processor.batch_decode(predicted_ids)[0]

    # Ground truth
    ground_truth = sample["transcription"]

    print(f"Sample {i+1}:")
    print(f"Transcription: {transcription.lower()}")
    print(f"Ground Truth: {ground_truth.lower()}")
    print("-" * 50)


