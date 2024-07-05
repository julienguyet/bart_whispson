# BART WHISPSON :robot:

---

## 1. Data Processing :studio_microphone:

Data is made of audio files, each corresponding to a human reading some text. Audios' length varies and can go up to 15 minutes. We have upload two samples for you to replicate locally and full dataset is available [here](https://www.kaggle.com/datasets/nfedorov/audio-summarization).

For each file, we were provided with a human transcript and summary. This is our ground truth and we will compare models transcripts/summaries to those.

To process our audio we used Whisper, a [Speech to Text model](https://platform.openai.com/docs/guides/speech-to-text) from OpenAI. To not have to rely on OpenAI API, we installed Whisper locally with [PIP](https://pypi.org/project/openai-whisper/).
Below is how you can do it:

```python
model = whisper.load_model("base")
result = model.transcribe('../data/audio/000001.mp3')
print(f' The text in audio: \n {result["text"]}')
```

Which would output the below (we truncated as audio are very long):
```
The text in audio: 
  The Castles of Athlean and Dunbane by Anne Radcliffe. Chapter 11 Allen was nowhere to be found. The Earl went himself in quest of him, but without success. As he returned from the terrace, she grinned and disappointed. He observed two persons cross the platform at some distance before him, and he could perceive by the dim moonlight which fell upon the spot that they were not of the castle, [...] them.
```

Now, to ensure we would have accurate transcripts only, we measured the cosine similarity between Whisper's outputs and the transcripts from the dataset. We decide to keep only outputs with a score of at least 0.70.

---

## 2. Model Fine-Tuning :telescope:

As a first try, we fine-tuned Flan-T5. More details on this are availble in the T5_fine_tuning notebook and below are the main steps.

First, we download the model and associated tokenizer:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name='google/flan-t5-base'

T5 = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

Then after defining some functions to process our data we create train and test datasets:

```python
tokenized_dataset = dataset.map(preprocess_function, batched=True)
split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
```

This will allow us to have prompts with correct summaries for training:

```
Training Prompt Example:
Input Text:
Summarize the following conversation. ### Input: ON the north-east coast of Scotland, in the most romantic part of the Highlands, [..], with the touching

Summary Text:
 The history of the family of the Earl of Athlin and the ongoing feud with Malcolm is introduced. 
```

Finally, we define a LoRA configuration and the training arguments, and we can launch training:

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    num_train_epochs=100,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=10_000,
    eval_steps=10_000,
    logging_dir='./logs',
    logging_steps=200,
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

peft_trainer.train()
```

Below is  an output example:

```
------------------------------------------------------------
HUMAN SUMMARY:
The history of the family of the Earl of Athlin and the ongoing feud with Malcolm is introduced.      
------------------------------------------------------------
ORIGINAL MODEL SUMMARY:
a strong desire to be a hero.
------------------------------------------------------------
FINE-TUNED MODEL SUMMARY:
A story of the life of the noble Earl of Athlin.
```

---

## 3. BART for Text Summarization :satellite:

Following our results with Flan-T5, we decided to use an LLM specialized in text summarization. We went for [BART] (https://huggingface.co/facebook/bart-large-cnn) and downloaded the large version, trained on CNN articles.

We linked it to Whisper, and build a python script for audio processing, transcription and summarization. 

You can call the model by typing in your terminal:

```bash
python bart_whispson.py <audio_file_path> <directory_to_save_results>
```

Below are some details on how it works behind the scene:

1. We import the models:

```python
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
```

2. We use a function to first call whisper to process the audio.
3. Then Whisper output becomes the input for BART.
4. We use the tokenizer to build a prompt to ask the model to summarize text.
5. We save transcript and generated summary in user's folder of choice for easy retrieving later. 

```python
def process_audio_and_summarize(audio_path, save_directory):

    transcript = get_audio_transcript(audio_path)
    
    inputs = tokenize_transcript(transcript)
    
    generated_ids = generate_summary(inputs)
    generated_summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    transcript_filename = os.path.basename(audio_path) + "_transcript.txt"
    transcript_filepath = os.path.join(save_directory, transcript_filename)
    
    with open(transcript_filepath, 'w', encoding='utf-8') as f:
        f.write(transcript)
    
    summary_filename = os.path.basename(audio_path) + "_summary.txt"
    summary_filepath = os.path.join(save_directory, summary_filename)
    
    with open(summary_filepath, 'w', encoding='utf-8') as f:
        f.write(generated_summary)
    
    return generated_summary
```
