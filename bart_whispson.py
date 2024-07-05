import os
import sys
from transformers import BartForConditionalGeneration, BartTokenizer
import whisper


tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

def get_audio_transcript(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    transcript = result["text"]
    return transcript

def tokenize_transcript(transcript):
    inputs = tokenizer("Summarize the following conversation.\n\n### Input:\n" + transcript + "\n\n### Summary:\n", return_tensors="pt", max_length=1024, truncation=True)
    return inputs

def generate_summary(inputs):
    generated_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
    return generated_ids

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

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python audio_summarizer.py <audio_path> <save_directory>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    save_directory = sys.argv[2]
    
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    generated_summary = process_audio_and_summarize(audio_path, save_directory)
    
    print("Generated Summary:\n", generated_summary)
