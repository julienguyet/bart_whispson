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
  The Castles of Athlean and Dunbane by Anne Radcliffe. Chapter 11 Allen was nowhere to be found. The Earl went himself in quest of him, but without success. As he returned from the terrace, she grinned and disappointed. He observed two persons cross the platform at some distance before him, and he could perceive by the dim moonlight which fell upon the spot that they were not of the castle. [...] them.
```

Now, to ensure we would have accurate transcripts only, we measured the cosine similarity between Whisper's outputs and the transcripts from the dataset. We decide to keep only outputs with a score of at least 0.70.
---
