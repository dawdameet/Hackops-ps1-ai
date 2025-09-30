from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import librosa
import torch

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "r-f/wav2vec-english-speech-emotion-recognition"
)
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "r-f/wav2vec-english-speech-emotion-recognition"
)

def predict_emotion(audio_path):
    audio, rate = librosa.load(audio_path, sr=16000)
    inputs = feature_extractor(audio, sampling_rate=rate, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_label = torch.argmax(predictions, dim=-1).item()
        emotion = model.config.id2label[predicted_label]
    return emotion

emotion = predict_emotion("sample_audio.wav")
print(f"Predicted emotion: {emotion}")
