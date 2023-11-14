import torchaudio
def load_audio(PATH):
    audio, _ = torchaudio.load(PATH)
    return audio
