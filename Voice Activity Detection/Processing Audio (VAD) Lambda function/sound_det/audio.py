import sys 
sys.path.append('/mnt/acess')
import webrtcvad
import collections
import contextlib
import sys
import wave
import numpy as np 
from pydub import AudioSegment
vad = webrtcvad.Vad(3)




def VAD_detection(file_path , threshold ) : 
    sound = AudioSegment.from_wav(file_path)
    sound = sound.set_frame_rate(32000)
    sound = sound.set_channels(1)
    sound = sound.set_sample_width(2)
    sound.export(file_path, format="wav")
    audio, sample_rate = read_wave(file_path)
    frames = frame_generator(10, audio, sample_rate)
    frames = list(frames)
    is_speech=[]
    for frame in frames:
          is_speech.append(vad.is_speech(frame.bytes, sample_rate))
    is_speech_np = np.array(is_speech)
    return is_speech,  np.sum(is_speech) , (is_speech_np.sum()/len(is_speech_np)>threshold)
  


def read_wave(path):

    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate



class Frame(object):

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):

    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

