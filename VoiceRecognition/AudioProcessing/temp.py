import pyaudio
import wave

wf = wave.open("./working_data/abbott_10.wav")

p = pyaudio.PyAudio()
stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),  
                channels = wf.getnchannels(),  
                rate = wf.getframerate(),  
                output = True)

CHUNK = 1024

data = wf.readframes(CHUNK)

while len(data) > 0:
    stream.write(data)
    data = wf.readframes(CHUNK)

stream.stop_stream()
stream.close()
p.terminate()
wf.close()