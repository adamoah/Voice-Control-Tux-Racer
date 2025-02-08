import torch
import speech_recognition as sr
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import io
from pydub import AudioSegment
from pynput.keyboard import Key, Controller
import warnings
import time
warnings.filterwarnings('ignore')

# import wav2vec from hugging face
model_n = 'facebook/wav2vec2-large-960h'
processor = Wav2Vec2Processor.from_pretrained(model_n)
model = Wav2Vec2ForCTC.from_pretrained(model_n)

# define movement keywords
LEFT, RIGHT, UP, DOWN, STOP, QUIT = 'MOVE LEFT', 'MOVE RIGHT', 'MOVE UP', 'MOVE DOWN', 'STOP', 'QUIT'

# audio and input control objects
r = sr.Recognizer()
quit = False
controller = Controller()
selected = None

# function to return keyboard input based on keyword detected
def select_key(output, prev):

    if LEFT in output:
        print('You Said: MOVE LEFT')
        return Key.left
    elif RIGHT in output:
        print('You Said: MOVE RIGHT')
        return Key.right
    elif UP in output:
        print('You Said: MOVE UP')
        return Key.up
    elif DOWN in output:
        print('You Said: MOVE DOWN')
        return Key.down
    elif STOP in output: # input to stop pressing current key
        print('You Said: STOP')
        return Key.space
    elif QUIT in output:
        print('You Said: QUIT')
        return Key.esc
    else:
        return prev

with sr.Microphone(sample_rate=16000) as source:
    print('recording audio from microphone')
    
    while not quit:
        audio = r.listen(source) # pyaudio object
        data = io.BytesIO(audio.get_wav_data()) # bytes array
        clip = AudioSegment.from_wav(data) # numpy array
        x    = torch.FloatTensor(clip.get_array_of_samples()) # tensor
        
        # pass data into the model and get outputs
        inputs = processor(x, sampling_rate=16000, return_tensors='pt', padding='longest').input_values
        logits = model(inputs).logits
        tokens = torch.argmax(logits, axis=-1)
        text = processor.batch_decode(tokens)[0]
        print(text)
        
        # determine key to select
        selected = select_key(text, selected)
        
        # initiate keybaord press or release
        if selected == Key.esc:
            quit = True
        elif selected == Key.space:
            controller.release(selected)
        elif selected is not None:
            controller.press(selected)
    
    print('Exiting...')
