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
LEFT, RIGHT, UP, DOWN, STOP, QUIT = 'LEFT', 'RIGHT', 'UP', 'DOWN', 'STOP', 'QUIT'

# audio and input control objects
r = sr.Recognizer()
quit = False
controller = Controller()
selected = []

# function to return keyboard input based on keyword detected
def select_key(output, prev):

    if LEFT in output:
        print('You Said: LEFT')
        return Key.left
    elif RIGHT in output:
        print('You Said: RIGHT')
        return Key.right
    elif UP in output:
        print('You Said: UP')
        return Key.up
    elif DOWN in output:
        print('You Said: DOWN')
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
        
        previous = selected[-2] if len(selected) > 1 else None
        # determine key to select
        selected.append(select_key(text, previous))
        
        # initiate keybaord press or release
        if selected[-1] == Key.esc:
            quit = True
        elif selected[-1] == Key.space:

            for input in selected: # release all previously store inputs
                if input is not None:
                    controller.release(input)
            selected = [] # set array to empty

        elif selected[-1] is not previous: # initiate press only if new key has been selected
            controller.press(selected[-1])
        
        print(selected)
    
    print('Exiting...')
