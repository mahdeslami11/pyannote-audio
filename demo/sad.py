import streamlit as st
import torch
from pyannote.audio.features import RawAudio
from pyannote.audio.utils.signal import Binarize
from matplotlib import pyplot as plt
from pyannote.core import notebook
from pyannote.core import SlidingWindowFeature
import numpy as np
import io
import base64

notebook.reset()

st.markdown('''
# Voice activity detection with pyannote.audio
''')

# helper function to make visualization prettier
plot_ready = lambda scores: SlidingWindowFeature(np.exp(scores.data[:, 1:]),
                                                 scores.sliding_window)

def always_speech():
    while True:
        yield 'speech'

@st.cache(allow_output_mutation=True)
def load_model(name):
    with st.spinner(text='Downloading pretrained model...'):
        model = torch.hub.load('pyannote/pyannote-audio:develop', name)
    return model
sad = load_model('sad_dihardx')

sample_rate = sad.sample_rate
raw_audio = RawAudio(sample_rate=sad.sample_rate)
pipeline_params = sad.pipeline_params_
binarize = Binarize(log_scale=True, **pipeline_params)

audio = st.file_uploader("")
if audio is not None:

    current_file = {'audio': audio}
    waveform_swf = raw_audio(current_file)
    current_file['waveform'] = waveform_swf.data

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figwidth(12)
    fig.set_figheight(2.)
    notebook.plot_feature(waveform_swf, ax=ax, time=True)
    plt.tight_layout()
    st.pyplot(fig=fig, clear_figure=True)

    with st.spinner(text='Computing raw speech scores...'):
        sad_scores = sad(current_file)

    st.markdown('## Raw speech scores')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figwidth(12)
    fig.set_figheight(2.)
    notebook.plot_feature(plot_ready(sad_scores), ax=ax, time=True)
    ax.set_ylim(-0.1, 1.1)
    plt.tight_layout()
    st.pyplot(fig=fig, clear_figure=True)

    with st.spinner(text='Computing final speech regions...'):
        speech = binarize.apply(sad_scores, dimension=1)

    st.markdown('## Final speech regions')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figwidth(12)
    fig.set_figheight(2.)
    notebook.plot_timeline(speech, ax=ax, time=True)
    plt.tight_layout()
    st.pyplot(fig=fig, clear_figure=True)

    # TODO get filename from st.file_uploader when it is updated to do so
    # see https://github.com/streamlit/streamlit/issues/896
    uri = 'pyannote'
    speech.uri = uri

    with io.StringIO() as fp:
        speech.to_annotation(generator=always_speech()).write_rttm(fp)
        content = fp.getvalue()

    b64 = base64.b64encode(content.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a download="{uri}.rttm" href="data:file/text;base64,{b64}">Download as RTTM</a>'
    st.markdown(href, unsafe_allow_html=True)

