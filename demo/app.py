import base64
import io

import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.backends.backend_agg import RendererAgg

from pyannote.audio.core.inference import Inference
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core import notebook

st.markdown(
    """
# Voice activity detection
"""
)

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is None:
    st.stop()


vad = Inference(
    "hbredin/VoiceActivityDetection-PyanNet-DIHARD",
    batch_size=8,
    device="cpu",
)

progress_bar = st.empty()


def progress_hook(chunk_idx, num_chunks):
    progress_bar.progress(chunk_idx / num_chunks)


vad.progress_hook = progress_hook
scores = vad({"audio": uploaded_file})

progress_bar.empty()

with RendererAgg.lock:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figwidth(12)
    fig.set_figheight(2.0)
    notebook.plot_feature(scores, ax=ax, time=True)
    ax.set_ylim(-0.1, 1.1)
    plt.tight_layout()
    st.pyplot(fig=fig, clear_figure=True)

pipeline = VoiceActivityDetection(scores="vad").instantiate(
    dict(min_duration_off=0.055, min_duration_on=0.071, offset=0.326, onset=0.833)
)

uri = uploaded_file.name
speech_regions = pipeline({"vad": scores, "uri": uri})

with RendererAgg.lock:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figwidth(12)
    fig.set_figheight(2.0)
    notebook.plot_timeline(speech_regions.get_timeline(), ax=ax, time=True)
    plt.tight_layout()
    st.pyplot(fig=fig, clear_figure=True)


with io.StringIO() as fp:
    speech_regions.write_rttm(fp)
    content = fp.getvalue()

b64 = base64.b64encode(content.encode()).decode()
href = f'<a download="vad.rttm" href="data:file/text;base64,{b64}">Download as RTTM</a>'
st.markdown(href, unsafe_allow_html=True)
