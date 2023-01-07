## Neural speaker diarization with pyannote.audio

Hourmazd Delvarianzadeh 40114140111031

University: Islamic Azad University, South Tehran branch

If I want to explain my GitHub project to you in simple language, I can say that:
Instead of changing the voice,
I write the song that is in the speech. And finally, I check the note in the speech song.
Actually pyannote.audio is an open-source toolkit written in Python for speaker diarization.
Based on PyTorch machine learning framework
it provides a set of trainable end-to-end neural building blocks 
that can be combined and jointly optimized to build speaker diarization pipelines.

##

## target:

Its purpose is to witness the process of partitioning an input audio stream into homogeneous parts in the output using the command.


Works that we did in this project : 
So far, Pyannote.audio has been updated 9 times,
but in the version 2 update, we see a complete rewrite in the program.
These changes include fundamental changes such as:

1.much better performance.Also you can see the Benchmark in the below:

## Benchmark

Out of the box, `pyannote.audio` default speaker diarization [pipeline] is expected to be much better (and faster) in v2.x than in v1.1. Those numbers are diarization error rates (in %)

| Dataset \ Version      | v1.1 | v2.0 | v2.1.1 (finetuned) |
| ---------------------- | ---- | ---- | ------------------ |
| AISHELL-4              | -    | 14.6 | 14.1 (14.5)        |
| AliMeeting (channel 1) | -    | -    | 27.4 (23.8)        |
| AMI (IHM)              | 29.7 | 18.2 | 18.9 (18.5)        |
| AMI (SDM)              | -    | 29.0 | 27.1 (22.2)        |
| CALLHOME (part2)       | -    | 30.2 | 32.4 (29.3)        |
| DIHARD 3 (full)        | 29.2 | 21.0 | 26.9 (21.9)        |
| VoxConverse (v0.3)     | 21.5 | 12.6 | 11.2 (10.7)        |
| REPERE (phase2)        | -    | 12.6 | 8.2 ( 8.3)         |
| This American Life     | -    | -    | 20.8 (15.2)        |



2.Python-first API


3.pretrained pipelines (and models) on model hub


4.multi-GPU training with pytorch-lightning


5.data augmentation with torch-audiomentations


6.Prodigy recipes for model-assisted audio annotation

My main source code:

https://github.com/pyannote/pyannote-audio

My main account in Github:

https://github.com/hourmazd98

My Linkedin account:

https://www.linkedin.com/in/hourmazd-delvarianzadeh-321187212

##

Files and articles related to the project are available in the links below:

https://drive.google.com/file/d/17lXIos9nXAe1hQse03x2SlPjaLLBW-1e/view?usp=sharing

https://drive.google.com/file/d/1AGN8UOkTIXeJYbfj4BfyhhomFvUWkJrq/view?usp=share_link

https://drive.google.com/file/d/1Q-rds9ltupNLZSCH7paTe1o_pVltUJjL/view?usp=share_link

https://drive.google.com/file/d/1PgptNDJOv8PIWNdADe34VoLrQQCkOspn/view?usp=share_link

https://drive.google.com/file/d/1SDRf46CXbJkUTseejn3QycOHLtq7Mr8Y/view?usp=share_link

https://drive.google.com/file/d/1Q6nuoDJDkS-xi1ExJupz_Z-AmbHQeFNF/view?usp=share_link

https://drive.google.com/file/d/1HoqOJCc2g0J01Tjjn7SG4hPY3k-htCFP/view?usp=share_link

https://drive.google.com/file/d/1NpBaq3Kp6Gso__qlEVQPG8vcw6oxvdgx/view?usp=share_link

https://drive.google.com/file/d/1bO0twBeo170RcJQTSmHhWhBcbxCI9gOS/view?usp=share_link

https://drive.google.com/file/d/1c_r-LqQ_QWzWASqExz9vGEoY-2GLtYdd/view?usp=share_link

https://drive.google.com/file/d/1c_r-LqQ_QWzWASqExz9vGEoY-2GLtYdd/view?usp=share_link

https://drive.google.com/file/d/1MAXsunyYVp3qx9WIFEXQ5GMJpapQiuX-/view?usp=share_link

https://drive.google.com/file/d/1rpagxfD2USWJ80RI8EQUGblQHL65QP5X/view?usp=share_link

https://drive.google.com/file/d/1yAeSRm3HQsnmpUY3LgBgubHxcO6DM7jx/view?usp=share_link

https://drive.google.com/file/d/1_o2o5XHaZSC5BfSx-UtLytqghPZFXMni/view?usp=share_link

https://drive.google.com/file/d/1Ep4Z1TBaXMsGFnMooQBu0GhQJG4iEkNq/view?usp=share_link

https://drive.google.com/file/d/1okbdsViYPbYuTKYbf7E79GxkmQMBG60a/view?usp=share_link

https://drive.google.com/file/d/1kaI3keD5iXqg6nkO6U244UsAQtr4YZks/view?usp=share_link

https://drive.google.com/file/d/12xh7pkV7uQ-H-rxbQ35Wkn7hnOGcuby9/view?usp=share_link

https://drive.google.com/file/d/12ONcRfdVd87gTj-L-2sGWqvtUHopDdzB/view?usp=share_link

https://drive.google.com/file/d/1kXwfqPP1BOM7qRKCX8OgUgaYsYM401KT/view?usp=share_link

https://drive.google.com/file/d/1mEpMz-zDaWu7FqZrq-E-1WNZ69BEtrNn/view?usp=share_link

https://drive.google.com/file/d/1pfhJFVISWDFg340QlE4Ncl7a35-8L8Dv/view?usp=share_link

https://drive.google.com/file/d/18gF38wtXw0r8tA8afllJwla4fJRBKa7e/view?usp=share_link

https://drive.google.com/file/d/1mXOjf_n2bFCaNpFVEgTQx6z8v9DC3g-0/view?usp=share_link

https://drive.google.com/file/d/11f1a3uEAgN7bgClN1QD0ZX9S7vKP5P6E/view?usp=share_link

https://drive.google.com/file/d/1pz-ty3tlXRmbly7dPh98nlJ7NOVt9qa-/view?usp=share_link

https://drive.google.com/file/d/19OYbs7sqi0KwZUUS3fet6tc1__6tiGHZ/view?usp=share_link

https://drive.google.com/file/d/13si6JtIhmCYNuVbjhN7VRaIwLe1sBl_r/view?usp=share_link

https://drive.google.com/file/d/1bFy0J3iSyh2F4_c5MXH63BaRtuA5tP_g/view?usp=share_link

https://drive.google.com/file/d/1YUsVga37unEbOtPK6A3TaF97AzrP5Iyl/view?usp=share_link

https://drive.google.com/file/d/1UHpdIODo86zCn2zR-ZogWsKjDOPWg46h/view?usp=share_link

https://drive.google.com/file/d/1xXILeS9gNjPNoEnBkHUCHTcAq8y9VBPN/view?usp=share_link

https://drive.google.com/file/d/1x1rw9_a6gS4OqEjFNOA5g37VzlUw-ssA/view?usp=share_link

https://drive.google.com/file/d/1gAllviqAtFMUmZp82LDztAqPiAN9C67B/view?usp=share_link
