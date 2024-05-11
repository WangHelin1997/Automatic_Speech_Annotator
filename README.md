# Automatic_Speech_Annotator
Automatic speech annotator processing speech with voice activaty detection, overlapping speech detection, speaker diarization and automatic speech recognition

## Pipeline
![Pipeline](https://github.com/WangHelin1997/Automatic_Speech_Annotator/blob/main/demo.png)

## Processing

1. Voice activity detection: We removed all the non-speech parts, e.g., silence, music, etc., with a voice activity detection (VAD) model.
2. Overlapping speech detection: Overlapping speech can degrade the performance of speaker diarization and speech recognition. To combat that, we used an overlap detection model to detect and remove the parts where more than one speaker talked simultaneously.
3. Speaker diarization: The clean data then served as input for an advanced speaker diarization system to obtain the timestamps, while noting which speaker was speaking at each time-step. After diarization, each resulting segment was tagged with a speaker identifier, e.g. Speaker_00.
4. Automatic speech recognition: We used a medium-size whisper model to transcribe these audio files.

## Citation

If you find this repo helpful, feel free to cite this paper:

[1] Finding Spoken Identifications: Using GPT-4 Annotation For An Efficient And Fast Dataset Creation Pipeline," Jahan, M., Wang, H. Thebaud, T., Sun, Y., Le, G., Fagyal, Z., Scharenborg, O., Hasegawa-Johnson, M., Moro-Velazquez, L., Dehak, N. International LREC-Coling Joint Conference, Torino, Italy, May 20-24, 2024.
