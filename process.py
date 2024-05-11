from pyannote.audio import Pipeline
import os
from pyannote.audio.pipelines import OverlappedSpeechDetection,VoiceActivityDetection
from pyannote.audio import Model
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
from pyannote.audio import Pipeline
import random
import json
import whisper

def VadOd_process(vadpipeline, odpipeline, audiopath, savepath):
    if os.path.exists(savepath):
        print(f'Already exists:{savepath}')
    else:
        print(f'VAD processing audio:{audiopath}')
        output = vadpipeline(audiopath)
        removes = []
        start = 0.
        end = 0.
        for i, speech in enumerate(output.get_timeline().support()):
            if i == 0:
                start = speech.start
                end = speech.end
            else:
                if speech.start - end > 1.:
                    removes.append([end,speech.start])
            start = speech.start
            end = speech.end

        print(f'OD processing audio:{audiopath}')
        output = odpipeline(audiopath)
        for speech in output.get_timeline().support():
            removes.append([speech.start, speech.end])

        print(f'Write audio:{audiopath}')
        (audio, sr) = librosa.load(audiopath, sr=16000)
        for r in removes:
            start = int(r[0]*sr)
            end = int(r[1]*sr)
            audio[start:end]=np.zeros(end-start)
        os.makedirs(savepath.rsplit('/',1)[0], exist_ok=True)
        sf.write(savepath, audio, sr)

def Sd_process(sdpipeline, filepath, savepath):
    if os.path.exists(savepath):
        print(f'Already exists:{savepath}')
    else:
        print(f'SD processing audio:{filepath}')
        diarization = sdpipeline(filepath)
        os.makedirs(savepath.rsplit('/',1)[0], exist_ok=True)
        text = ''
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            text += str(turn.start)+'\t'+str(turn.end)+'\t'+str(speaker)+'\n'
        with open(savepath, "w") as outfile:
            outfile.write(text)

def resolve_overlaps(time_dicts):
    sorted_dicts = sorted(time_dicts, key=lambda x: x['start'])
    result = []
    for current in sorted_dicts:
        overlap = False
        for i, existing in enumerate(result):
            if (current['start'] < existing['end']) and (current['end'] > existing['start']):
                overlap = True
                new_start = max(current['start'], existing['start'])
                new_end = min(current['end'], existing['end'])
                # Create the overlapping segment
                new_segment = {
                    'start': new_start,
                    'end': new_end,
                    'speaker': list(set(existing['speaker'] if isinstance(existing['speaker'], list) else [existing['speaker']] + [current['speaker']]))
                }
                # Update the existing segment
                if existing['end'] > new_end:
                    result.append({'start': new_end, 'end': existing['end'], 'speaker': existing['speaker']})
                existing['end'] = new_start
                # Update the current segment
                current['start'] = new_end
                result.append(new_segment)
                # Remove the segment if it has zero length
                if existing['start'] == existing['end']:
                    result.pop(i)
        if not overlap or (current['start'] != current['end']):
            result.append(current)
    return sorted(result, key=lambda x: x['start'])

def asr_process(asrmodel, txtfile, audiopath, tmp, savepath):
    if os.path.exists(savepath):
        print(f'Already exists:{savepath}')
    else:
        print(f'ASR processing audio:{audiopath}')
        with open(txtfile, 'r') as fi:
            lines = fi.readlines()
        spks = []
        for line in lines:
            line = line.split('\n')[0]
            spks.append({'start':float(line.split('\t')[0]),'end':float(line.split('\t')[1]),'speaker':line.split('\t')[2]})

        time_dicts = spks
        resolved = resolve_overlaps(time_dicts)
        results = {}
        os.makedirs(tmp.rsplit('/',1)[0], exist_ok=True)
        for i in tqdm(range(len(resolved))):
            s = resolved[i]
            x, sr = librosa.load(audiopath, offset=s['start'],duration=s['end']-s['start'])
            sf.write(tmp, x, sr)
            asr_result = asrmodel.transcribe(tmp)
            os.remove(tmp)
            s['transcript'] = asr_result['text']
            results[str(i)] = s
        os.makedirs(savepath.rsplit('/',1)[0], exist_ok=True)
        with open(savepath, "w") as outfile:
            json.dump(results, outfile)

def gecko(jsonfile, savepath):
    if os.path.exists(savepath):
        print(f'Already exists:{savepath}')
    else:
        print(f'Gecko processing audio:{jsonfile}')
        with open(jsonfile, 'r') as openfile:
            spks = json.load(openfile)
        dictText={'schemaVersion': '2.0', 'monologues':[]}
        for k in spks.keys():
            lTerms=[]
            s = spks[k]
            for word in s['transcript'].split():
                dictWord={'text': word,
                            'speaker': {'id': ''},
                            'start': float(s['start']),
                            'end': float(s['end']),
                            'type': 'WORD'}
                lTerms.append(dictWord)
            if type(s['speaker']) is list:
                spkid = '+'.join(map(str, s['speaker']))
            else:
                spkid = s['speaker']
            dSentence={'speaker': {'id': spkid, 'name': ''},
                        'start': float(s['start']),
                        'end': float(s['end']),
                        'terms':lTerms}
            dictText['monologues'].append(dSentence)
        os.makedirs(savepath.rsplit('/',1)[0], exist_ok=True)
        with open(savepath, "w") as outfile:
            json.dump(dictText, outfile)

def chatgpt(jsonfile, savepath):
    if os.path.exists(savepath):
        print(f'Already exists:{savepath}')
    else:
        print(f'ChatGPT processing audio:{jsonfile}')
        with open(jsonfile, 'r') as openfile:
            spks = json.load(openfile)
        
        temp_speaker = None
        temp_text = None
        alltxt = ''
        for k in spks.keys():
            s = spks[k]
            if 'transcript' in s.keys() and len(s['transcript']) > 0:
                if temp_speaker == None:
                    temp_speaker = s['speaker']
                    temp_text = s['transcript']
                else:
                    if s['speaker'] == temp_speaker:
                        temp_text += ' ' + s['transcript']
                    else:
                        if isinstance(temp_speaker, list):
                            temp_speaker = '+'.join(temp_speaker)
                        alltxt += temp_speaker + ': ' + temp_text + '\n'
                        temp_speaker = s['speaker']
                        temp_text = s['transcript']
        alltxt += temp_speaker + ': ' + temp_text + '\n'
        os.makedirs(savepath.rsplit('/',1)[0], exist_ok=True)
        with open(savepath, "w") as outfile:
            outfile.write(alltxt)

def main(datadir, savedir, filename):
    audiopath = os.path.join(datadir, filename+'.wav')
    saveaudiopath = os.path.join(savedir, 'VADAudios', filename+'.wav')
    savesdpath = os.path.join(savedir, 'SpeakerDiarization', filename+'.txt')
    saveasrpath = os.path.join(savedir, 'Transcriptions', filename+'.json')
    tmpfile = os.path.join(savedir, 'tmp', filename+'.wav')
    savegeckopath = os.path.join(savedir, 'Gecko', filename+'.json')
    savechatgptpath = os.path.join(savedir, 'ChatGPT', filename+'.txt')

    vadpipeline = VoiceActivityDetection("pyannote/segmentation",
                                        use_auth_token="") # Fill in with your huggingface token
    odpipeline = OverlappedSpeechDetection("pyannote/segmentation", 
                                use_auth_token="") # Fill in with your huggingface token
    sdpipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                            use_auth_token="") # Fill in with your huggingface token
    asrmodel = whisper.load_model("medium.en")
    VadOd_process(vadpipeline, odpipeline, audiopath, saveaudiopath)
    Sd_process(sdpipeline, saveaudiopath, savesdpath)
    asr_process(asrmodel, savesdpath, saveaudiopath, tmpfile, saveasrpath)
    gecko(saveasrpath, savegeckopath)
    chatgpt(saveasrpath, savechatgptpath)

if __name__ == '__main__':
    datadir = './fairness/data/japanesedata/Audios' # Audio path to process
    savedir = './fairness/data/japanesedata/' # Save path
    for root, dir, files in os.walk(datadir):
        for f in tqdm(files):
            if f.endswith('.wav'):
                main(datadir, savedir, f.split('.wav')[0])
