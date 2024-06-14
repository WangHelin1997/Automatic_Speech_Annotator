import os
from tqdm import tqdm
import json
import string
import pandas as pd


def clean_sentence(sentence):
    # Remove punctuation
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))
    # Replace multiple spaces with a single space
    sentence = ' '.join(sentence.split())
    return sentence


def postprocess(jsonfile, savepath, meta):
    # if os.path.exists(savepath):
    #     print(f'Already exists:{savepath}')
    # else:
    #     print(f'ChatGPT processing audio:{jsonfile}')
        with open(jsonfile, 'r') as openfile:
            spks = json.load(openfile)

        dictText={'schemaVersion': '2.0', 'monologues':[]}
        temp_speaker = None
        temp_text = None
        temp_start = None
        temp_end = None
        count = 0
        tags = [True]*len(list(meta['Line']))
        # print(spks.keys())
        for k in spks.keys():
            s = spks[k]
            if 'transcript' in s.keys() and len(s['transcript']) > 0:
                spk = s['speaker']
                if isinstance(spk, list):
                    spk = '+'.join(spk)
                if temp_speaker == None:
                    temp_speaker = spk
                    temp_text = clean_sentence(s['transcript']).lower()
                    temp_start = s['start']
                    temp_end = s['end']
                else:
                    if temp_speaker == spk:
                        temp_text += ' ' + clean_sentence(s['transcript']).lower()
                        temp_end = s['end']
                    else:
                        for idx, target_text in enumerate(list(meta['Line'])):
                            tmp_text = clean_sentence(target_text).lower()
                            if tags[idx]:
                                if tmp_text in temp_text:
                                    if list(meta['Type'])[idx] == 'self-identification':
                                        temp_text = temp_text.replace(tmp_text, '<si race=\"'+list(meta['Race'])[idx]+'\" national_origin_group=\"'+list(meta['National Origin'])[idx] + '\" ethnicity=\"'+list(meta['Ethnicity'])[idx] + '\"> '+tmp_text+" <ei> ")
                                        count += 1
                                    else:
                                        temp_text = temp_text.replace(tmp_text, '<2i race=\"'+list(meta['Race'])[idx]+'\" national_origin_group=\"'+list(meta['National Origin'])[idx] + '\" ethnicity=\"'+list(meta['Ethnicity'])[idx] + '\" SPEAKER=\"' + list(meta['Target'])[idx] + '\"> '+tmp_text+" <ei> ")
                                        count += 1
                                    # print(temp_text)
                                    tags[idx] = False

                        lTerms=[]
                        for word in temp_text.split():
                            dictWord={
                                'text': word,
                                'speaker': {'id': ''},
                                'start': float(temp_start),
                                'end': float(temp_end),
                                'type': 'WORD'
                                }
                            lTerms.append(dictWord)
                        dSentence={'speaker': {'id': temp_speaker, 'name': ''},
                            'start': float(temp_start),
                            'end': float(temp_end),
                            'terms':lTerms
                            }
                        # print(temp_text)
                        dictText['monologues'].append(dSentence)

                        temp_speaker = spk
                        temp_text = clean_sentence(s['transcript']).lower()
                        temp_start = s['start']
                        temp_end = s['end']

        if temp_text is not None:
            for idx, target_text in enumerate(list(meta['Line'])):
                tmp_text = clean_sentence(target_text).lower()
                if tags[idx]:
                    if tmp_text in temp_text:
                        if list(meta['Type'])[idx] == 'self-identification':
                            temp_text = temp_text.replace(tmp_text, '<si race=\"'+list(meta['Race'])[idx]+'\" national_origin_group=\"'+list(meta['National Origin'])[idx] + '\" ethnicity=\"'+list(meta['Ethnicity'])[idx] + '\"> '+tmp_text+"<ei> ")
                            count += 1
                        else:
                            temp_text = temp_text.replace(tmp_text, '<2i race=\"'+list(meta['Race'])[idx]+'\" national_origin_group=\"'+list(meta['National Origin'])[idx] + '\" ethnicity=\"'+list(meta['Ethnicity'])[idx] + '\" SPEAKER=\"' + list(meta['Target'])[idx] + '\"> '+tmp_text+" <ei> ")
                            count += 1
                        # print(temp_text)
                        tags[idx] = False
            lTerms=[]
            for word in temp_text.split():
                dictWord={
                    'text': word,
                    'speaker': {'id': ''},
                    'start': float(temp_start),
                    'end': float(temp_end),
                    'type': 'WORD'
                    }
                lTerms.append(dictWord)
            dSentence={'speaker': {'id': temp_speaker, 'name': ''},
                'start': float(temp_start),
                'end': float(temp_end),
                'terms':lTerms
                }
            dictText['monologues'].append(dSentence)

        print(count, len(meta['Line']))
        # print(tags)
        os.makedirs(savepath.rsplit('/',1)[0], exist_ok=True)
        with open(savepath, "w") as outfile:
            json.dump(dictText, outfile)


if __name__ == '__main__':
    # datadir = './fairness/data/japanesedata/Audios' # Audio path to process
    # savedir = './fairness/data/japanesedata/' # Output path
    # for root, dir, files in os.walk(datadir):
    #     for f in tqdm(files):
    #         if f.endswith('.wav'):
    #             main(datadir, savedir, f.split('.wav')[0])
    os.makedirs('/Users/westbrookrussell/Documents/fairness/japanesedata/PostGecko/', exist_ok=True)
    csvfile = '/Users/westbrookrussell/Documents/fairness/japanesedata/japanese_t_0.2_m_gpt-4_v0.4.1_trim.csv'
    df = pd.read_csv(csvfile)
    for root, dir, files in os.walk('/Users/westbrookrussell/Documents/fairness/japanesedata/Transcriptions'):
        for f in files:
            name = f.replace('.json','.txt')
            print(name)
            postprocess(
                os.path.join(root, f),
                os.path.join('/Users/westbrookrussell/Documents/fairness/japanesedata/PostGecko/', f),
                df.loc[df["File"] == name]
            )
