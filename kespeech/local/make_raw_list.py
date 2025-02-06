#!/usr/bin/env python3

import argparse
import json

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--segments", default=None, help="segments file")
    parser.add_argument("wav_file", help="wav file")
    parser.add_argument("text_file", help="text file")
    parser.add_argument("utt2subdialect_file", help="utt2subdialect file")
    parser.add_argument("utt2dur_file", help="utt2dur file")
    parser.add_argument("utt2spk_file", help="utt2spk file")
    parser.add_argument("spk2age_file", help="spk2age file")
    parser.add_argument("spk2gender_file", help="spk2gender file")
    parser.add_argument("output_file", help="output list file")
    args = parser.parse_args()

    # 年龄分3个桶
    spk2age_label = {}
    with open(args.spk2age_file, "r", encoding="utf8") as spk2age:
        ages = []
        for line in spk2age:
            arr = line.strip().split(maxsplit=1)
            key = arr[0]
            age = int(arr[1]) if len(arr) > 1 else 0
            ages.append(age)
        age_bins = np.percentile(np.sort(ages), [33, 66])
        spk2age.seek(0)
        for line in spk2age:
            arr = line.strip().split(maxsplit=1)
            key = arr[0]
            age = int(arr[1]) if len(arr) > 1 else 0
            age_label = np.digitize(age, age_bins).tolist()
            spk2age_label[key] = age_label
    # print(spk2age_label)

    spk2gender_label = {}
    with open(args.spk2gender_file, "r", encoding="utf8") as spk2gender:
        for spk2gender_line in spk2gender:
            arr = spk2gender_line.strip().split(maxsplit=1)
            key = arr[0]
            gender = arr[1] if len(arr) > 1 else ""
            gender = 1 if gender == "Male" else 0
            spk2gender_label[key] = gender
    # print(spk2gender_label)

    # 语速分5个桶
    utt2speed_label = {}
    with open(args.utt2dur_file, "r", encoding="utf8") as utt2dur, open(
        args.text_file, "r", encoding="utf8"
    ) as text:
        speeds = []
        for utt2dur_line, text_line in zip(utt2dur, text):
            arr = utt2dur_line.strip().split(maxsplit=1)
            key = arr[0]
            dur = float(arr[1]) if len(arr) > 1 else 0.0
            arr2 = text_line.strip().split(maxsplit=1)
            key2 = arr2[0]
            txt = arr2[1] if len(arr2) > 1 else ""
            assert key == key2
            speed = len(txt) / dur  # tokens/second
            speeds.append(speed)
        speed_bins = np.percentile(np.sort(speeds), [20, 40, 60, 80])
        utt2dur.seek(0)
        text.seek(0)
        for utt2dur_line, text_line in zip(utt2dur, text):
            arr = utt2dur_line.strip().split(maxsplit=1)
            key = arr[0]
            dur = float(arr[1]) if len(arr) > 1 else 0.0
            arr2 = text_line.strip().split(maxsplit=1)
            key2 = arr2[0]
            txt = arr2[1] if len(arr2) > 1 else ""
            assert key == key2
            speed = len(txt) / dur  # tokens/second
            speed_label = np.digitize(speed, speed_bins).tolist()
            utt2speed_label[key] = speed_label
    # print(utt2speed_label)

    utt2expression_habit = {}
    expression_habits = []
    with open(args.utt2spk_file, "r", encoding="utf8") as utt2spk:
        for utt2spk_line in utt2spk:
            arr = utt2spk_line.strip().split(maxsplit=1)
            key = arr[0]
            spk = arr[1] if len(arr) > 1 else ""
            age_label = spk2age_label[spk]
            gender_label = spk2gender_label[spk]
            speed_label = utt2speed_label[key]
            # 笛卡尔积编码
            expression_habit = age_label * (2 * 5) + gender_label * 5 + speed_label
            expression_habits.append(expression_habit)
            utt2expression_habit[key] = expression_habit
    # print(utt2expression_habit)

    # 计算权重
    # hits, bins = np.histogram(expression_habits, bins=range(31))
    # hits = hits.tolist()
    # weights = [1.0 / hits[idx] for idx in range(len(hits))]

wav_table = {}
with open(args.wav_file, "r", encoding="utf8") as fin:
    for line in fin:
        arr = line.strip().split()
        assert len(arr) == 2
        wav_table[arr[0]] = arr[1]

if args.segments is not None:
    segments_table = {}
    with open(args.segments, "r", encoding="utf8") as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 4
            segments_table[arr[0]] = (arr[1], float(arr[2]), float(arr[3]))

with open(args.text_file, "r", encoding="utf8") as fin, open(
    args.utt2subdialect_file, "r", encoding="utf8"
) as fdialect, open(args.output_file, "w", encoding="utf8") as fout:
    for line, dialect in zip(fin, fdialect):
        arr = line.strip().split(maxsplit=1)
        key = arr[0]
        txt = arr[1] if len(arr) > 1 else ""
        arr2 = dialect.strip().split(maxsplit=1)
        key2 = arr2[0]
        subdialect = arr2[1] if len(arr2) > 1 else ""
        assert key == key2
        expression_habit = utt2expression_habit[key]
        if args.segments is None:
            assert key in wav_table
            wav = wav_table[key]
            line = dict(
                key=key,
                wav=wav,
                txt=txt,
                subdialect=subdialect,
                expression_habit=expression_habit,
            )
        else:
            assert key in segments_table
            wav_key, start, end = segments_table[key]
            wav = wav_table[wav_key]
            line = dict(
                key=key,
                wav=wav,
                txt=txt,
                subdialect=subdialect,
                expression_habit=expression_habit,
                start=start,
                end=end,
            )
        json_line = json.dumps(line, ensure_ascii=False)
        fout.write(json_line + "\n")
