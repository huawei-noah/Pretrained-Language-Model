# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE

import os
import json
import re
import pandas as pd
from typing import Union
from pathlib import Path
from collections import Counter

import config
from prepare_for_parse import create_or_clean

PAR_THRESHOLD = 10

SCENE_CHANGE = [' INT ', ' EXT ', ' INT.', ' INT:', ' EXT:', ' EXT.', ' INT/EXT', ' EXT/INT', 'FADE IN:', 'EXTERIOR',
                'INTERIOR', 'CUT FROM', 'CUT TO', 'TO:', 'INTERCUTS', 'I/E.', ' TITLE', ' TTTLE', 'FADE',
                'TRANSITION TO', 'INTERCUT', 'SCENE', 'SFX OVER', 'VFX:', 'DISSOLVE', 'PART I']

SCENE_CHANGE_NOT_UPPER = [' INT ', ' EXT ', ' INT.', ' INT:', ' EXT:', ' EXT.', ' INT/EXT', ' EXT/INT', 'FADE IN:',
                          'EXTERIOR', 'INTERIOR', 'CUT FROM', 'CUT TO', ' TO:', 'INTERCUTS', 'I/E.', ' TITLE', ' TTTLE',
                          'TRANSITION TO', 'INTERCUT', 'SFX OVER', 'VFX:', 'DISSOLVE', 'PART I']

DIRECTIONS = ['ROOFTOP', 'OMITTED', 'Omitted', 'OMIT', '(ECU)', 'Shooting Script', 'CONTINUED', 'BACK TO',
              'CLOSE ON', 'DISSOLVE TO', 'ONSCREEN', 'TO CAMERA', 'CLOSEUP', 'SUPERIMPOSE', '9/27/13', 'Revised',
              'REVISED', 'Revision', 'PAGES', 'SCREENPLAY', 'SHOOTING SCRIPT', '"IN DARKNESS" SCRIPT']

SPEAKER = ['(V.O.)', '(O.S.)', "(CONT'D)", '(SUBTITLE)']
LOWER_SPEAKER = [s.lower() for s in SPEAKER]

preformat_map = {'\t': '   ', '*': '', "’": "'", '(MORE)': '', "cont'd": "CONT'D", '(Cont.)': "(CONT'D)"}


class Turn:
    def __init__(self, speaker):
        self.speaker = speaker
        self.utterances = []

    def add_utterance(self, utter):
        self.utterances.append(utter)

    def __str__(self):
        return self.speaker + '\t' + ' '.join(self.utterances).replace('--', '')


def match(li, s):
    return bool([True for part in li if part in s])


def is_scene_changer(s):
    return (s.isupper() and match(SCENE_CHANGE, ' ' + s.lstrip())) or match(SCENE_CHANGE_NOT_UPPER, ' ' + s.lstrip())


def fix_speaker(s):
    return re.sub('\(.*\)', '', s).strip()


def double_word(s):
    s = s.strip()
    if (len(s) - 1) % 2 or len(s) < 3:
        return False
    mid = len(s) // 2
    return s[mid] == ' ' and s[:mid] == s[mid + 1:]


def is_speaker_without_indent(s, is_new_scene, not_start):
    return s.isupper() and (not is_new_scene) and not_start


def is_speaker(s, is_new_scene, not_start, indent_change, speaker_names, upper_speaker_sign, lower_speaker_sign):
    s = s.strip()
    cur_is_potential_speaker = is_speaker_without_indent(s, is_new_scene, not_start)
    cur_is_speaker = cur_is_potential_speaker and (indent_change > 0 or upper_speaker_sign)
    if not cur_is_speaker:
        cur_is_speaker = (not is_new_scene) and not_start and lower_speaker_sign and fix_speaker(s).isupper()
        cur_is_potential_speaker = max(cur_is_potential_speaker, cur_is_speaker)

    if s[0] == '-' or s[-1] == '-' or ':' in s or 'EVENING' in s or 'MONTAGE' in s:
        return False, False
    if ' -' in s and 'AGENT -' not in s:
        return False, False
    if double_word(s) and set(s) & set('0123456789'):
        return False, False
    if s == 'THE END':
        return False, False
    if len(s) > 32 and ' ' * 4 not in s:
        return False, False

    s = fix_speaker(s)
    if cur_is_speaker or s in speaker_names:
        speaker_names[s] += 1
        cur_is_speaker = True

    return cur_is_speaker, cur_is_potential_speaker


def is_direction(s):
    return match(DIRECTIONS, s.strip())


def remove_all_parentheses(text):
    n = 1
    while n:
        text, n = re.subn(r'\([^()]*\)', '', text)
    return text


def modify_balance(symb, balance):
    if symb == '(':
        balance += 1
    elif symb == ')':
        balance -= 1
    return balance


def fix_slice_by_replacing(s, st, end, old, new):
    if old in s[st:end]:
        return s[:st] + s[st:end].replace(old, new, 1) + s[end:], True
    return s, False


def fix_positive_balance(s, i, par_balance, next_lines):
    check_balance = par_balance
    for j, check_line in enumerate([s] + next_lines):
        start = 0 if j else i + 1
        for symb in check_line[start:]:
            check_balance = modify_balance(symb, check_balance)
            if check_balance == 0:
                return s, False

    s, ok = fix_slice_by_replacing(s, i, len(s), '}', ')')
    if ok:
        return s, True
    s, ok = fix_slice_by_replacing(s, i, len(s), ']', ')')
    if ok:
        return s, True
    return s + ')', True


def fix_negative_balance(s, i):
    if re.search(r'\)[A-Za-z]\(', s[i:i + 3]):
        return s[:i] + ' ' + s[i + 3:], i
    if i < 3:
        return s[:i] + ']' + s[i + 1:], i
    s, ok = fix_slice_by_replacing(s, 0, i, '{', '(')
    if ok:
        return s, i
    s, ok = fix_slice_by_replacing(s, 0, i, '[', '(')
    if ok:
        return s, i
    return '(' + s, i + 1


def preformat(s, par_balance, closing_par_exists, next_lines, prev_lines):
    s = s.strip()
    for m in preformat_map.items():
        s = s.replace(*m)

    if 'draft' in s.lower():
        return '', par_balance, closing_par_exists

    prev_par_count = par_balance
    i = 0
    while i < len(s):
        symb = s[i]
        par_balance = modify_balance(symb, par_balance)
        if par_balance > 0 and not closing_par_exists:
            s, changed = fix_positive_balance(s, i, par_balance, next_lines)
            if changed:
                '''
                print('POSITIVE FIX')
                print(s)
                for l in next_lines:
                    print(l, end='')
                print('-' * 80)
                '''
        elif par_balance == 0:
            closing_par_exists = False
        elif par_balance == -1:
            s, i = fix_negative_balance(s, i)
            '''
            print('NEGATIVE FIX')
            for l in prev_lines:
                print(l, end='')
            print(s)
            print('-' * 80)
            '''
            par_balance = 0

        i += 1

    s = '(' * prev_par_count + s + ')' * par_balance
    return s, par_balance, closing_par_exists


def parse_scenes(file_path: Union[str, Path]):
    scenes = []
    descriptions = []
    speaker_names = Counter()

    with open(file_path, encoding='utf-8') as f:
        prev_indent = 0
        prev_is_speaker = False
        in_dialog = False
        par_balance = 0
        closing_par_exists = False

        speaker_count = 0
        potential_speaker_count = 0

        lines = f.readlines()
        film_name, lines = lines[0].strip(), lines[1:]
        for i, s in enumerate(lines):
            indent = len(s) - len(s.lstrip())
            s, par_balance, closing_par_exists = preformat(s, par_balance, closing_par_exists,
                                                           lines[i + 1:i + PAR_THRESHOLD], lines[i - PAR_THRESHOLD:i])

            upper_speaker_sign = match(SPEAKER, s)
            lower_speaker_sign = match(LOWER_SPEAKER, s)
            s = ' ' * indent + remove_all_parentheses(s).strip()

            if not s.strip():
                if DEBUG:
                    print('EMPTY', s)
                continue
            if is_direction(s):
                if descriptions:
                    descriptions[-1].append(s.strip())
                if DEBUG:
                    print('DIRECTION', s)
                continue

            indent_change = indent - prev_indent

            is_new_scene = is_scene_changer(s)
            cur_is_speaker, pot = is_speaker(s, is_new_scene, bool(scenes), indent_change, speaker_names,
                                             upper_speaker_sign, lower_speaker_sign)
            speaker_count += int(cur_is_speaker)
            potential_speaker_count += int(pot)

            if cur_is_speaker:
                is_new_scene = False  # case when speaker was in set
            if not scenes and i > 5:
                is_new_scene = True

            if is_new_scene or cur_is_speaker:
                in_dialog = False
            elif prev_is_speaker and indent_change <= 0:
                in_dialog = True
            else:
                in_dialog &= indent_change == 0

            if is_new_scene:
                if DEBUG:
                    print('NEW SCENE', cur_is_speaker, in_dialog, s)
                scenes.append([])
                descriptions.append([s.strip()])
            elif cur_is_speaker and scenes:
                if DEBUG:
                    print('SPEAKER', in_dialog, s)
                scenes[-1].append(Turn(s.strip().upper()))
            elif in_dialog:
                if DEBUG:
                    print('IN DIALOG', in_dialog, s)
                scenes[-1][-1].add_utterance(s.strip())
            elif descriptions:
                descriptions[-1].append(s.strip())

            prev_indent = indent
            prev_is_speaker = cur_is_speaker

    json_scenes = [{
        'synopsis': '',
        'description': ' '.join(descr),
        'turns': [{'speaker': turn.speaker, 'utterance': ' '.join(turn.utterances)} for turn in scene]
    } for scene, descr in zip(scenes, descriptions) if scene]

    parsed_film = {'filename': file_path.name, 'film_name': film_name, 'full_synopsis': "", 'scenes': json_scenes}
    return parsed_film, len(json_scenes), speaker_names, speaker_count / (potential_speaker_count + 1e-6)


if __name__ == '__main__':
    scripts_path = config.PIPELINE_PATH / 'flattened_scripts'
    output_path = config.PIPELINE_PATH / 'parsed_scenes'
    create_or_clean(output_path)

    film_count = 0
    unhandled_count = 0
    dialog_count = 0

    film_df = pd.read_csv(config.PIPELINE_PATH / 'only_names.csv', na_filter=False).set_index('filename')
    film_df['roles'] = None
    film_df['film_name'] = ''

    DEBUG = False

    for film in os.listdir(scripts_path):
        parsed_film, count, speakers, speaker_indent_factor = parse_scenes(scripts_path / film)
        film_df.loc[film, 'film_name'] = parsed_film['film_name']
        film_count += 1
        dialog_count += count
        if count < 5:
            unhandled_count += 1
            film_df.loc[film, 'roles'] = set()
        else:
            with open(output_path / (film[:-3] + 'json'), 'w') as f:
                json.dump(parsed_film, f)
                film_df.loc[film, 'roles'] = set(speakers.keys())

        speakers = sorted(list(speakers.items()), key=lambda x: x[1])

        print(film_count, film, round(speaker_indent_factor, 3), count, speakers)

    print(f'DIALOGS: {dialog_count}')
    print(f'FILMS: {film_count - unhandled_count} handled/{film_count} total')

    print(film_df.head())
    film_df.reset_index().to_csv(config.PIPELINE_PATH / 'names_and_roles.csv', index=False)
