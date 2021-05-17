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

import json
import re
import os
import numpy as np
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
from pathlib import Path
from nltk import word_tokenize
from nltk.corpus import stopwords
from spacy.lang.en import English

import config


# feature extraction

def symmetric_jaccard(x, y):
    return len(x & y) / (len(x | y) + 1e-6)


def jaccard(plot, scene):
    return len(plot & scene) / (len(plot) + 1e-6)


class RoleKeeper:
    def __init__(self, roles, scenes):
        self.roles_ = roles
        self.scenes_ = scenes

        self.role_sets_ = self.get_role_sets_()
        self.norm_speaker2role_ = self.get_norm_speaker2role_()

    # public

    def roles_from_plot(self, plot_with_id):
        window_size = 3
        res = set()

        pad = [''] * (window_size - 1)
        tokenized = pad + word_tokenize(plot_with_id['sent']) + pad
        for start_pos in range(len(tokenized) - window_size + 1):
            window = set(tokenized[start_pos:start_pos + window_size])
            if [word for word in window if word.istitle()]:
                scores = np.array([symmetric_jaccard({word.upper() for word in window}, cand_set)
                                   for cand_set in self.role_sets_])
                if scores.max() > 0:
                    res.add(self.roles_[np.argmax(scores)])

        return res

    def role_from_speaker(self, speaker):
        return self.norm_speaker2role_.get(self.normalize_speaker_(speaker))

    def roles_from_scene(self, scene):
        res = set()
        for turn in scene['turns']:
            role = self.role_from_speaker(turn[0])
            if role:
                res.add(role)
        return res

    def scene2str(self, scene):
        turns = []
        for turn in scene['turns']:
            role = self.role_from_speaker(turn[0])
            if not role:
                role = turn[0]
            utter = turn[1]
            turns.append(role + ': ' + utter)

        return ' '.join(turns)

    # private

    @staticmethod
    def normalize_speaker_(speaker):
        return re.sub('\(.*\)', '', speaker).strip()

    def get_role_sets_(self):
        to_exclude = set(w.upper() for w in stopwords.words('english'))
        return [set(role.upper().split()) - to_exclude for role in self.roles_]

    def get_norm_speaker2role_(self):
        norm_speakers = list(set([self.normalize_speaker_(turn[0])
                                  for scene in self.scenes_ for turn in scene['turns']]))

        norm_speaker2role = dict()
        for speaker in norm_speakers:
            speaker_set = set(speaker.split())

            cand_scores = [len(speaker_set & cand) for cand in self.role_sets_]
            m = np.argmax(cand_scores)
            if cand_scores[m] > 0:
                norm_speaker2role[speaker] = self.roles_[m]

        return norm_speaker2role


# embeddings

class Sentencizer:
    def __init__(self):
        self.nlp = English()
        sentencizer = self.nlp.create_pipe("sentencizer")
        self.nlp.add_pipe(sentencizer)

    def __call__(self, text):
        return [s.text for s in self.nlp(text).sents]


class SentenceEncoderWrapper:
    def __init__(self, model, model_name_or_path=None):
        possible_models = ['use', 'roberta']
        if model == 'use':
            if model_name_or_path is None:
                model_name_or_path = 'https://tfhub.dev/google/universal-sentence-encoder-large/5'
            self.model_ = hub.load(model_name_or_path)
            self.encode_ = lambda sents: self.model_(sents).numpy()
        elif model == 'roberta':
            if model_name_or_path is None:
                model_name_or_path = 'roberta-base-nli-stsb-mean-tokens'
            self.model_ = SentenceTransformer(model_name_or_path)
            self.encode_ = lambda sents: np.array(self.model_.encode(sents))
        else:
            raise NotImplementedError(f"Not found model '{model}' in {possible_models}")
        self.sentencizer_ = Sentencizer()

    # public

    def get_similarity_matrix(self, plot_sents_with_ids, scene_sents):
        plot_sents = [pair['sent'] for pair in plot_sents_with_ids]
        plot_feats = self.get_plot_emb_(plot_sents)
        return self.get_feats_max_(plot_feats, scene_sents), self.get_feats_mean_(plot_feats, scene_sents)

    # private

    def get_sent_embeddings_(self, sent_list):
        return self.encode_(sent_list)

    @staticmethod
    def normalize_(emb):
        emb = np.array(emb)
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    def get_plot_emb_(self, plot_sent_list):
        return self.normalize_(self.get_sent_embeddings_(plot_sent_list))

    def get_scene_emb_(self, sent_list, multi_sent_scene_handling):
        sents, counts = [], []
        for text in sent_list:
            cur_sents = self.sentencizer_(text)
            sents += cur_sents
            counts.append(len(cur_sents))

        embs = self.get_sent_embeddings_(sents)

        text_embs = []
        cur_pos = 0
        for count in counts:
            if multi_sent_scene_handling == 'mean':
                text_embs.append(embs[cur_pos:cur_pos + count].mean(0))
            else:
                text_embs.append(self.normalize_(embs[cur_pos:cur_pos + count]))
            cur_pos += count

        if multi_sent_scene_handling == 'mean':
            return self.normalize_(text_embs)
        return text_embs

    def score_(self, embs, sent_embs):
        return embs.dot(sent_embs.T).max(1)

    def get_feats_mean_(self, plot_feats, scene_sents):
        scene_feats = self.get_scene_emb_(scene_sents, 'mean')
        return plot_feats.dot(scene_feats.T)

    def get_feats_max_(self, plot_feats, scene_sents):
        scene_feats = self.get_scene_emb_(scene_sents, 'max')
        return np.array([self.score_(plot_feats, scene_f) for scene_f in scene_feats]).T


def get_plot_sents(plot, sentencizer):
    def remove_all_parentheses(text):
        n = 1
        while n:
            text, n = re.subn(r'\([^()]*\)', '', text)
        return text

    plot = remove_all_parentheses(plot).replace('  ', ' ').strip()  # to exclude (Samuel L. Jackson)
    plot = re.sub(r'([a-z][.,?!])([A-Z])', r'\1 \2', plot)  # to add spaces between paragraphs
    plot = re.sub(r' ([.,?!])', r'\1', plot)  # to remove spaces before punctuation signs

    return sentencizer(plot)


def add_plot_sents_ids(plot_sents, add_empty_strings):
    plot_sents = [{'id': i, 'sent': sent} for i, sent in enumerate(plot_sents)]
    if add_empty_strings:
        dummy_syn = {'id': -1, 'sent': ''}
        plot_sents = [dummy_syn] + sum([[elem, dummy_syn] for elem in plot_sents], [])
    return plot_sents


def get_prob(arr, eps=0.01):
    return arr


def get_jaccard_similarity_matrix(plot_roles, scene_roles, similarity_func):
    return np.array([[similarity_func(plot_roles[i], scene_roles[j])
                      for j in range(len(scene_roles))] for i in range(len(plot_roles))])


class SimilarityScorer:
    def __init__(self, similarity_max, similarity_mean, symmetric_jaccard_sim, one_sided_jaccard_sim, debug=False):
        self.sim_max_ = get_prob(similarity_max)
        self.sim_mean_ = get_prob(similarity_mean)
        self.symmetric_jac_ = get_prob(symmetric_jaccard_sim)
        self.one_sided_jac_ = get_prob(one_sided_jaccard_sim)
        self.shape = self.sim_max_.shape

    def __call__(self, i, j, coefs):
        assert set(coefs.keys()) == {'sim_max', 'sim_mean', 'sym_jac', 'one_sided_jac'}
        numerator = self.sim_max_[i, j] * coefs['sim_max'] + self.sim_mean_[i, j] * coefs['sim_mean']
        numerator += self.symmetric_jac_[i, j] * coefs['sym_jac'] + self.one_sided_jac_[i, j] * coefs['one_sided_jac']
        return numerator

    def remove_empty(self):
        self.sim_max_ = self.sim_max_[1::2]
        self.sim_mean_ = self.sim_mean_[1::2]
        self.symmetric_jac_ = self.symmetric_jac_[1::2]
        self.one_sided_jac_ = self.one_sided_jac_[1::2]

        self.shape = self.sim_max_.shape


# matching algorithm

def dtw_restricted(scorer, coefs):
    dtw = -np.ones(np.array(scorer.shape) + 1) * np.inf  # 1s to start not from first to first
    dtw[0, 0] = 0
    parent = -np.ones_like(dtw, dtype=int)

    for i in range(1, scorer.shape[0] + 1):
        for j in range(1, scorer.shape[1] + 1):
            direction = [dtw[i - 1, j], dtw[i - 1, j - 1]]
            if i % 2:  # empty syn case
                direction.append(dtw[i, j - 1])
            parent[i, j] = np.argmax(direction)

            dtw[i, j] = scorer(i - 1, j - 1, coefs) + max(direction)

    # decoding

    dir_change = {0: (-1, 0), 1: (-1, -1), 2: (0, -1)}

    plot_pos, scene_pos = np.array(dtw.shape) - 1
    plot2scene = [0] * scorer.shape[0]
    while plot_pos * scene_pos:
        plot2scene[plot_pos - 1] = scene_pos - 1
        change = dir_change[parent[plot_pos, scene_pos]]
        plot_pos += change[0]
        scene_pos += change[1]

    return plot2scene


# utilities

def preprocess(film_json, sent_encoder, add_empty_strings, plot_sl=slice(None), sc_sl=slice(None), debug=False):
    scenes = sum([cluster['scenes'] for cluster in film_json['clusters']], [])[sc_sl]
    role_keeper = RoleKeeper(film_json['roles'], scenes)

    plot_sents_with_ids = add_plot_sents_ids(film_json['plot_sents'], add_empty_strings)[plot_sl]
    scene_sents = list(map(role_keeper.scene2str, scenes))
    plot_roles = list(map(role_keeper.roles_from_plot, plot_sents_with_ids))
    scene_roles = list(map(role_keeper.roles_from_scene, scenes))

    similarity_max, similarity_mean = sent_encoder.get_similarity_matrix(plot_sents_with_ids, scene_sents)
    symmetric_jac_sim = get_jaccard_similarity_matrix(plot_roles, scene_roles, symmetric_jaccard)
    one_sided_jac_sim = get_jaccard_similarity_matrix(plot_roles, scene_roles, jaccard)
    if add_empty_strings:  # assigning minimal score to dummy strings
        similarity_max[::2] = 0
        similarity_mean[::2] = 0
        symmetric_jac_sim[::2] = 0
        one_sided_jac_sim[::2] = 0
    scorer = SimilarityScorer(similarity_max, similarity_mean, symmetric_jac_sim, one_sided_jac_sim, debug=debug)

    return plot_sents_with_ids, scene_sents, plot_roles, scene_roles, scorer, role_keeper


def add_plot_to_scenes(plot_sents_with_ids, clusters, plot2scene):
    def remove_double_spaces(s):
        while '  ' in s:
            s = s.replace('  ', ' ')
        return s.strip()

    for syn, scene in enumerate(plot2scene):
        plot_pair = plot_sents_with_ids[syn]
        cluster_dict = clusters[scene]
        cluster_dict['plot_ids'].append(plot_pair['id'])
        cluster_dict['plot'].append(plot_pair['sent'])
    return clusters


# scenes merging

def remove_empty(plot_sents_with_ids, plot2scene, scorer):
    plot_sents_with_ids = plot_sents_with_ids[1::2]
    plot2scene = plot2scene[1::2]
    scorer.remove_empty()
    return plot_sents_with_ids, plot2scene, scorer


def merge_two_clusters(cluster_a, cluster_b):
    return {key: cluster_a[key] + cluster_b[key] for key in cluster_a.keys()}


def merge_scenes(clusters, scorer, plot2scene, coefs):
    def merge_to_left(left, start, end):
        for cluster in clusters[start:end]:
            left = merge_two_clusters(left, cluster)
        return left

    new_scenes = [merge_to_left(clusters[0], 1, plot2scene[0] + 1)]

    for left_syn, left_scene in enumerate(plot2scene[:-1]):
        right_syn = left_syn + 1
        right_scene = plot2scene[right_syn]
        if left_scene == right_scene:
            continue

        best_last = left_scene
        best_score = score = sum([scorer(right_syn, pos, coefs) for pos in range(left_scene + 1, right_scene)])
        for last_scene_to_left in range(left_scene + 1, right_scene):
            score += scorer(left_syn, last_scene_to_left, coefs) - scorer(right_syn, last_scene_to_left, coefs)
            if score > best_score:
                best_last, best_score = last_scene_to_left, score

        new_scenes[-1] = merge_to_left(new_scenes[-1], left_scene + 1, best_last + 1)
        new_scenes.append(merge_to_left(clusters[best_last + 1], best_last + 2, right_scene + 1))

    new_scenes[-1] = merge_to_left(new_scenes[-1], plot2scene[-1] + 1, None)
    return new_scenes


# end-to-end

def map_plots(j, sent_encoder, coefs, algo=dtw_restricted, add_empty_strings=True):
    prep = preprocess(j, sent_encoder, add_empty_strings=add_empty_strings)
    plot_sents_with_ids, scene_sents, _, _, scorer, role_keeper = prep

    mappings = [algo(scorer, option) for option in coefs]
    optimal_option_id = int(np.argmax([len(set(mapping)) for mapping in mappings]))
    plot2scene = mappings[optimal_option_id]
    j['coefs'] = optimal_coef = coefs[optimal_option_id]

    if add_empty_strings:
        plot_sents_with_ids, plot2scene, scorer = remove_empty(plot_sents_with_ids, plot2scene, scorer)
    add_plot_to_scenes(plot_sents_with_ids, j['clusters'], plot2scene)
    scorer_shape = scorer.shape

    j['clusters'] = merge_scenes(j['clusters'], scorer, plot2scene, optimal_coef)
    return j, scorer_shape, role_keeper


# supplementary

def compute_mae(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.mean(np.abs(x - y))


def compute_intrinsic_score(plot2scene, scorer):
    return sum([scorer(i, j) for i, j in enumerate(plot2scene)])


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def show_annotated(plot2scene, scene_syn, scenes, role_keeper, descriptions=False):
    for scene_num, (syn, sc) in enumerate(zip(scene_syn, scenes)):
        if descriptions:
            print('DESCRIPTION:', sc['description'])
            print('-' * 10)
        print('SYN ROLES', role_keeper.roles_from_syn(syn))
        print()
        print('plot:', syn)
        print('-' * 10)
        print(scene_num, [i for i, scene in enumerate(plot2scene) if scene == scene_num], 'SCENE ROLES',
              role_keeper.roles_from_scene(sc))
        print()
        for turn in sc['turns']:
            print(f'{turn["speaker"]}: {turn["utterance"]}')
        print('-' * 80)


def show_annotated_both(plot2scene_one, plot2scene_two, plot_sents, scenes):
    scene_plot_one = get_scene_plots(plot2scene_one, plot_sents, len(scenes))
    scene_plot_two = get_scene_plots(plot2scene_two, plot_sents, len(scenes))

    for scene_num, (plot_one, plot_two, scene) in enumerate(zip(scene_plot_one, scene_plot_two, scenes)):
        print('plot ONE:', [i for i, scene in enumerate(plot2scene_one) if scene == scene_num], plot_one)
        print('-' * 10)
        print('plot TWO:', [i for i, scene in enumerate(plot2scene_two) if scene == scene_num], plot_two)
        print('-' * 10)
        print('SCENE', scene_num)
        for turn in scene['turns']:
            print(f'{turn["speaker"]}: {turn["utterance"]}')
        print('-' * 80)


def show_annotated_j(plot2scene, scenes, role_keeper):
    for scene_num, scene in enumerate(scenes):
        print('PLOT ROLES', role_keeper.roles_from_syn(scene['plot']))
        print()
        print('plot:', scene['plot'])
        print('-' * 10)
        print(scene_num, [i for i, scene in enumerate(plot2scene) if scene == scene_num], 'SCENE ROLES',
              role_keeper.roles_from_scene(scene))
        print()
        for turn in scene['turns']:
            print(f'{turn["speaker"]}: {turn["utterance"]}')
        print('-' * 80)


def process_pipeline(pipeline_path, sent_encoder_model, coefs):
    sent_encoder = SentenceEncoderWrapper(sent_encoder_model)

    pipeline_path = Path(pipeline_path)
    scripts_path = pipeline_path / 'scenes_and_plot'
    output_path = pipeline_path / 'mapped_scenes'
    mkdir_if_not_exist(output_path)
    scenes_total = 0
    first_type_like, total = 0, 0

    films = sorted(os.listdir(scripts_path))
    processed_films = set(os.listdir(output_path))
    for i, filename in enumerate(films):
        if not filename.endswith('json'):
            continue
        if filename in processed_films:
            continue

        with open(scripts_path / filename) as f:
            j = json.load(f)

        j, scorer_shape, _ = map_plots(j, sent_encoder, coefs=coefs)

        with open(output_path / filename, 'w') as f:
            json.dump(j, f)

        scene_count = len(j['clusters'])
        scenes_total += scene_count
        est = int(scenes_total * len(films) / (i + 1))

        total += 1
        if j['coefs']['one_sided_jac'] == coefs[0]['one_sided_jac']:
            first_type_like += 1
        cur_first_percent = first_type_like * 100 // total

        log = (f'{i + 1}/{len(films)} Processing {filename}, (syn, scene) = {scorer_shape}, got {scene_count} scenes.' +
               f' Now have {scenes_total}, estimated {est} total. {first_type_like}/{total}, {cur_first_percent}% first-like.')
        print(log)
        with open('log.txt', 'a') as f:
            print(log, file=f)


if __name__ == '__main__':
    pipeline_path = config.PIPELINE_PATH
    sent_encoder_model = 'use'
    coefs = [{'sim_max': 0.7, 'sim_mean': 0.1, 'sym_jac': 0.2, 'one_sided_jac': 0.1},
             {'sim_max': 0.2, 'sim_mean': 0.3, 'sym_jac': 0.0, 'one_sided_jac': 0.4}]
    process_pipeline(pipeline_path, sent_encoder_model, coefs)
