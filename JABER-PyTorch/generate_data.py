
# 2021.09.29-Modified slightly on Arabic text preprocessing as well as added method for preprocessing 
# ALUE data; prepared ALUE test set submission
#              Huawei Technologies Co., Ltd. 

# Copyright 2021 Huawei Technologies Co., Ltd.


import string, warnings
import os, pickle, copy
import random, math

from collections import defaultdict, Counter
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
from tqdm import tqdm

import html
import logging
import re, locale, optparse

import pyarabic.araby as araby
from tokenizationBBPE import FullTokenizer

optparser = optparse.OptionParser()

optparser.add_option('--mode',
                     default="train",
                     choices=["train", "test"],
                     help="generate train/test data")

opts = optparser.parse_args()[0]


locale.setlocale(locale.LC_ALL, 'en_US.utf-8')
warnings.filterwarnings("ignore")


############################
## Code to download ALUE ###
############################
class ArabertPreprocessor:
    def __init__(self, model_name):
        print("Warning!!!! This is just an empty a placeholder class. "
              "Follow instructions in README.md file to obtain the true `ArabertPreprocessor` class")

    def preprocess(self, text):
        return text

def normalize_text(pp_text):
    # map some weired characters
    mapping = {"ھ": "ه", "گ": "ك", r'\s': " ", "٪": "%", "℃": "C", "·": ".", "…": ".",
               'ـــ': '-', 'ـ': '-', ",": "،", "\?": "؟", "“": '"', '”': '"'}
    # map indic to arabic digits
    digit_lst = ["٠", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"]
    mapping.update({dig: "%s" % idx for idx, dig in enumerate(digit_lst)})

    for a, b in mapping.items():
        pp_text = re.sub(a, b, pp_text)

    return pp_text



############################
## Code to download ALUE ###
############################
class FTProcessor():
    def __init__(self, max_seq_length, bert_type):
        self.task_type_dict = {"SVREG": "reg",
                               "SEC": "ml_cls"}

        self.max_seq_length = max_seq_length
        self.bert_type = bert_type
        assert bert_type == "jaber", "no support for other baselines for now"
        vocab_file = os.path.join("./pretrained_models/JABER/vocab.txt")
        do_lower_case = False
        self.tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
        self.arabert_prep = ArabertPreprocessor("bert-large-arabertv02")

    def get_tag_to_id(self, task, exp_lst):
        if task not in self.task_type_dict:
            tag_to_id = get_tag_to_id_cls(exp_lst)
        else:
            tag_to_id = None

        return tag_to_id

    def to_bert_input(self, tag_to_id, exp_dict):

        tok_lst = [self._tokenize_text(text) for text in exp_dict["s_lst"]]
        tokens_a = tok_lst[0]
        tokens_b = tok_lst[1] if len(tok_lst) > 1 else {"input_ids": [], "positions": []}

        input_ids, segment_ids, positions = \
            _truncate_seq_pair(self.tokenizer, tokens_a, tokens_b, self.max_seq_length)

        target = None

        if "lbl" in exp_dict:
            if not tag_to_id:
                target = exp_dict["lbl"]
            else:
                target = tag_to_id[exp_dict["lbl"]]

        dico = {"idx": exp_dict["idx"], "input_ids": input_ids,
                "segment_ids": segment_ids, "target": target,
                "positions": positions}

        if target is None:
            del dico["target"]

        return dico

    def _text_to_tokens(self, text):
        pp_text = self.arabert_prep.preprocess(text)
        return normalize_text(pp_text).split()

    def _tokenize_text(self, text):
        word_lst = self._text_to_tokens(text)

        input_ids, positions = [], []
        for w in word_lst:
            bert_lst = self.tokenizer.tokenize(w)
            positions.append(len(input_ids))
            input_ids += self.tokenizer.convert_tokens_to_ids(bert_lst)

        return {"input_ids": input_ids, "positions": positions}

def _truncate_seq_pair(tokenizer, tokens_a, tokens_b, max_seq_length):
    """Truncates a pair of sequences to a maximum sequence length."""
    # print(len(tokens_a["input_ids"]) + len(tokens_b["input_ids"]))

    while True:
        total_length = len(tokens_a["input_ids"]) + len(tokens_b["input_ids"]) + 3 # cls + 2 sep
        if total_length <= max_seq_length:
            break

        token_trunc = tokens_a if len(tokens_a["positions"]) > len(tokens_b["positions"]) else tokens_b
        assert len(token_trunc["positions"]) > 1

        token_trunc["input_ids"] = token_trunc["input_ids"][:token_trunc["positions"][-1]]
        token_trunc["positions"] = token_trunc["positions"][:-1]

    # merge
    input_ids, segment_ids, positions = tokenizer.convert_tokens_to_ids(["[CLS]"]), [], []
    positions.extend([p + len(input_ids) for p in tokens_a["positions"]])
    input_ids.extend(tokens_a["input_ids"])
    input_ids.extend(tokenizer.convert_tokens_to_ids(["[SEP]"]))
    segment_ids.extend([0] * len(input_ids))

    assert len(input_ids) <= max_seq_length
    if not tokens_b:
        return input_ids, segment_ids, positions

    positions.extend([p + len(input_ids) for p in tokens_b["positions"]])
    input_ids.extend(tokens_b["input_ids"])
    input_ids.extend(tokenizer.convert_tokens_to_ids(["[SEP]"]))
    segment_ids.extend([1] * (len(tokens_b["input_ids"])+1))

    assert len(input_ids) <= max_seq_length

    return input_ids, segment_ids, positions


def get_tag_to_id_cls(exp_lst):
    counter = Counter([exp["lbl"] for exp in exp_lst])
    counter = sorted(counter.items(), key= lambda x:(x[1], x[0]), reverse=True)
    tag_to_id = {tag: idx for idx, (tag, _) in enumerate(counter)}
    return tag_to_id


def load_mq2q_dev():
    filename = "./raw_datasets/mq2q.dev.tsv"
    if not os.path.exists(filename):
        raise ValueError ("Please check the README to generate MQ2Q dev set")
    lst = []
    for idx, line in enumerate(open(filename)):
        lbl, s1, s2 = line.strip().split("\t")
        lst.append({"idx": len(lst), "s_lst": [s1, s2], "lbl": int(lbl)})
    return lst


def process_alue():
    # https://www.alue.org/tasks
    bench_dict = {}
    data_dir = "./raw_datasets"

    # MQ2Q
    bench_dict["MQ2Q"] = defaultdict(list)
    df_train = pd.read_csv(os.path.join(data_dir, "q2q_similarity_workshop_v2.1.tsv"), sep="\t")
    df_test = pd.read_csv(os.path.join(data_dir, "q2q_no_labels_v1.0 - q2q_no_labels_v1.0.tsv"), sep="\t")

    for idx, (s1, s2, lbl) in enumerate(zip(df_train["question1"], df_train["question2"], df_train["label"])):
        exp = {"idx": idx, "s_lst": [s1, s2], "lbl": lbl}
        bench_dict["MQ2Q"]["train"].append(exp)
    for s1, s2, idx in zip(df_test["question1"], df_test["question2"], df_test["QuestionPairID"]):
        exp = {"idx": idx, "s_lst": [s1, s2]}
        bench_dict["MQ2Q"]["test"].append(exp)

    # add dev set from translated QQP
    bench_dict["MQ2Q"]["dev"] = load_mq2q_dev()

    # OOLD and OHSD
    bench_dict["OOLD"], bench_dict["OHSD"] = defaultdict(list), defaultdict(list)
    filename = os.path.join(data_dir, "OSACT2020-sharedTask-%s.txt")
    for task, lbl_key in [("OOLD", "offensive"), ("OHSD", "hate")]:
        for portion in ["train", "dev", "test"]:
            names = ["text"] if portion == "test" else ["text", "offensive", "hate"]
            df = pd.read_csv(filename % portion, sep="\t", quotechar='▁', header=None, names=names)
            if portion != "test":
                for idx, (s, lbl) in enumerate(zip(df["text"], df[lbl_key])):
                    exp = {"idx": idx, "s_lst": [s], "lbl": lbl}
                    bench_dict[task][portion].append(exp)
            else:
                for idx, s in enumerate(df["text"]):
                    exp = {"idx": idx, "s_lst": [s]}
                    bench_dict[task][portion].append(exp)

    # SVREG
    bench_dict["SVREG"] = defaultdict(list)
    filename = os.path.join(data_dir, "SemEval2018-Task1-all-data/Arabic/V-reg/2018-Valence-reg-Ar-%s.txt")
    for portion in ["train", "dev", "test"]:
        df = pd.read_csv(filename% portion, sep="\t")
        for idx, (s, lbl) in enumerate(zip(df["Tweet"], df["Intensity Score"])):
            exp = {"idx": idx, "s_lst": [s], "lbl": lbl}
            bench_dict["SVREG"][portion].append(exp)

        if portion == "test":
            for exp in bench_dict["SVREG"]["test"]:
                del exp["lbl"]

    # SEC
    bench_dict["SEC"] = defaultdict(list)
    filename = os.path.join(data_dir, "SemEval2018-Task1-all-data/Arabic/E-c/2018-E-c-Ar-%s.txt")
    for portion in ["train", "dev", "test", "test-gold"]:
        df = pd.read_csv(filename % portion, sep="\t")

        for row in df.to_dict(orient="records"):
            exp = {"idx": row["ID"], "s_lst": [row["Tweet"]]}
            if portion != "test":
                exp["lbl"] = [int(v) for _, v in list(row.items())[2:]]
            bench_dict["SEC"][portion].append(exp)

    # FID
    bench_dict["FID"] = defaultdict(list)
    filename = os.path.join(data_dir, "IDAT_data", "IDAT_%s_text.csv")
    for portion in ["train", "test"]:
        key = portion if portion == "test" else "training"
        df = pd.read_csv(filename % key, sep=",")
        for idx, (s, lbl) in enumerate(zip(df["text"], df["label"])):
            exp = {"idx": idx, "s_lst": [s], "lbl": lbl}
            bench_dict["FID"][portion].append(exp)
            if portion == "test":
                bench_dict["FID"]["dev"].append(copy.deepcopy(exp))

    # XNLI
    bench_dict["XNLI"] = defaultdict(list)
    filename = os.path.join(data_dir, "XNLI", "arabic_%s.tsv")
    for portion in ["train", "dev", "diag"]:
        df = pd.read_csv(filename % portion, sep="\t")
        for idx, s1, s2, lbl in zip(df["pairID"], df["sentence1"], df["sentence2"], df["gold_label"]):
            exp = {"idx": idx, "s_lst": [s1, s2], "lbl": lbl}
            bench_dict["XNLI"][portion].append(exp)
            if portion == "dev":
                bench_dict["XNLI"]["test"].append(copy.deepcopy(exp))
                #del bench_dict["XNLI"]["test"][-1]["lbl"]

    bench_dict["MDD"] = defaultdict(list)
    filename = os.path.join(data_dir, "MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021/MADAR-Corpus-26-%s.tsv")

    for portion in ["train", "dev", "test"]:
        df = pd.read_csv(filename % portion, sep="\t", header=None, names=["Text", "label"])
        for idx, (s, lbl) in enumerate(zip(df["Text"], df["label"])):
            exp = {"idx": idx, "s_lst": [s], "lbl": lbl}
            bench_dict["MDD"][portion].append(exp)

    # generate tag_to_id
    ft_proc = FTProcessor(128, "jaber")
    for task, portion_dict in bench_dict.items():
        bench_dict[task]["tag_to_id"] = ft_proc.get_tag_to_id(task, portion_dict["train"])

    return bench_dict


def create_finetune_data(ds_name="alue", bert_type="jaber", max_seq_len=128):
    """

    :param conf_name: we use the same configuration of the pretrain model
    :return:
    """

    ft_proc = FTProcessor(max_seq_len, bert_type)
    bench_dict = process_alue()

    output_dict = {}
    for task, portion_dict in bench_dict.items():
        print(task)
        output_dict[task] = defaultdict(list)
        tag_to_id = portion_dict["tag_to_id"]
        for portion, exp_lst in portion_dict.items():
            if portion == "tag_to_id":
                output_dict[task]["tag_to_id"] = tag_to_id
                continue
            output_dict[task][portion] = [ft_proc.to_bert_input(tag_to_id, exp_dict) for exp_dict in tqdm(exp_lst)]

    key = (ds_name, max_seq_len, bert_type)
    file_output = os.path.join("./raw_datasets", "%s.%s.%s.pkl" % key)
    with open(file_output, 'wb') as handle:
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_finetune_test():

    # load id_to_tag
    print("loading data")
    key = ("alue", "128", "jaber")

    filename = os.path.join("./raw_datasets", "%s.%s.%s.pkl" % key)
    with open(filename, 'rb') as fp:
        instances_dict = pickle.load(fp)
    id_to_tag_dict = {}
    y_true_dict = {}
    for task_name, dico in instances_dict.items():
        tag_to_id = dico["tag_to_id"] if "tag_to_id" in dico else None
        id_to_tag_dict[task_name] = {v:k for k, v in tag_to_id.items()} if tag_to_id else None
        for portion in ["test", "diag"]:
            if portion in dico:
                if "target" not in dico[portion][0]: continue
                y_true_dict[(task_name, portion)] = [exp_dict["target"] for exp_dict in dico[portion]]

    # load best pred
    test_dir = os.path.join("alue_predictions")
    best_pred_dict = {}
    best_score_dict = {}
    for filename in os.listdir(test_dir):
        if not filename.endswith(".pkl"): continue
        model_name, task_name, dev_score = filename[:-4].split("_")
        dev_score = float(dev_score)
        key = (model_name, task_name)
        if key in best_score_dict and best_score_dict[key] > dev_score: continue
        best_score_dict[key] = dev_score
        with open(os.path.join(test_dir, filename), 'rb') as fp:
            best_pred_dict[key] = pickle.load(fp)

    for (model_name, task_name), pred_dict in best_pred_dict.items():
        id_to_tag = id_to_tag_dict[task_name]
        save_prediction(task_name, pred_dict, id_to_tag, y_true_dict)


def save_prediction(task_name, best_pred_dict, id_to_tag, y_true_dict):

    data_dir = os.path.join("./raw_datasets")
    output_dir = os.path.join("./alue_test_submission")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if task_name == "MQ2Q":
        predictions = [id_to_tag[y] for x, y in best_pred_dict["test"]]
        df_test = pd.read_csv(os.path.join(data_dir, "q2q_no_labels_v1.0 - q2q_no_labels_v1.0.tsv"), sep="\t")
        df_preds = pd.DataFrame(data=predictions, columns=["prediction"], index=df_test["QuestionPairID"])
        df_preds.reset_index(inplace=True)
        df_preds.to_csv(os.path.join(output_dir, "q2q.tsv"), index=False, sep="\t")

    if task_name in ["OOLD", "OHSD"]:
        predictions = [id_to_tag[y] for x, y in best_pred_dict["test"]]
        df_preds = pd.DataFrame(data=predictions, columns=["prediction"])
        df_preds.reset_index(inplace=True)
        k = {"OOLD": "offensive", "OHSD": "hate"}[task_name]
        df_preds.to_csv(os.path.join(output_dir, "%s.tsv" % k), index=False, header=False, sep="\t")

    if task_name == "SVREG":
        filename = os.path.join(data_dir, "SemEval2018-Task1-all-data/Arabic/V-reg/2018-Valence-reg-Ar-%s.txt")
        df_test = pd.read_csv(filename % "test", sep="\t")
        predictions = [y for _, y in best_pred_dict["test"]]
        df_preds = pd.DataFrame(data=predictions, columns=["prediction"], index=df_test["ID"])
        df_preds.reset_index(inplace=True)
        df_preds.to_csv(os.path.join(output_dir,"v_reg.tsv"), index=False, sep="\t")

    if task_name == "SEC":
        filename = os.path.join(data_dir, "SemEval2018-Task1-all-data/Arabic/E-c/2018-E-c-Ar-%s.txt")
        df_train, df_test = pd.read_csv(filename % "train", sep="\t"), pd.read_csv(filename % "test", sep="\t")
        predictions = [[int(i) for i in y] for _, y in best_pred_dict["test"]]
        df_preds = pd.DataFrame(data=predictions, columns=df_train.columns[2:].tolist(), index=df_test["ID"])
        df_preds.reset_index(inplace=True)
        df_preds.to_csv(os.path.join(output_dir,"E_c.tsv"), index=False, sep="\t")

    if task_name == "FID":
        df_test = pd.read_csv(os.path.join(data_dir, "IDAT_data", "IDAT_test_text.csv"), sep=",")
        predictions = [id_to_tag[y] for x, y in best_pred_dict["test"]]
        y_true = [id_to_tag[y] for y in y_true_dict[(task_name, "test")]]
        print("FID", "%.2f" % (100 * f1_score(y_true, predictions, average="macro")))
        df_preds = pd.DataFrame(data=predictions, columns=["prediction"], index=df_test["id"])
        df_preds.reset_index(inplace=True)
        df_preds.to_csv(os.path.join(output_dir, "irony.tsv"), index=False, sep="\t")

    if task_name == "XNLI":
        for portion in ["test", "diag"]:
            fn = "dev" if portion == "test" else portion
            df_test = pd.read_csv(os.path.join(data_dir, "XNLI", "arabic_%s.tsv" % fn), sep="\t")
            y_true = [id_to_tag[y] for y in y_true_dict[(task_name, portion)]]
            predictions = [id_to_tag[y] for x, y in best_pred_dict[portion]]
            if portion == "diag":
                print(task_name, portion, "%.2f" % (100* f1_score(y_true, predictions, average="macro")))
            else:
                print(task_name, portion, "%.2f" % (100 * accuracy_score(y_true, predictions)))
            df_preds = pd.DataFrame(data=predictions, columns=["prediction"], index=df_test["pairID"])
            df_preds.reset_index(inplace=True)
            fn = "xnli" if portion == "test" else "diagnostic"
            df_preds.to_csv(os.path.join(output_dir, "%s.tsv" % fn), index=False, sep="\t")

    if task_name == "MDD":
        predictions = [id_to_tag[y] for x, y in best_pred_dict["test"]]
        y_true = [id_to_tag[y] for y in y_true_dict[(task_name, "test")]]
        print(task_name, "%.2f" % (100 * f1_score(y_true, predictions, average="macro")))
        df_preds = pd.DataFrame(data=predictions)
        df_preds.to_csv(os.path.join(output_dir, "madar.tsv"), index=False, header=False, sep="\t")


if opts.mode == "train":
    create_finetune_data()
elif opts.mode == "test":
    create_finetune_test()
else:
    raise ValueError("Not supported option %s !!!!!" % opts.mode)
