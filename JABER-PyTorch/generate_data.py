# 2021.09.29-Modified slightly on Arabic text preprocessing as well as added method for preprocessing
# ALUE data; prepared ALUE test set submission
#              Huawei Technologies Co., Ltd.

# Copyright 2021 Huawei Technologies Co., Ltd.

import sys, re, json, argparse
sys.path.insert(0,'..')
import warnings, locale, random, shutil, os, pickle, copy
from collections import defaultdict, Counter

import pandas as pd
from tqdm import tqdm
import datasets
import numpy as np

import tokenizationBBPE as tokenization
from compute_metrics import alue_compute_metrics as compute_metrics
# from eval_squad import *

from difflib import get_close_matches
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer, scoring

warnings.filterwarnings("ignore")
locale.setlocale(locale.LC_ALL, 'en_US.utf-8')
############################################
#### Arabert Preprocessor Place holder #####
############################################
# from preprocess import ArabertPreprocessor

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

##############################
#### Data loader methods #####
##############################
ARROW_FEATURES = """
{
    "input_ids": {
      "feature": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
      },
      "length": -1,
      "id": null,
      "_type": "Sequence"
    },
    "labels": {
      "feature": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
      },
      "length": -1,
      "id": null,
      "_type": "Sequence"
    }
  }

"""

ARROW_FEATURES_REG = """
{
    "input_ids": {
      "feature": {
        "dtype": "int32",
        "id": null,
        "_type": "Value"
      },
      "length": -1,
      "id": null,
      "_type": "Sequence"
    },
    "labels": {
      "feature": {
        "dtype": "float32",
        "id": null,
        "_type": "Value"
      },
      "length": -1,
      "id": null,
      "_type": "Sequence"
    }
  }
"""

LABEL_TO_TEXT = {"MQ2Q": {0: "غير مكرر", 1: "مكرر"},
                 "FID":  {0: "سخرية", 1: "ليس سخرية"},
                 "OOLD":  {"NOT_OFF": "غير مهين", "OFF": "مهين"},
                 "OHSD": {"NOT_HS": "لا يحض على الكراهية", "HS": "خطاب كراهية"},
                 "XNLI": {"neutral": "علاقة غير مترابطة", "entailment": "علاقة مترابطة", "contradiction": "علاقة متناقضة"},
                 "MDD":  {'SFX': 'صفاقس', 'ALX': 'الإسكندرية', 'ALE': 'حلب', 'FES': 'فاس', 'TRI': 'طرابلس', 'MSA': 'العربية',
                          'CAI': 'القاهرة', 'ASW': 'أسوان', 'AMM': 'عمان', 'TUN': 'تونس', 'DOH': 'الدوحة', 'RIY': 'الرياض',
                          'ALG': 'الجزائر', 'KHA': 'الخرطوم', 'DAM': 'دمشق', 'RAB': 'الرباط', 'SAN': 'صنعاء', 'BEI': 'بيروت',
                          'JER': 'القدس', 'JED': 'جدة', 'BAS': 'البصرة', 'BEN': 'بنغازي', 'SAL': 'سل', 'MUS': 'مسقط',
                          'MOS': 'الموصل', 'BAG': 'بغداد'
                          }
                 }

MLC_LBL_DICT = {"SEC": ['الغضب', 'الترقب', 'الإشمئزاز', 'الخوف', 'الفرح', 'الحب', 'التفاؤل', 'التشاؤم', 'الحزن', 'المفاجئة', 'الثقة']}
for tn, lst in MLC_LBL_DICT.items():
    LABEL_TO_TEXT[tn] = {i: name for i, name in enumerate(lst)}

TASK_TYPE = {"SVREG": "reg", "SEC": "mlc", "MDD": "cls", "XNLI": "cls",
             "OHSD": "cls", "OOLD": "cls", "FID": "cls", "MQ2Q": "cls",
             "TS": "gen", "QA": "gen", "QG": "gen", "EMD": "gen"}

EVAL_METRIC = {"SVREG": "pearson", "SEC": "jaccard", "MDD": "f1", "XNLI": "acc",
               "OHSD": "f1", "OOLD": "f1", "FID": "f1", "MQ2Q": "f1",
               "TS": "rougeL", "QA": "em", "QG": "bleu", "EMD": "bleu"
               }
LOAD_FN = {"SVREG": "load_alue", "SEC": "load_alue", "MDD": "load_alue", "XNLI": "load_alue",
           "OHSD": "load_alue", "OOLD": "load_alue", "FID": "load_alue", "MQ2Q": "load_alue",
           "TS": "load_gen", "QA": "load_gen", "QG": "load_gen", "EMD": "load_gen"
           }

SEQ_PAIR_TASK = {"MQ2Q", "XNLI"}

MODEL_ARCH_MAP = {"bert": {"JABER", "SABER"},
                  "t5": {"AT5S", "AT5B"}
                 }

ALUE_TASKS = ["MQ2Q", "OOLD", "OHSD", "SVREG", "SEC", "FID", "XNLI", "MDD"]

HP_LST = ["per_gpu_train_batch_size", "learning_rate", "dropout_rate"]


def load_alue(raw_dataset_dir, task_name):
    # https://www.alue.org/tasks
    bench_dict = defaultdict(list)

    if task_name == "MQ2Q":
        df_train = pd.read_csv(os.path.join(raw_dataset_dir, "q2q_similarity_workshop_v2.1.tsv"), sep="\t")
        df_test = pd.read_csv(os.path.join(raw_dataset_dir, "q2q_no_labels_v1.0 - q2q_no_labels_v1.0.tsv"), sep="\t")

        for idx, (s1, s2, lbl) in enumerate(zip(df_train["question1"], df_train["question2"], df_train["label"])):
            exp = {"idx": idx, "s_lst": [s1, s2], "lbl": lbl}
            bench_dict["train"].append(exp)
        for s1, s2, idx in zip(df_test["question1"], df_test["question2"], df_test["QuestionPairID"]):
            exp = {"idx": idx, "s_lst": [s1, s2]}
            bench_dict["test"].append(exp)

        # add dev set from translated QQP
        bench_dict["dev"] = load_mq2q_dev()

    if task_name in ["OOLD", "OHSD"]:
        lbl_key = "offensive" if task_name == "OOLD" else "hate"
        filename = os.path.join(raw_dataset_dir, "OSACT2020-sharedTask-%s.txt")

        for portion in ["train", "dev", "test"]:
            names = ["text"] if portion == "test" else ["text", "offensive", "hate"]
            df = pd.read_csv(filename % portion, sep="\t", quotechar='▁', header=None, names=names)
            if portion != "test":
                for idx, (s, lbl) in enumerate(zip(df["text"], df[lbl_key])):
                    exp = {"idx": idx, "s_lst": [s], "lbl": lbl}
                    bench_dict[portion].append(exp)
            else:
                for idx, s in enumerate(df["text"]):
                    exp = {"idx": idx, "s_lst": [s]}
                    bench_dict[portion].append(exp)

    # SVREG
    if task_name == "SVREG":
        filename = os.path.join(raw_dataset_dir, "SemEval2018-Task1-all-data/Arabic/V-reg/2018-Valence-reg-Ar-%s.txt")
        for portion in ["train", "dev", "test"]:
            df = pd.read_csv(filename% portion, sep="\t")
            for idx, (s, lbl) in enumerate(zip(df["Tweet"], df["Intensity Score"])):
                exp = {"idx": idx, "s_lst": [s], "lbl": lbl}
                bench_dict[portion].append(exp)

            if portion == "test":
                for exp in bench_dict["test"]:
                    del exp["lbl"]
            # print(portion, len(bench_dict[portion]))
    if task_name == "SEC":
        filename = os.path.join(raw_dataset_dir, "SemEval2018-Task1-all-data/Arabic/E-c/2018-E-c-Ar-%s.txt")
        for portion in ["train", "dev", "test", "test-gold"]:
            df = pd.read_csv(filename % portion, sep="\t")

            for row in df.to_dict(orient="records"):
                exp = {"idx": row["ID"], "s_lst": [row["Tweet"]]}
                if portion != "test":
                    exp["lbl"] = [int(v) for _, v in list(row.items())[2:]]
                bench_dict[portion].append(exp)

        # print(portion, len(bench_dict["SEC"][portion]))

    # FID
    if task_name == "FID":
        filename = os.path.join(raw_dataset_dir, "IDAT_data", "IDAT_%s_text.csv")
        for portion in ["train", "test"]:
            key = portion if portion == "test" else "training"
            df = pd.read_csv(filename % key, sep=",")
            for idx, (s, lbl) in enumerate(zip(df["text"], df["label"])):
                exp = {"idx": idx, "s_lst": [s], "lbl": lbl}
                bench_dict[portion].append(exp)
                if portion == "test":
                    bench_dict["dev"].append(copy.deepcopy(exp))

    # XNLI
    if task_name == "XNLI":
        filename = os.path.join(raw_dataset_dir, "XNLI", "arabic_%s.tsv")
        for portion in ["train", "dev", "diag"]:
            df = pd.read_csv(filename % portion, sep="\t")
            for idx, s1, s2, lbl in zip(df["pairID"], df["sentence1"], df["sentence2"], df["gold_label"]):
                exp = {"idx": idx, "s_lst": [s1, s2], "lbl": lbl}
                bench_dict[portion].append(exp)
                if portion == "dev":
                    bench_dict["test"].append(copy.deepcopy(exp))

    if task_name == "MDD":
        filename = os.path.join(raw_dataset_dir,
                                "MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021/MADAR-Corpus-26-%s.tsv")

        for portion in ["train", "dev", "test"]:
            df = pd.read_csv(filename % portion, sep="\t", header=None, names=["Text", "label"])
            for idx, (s, lbl) in enumerate(zip(df["Text"], df["label"])):
                exp = {"idx": idx, "s_lst": [s], "lbl": lbl}
                bench_dict[portion].append(exp)
                # if not idx:
                #     print(exp)
            # print(portion, len(bench_dict["MDD"][portion]))

    return bench_dict


def save_alue_leaderboard(raw_dataset_dir, output_dir, task_name, portion, y_pred, score_ext=""):
    # if task_name != "MDD": return

    if task_name == "MQ2Q":
        df_test = pd.read_csv(os.path.join(raw_dataset_dir, "q2q_no_labels_v1.0 - q2q_no_labels_v1.0.tsv"), sep="\t")
        df_preds = pd.DataFrame(data=y_pred, columns=["prediction"], index=df_test["QuestionPairID"])
        df_preds.reset_index(inplace=True)
        df_preds.to_csv(os.path.join(output_dir, "q2q_%s.tsv" % score_ext), index=False, sep="\t")

    if task_name in ["OOLD", "OHSD"]:
        df_preds = pd.DataFrame(data=y_pred, columns=["prediction"])
        df_preds.reset_index(inplace=True)
        k = {"OOLD": "offensive", "OHSD": "hate"}[task_name]
        df_preds.to_csv(os.path.join(output_dir, "%s_%s.tsv" % (k, score_ext)), index=False, header=False, sep="\t")

    if task_name == "SVREG":
        filename = os.path.join(raw_dataset_dir, "SemEval2018-Task1-all-data/Arabic/V-reg/2018-Valence-reg-Ar-%s.txt")
        df_test = pd.read_csv(filename % "test", sep="\t")
        df_preds = pd.DataFrame(data=y_pred, columns=["prediction"], index=df_test["ID"])
        df_preds.reset_index(inplace=True)
        df_preds.to_csv(os.path.join(output_dir,"v_reg_%s.tsv" % score_ext), index=False, sep="\t")

    if task_name == "SEC":
        filename = os.path.join(raw_dataset_dir, "SemEval2018-Task1-all-data/Arabic/E-c/2018-E-c-Ar-%s.txt")
        df_train, df_test = pd.read_csv(filename % "train", sep="\t"), pd.read_csv(filename % "test", sep="\t")
        df_preds = pd.DataFrame(data=y_pred, columns=df_train.columns[2:].tolist(), index=df_test["ID"])
        df_preds.reset_index(inplace=True)
        df_preds.to_csv(os.path.join(output_dir,"E_c_%s.tsv" % score_ext), index=False, sep="\t")

    if task_name == "FID":
        df_test = pd.read_csv(os.path.join(raw_dataset_dir, "IDAT_data", "IDAT_test_text.csv"), sep=",")
        df_preds = pd.DataFrame(data=y_pred, columns=["prediction"], index=df_test["id"])
        df_preds.reset_index(inplace=True)
        df_preds.to_csv(os.path.join(output_dir, "irony_%s.tsv" % score_ext), index=False, sep="\t")

    if task_name == "XNLI":
        fn = "dev" if portion == "test" else portion
        df_test = pd.read_csv(os.path.join(raw_dataset_dir, "XNLI", "arabic_%s.tsv" % fn), sep="\t")
        df_preds = pd.DataFrame(data=y_pred, columns=["prediction"], index=df_test["pairID"])
        df_preds.reset_index(inplace=True)
        fn = "xnli" if portion == "test" else "diagnostic"
        df_preds.to_csv(os.path.join(output_dir, "%s_%s.tsv" % (fn, score_ext)), index=False, sep="\t")

    if task_name == "MDD":
        df_preds = pd.DataFrame(data=y_pred)
        df_preds.to_csv(os.path.join(output_dir, "madar_%s.tsv" % score_ext), index=False, header=False, sep="\t")


def load_mq2q_dev():
    filename = "./raw_datasets/mq2q.dev.tsv"
    if not os.path.exists(filename):
        raise ValueError ("Please check the README to generate MQ2Q dev set")
    lst = []
    for idx, line in enumerate(open(filename)):
        lbl, s1, s2 = line.strip().split("\t")
        lst.append({"idx": len(lst), "s_lst": [s1, s2], "lbl": int(lbl)})
    return lst

#########################
### generative tasks ####
#########################
def load_EASC(raw_dataset_dir):
    input_dir = os.path.join(raw_dataset_dir,  "TS", "EASC")
    exp_lst = []
    for idx, dir_name in enumerate(os.listdir(os.path.join(input_dir, "Articles"))):
        tmp_dir = os.path.join(input_dir, "Articles", dir_name)
        source = " ".join([l.strip() for l in open(os.path.join(tmp_dir, os.listdir(tmp_dir)[0])) if l.strip()])
        tmp_dir = os.path.join(input_dir, "MTurk", dir_name)
        target_lst = []
        for filename in os.listdir(tmp_dir):
            target = " ".join([l.strip() for l in open(os.path.join(tmp_dir, filename)) if l.strip()])
            if not (source.strip() and target.strip()): continue
            target_lst.append(target)

        exp_lst.append({"idx": str(idx), "s_lst": [source], "lbl": target_lst})

    return {"test_easc": exp_lst}


def load_WikiLingua(raw_dataset_dir, seed=42):

    filename = os.path.join(raw_dataset_dir, "TS", "WikiLingua.pkl")
    with open(filename, 'rb') as fp:
        instance_dict = pickle.load(fp)
    exp_lst = []
    for i, doc_dict in enumerate(instance_dict.values()):
        for j, doc in enumerate(doc_dict.values()):
            if not doc["document"].strip() or not doc["summary"].strip():
                continue
            exp_lst.append({"idx": "%s_%s" % (i, j), "s_lst": [doc["document"]], "lbl": doc["summary"]})

    random.seed(seed)
    random.shuffle(exp_lst)
    instance_dict = defaultdict(list)
    idx_train = int(len(exp_lst) * .8)
    instance_dict["train"] = exp_lst[:idx_train]
    idx_dev = int(len(exp_lst) * .9)
    instance_dict["dev"] = exp_lst[idx_train:idx_dev]
    instance_dict["test"] = exp_lst[idx_dev:]

    return instance_dict


def convert_qa(passage, question, answer, task_type="QA"):
    passage_prefix = "النص:"
    if task_type == "QA":
        key = (passage_prefix, passage, "السؤال:", question)
        lbl = answer
    elif task_type == "QG":
        key = (passage_prefix, passage, "الاجابة:", answer)
        lbl = question
    else:
        raise ValueError("Unsupported task type= %s" % task_type)

    return {"s_lst": ["%s %s %s %s" % key], "lbl": lbl}


def load_QA(raw_dataset_dir, task_type="QA"):

    def clean_text(text):
        lst = [u'\u200b', u'\u200c', u'\u200d', u'\u200e', u'\u200f', u'\u202b',
               u'\u202c', u'\u202d', u'\u202e', u'\u202f', '\ufeff']
        for uni in lst:
            text = text.replace(uni, " ")
        return text
    instance_dict = defaultdict(list)
    file_dict = {"SQuAD_translate-train_squad.translate.train.en-ar": "train",
                 "arcd": "test",
                 "dev-context-ar-question-ar": "dev",
                 "test-context-ar-question-ar": "test",
                 "tydiqa.goldp.ar.train": "train",
                 "tydiqa.goldp.ar.dev": "test",
                 "xquad.ar": "test"}
    for filename, portion in file_dict.items():
        for doc in json.load(open(os.path.join(raw_dataset_dir, "QA", "%s.json" % filename)))["data"]:
            for par in doc["paragraphs"]:
                for qus in par["qas"]:
                    if len(qus["answers"]) > 1:
                        print(qus)
                    question = qus["question"].strip()
                    answer = qus["answers"][0]["text"].strip()
                    passage = clean_text(par["context"].strip())
                    if not re.search('[\u0621-\u064a\ufb50-\ufdff\ufe70-\ufefc0-9A-Za-z]', answer):
                        # print(question)
                        # print(qus["answers"], answer)
                        # print()
                        continue
                    dico = convert_qa(passage, question, answer, task_type)
                    dico["idx"] = str(len(instance_dict[portion]))
                    instance_dict[portion].append(dico)

    return instance_dict


def load_emd_dataset():
    from datasets import load_dataset
    all_data = load_dataset("ArabicEmpatheticDialogues.py")
    train_data = all_data['train'].train_test_split(test_size=0.1, seed=42)['train']
    val_data = all_data['train'].train_test_split(test_size=0.1, seed=42)['test']
    dev_data = val_data.train_test_split(test_size=0.5, seed=42)['train']
    test_data = val_data.train_test_split(test_size=0.5, seed=42)['test']

    dico = {"train": train_data, "dev": dev_data, "test": test_data}

    for portion, exp_lst in dico.items():
        dico[portion] = [{"idx": str(idx), "s_lst": [exp["context"]], "lbl": exp["response"]}
                         for idx, exp in enumerate(exp_lst)]

    return dico


def load_gen(raw_dataset_dir, task_name):
    bench_dict = {}

    if task_name == "EMD":
        bench_dict = load_emd_dataset()

    if task_name == "TS":
        bench_dict = load_WikiLingua(raw_dataset_dir)
        bench_dict.update(load_EASC(raw_dataset_dir))

    if task_name == "QA":
        bench_dict = load_QA(raw_dataset_dir, task_type="QA")

    if task_name == "QG":
        bench_dict = load_QA(raw_dataset_dir, task_type="QG")

    return bench_dict

###################################
#### main class data processor ####
###################################

class DataProcessor(object):
    def __init__(self, task_name, model_name):

        self.task_name = task_name
        self.task_type = TASK_TYPE[self.task_name]
        self.model_name = model_name
        self.is_gen = model_name in MODEL_ARCH_MAP["t5"]
        do_lower_case = False
        vocab_file = "./pretrained_models/%s/vocab.txt" % model_name
        arabert_processor = "bert-large-arabertv02"
        self.arabert_prep = ArabertPreprocessor(arabert_processor)
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

        self.label2id, self.id2label = None, None
        self.t5_label2id, self.t5_id2label, self.t5_name2id = None, None, None
        self.t5_entity2id, self.t5_id2entity = None, None

        self.pair_seq_keys = self._t5_pair_seq_keys()

        self.data_dict = {}
        self.y_true, self.y_pred, self.y_logits = defaultdict(list), defaultdict(list), defaultdict(list)
        self.eval_counter = defaultdict(int)

        self.pad_idx = self.tokenizer.convert_tokens_to_ids(["[PAD]"])[0]
        self.bos_idx = self.tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
        self.eos_idx = self.tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
        self.start_sentinel_id, self.end_sentinel_id = 8, 99

    def dump_train_data(self):
        load_fn = LOAD_FN[self.task_name]
        raw_dataset_dir = os.path.join("./raw_datasets")
        bench_dict = globals()[load_fn](raw_dataset_dir, self.task_name)

        # build the mapping for disc tasks
        if self.task_type not in ["gen", "reg"]:
            self.label2id, self.id2label = _get_tag_to_id(bench_dict["train"], self.task_type)

        # initialize mapping
        if self.task_name in LABEL_TO_TEXT:
            self.t5_label2id, self.t5_id2label, self.t5_name2id = self._t5_get_label_mapping()

        for portion, exp_lst in bench_dict.items():
            self._preprocess_data(portion, exp_lst)

    def _preprocess_data(self, portion, exp_lst):
        print(portion)
        if self.is_gen:
            parse_exp_lst = [self.parse_gen_exp(exp_dict) for exp_dict in tqdm(exp_lst)]
            if self.task_type == "gen":
                parse_exp_lst = [_edit_long_gen_seq(self.task_name, item) for item in parse_exp_lst]
                # old_len = len(parse_exp_lst)
                # parse_exp_lst = [item for item in parse_exp_lst if not _drop_long_gen_seq(item, self.task_name)]
                # if len(parse_exp_lst) != old_len:
                #     print("%s examples dropped due to long sequence length" % (old_len-len(parse_exp_lst)))
        elif self.task_type in ["cls", "reg", "mlc"]:
            parse_exp_lst = [self.parse_cls_exp(exp_dict) for exp_dict in tqdm(exp_lst)]
        else:
            raise ValueError("Task Type %s is not supported" % self.task_type)

        feature_dict = {"input_ids": [], "labels": []}

        def _read_single_exp(inst_dict):
            if "labels" not in inst_dict:
                feature_dict["labels"].append([0])
            for k, v in inst_dict.items():
                if k == "labels" and not isinstance(v, list): v = [v]
                if k not in feature_dict: continue
                feature_dict[k].append(v)

        [_read_single_exp(inst_dict) for inst_dict in parse_exp_lst]

        # save arrow dataset
        dir_name = "dataset_%s_%s_%s_%s" % (self.task_name, self.model_name, self.is_gen, portion)
        tmp_dataset_dir = os.path.join("./raw_datasets", dir_name)
        self.save_dataset(tmp_dataset_dir, feature_dict)

        # save exp_lst if is not the train, used for prediction
        if portion != "train":
            self.data_dict[portion] = {"exp_lst": exp_lst, "parse_exp_lst": parse_exp_lst}
            self.set_y_true(portion)

    def save_dataset(self, dataset_dir, feature_dict):

        if not self.is_gen and self.task_type == "reg":
            af = ARROW_FEATURES_REG
        else:
            af = ARROW_FEATURES

        features = datasets.features.Features.from_dict(json.loads(af))
        encoded_dataset = datasets.arrow_dataset.Dataset.from_dict(feature_dict, features=features)

        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
        encoded_dataset.save_to_disk(dataset_dir)

    def set_y_true(self, portion):
        exp_lst, parse_exp_lst = self.data_dict[portion]["exp_lst"], self.data_dict[portion]["parse_exp_lst"]

        if self.task_type == "gen":
            for exp_dict in parse_exp_lst:
                if isinstance(exp_dict["labels"][0], list):
                    self.y_true[portion].append([self._ids_to_text(lst) for lst in exp_dict["labels"]])
                else:
                    self.y_true[portion].append(self._ids_to_text(exp_dict["labels"]))

        if self.task_type in ["reg", "mlc"]:
            if "lbl" not in exp_lst[0]: return
            self.y_true[portion] = [exp_dict["lbl"] for exp_dict in exp_lst]
        if self.task_type == "cls":
            if "lbl" not in exp_lst[0]: return
            self.y_true[portion] = [self.label2id[exp_dict["lbl"]] for exp_dict in exp_lst]

    def parse_gen_exp(self, exp_dict):
        if self.task_type == "gen":
            input_ids = self.tokenize(" ".join(exp_dict["s_lst"]))
            labels = self.tokenize(exp_dict["lbl"])
            return {"input_ids": input_ids, "labels": labels}
        elif self.task_type in ["cls", "mlc", "reg"]:
            input_ids = self._t5_get_input_ids(exp_dict)
            labels = self._t5_get_labels(exp_dict)
            return {"input_ids": input_ids, "labels": labels}

        raise ValueError("Task Type %s is not supported" % self.task_type)

    def parse_cls_exp(self, exp_dict):
        input_ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"])

        for idx, s in enumerate(exp_dict["s_lst"]):
            input_ids += self.tokenize(s)
            if not idx:
                input_ids.append(self.eos_idx)

        dico = {"input_ids": input_ids}
        if "lbl" in exp_dict:
            if self.label2id:
                dico["labels"] = self.label2id[exp_dict["lbl"]]
            else:
                dico["labels"] = exp_dict["lbl"]

        return dico

    def tokenize(self, text, return_ids=True):
        pp_text = self.norm_arabic(text)
        bert_lst = self.tokenizer.tokenize(pp_text)
        if not return_ids: return bert_lst
        return self.tokenizer.convert_tokens_to_ids(bert_lst)

    def norm_arabic(self, text):
        norm_text = self.arabert_prep.preprocess(text)

        if not norm_text.strip():
            norm_text = text
            # raise ValueError("Text `%s` cannot be processed!!!" % text)
        norm_text = normalize_text(norm_text)
        norm_text = norm_text.strip()
        return norm_text

    def _t5_get_labels(self, exp_dict):
        if "lbl" not in exp_dict: return []
        if self.task_type == "cls":
            return self.t5_id2label[self.label2id[exp_dict["lbl"]]]
        elif self.task_type == "mlc":
            return sum([self.t5_id2label[idx] for idx, b in enumerate(exp_dict["lbl"]) if b], [])
        elif self.task_type == "reg":
            return self.tokenize(str(exp_dict["lbl"]))

        raise ValueError("Task Type %s is not supported" % self.task_type)

    def _t5_get_input_ids(self, exp_dict):

        input_ids = copy.deepcopy(self.pair_seq_keys[0]) if self.task_name in SEQ_PAIR_TASK else []

        for idx, s in enumerate(exp_dict["s_lst"]):
            input_ids += self.tokenize(s)
            if not idx and self.task_name in SEQ_PAIR_TASK:
                input_ids += self.pair_seq_keys[1]

        return input_ids

    def _t5_pair_seq_keys(self):
        if self.task_name == "MQ2Q":
            return [self.tokenize("السؤال الأول:"), self.tokenize("السؤال الثاني:")]
        if self.task_name == "XNLI":
            return [self.tokenize("المقدمة:"), self.tokenize("الفرضية:")]

        return None

    def _t5_decode_disc(self, ids):
        if self.eos_idx in ids:
            ids = ids[:ids.index(self.eos_idx)]
        sentinel_id_lst = list(range(self.start_sentinel_id, self.end_sentinel_id))
        skip_set = set([self.pad_idx, self.bos_idx, self.eos_idx, -100] + sentinel_id_lst)
        ids = [idx for idx in ids if idx not in skip_set]
        if self.task_type == "cls": return self._t5_decode_cls(ids)
        if self.task_type == "reg": return self._t5_decode_reg(ids)
        if self.task_type == "mlc": return self._t5_decode_mlc(ids)

        raise ValueError("Task Type %s is not supported" % self.task_type)

    def _t5_decode_gen(self, portion):

        def decode(ids):
            # remove pad ans eos tokens
            if self.eos_idx in ids:
                ids = ids[:ids.index(self.eos_idx)]
            ids = [idx for idx in ids if idx not in [self.pad_idx, -100]]
            text = self._ids_to_text(ids)
            return text

        # parse y_pred
        self.y_pred[portion] = [decode(lst) for lst in self.y_pred[portion]]

    def _ids_to_text(self, decoder_output_ids):
        tok_lst = self.tokenizer.convert_ids_to_tokens(decoder_output_ids)
        tok_lst = [tokenization.printable_text_byte(x) for x in tok_lst]
        lst = [" "]
        for tok in tok_lst:
            if tok.startswith("##"):
                lst[-1] += tok[2:]
            else:
                lst.append(tok)

        text = " ".join(lst).strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _t5_decode_cls(self, decoder_output_ids):
        pred_str = " ".join(list(map(str, decoder_output_ids)))
        if pred_str in self.t5_label2id:
            return self.t5_label2id[pred_str]

        pred_str = self._ids_to_text(decoder_output_ids)
        name = get_close_matches(pred_str, self.t5_name2id.keys())
        if not name: return random.choice(list(self.t5_id2label.keys()))
        return self.t5_name2id[name[0]]

    def _t5_decode_mlc(self, decoder_output_ids):
        pred_str = " ".join(list(map(str, decoder_output_ids)))
        pred = [0] * len(self.t5_label2id)

        for lbl_str, tid in self.t5_label2id.items():
            if lbl_str in pred_str:
                pred[tid] = 1

        return pred

    def _t5_decode_reg(self, decoder_output_ids):
        pred_str = self._ids_to_text(decoder_output_ids).replace(" ", "")
        try:
            return float(pred_str)
        except ValueError:
            return 0.0

    def _t5_get_label_mapping(self):
        if self.task_type == "mlc":
            t5_name2id = {lbl: i for i, lbl in enumerate(MLC_LBL_DICT[self.task_name])}
        else:
            t5_name2id = {text: self.label2id[lbl] for lbl, text in LABEL_TO_TEXT[self.task_name].items()}
        t5_id2label = {idx: self.tokenize(text) for text, idx in t5_name2id.items()}
        t5_label2id = {" ".join(map(str, labels)): lbl for lbl, labels in t5_id2label.items()}

        return t5_label2id, t5_id2label, t5_name2id

    def compute_score(self, portion):

        if self.task_type == "gen":
            scores = self.compute_score_gen(portion)
        else:
            scores = compute_metrics(self.task_name.lower(), self.y_pred[portion], self.y_true[portion])
            scores = {k: round(100 * v, 2) for k, v in scores.items()}

        return scores

    def process_logits(self, portion):

        if not self.is_gen:
            self.y_logits[portion] = np.asarray([item for sublist in self.y_logits[portion] for item in sublist])
            if self.task_type == "reg":
                self.y_logits[portion] = np.squeeze(self.y_logits[portion])

        if self.is_gen:
            self.y_pred[portion] = [item for sublist in self.y_pred[portion] for item in sublist]
            if self.task_type == "gen":
                self._t5_decode_gen(portion)
            else:
                self.y_pred[portion] = [self._t5_decode_disc(arr) for arr in self.y_pred[portion]]
        else:
            self._process_logits_cls(portion)

    def _process_logits_cls(self, portion):
        if self.task_type == "reg":
            self.y_pred[portion] = np.squeeze(self.y_logits[portion])
        elif self.task_type == "mlc":
            self.y_pred[portion] = np.where(self.y_logits[portion] > 0.25, 1, 0).astype(np.int32)
        else:
            self.y_pred[portion] = np.argmax(self.y_logits[portion], axis=-1)

    def final_metric(self, scores):
        return scores[EVAL_METRIC[self.task_name]]

    def reset_pred(self):
        self.y_pred, self.y_logits, self.eval_counter = defaultdict(list), defaultdict(list), defaultdict(tuple)

    def compute_score_gen(self, portion):
        if self.task_name in ["EMD", "QG"]:
            results = {"bleu": round(corpus_bleu(self.y_pred[portion],
                                                 [self.y_true[portion]],
                                                 force=True,
                                                 lowercase=False,
                                                 tokenize='none'
                                                 ).score, 4)}
        elif self.task_name == "QA":
            gold_lst = [text.split('*****') for text in self.y_true[portion]]
            res = evaluate_squad(gold_lst, self.y_pred[portion])
            results = {'em': res["exact_match"], 'f1': res["f1"]}

        elif self.task_name == "TS":
            rouge_types = ["rouge1", "rouge2", "rougeL"]
            scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)
            aggregator = scoring.BootstrapAggregator()

            for ref, pred in zip(self.y_true[portion], self.y_pred[portion]):
                pred = rouge_postprocessor(pred)
                ref = [rouge_postprocessor(r) for r in ref] if isinstance(ref, list) else rouge_postprocessor(ref)
                if not isinstance(ref, list):
                    score = scorer.score(ref, pred)
                else:
                    lst = [(r, scorer.score(r, pred)["rougeL"]) for r in ref]
                    lst.sort(key=lambda x:x[1], reverse=True)
                    score = scorer.score(lst[0][0], pred)

                aggregator.add_scores(score)

            results = aggregator.aggregate()
            results = {key: value.mid.fmeasure * 100 for key, value in results.items()}

        else:
            raise ValueError("Not supported Task= %s" % self.task_name)

        return results


def rouge_postprocessor(text):
    for p in "!.؟?":
        text = text.replace(" %s " % p, " %s\n" % p)
    return text


######################
### Utils Methods ####
######################
def _drop_long_gen_seq(task_name, exp_dict):
    max_seq_len = {"QA": 1024, "QG": 1024, "TS": 1024, "EMD": 80}[task_name]
    return not (len(exp_dict["input_ids"]) < max_seq_len and len(exp_dict["labels"]) < max_seq_len)


def _edit_long_gen_seq(task_name, exp_dict):
    max_seq_len = {"QA": 512, "QG": 512, "TS": 1024, "EMD": 128}[task_name]
    return {"input_ids": exp_dict["input_ids"][:max_seq_len], "labels": exp_dict["labels"][:max_seq_len]}


def _get_tag_to_id(exp_lst, task_type):
    if task_type == "cls":
        tag_to_id = _get_tag_to_id_cls(exp_lst)
    elif task_type in ["mlc", "reg"]:
        tag_to_id = None
    else:
        raise ValueError("Not supported task type=%s" % task_type)

    id_to_tag = {v: k for k, v in tag_to_id.items()} if tag_to_id else None

    return tag_to_id, id_to_tag


def _get_tag_to_id_cls(exp_lst):
    counter = Counter([exp["lbl"] for exp in exp_lst])
    counter = sorted(counter.items(), key= lambda x:(x[1], x[0]), reverse=True)
    tag_to_id = {tag: idx for idx, (tag, _) in enumerate(counter)}
    return tag_to_id


def gather_alue_leaderboard(args):

    pred_dir = os.path.join("./alue_predictions", args.model_name)
    assert os.path.exists(pred_dir)

    group_dict = {}
    for filename in os.listdir(pred_dir):
        if not filename.endswith(".tsv"): continue
        name = filename.split(".tsv")[0]
        idx = name.rindex("_")
        name, score = "%s.tsv" % name[:idx],  float(name[idx+1])
        if name not in group_dict:
            group_dict[name] = (filename, score)
        else:
            if group_dict[name][1] > score:
                group_dict[name] = (filename, score)

    best_dir = os.path.join("./alue_test_submission", args.model_name)
    if not os.path.exists(best_dir):
        os.makedirs(best_dir)

    for name, (filename, _) in group_dict.items():
        scr = os.path.join(pred_dir, filename)
        dst = os.path.join(best_dir, name)
        shutil.copy(scr, dst)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--mode",
                        default="parse_raw",
                        choices=["parse_raw", "gather_alue"],
                        help="Choose which functionality to run")

    parser.add_argument(
        "--model_name",
        default="AT5S",
        type=str,
        help="model_name",
    )

    args = parser.parse_args()

    for task_name in TASK_TYPE.keys():
        # if task_name != "XNLI": continue#not in ["QA", "QG", "TS"]: continue
        print(args.model_name, task_name)
        data_processor = DataProcessor(task_name, args.model_name)
        if data_processor.task_type == "gen" and not data_processor.is_gen: continue
        data_processor.dump_train_data()
        filename = os.path.join("./raw_datasets", "dp.%s.%s.pkl" % (task_name, args.model_name))
        with open(filename, 'wb') as handle:
            pickle.dump(data_processor, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    main()
