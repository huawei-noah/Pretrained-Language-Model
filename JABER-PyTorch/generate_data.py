# 2021.09.29-Modified slightly on Arabic text preprocessing as well as added method for preprocessing
# ALUE data; prepared ALUE test set submission
#              Huawei Technologies Co., Ltd.

# Copyright 2021 Huawei Technologies Co., Ltd.

import sys, re, json, argparse, math
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
from datasets import load_dataset, load_metric
import pyarabic.araby as araby

from datasets import load_from_disk, Features, Sequence, Value, ClassLabel, arrow_dataset

try:
    from cfuzzyset import cFuzzySet as FuzzySet
except ImportError:
    from fuzzyset import FuzzySet

warnings.filterwarnings("ignore")
locale.setlocale(locale.LC_ALL, 'en_US.utf-8')
ar_char = "\u0621-\u064a\ufb50-\ufdff\ufe70-\ufefc"

############################################
#### Arabert Preprocessor Place holder #####
############################################
# from preprocess import ArabertPreprocessor

class ArabertPreprocessor:
    def __init__(self, model_name, use_old_elongation=False, replace_slash_with_dash=False):
        print("Warning!!!! This is just an empty a placeholder class. "
              "Follow instructions in README.md file to obtain the true `ArabertPreprocessor` class")

    def preprocess(self, text):
        return text


def norm_arabic(arabert_prep, text, conf_name):
    norm_text = arabert_prep.preprocess(text)
    if not norm_text.strip(): return None
    norm_text = normalize_text(norm_text, conf_name)
    norm_text = norm_text.strip()
    return norm_text


def normalize_text(pp_text, conf_name):
    # map some weired characters
    mapping = {"ھ": "ه", "گ": "ك", r'\s': " ", "٪": "%", "℃": "C", "·": ".", "…": ".",
               "ا ٔ ": "أ", "ا ٕ ": "إ", "و ٔ ": "ؤ", " ٔ": "ئ", "ٕ ا": "إ",
               'ـــ': '-', 'ـ': '-', ",": "،", "\?": "؟", "“": '"', '”': '"',
               #
               }
    if conf_name == "pretrain_v5":
        mapping.update({"أ": "ا", "إ": "ا", "آ": "ا"})

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
                          },

                 # orca
                 "xlni": {"neutral": "علاقة غير مترابطة", "entailment": "علاقة مترابطة", "contradiction": "علاقة متناقضة"},
                 "mq2q":  {'no': "غير مكرر", 'yes': "مكرر"},
                 "baly-stance": {'unrelated': 'غير مرتبط', 'agree': 'موافق', 'discuss': 'مناقشة', 'disagree': 'غير موافق'},
                 "ans-stance": {'disagree': 'غير موافق', 'agree': 'موافق', 'other': 'غير ذلك'},
                 "wsd": {"1": "نفس المعنى", "0": "معنى مختلف"},
                 "abusive": {'normal': "عادي", 'abusive': "مسيئ", 'hate': 'كراهية'},
                 "sarcasm": {'false_sarcasm': "ليس سخرية", 'true_sarcasm': "سخرية"}, # {'السخرية الكاذبة': 0، 'السخرية الحقيقية': 1}
                 "irony": {'irony': "سخرية", 'NOT': "ليس سخرية"},
                 "offensive": {"NOT_OFF": "غير مهين", "OFF": "مهين"},
                 "hate-speech": {"NOT_HS": "لا يحض على الكراهية", "HS": "خطاب كراهية"},
                 "emotion": {'happy': 'الفرح', 'fear': 'الخوف', 'sad': 'الحزن', 'trust': 'الثقة', 'anticipation': 'الترقب', 'surprise': 'المفاجئة', 'anger': 'الغضب', 'disgust': 'الإشمئزاز'},
                 "dangerous": {'NOT': "غير خطير", 'DANG': "خطير"},
                 "adult": {'NOT_ADULT': "غير بالغ", 'ADULT': "بالغ"},
                 "gender": {'Male': "ذكر", 'Female': "أنثى"},
                 "age": {'25 until 34': '25 حتى 34', 'under 25': 'أقل من 25', '35 and up': '35 وما فوق'},
                 "machine-generation": {'Orginal': "تم إنشاؤها بواسطة الإنسان", 'Machine': "تم إنشاؤها بواسطة الآلة"},
                 "ans-claim": {'Not_fake': 'غير مزيف', 'Fake': 'مزيف'},
                 "topic": {'international_news': "أخبار دولية", 'sports': "رياضة", 'middle_east': "الشرق الأوسط", 'economy': "اقتصاد", 'family': "عائلة",
                           'history': "تاريخ", 'religious': "ديني", 'technology': "تكنولوجيا", 'recipes': "وصفات", 'local_news': 'أخبار محلية',
                           'health': 'صحة', 'culture': 'ثقافة', 'law': 'قانون', 'stories': 'قصص', 'space': 'فضاء'},
                "dialect-country": {'Saudi_Arabia': "المملكة العربية السعودية", 'Egypt': "مصر", 'Kuwait': "الكويت", 'Palestine': "فلسطين", 'Libya': "ليبيا", 'Qatar': "قطر",
                                    'Jordan': "الأردن", 'Lebanon': "لبنان", 'UAE': "الإمارات", 'Bahrain': "البحرين", 'Oman': "عمان", 'Iraq': "العراق",
                                    'Algeria': "الجزائر", 'Sudan': "السودان", 'Yemen': "اليمن", 'Syria': "سوريا", 'Tunisia': "تونس", 'Morocco': "المغرب",
                                    'Somalia': 'الصومال', 'Mauritania': "موريتانيا", 'Djibouti': "جيبوتي"},
                "dialect-binary": {'MSA': "الفصحى", 'DA': "اللهجة"},
                "dialect-region": {'Gulf': "الخليج", 'Egypt': "مصر", 'Levnt': "شرقي", 'Magreb': "المغرب"},
                 "sentiment": {'neg': "سلبي", 'pos': "إيجابي", 'neut': "حيادي"}
                 }

MLC_LBL_DICT = {"SEC": ['الغضب', 'الترقب', 'الإشمئزاز', 'الخوف', 'الفرح', 'الحب', 'التفاؤل', 'التشاؤم', 'الحزن', 'المفاجئة', 'الثقة']}
for tn, lst in MLC_LBL_DICT.items():
    LABEL_TO_TEXT[tn] = {i: name for i, name in enumerate(lst)}

TASK_TYPE = {"SVREG": "reg", "SEC": "mlc", "MDD": "cls", "XNLI": "cls",
             "OHSD": "cls", "OOLD": "cls", "FID": "cls", "MQ2Q": "cls",
             "TS": "gen", "QA": "gen", "QG": "gen", "EMD": "gen",

             # orca
             "arabic-ner": "ner", "aqmar-ner": "ner", "msa-pos": "ner", "dialect-pos": "ner",
             "qa": "qa",

             'sentiment': 'cls', 'dialect-region': 'cls', 'dialect-binary': 'cls', 'dialect-country': 'cls',
             'topic': 'cls', 'ans-claim': 'cls', 'machine-generation': 'cls',
             'age': 'cls', 'gender': 'cls', 'adult': 'cls', 'dangerous': 'cls', 'emotion': 'cls',
             'hate-speech': 'cls', 'offensive': 'cls', 'irony': 'cls',
             'sarcasm': 'cls', 'abusive': 'cls', 'wsd': 'cls', 'ans-stance': 'cls',
             'baly-stance': 'cls', 'mq2q': 'cls', 'xlni': 'cls',

             'sts': 'reg', 'emotion-reg': 'reg'
             }

EVAL_METRIC = {"SVREG": "pearson", "SEC": "jaccard", "MDD": "f1", "XNLI": "acc",
               "OHSD": "f1", "OOLD": "f1", "FID": "f1", "MQ2Q": "f1",
               "TS": "rougeL", "QA": "em", "QG": "bleu", "EMD": "bleu",

               # orca
               "arabic-ner": "f1", "aqmar-ner": "f1", "msa-pos": "f1", "dialect-pos": "f1",
               "qa": "f1",

               'sentiment': 'f1', 'dialect-region': 'f1', 'dialect-binary': 'f1', 'dialect-country': 'f1',
               'topic': 'f1', 'ans-claim': 'f1', 'machine-generation': 'f1',
               'age': 'f1', 'gender': 'f1', 'adult': 'f1', 'dangerous': 'f1', 'emotion': 'f1',
               'hate-speech': 'f1', 'offensive': 'f1', 'irony': 'f1',
               'sarcasm': 'f1', 'abusive': 'f1', 'wsd': 'f1', 'ans-stance': 'f1',
               'baly-stance': 'f1', 'mq2q': 'f1', 'xlni': 'f1',

               'sts': 'spearman', 'emotion-reg': 'spearman'



               }
LOAD_FN = {"SVREG": "load_alue", "SEC": "load_alue", "MDD": "load_alue", "XNLI": "load_alue",
           "OHSD": "load_alue", "OOLD": "load_alue", "FID": "load_alue", "MQ2Q": "load_alue",
           "TS": "load_gen", "QA": "load_gen", "QG": "load_gen", "EMD": "load_gen",

           # orca
           "arabic-ner": "load_orca", "aqmar-ner": "load_orca", "msa-pos": "load_orca", "dialect-pos": "load_orca",
           "qa": "load_orca",

           'sentiment': 'load_orca', 'dialect-region': 'load_orca', 'dialect-binary': 'load_orca',
           'dialect-country': 'load_orca',
           'topic': 'load_orca', 'ans-claim': 'load_orca', 'machine-generation': 'load_orca',
           'age': 'load_orca', 'gender': 'load_orca', 'adult': 'load_orca', 'dangerous': 'load_orca',
           'emotion': 'load_orca',
           'hate-speech': 'load_orca', 'offensive': 'load_orca', 'irony': 'load_orca',
           'sarcasm': 'load_orca', 'abusive': 'load_orca', 'wsd': 'load_orca', 'ans-stance': 'load_orca',
           'baly-stance': 'load_orca', 'mq2q': 'load_orca', 'xlni': 'load_orca',

           'sts': 'load_orca', 'emotion-reg': 'load_orca'

           }

SEQ_PAIR_TASK = {"MQ2Q", "XNLI",
                 'ans-stance', 'baly-stance', "xnli", 'wsd' }

MODEL_ARCH_MAP = {"bert": {"JABER", "SABER", "JABERv2", "JABERv2-6L"},
                  "t5": {"AT5S", "AT5B", "AT5Sv2", "AT5Bv2"}
                 }
MODEL_CONF_MAP = {"JABER": "pretrain_jaber", "SABER": "pretrain_jaber",
                  "AT5S": "pretrain_jaber", "AT5B": "pretrain_jaber",
                  "JABERv2": "pretrain_v5", "JABERv2-6L": "pretrain_v5",
                  "AT5Sv2": "pretrain_v5", "AT5Bv2": "pretrain_v5",
                  }

CONF_MAP = {"pretrain_jaber": {"do_lower_case": False, "arabert_processor": "bert-large-arabertv02"},
            "pretrain_v5": {"do_lower_case": True, "arabert_processor": "bert-large-arabertv02-twitter"}}


ALUE_TASKS = ["MQ2Q", "OOLD", "OHSD", "SVREG", "SEC", "FID", "XNLI", "MDD"]

ORCA_TASKS = [

    'abusive', 'adult', 'age', 'ans-claim', 'dangerous', 'dialect-binary', 'dialect-region',  'dialect-country',
    'emotion',  'emotion-reg', 'gender',   'hate-speech', 'irony', 'offensive', 'machine-generation', 'sarcasm', 'sentiment',
    "arabic-ner", "aqmar-ner",  "dialect-pos","msa-pos",
    'ans-stance', 'baly-stance', 'xlni',
     'sts', 'mq2q', 'topic', "qa", 'wsd',

]


_TOKENS_CLASSIFICATION_TASKS = ['arabic-ner', 'aqmar-ner', 'msa-pos', 'dialect-pos']

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

######################
### orca tasks #######
######################
def _parse_orca_row(task_name, row):
    exp_dict = {}
    _MULTIPLE_INPUTS_TASKS_cols = {
        'mq2q': ("question1", "question2"),
        'ans-stance': ("s1", "s2"),
        'baly-stance': ("claim", "article"),
        'xlni': ("sentence1", "sentence2"),
        'sts': ("sentence1", "sentence2"),
        'emotion-reg': ("emotion", "content"),
        'wsd': ("sense", "sentence"),
    }
    AQMAR_MAP = {"I--ORG": "I-ORG", 'B-MISS1': 'B-MIS1',
                 'B-MIS1`': 'B-MIS1', 'B-MIS-1': 'B-MIS1', 'B-MIS-2': 'B-MIS2'}
    if task_name in _MULTIPLE_INPUTS_TASKS_cols:
        exp_dict["s_lst"] = [row[key] for key in  _MULTIPLE_INPUTS_TASKS_cols[task_name]]
        exp_dict["lbl"] = row["label"]
    elif task_name in _TOKENS_CLASSIFICATION_TASKS:
        exp_dict["tokens"] = row["tokens"]
        exp_dict["tags"] = row["tags"]
        if task_name == "aqmar-ner":
            exp_dict["tags"] = [AQMAR_MAP[t] if t in AQMAR_MAP else t for t in exp_dict["tags"]]
    elif task_name == "qa":
        exp_dict = copy.deepcopy(row)
        del exp_dict["id"]
        del exp_dict["title"]
        exp_dict["id"] = row["id"]
    else:
        if not row["content"].strip():
            row["content"] = "N/A"
        exp_dict["s_lst"], exp_dict["lbl"] = [row["content"]], row["label"]

    return exp_dict


def load_orca(dataset_dir, task_name):

    bench_dict = defaultdict(list)
    dataset = datasets.load_dataset("UBC-NLP/orca", task_name)

    for portion, rows in dataset.items():
        # if portion != "validation": continue
        for idx, row in enumerate(rows):
            exp_dict = _parse_orca_row(task_name, row)
            exp_dict["idx"] = idx

            if task_name == "mq2q" and portion != "test" and not exp_dict["lbl"]:
                exp_dict["lbl"] = "no"

            if portion == "test" and "lbl" in exp_dict: del exp_dict["lbl"]
            if portion == "validation": portion = "dev"
            bench_dict[portion].append(exp_dict)

    return bench_dict

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

###################
### QA context ####
###################
def split_qa_context(context, max_len=80):
    split_idx_lst = [m.end()-1 for m in re.finditer(f'[\d{ar_char} ]{3,}\. ', context)] + [len(context)]
    begin = 0
    counter = 0

    context_tmp, context_lst = "", []

    for idx in split_idx_lst:
        c = context[begin:idx].count(" ")
        if counter + c > max_len:
            context_lst.append(context_tmp)
            counter = 0
            context_tmp = ""

        context_tmp += context[begin:idx]
        counter += c
        begin = idx+1

    if context_tmp:
        context_lst.append(context_tmp)
    if not context_lst:
        context_lst.append(context)

    return context_lst

def fix_qa_ans_offset(context, answer_text, answer_start):

    begin, end = answer_start, answer_start+len(answer_text)

    # remove tags
    for i in reversed(range(begin, end)):
        if context[i] in "</":
            end -= 1
        elif context[i] == " ":
            continue
        else:
            break

    if end < len(context) and context[end-1] == " ": end -= 1
    if context[begin] == " ": begin += 1


    # fix end
    for i in range(end, len(context)):
        if not re.match(f'[{ar_char}A-Za-z0-9]', context[i]) and context[i] not in araby.TASHKEEL: break
        end += 1

    # fix begin
    for i in reversed(range(0, begin)):
        if not re.match(f'[{ar_char}A-Za-z0-9]', context[i]) and context[i] not in araby.TASHKEEL: break
        begin -= 1

    # if answer_text != context[begin:end]:
    #     print(answer_text)
    #     print(context[begin:end])
    #     print(context)
    #     print()

    answer_text = context[begin:end]

    if answer_text == "unlabeled":
        answer_text = ""
        begin = 0

    if len(answer_text) > 1 and answer_text[-1] in ".,،":
        answer_text = answer_text[:-1]

    return answer_text, begin

def find_qa_idx(context, answer_text, s_idx, cxt_ii, ans_ii):
    ctx_str = " ".join(map(str, cxt_ii))
    ctx_str = " " + ctx_str + " "
    ans_str = " ".join(map(str, ans_ii))

    s_idx_lst = [ctx_str[:m.start()].count(" ") for m in re.finditer(" "+ans_str+" ", ctx_str)]
    idx_lst = [(s, s+len(ans_ii)) for s in s_idx_lst]
    reg_skip = f"[^{ar_char}A-Za-z0-9]"
    b_idx, b_idx_lst = 0, [("\b", "\b"),
                           (reg_skip, reg_skip), (reg_skip, ""), ("", reg_skip),
                           ("\b", ""), ("", "\b"), ("", "")]
    if not idx_lst:
        return -1, -1
        # print("Answer Text", answer_text)
        # print(context)
        # print()
    if len(idx_lst) > 1:
        m_lst = []
        while not m_lst:
            p, s = b_idx_lst[b_idx]
            m_lst = [m.start(0) for m in re.finditer(r"%s(%s)%s" % (p, re.escape(answer_text), s), context)]
            b_idx += 1

        abs_lst = [math.fabs(m-s_idx) for m in m_lst]
        min_idx = abs_lst.index(min(abs_lst))
        if len(m_lst) == len(idx_lst):
            idx_lst = [idx_lst[min_idx]]
        else:
            ratio = m_lst[min_idx] /len(context)
            l = [math.fabs(ratio - (j/len(cxt_ii))) for j in s_idx_lst]
            new_min_idx = l.index(min(l))
            idx_lst = [idx_lst[new_min_idx]]
            # print("Multiple answers", answer_text)
            # print(ans_ii)
            # print(context)
            # print(m_lst.index(min(m_lst)), new_min_idx, len(m_lst), len(idx_lst))
            # print()

    return idx_lst[0]

def qa_get_top_pred(input_ids, start_logits, end_logits, sep_idx=3, n_best_size=5, max_answer_length=30):
    skip_before = input_ids.index(sep_idx)

    # Update minimum null prediction.
    feature_null_score = start_logits[0] + end_logits[0]
    min_null_prediction = {
        "offsets": (0, 1),
        "score": feature_null_score,
        "start_logit": start_logits[0],
        "end_logit": end_logits[0],
    }

    prelim_predictions = []

    # Go through all possibilities for the `n_best_size` greater start and end logits.
    # start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
    # end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
    start_indexes = np.argsort(start_logits).tolist()[::-1][:n_best_size]
    end_indexes = np.argsort(end_logits).tolist()[::-1][:n_best_size]
    for start_index in start_indexes:
        for end_index in end_indexes:
            # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
            # to part of the input_ids that are not in the context.
            if (
                    start_index >= len(input_ids)
                    or end_index >= len(input_ids)
                    or start_index <= skip_before
                    or end_index <= skip_before

            ):
                continue
            # Don't consider answers with a length that is either < 0 or > max_answer_length.
            if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                continue

            prelim_predictions.append(
                {
                    "offsets": (start_index, end_index+1),
                    "score": start_logits[start_index] + end_logits[end_index],
                    "start_logit": start_logits[start_index],
                    "end_logit": end_logits[end_index],
                }
            )

    prelim_predictions.append(min_null_prediction)
    prelim_predictions.sort(key= lambda x:x["score"], reverse=True)

    return prelim_predictions


def qa_locate_pred(context_raw, ans_str):
    n = ans_str.count(" ")
    words = context_raw.split(" ")
    from nltk import ngrams
    a = FuzzySet()
    for j in range(1, max(2, n+2)):
        for item in ngrams(words, j):
            a.add(" ".join(item))

    lst = a.get(ans_str)
    if not lst:
        return ""
    else:
        s = lst[0][1]
        if len(s) > 1 and s[-1] in ".,،":
            s = s[:-1]
        return s


##########################
### NER methods ##########
##########################

def load_ner_file(filename, sep=" "):
    sentences, tokens, tags = [], [], []
    for line in open(filename, 'r').readlines():
        if line.strip():
            # print(line.strip())
            token, tag = line.strip().split(sep)
            tokens.append(token)
            tags.append(tag)
        elif tokens:
            sentences.append({"tokens": tokens, "tags": tags})
            tokens, tags = [], []

    if tokens:
        sentences.append({"tokens": tokens, "tags": tags})

    for idx, sent in enumerate(sentences):
        seq_lst = split_long_sequence(sent["tokens"], sent["tags"], max_seq_len=128)

        for sent_part_num, (tok_lst, tag_lst) in enumerate(seq_lst):
            if not tok_lst: continue
            sentences[idx] = {"idx": "%s.%s" % (idx, sent_part_num), "tokens": tok_lst, "tags": tag_lst}

    return sentences


def split_long_sequence(tokens, tags, max_seq_len=128):
    if len(tokens) < max_seq_len:
        return [(tokens, tags)]

    s = "".join([t[1:1] for t in tags])
    pick_lst = [m.start(0) + 2 for m in re.finditer("OOOOO", s)]

    if not pick_lst:
        mid = len(tokens) // 2
    else:
        index_lst = [(i, math.fabs(i - len(tokens) // 2)) for i in pick_lst]
        index_lst.sort(key=lambda x: x[1])
        mid = index_lst[0][0]

    l1 = split_long_sequence(tokens[:mid], tags[:mid])
    l2 = split_long_sequence(tokens[mid:], tags[mid:])

    return l1 + l2


def split_long_sequence_ner(input_ids, positions, tags, max_seq_len=256):
    if len(input_ids) < max_seq_len:
        return [(input_ids, positions, tags)]

    s = "".join([t[1:1] for t in tags])
    pick_lst = [m.start(0) + 2 for m in re.finditer("OOOOO", s)]

    if not pick_lst:
        mid = len(positions) // 2
    else:
        index_lst = [(i, math.fabs(i - len(positions) // 2)) for i in pick_lst]
        index_lst.sort(key=lambda x: x[1])
        mid = index_lst[0][0]

    l1 = split_long_sequence_ner(input_ids[:positions[mid]], positions[:mid], tags[:mid], max_seq_len)
    new_positions = [i-positions[mid] for i in positions[mid:]]
    l2 = split_long_sequence_ner(input_ids[positions[mid]:], new_positions, tags[mid:], max_seq_len)

    return l1 + l2


def get_pred_tags(mentions, exp_dict):

    input_ids, positions = exp_dict["input_ids"], exp_dict["positions"]
    wp_to_positions = {}
    for i in range(len(positions)):
        b, e = positions[i], positions[i+1] if i != len(positions)-1 else len(input_ids)
        for j in range(b, e):
            wp_to_positions[j] = i
    # find all occurrences of generated span
    # this will automatically discard invalid span (e.g. non consecutive ones)
    # also this will lead to potential multiple matches
    men_index_lst = [_find_sub_list(wp_lst, input_ids) for wp_lst, _ in mentions]

    # discard non found span mentions
    lst = [(men_index_lst[i], mentions[i][1]) for i in range(len(mentions)) if men_index_lst[i]]
    # convert wp indexes to word indexes
    mentions = []

    for idx, (index_lst, tag) in enumerate(lst):
        index_lst = [sorted(list(set([wp_to_positions[j] for j in range(b, e)]))) for b, e in index_lst]
        index_lst = [(lst[0], lst[-1]+1) for lst in index_lst]
        if len(index_lst) == 1: index_lst = index_lst[0]
        mentions.append((index_lst, tag))

    word_len = len(positions)
    for i in range(2):
        mentions = _remove_ambiguous_mention(mentions, word_len)
        mentions = _remove_duplicate_mentions(mentions)
        mentions = _remove_embedded_mentions(mentions)

    mentions = _solve_mention_ambiguity(mentions)
    pred_tags = mentions_to_tags(mentions, word_len)

    return pred_tags


def _remove_embedded_mentions(mentions):
    # remove embedded entities of same type
    lst = []
    for m_idx, men in enumerate(mentions):
        index_lst, tag = men
        if not isinstance(index_lst, list): index_lst = [index_lst]
        for i_idx, (b, e) in enumerate(index_lst):
            item = [tag] + list(range(b, e)) + [m_idx, i_idx]
            lst.append(item)
    lst.sort(key=lambda x: len(x), reverse=True)

    done = set()
    rm_set = set()
    for i in range(len(lst)):
        tag, sub_lst, m_idx, i_idx = lst[i][0], lst[i][1:-2], lst[i][-2], lst[i][-1]

        if all([(k, tag) in done for k in sub_lst]):
            rm_set.add((m_idx, i_idx))
        else:
            done.update([(k, tag) for k in sub_lst])

    rm_men_id = set()
    for m_idx, i_idx in rm_set:
        if isinstance(mentions[m_idx][0], tuple):
            rm_men_id.add(m_idx)
            continue
        else:
            index_lst = mentions[m_idx][0]
            index_lst = index_lst[:i_idx] + index_lst[i_idx+1:]
            if len(index_lst) == 1: index_lst = index_lst[0]
            mentions[m_idx] = (index_lst, mentions[m_idx][1])

    mentions = [men for idx, men in enumerate(mentions) if idx not in rm_men_id]
    mentions = sorted(mentions, key=lambda x: x[0][0] if isinstance(x[0], tuple) else x[0][0][0])

    return mentions


def _remove_duplicate_mentions(mentions):

    # remove duplicate entity
    rm_set, new_mentions = set(), []
    for i in range(len(mentions)):
        if i in rm_set: continue
        lst = [j for j in range(i + 1, len(mentions)) if mentions[j] == mentions[i]]
        if lst:
            #assert len(lst) + 1 == len(mentions[i][0])
            if not isinstance(mentions[i][0], list): mentions[i] = ([mentions[i][0]], mentions[i][1])
            new_mentions += [((b, e), mentions[i][1]) for b, e in mentions[i][0]]
            rm_set.update(lst + [i])

    mentions = [men for idx, men in enumerate(mentions) if idx not in rm_set]
    mentions += new_mentions
    mentions = sorted(mentions, key=lambda x: x[0][0] if isinstance(x[0], tuple) else x[0][0][0])

    return mentions


def _find_sub_list(sl, l):
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            results.append((ind, ind + sll))# - 1

    return results


def _remove_ambiguous_mention(mentions, word_len, max_try=3):
    multi_span_lst = [i for i, men in enumerate(mentions) if isinstance(men[0], list)]

    while multi_span_lst and max_try:
        rm_set = set()
        for idx in multi_span_lst:

            start_after, end_before = -1, word_len

            # get nearest preceding index
            for j in reversed(range(idx)):
                if isinstance(mentions[j][0], list): continue
                start_after = mentions[j][0][1]
                break

            # get nearest proceeding index
            for j in range(idx+1, len(mentions)):
                if isinstance(mentions[j][0], list): continue
                end_before = mentions[j][0][0]
                break

            index_lst = mentions[idx][0]
            lst = [(i, j) for i, j in index_lst if i >= start_after and j <= end_before]
            if len(lst) == 1:
                rm_set.add(idx)
                mentions[idx] = (lst[0], mentions[idx][1])

        multi_span_lst = [idx for idx in multi_span_lst if idx not in rm_set]
        max_try -= 1

    return mentions


def _solve_mention_ambiguity(mentions):
    multi_span_lst = [i for i, men in enumerate(mentions) if isinstance(men[0], list)]
    # select all ambiguous spans!!!!
    if multi_span_lst:
        for idx in multi_span_lst:
            index_lst, tag = mentions[idx]
            for begin, end in index_lst:
                mentions.append({"begin": begin, "end": end, "tag": tag})

        multi_span_lst = set(multi_span_lst)
        mentions = [men for idx, men in enumerate(mentions) if idx not in multi_span_lst]

    for idx, men in enumerate(mentions):
        if isinstance(men, dict): break
        mentions[idx] = {"begin": men[0][0], "end": men[0][1], "tag": men[1]}

    mentions = sorted(mentions, key=lambda x: x["begin"])

    return mentions


def tags_to_mentions(tag_lst, is_iob=True):
    if not is_iob: raise ValueError("Input should be in IOB format")
    mentions = []
    begin, value = -1, None

    for idx, tag in enumerate(tag_lst):
        if (begin == -1 and tag.startswith("I-")) or \
                (begin != -1 and tag.startswith("I-") and not tag.endswith(value)):
            tag = "B-%s" % tag[2:]
            tag_lst[idx] = tag
        if begin != -1 and tag != 'I-%s' % value:
            men = {"begin": begin, "end": idx, "tag": value}
            mentions.append(men)
            begin, value = -1, None

        if tag.startswith('B-'):
            begin, value = idx, tag[2:]

    if begin != -1:
        men = {"begin": begin, "end": len(tag_lst), "tag": value}
        mentions.append(men)

    return mentions, tag_lst



def mentions_to_tags(mentions, word_len, is_iob=True):
    if not is_iob: raise ValueError("output will be in IOB format")
    tag_lst = ["O"] * word_len

    if not mentions: return tag_lst

    for men in mentions:
        for i in range(men["begin"], men["end"]):
            ext = "B-" if i == men["begin"] else "I-"
            tag_lst[i] = ext + men["tag"]

    return tag_lst

def get_positions(bert_lst):
    rnd = np.random.uniform(size=(len(bert_lst),))
    is_skip = []
    for idx, t in enumerate(bert_lst):
        c = re.search(f'[{ar_char}A-Za-z0-9]', t)
        is_skip.append(not c and rnd[idx] < 0.5)

    counter = 0
    positions = [0] * len(bert_lst)
    # prv_is_sub = False
    for idx, t in enumerate(bert_lst):
        if is_skip[idx]: continue
        if not t.startswith("##"): counter += 1
        # if prv_is_sub and not t.startswith("##"): counter += 1
        # if not prv_is_sub and not t.startswith("##"): counter += 1
        # prv_is_sub = t.startswith("##")
        positions[idx] = counter

    return positions
###################################
#### main class data processor ####
###################################

class DataProcessor(object):
    def __init__(self, task_name, model_name, is_gen):

        self.task_name = task_name
        self.task_type = TASK_TYPE[self.task_name]
        self.model_name = model_name
        self.is_gen = is_gen #model_name in MODEL_ARCH_MAP["t5"]

        self.conf_name = MODEL_CONF_MAP[model_name]
        arabert_processor = CONF_MAP[MODEL_CONF_MAP[model_name]]["arabert_processor"]

        use_old_elongation = self.conf_name == "pretrain_jaber" or task_name in ["XNLI", "MDD"]
        replace_slash_with_dash = not (self.conf_name == "pretrain_jaber" or task_name in ["XNLI", "MDD"])

        self.arabert_prep = ArabertPreprocessor(arabert_processor,
                                                use_old_elongation=use_old_elongation,
                                                replace_slash_with_dash=replace_slash_with_dash)

        vocab_file = "./pretrained_models/%s/vocab.txt" % model_name
        do_lower_case = CONF_MAP[MODEL_CONF_MAP[model_name]]["do_lower_case"]
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
        if self.task_type not in ["gen", "reg", "qa"]:
            self.label2id, self.id2label = get_tag_to_id(bench_dict["train"], self.task_type)

        # initialize mapping
        if self.task_name in LABEL_TO_TEXT:
            self.t5_label2id, self.t5_id2label, self.t5_name2id = self._t5_get_label_mapping()
        if self.task_type == "ner" and self.is_gen:
            self.t5_entity2id, self.t5_id2entity = self._t5_ner_entity2id()

        # bench_dict["train"] = bench_dict["dev"]
        for portion, exp_lst in bench_dict.items():
            # if portion != "test": continue
            self._preprocess_data(portion, exp_lst)

    def _preprocess_data(self, portion, exp_lst):
        print(portion)
        if self.is_gen:
            # exp_lst = exp_lst[:1000]
            parse_exp_lst = [self.parse_gen_exp(exp_dict) for exp_dict in tqdm(exp_lst)]
        elif self.task_type in ["cls", "reg", "mlc"]:
            parse_exp_lst = [self.parse_cls_exp(exp_dict) for exp_dict in tqdm(exp_lst)]
        elif self.task_type == "ner":
            parse_exp_lst = [self.parse_ner_exp(exp_dict) for exp_dict in tqdm(exp_lst)]
        elif self.task_type == "qa":
            # exp_lst = exp_lst[:1000]
            parse_exp_lst = [self.parse_qa_exp(exp_dict) for exp_dict in tqdm(exp_lst)]
        else:
            raise ValueError("Task Type %s is not supported" % self.task_type)

        feature_dict = {"input_ids": [], "labels": []}
        if not self.is_gen:
            feature_dict["positions"] = []

        def _read_single_exp(inst_dict):
            if "labels" not in inst_dict:
                if self.task_type == "mlc":
                    feature_dict["labels"].append([0] * len(MLC_LBL_DICT[self.task_name]))
                else:
                    feature_dict["labels"].append([0])
            for k, v in inst_dict.items():
                if k == "labels" and not isinstance(v, list): v = [v]
                if k not in feature_dict: continue
                feature_dict[k].append(v)

        if self.task_type in ["ner", "qa"]:
            for sent_inst_lst in parse_exp_lst:
                [_read_single_exp(inst_dict) for inst_dict in sent_inst_lst]
        else:
            [_read_single_exp(inst_dict) for inst_dict in parse_exp_lst]

        # save arrow dataset
        dir_name = "dataset_%s_%s_%s_%s" % (self.task_name, self.model_name, self.is_gen, portion)
        tmp_dataset_dir = os.path.join("./raw_datasets", dir_name)
        self.save_dataset(tmp_dataset_dir, feature_dict)

        # save exp_lst if is not the train, used for prediction
        if portion != "train":
            self.data_dict[portion] = {"exp_lst": exp_lst, "parse_exp_lst": parse_exp_lst}
            self.set_y_true(portion)

    def save_dataset(self, dataset_dir, feature_dict, with_logits=False):
        lbl_dtype = "float32" if not self.is_gen and self.task_type == "reg" else "int32"
        meta_data_features = {
            'input_ids': Sequence(feature=Value(dtype='int32', id=None), length=-1, id=None),
            'positions': Sequence(feature=Value(dtype='int32', id=None), length=-1, id=None),
            'labels': Sequence(feature=Value(dtype=lbl_dtype, id=None), length=-1, id=None),
        }

        if self.is_gen:
            del meta_data_features['positions']
        if with_logits:
            meta_data_features["logits"] = Sequence(feature=Value(dtype='float32', id=None), length=-1, id=None)

        features = Features(meta_data_features)
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

        if self.task_type == "ner":
            if "tags" not in exp_lst[0]: return
            self.y_true[portion] = [exp_dict["tags"] for exp_dict in exp_lst]
        if self.task_type in ["reg", "mlc"]:
            if "lbl" not in exp_lst[0]: return
            self.y_true[portion] = [exp_dict["lbl"] for exp_dict in exp_lst]
        if self.task_type == "cls":
            if "lbl" not in exp_lst[0]: return
            self.y_true[portion] = [self.label2id[exp_dict["lbl"]] for exp_dict in exp_lst]
        if self.task_type == "qa":
            self.y_true[portion] = [{"id": exp_dict["id"], "answers": exp_dict["answers"]} for exp_dict in exp_lst]

    def parse_gen_exp(self, exp_dict):
        if self.task_type == "gen":
            input_ids = self.tokenize(" ".join(exp_dict["s_lst"]))
            labels = self.tokenize(exp_dict["lbl"])
            return {"input_ids": input_ids, "labels": labels}
        elif self.task_type in ["cls", "mlc", "reg"]:
            input_ids = self._t5_get_input_ids(exp_dict)
            labels = self._t5_get_labels(exp_dict)
            return {"input_ids": input_ids, "labels": labels}
        elif self.task_type == "ner":
            t5_exp_lst = self.t5_parse_ner(exp_dict)
            return t5_exp_lst
        elif self.task_type == "qa":
            t5_exp_lst = self.t5_parse_qa(exp_dict)
            return t5_exp_lst

        raise ValueError("Task Type %s is not supported" % self.task_type)

    def parse_cls_exp(self, exp_dict):
        # input_ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"])
        input_ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"])
        skip_lst = [0]
        for idx, s in enumerate(exp_dict["s_lst"]):
            input_ids += self.tokenize(s)
            if not idx:
                skip_lst.append(len(input_ids))
                input_ids += self.tokenizer.convert_tokens_to_ids(["[SEP]"])

        bert_lst = [tokenization.printable_text_byte(x) for x in self.tokenizer.convert_ids_to_tokens(input_ids)]
        positions = get_positions(bert_lst)
        positions = [p if i not in skip_lst else 0 for i, p in enumerate(positions)]

        dico = {"input_ids": input_ids, "positions": positions}
        if "lbl" in exp_dict:
            if self.label2id:
                dico["labels"] = self.label2id[exp_dict["lbl"]]
            else:
                dico["labels"] = exp_dict["lbl"]

        return dico

    def parse_qa_exp(self, exp_dict):
        output_lst = []

        cls_index = self.tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
        sep_index = self.tokenizer.convert_tokens_to_ids(["[SEP]"])[0]

        qus_ii = self.tokenize(exp_dict['question'])
        answer_text, answer_start = exp_dict["answers"]["text"][0], exp_dict["answers"]["answer_start"][0]
        answer_text, answer_start_char = fix_qa_ans_offset(exp_dict["context"], answer_text, answer_start)
        ans_ii = self.tokenize(answer_text) if answer_text.strip() else []
        context_lst = split_qa_context(exp_dict["context"], max_len=384)
        char_start = 0
        is_ans_missing = True and len(ans_ii) > 0

        for i, context in enumerate(context_lst):
            sp, ep = 0, 0
            cxt_ii = self.tokenize(context)

            input_ids = [cls_index] + qus_ii + [sep_index]
            if ans_ii and char_start <= answer_start_char < len(context):
                s_idx = answer_start_char - char_start
                sp, ep = find_qa_idx(context, answer_text, s_idx, cxt_ii, ans_ii)
                if sp != -1 and ep != -1:
                    sp += len(input_ids)
                    ep += len(input_ids) - 1  # TODO this is to be systematic everywhere
                    is_ans_missing = False
                else:
                    sp, ep = 0, 0

            input_ids += cxt_ii + [sep_index]
            char_start += len(context)

            dico = \
                {
                    "idx": "%s.%s" % (exp_dict["idx"], i),
                    "input_ids": input_ids,
                    "positions": [0] * len(input_ids),
                    "labels": [sp, ep],
                    "context": context,
                    "answer": answer_text if sp != 0 and ep != 0 else ""
                }
            output_lst.append(dico)

        # if is_ans_missing:
        #     print(exp_dict)
        return output_lst

    def parse_ner_exp(self, exp_dict):

        pp_text = [norm_arabic(self.arabert_prep, token, self.conf_name) for token in exp_dict["tokens"]]
        pp_text = [pt if pt else "," for pt in pp_text]
        bert_lst = [self.tokenizer.tokenize(pp_token) for pp_token in pp_text]
        ner_max_seq_len = 180
        positions = [0]
        for i in range(1, len(bert_lst)):
            positions.append(positions[-1] + len(bert_lst[i - 1]))
        input_ids = [self.tokenizer.convert_tokens_to_ids(bl) for bl in bert_lst]
        input_ids = [x for xs in input_ids for x in xs]

        multi_lst = split_long_sequence_ner(input_ids, positions, exp_dict["tags"], ner_max_seq_len - 2)
        output_lst = []

        for i, (input_ids, positions, target) in enumerate(multi_lst):
            input_ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"]) + input_ids
            input_ids += self.tokenizer.convert_tokens_to_ids(["[SEP]"])
            positions = [p + 1 for p in positions]
            target = [self.label2id[tag] for tag in target]
            assert len(input_ids) <= ner_max_seq_len
            assert len(target) == len(positions)

            dico = {"idx": "%s.%s" % (exp_dict["idx"], i), "input_ids": input_ids, "positions": positions,
                    "labels": target}

            output_lst.append(dico)

        return output_lst

    def tokenize(self, text, return_ids=True):
        pp_text = norm_arabic(self.arabert_prep, text, self.conf_name)
        if not pp_text:
            # print("Warning!!! The post process text is `None` while the original one is not.")
            # print("Original Text", text)
            pp_text = text
        bert_lst = self.tokenizer.tokenize(pp_text)
        if not return_ids: return bert_lst
        return self.tokenizer.convert_tokens_to_ids(bert_lst)

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



    def t5_parse_ner(self, exp_dict):
        output_lst = []

        dico_lst = self.parse_ner_exp(exp_dict)
        for dico in dico_lst:
            input_ids, positions, iob_labels = dico["input_ids"], dico["positions"], dico["labels"]

            # get word level mentions
            iob_target = [self.id2label[l] for l in iob_labels]
            mentions, tag_lst = tags_to_mentions(iob_target)

            # labels
            labels = []
            for men in mentions:
                wp_begin = positions[men["begin"]]
                wp_end = positions[men["end"]] if men["end"] < len(positions) else len(input_ids)

                # copy input_ids
                labels += [input_ids[i] for i in range(wp_begin, wp_end)]

                # add labels special token at the end
                labels.append(self.t5_entity2id[men["tag"]])

            dico = {"positions": positions, "tags": tag_lst,
                    "input_ids": input_ids, "labels": labels}
            output_lst.append(dico)

        return output_lst

    def t5_parse_qa(self, exp_dict):
        cls_seq, sep_seq = self.pair_seq_keys
        qus_ii = self.tokenize(exp_dict['question'])
        answer_text, answer_start = exp_dict["answers"]["text"][0], exp_dict["answers"]["answer_start"][0]
        answer_text, answer_start_char = fix_qa_ans_offset(exp_dict["context"], answer_text, answer_start)
        labels = self.tokenize(answer_text) if answer_text.strip() else []
        context_lst = split_qa_context(exp_dict["context"], max_len=464)

        output_lst = []
        for i, context in enumerate(context_lst):
            cxt_ii = self.tokenize(context)
            input_ids = cls_seq + qus_ii + sep_seq + cxt_ii

            dico = \
                {
                    "idx": "%s.%s" % (exp_dict["idx"], i),
                    "input_ids": input_ids,
                    "labels": labels,
                    "context": context,
                    "answer": answer_text
                }
            output_lst.append(dico)

        return output_lst

    def _t5_pair_seq_keys(self):
        if self.task_name == "MQ2Q":
            return [self.tokenize("السؤال الأول:"), self.tokenize("السؤال الثاني:")]
        if self.task_name == "XNLI":
            return [self.tokenize("المقدمة:"), self.tokenize("الفرضية:")]
        # orca
        if self.task_name == "mq2q":
            return [self.tokenize("السؤال الأول:"), self.tokenize("السؤال الثاني:")]
        if self.task_name == "xlni":
            return [self.tokenize("المقدمة:"), self.tokenize("الفرضية:")]
        if self.task_name in ['ans-stance', 'baly-stance', 'wsd']:
            return [self.tokenize("الجملة الأولى:"), self.tokenize("الجملة الثانية:")]
        if self.task_name == "qa":
            return [self.tokenize("السؤال:"), self.tokenize("النص:")]
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

    def _t5_decode_ner(self, decoder_output_ids, portion):
        # print(decoder_output_ids)
        # remove pad ans eos tokens

        def clean_tok(t):
            if "[" in t and "nu" in t and "se" in t:
                return t.replace("0x", "")
            return t

        if self.eos_idx in decoder_output_ids:
            decoder_output_ids = decoder_output_ids[:decoder_output_ids.index(self.eos_idx)]
        decoder_output_ids = [idx for idx in decoder_output_ids if idx not in [self.pad_idx, -100]]

        # start extracting entities
        mentions = []
        wp_lst = []
        # print(decoder_output_ids)
        # last span is discarded if no tag is predicted at the end
        for wp_id in decoder_output_ids:
            if wp_id in self.t5_id2entity:
                tag = self.t5_id2entity[wp_id]
                # avoid adding empty mentions
                if wp_lst:
                    mentions.append((wp_lst, tag))
                    wp_lst = []
            else:
                wp_lst.append(wp_id)

        if not self.eval_counter[portion]:
            i, j = (0, 0)
        else:
            i, j = self.eval_counter[portion]
            if j+1 < len(self.data_dict[portion]["parse_exp_lst"][i]):
                i, j = i, j+1
            else:
                i, j = i+1, 0

        exp_dict = self.data_dict[portion]["parse_exp_lst"][i][j]
        self.eval_counter[portion] = (i, j)


        # print(mentions)
        # extract mentions and convert them to tag_lst
        # write the prediction in a file for evaluation
        pred_tags = get_pred_tags(mentions, exp_dict)
        assert len(pred_tags) == len(exp_dict["positions"])
        # print(pred_tags)
        # print(exp_dict["tags"])
        # print()

        return pred_tags

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

    def _t5_ner_entity2id(self):
        entity2id = {}

        for tag, idx in self.label2id.items():
            if "-" not in tag: continue
            t = tag.split("-")[1]
            if t in entity2id: continue
            entity2id[t] = len(entity2id) + self.start_sentinel_id

        id2entity = {v: k for k, v in entity2id.items()}

        return entity2id, id2entity

    def compute_score(self, portion):

        if self.task_type == "gen":
            scores = self.compute_score_gen(portion)
        elif self.task_type == "ner":
            scores = self.compute_score_ner(portion)
        elif self.task_type == "qa":
            scores = self.compute_score_qa(portion)
        else:
            scores = compute_metrics(self.task_name.lower(), self.y_pred[portion], self.y_true[portion])
            scores = {k: round(100 * v, 2) for k, v in scores.items()}

        return scores

    def process_logits(self, portion):

        if not self.is_gen and not self.task_type in ["qa", "ner"]:
            self.y_logits[portion] = np.asarray([item for sublist in self.y_logits[portion] for item in sublist])
        if not self.is_gen and self.task_type == "reg":
            self.y_logits[portion] = np.squeeze(self.y_logits[portion])

        if self.is_gen:
            self.y_pred[portion] = [item for sublist in self.y_pred[portion] for item in sublist]
            if self.task_type == "ner":
                self.y_pred[portion] = [self._t5_decode_ner(arr, portion) for arr in self.y_pred[portion]]
            elif self.task_type == "qa":
                self.y_pred[portion] = self._t5_decode_qa(portion)
            elif self.task_type == "gen":
                self._t5_decode_gen(portion)
            else:
                self.y_pred[portion] = [self._t5_decode_disc(arr) for arr in self.y_pred[portion]]
                if portion == "test":
                    self.y_pred[portion] = [self.id2label[p] for p in self.y_pred[portion]]
        else:
            if self.task_type == "ner":
                self._process_logits_ner(portion)
            elif self.task_type == "qa":
                self._process_logits_qa(portion)
            else:
                self._process_logits_cls(portion)

    def _process_logits_ner(self, portion):
        # pad max_seq_len
        msl = max([len(batch[0]) for batch in self.y_logits[portion]])
        word_pad = [0] * len(self.label2id)
        # print(msl)
        torch_pred_lst = []
        for batch in self.y_logits[portion]:
            for row in batch:
                lst = row + [word_pad for _ in range(msl - len(row))]
                torch_pred_lst.append(lst)
        torch_pred_lst = np.argmax(np.asarray(torch_pred_lst), axis=-1).tolist()
        exp_lst, parse_exp_lst = self.data_dict[portion]["exp_lst"], self.data_dict[portion]["parse_exp_lst"]

        i = 0
        self.y_pred[portion] = []
        for exp_dict, p_exp_lst in zip(exp_lst, parse_exp_lst):
            pred_lst = []
            for p_exp_dict in p_exp_lst:
                seq_len = len(p_exp_dict["labels"])
                pred_lst.extend([self.id2label[idx] for idx in torch_pred_lst[i][:seq_len]])
                i += 1
            self.y_pred[portion].append(pred_lst)

    def _t5_decode_qa(self, portion):
        self._t5_decode_gen(portion)
        exp_lst, parse_exp_lst = self.data_dict[portion]["exp_lst"], self.data_dict[portion]["parse_exp_lst"]
        parse_exp_lst = [item for sublist in parse_exp_lst for item in sublist]
        group_exp_dict = {}

        for dico in parse_exp_lst:
            exp_idx, exp_sub_idx = dico["idx"].split(".")
            group_exp_dict[dico["idx"]] = int(exp_idx)

        predictions_dict = defaultdict(list)
        for p_text, dico in zip(self.y_pred[portion], parse_exp_lst):
            text = qa_locate_pred(dico["context"], p_text) if p_text.strip() else ""
            predictions_dict[group_exp_dict[dico["idx"]]].append(text)

        grouped_y_pred = []
        for k, exp_dict in enumerate(exp_lst):
            lst = predictions_dict[exp_dict["idx"]]
            prediction_text = sorted(lst, key=lambda x: len(x), reverse=True)[0]
            d = {"id": exp_dict["id"], "prediction_text": prediction_text, "no_answer_probability": 0.0}#
            grouped_y_pred.append(d)

        return grouped_y_pred

    def _process_logits_qa(self, portion):
        sep_idx = self.tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
        exp_lst, parse_exp_lst = self.data_dict[portion]["exp_lst"], self.data_dict[portion]["parse_exp_lst"]
        parse_exp_lst = [item for sublist in parse_exp_lst for item in sublist]
        group_exp_dict = {}

        for dico in parse_exp_lst:
            exp_idx, exp_sub_idx = dico["idx"].split(".")
            group_exp_dict[dico["idx"]] = int(exp_idx)

        start_pred, end_pred = [], []
        for sublist in self.y_logits[portion]:
            n = len(sublist)
            start_pred += sublist[:n//2]
            end_pred += sublist[n//2:]

        prelim_predictions_dict = defaultdict(list)
        tmp_dict = defaultdict(list)
        for i, (sp, ep, dico) in enumerate(zip(start_pred, end_pred, parse_exp_lst)):
            prelim_predictions = qa_get_top_pred(dico["input_ids"], sp, ep, sep_idx=sep_idx)
            top_can = prelim_predictions[0]
            if sum(top_can["offsets"]) == 1:
                text = ""
            else:
                # if portion == "test":
                ans_str = self._ids_to_text(dico["input_ids"][top_can["offsets"][0]:top_can["offsets"][1]])
                tmp_dict[group_exp_dict[dico["idx"]]].append(ans_str)
                text = qa_locate_pred(dico["context"], ans_str)
                # else:
                #     text = self._ids_to_text(dico["input_ids"][top_can["offsets"][0]:top_can["offsets"][1]])
            prelim_predictions[0]["text"] = text
            prelim_predictions_dict[group_exp_dict[dico["idx"]]].extend(prelim_predictions)

        for k, exp_dict in enumerate(exp_lst):
            lst = prelim_predictions_dict[exp_dict["idx"]]
            prediction_text = sorted(lst, key= lambda x:x["score"], reverse=True)[0]["text"]
            d = {"id": exp_dict["id"], "prediction_text": prediction_text, "no_answer_probability": 0.0} #
            self.y_pred[portion].append(d)
            # print(self.y_true[portion][k])
            # print(tmp_dict[exp_dict["idx"]])
            # print(d)
            # print()
        # print(self.compute_score_qa(portion))

    def _process_logits_cls(self, portion):
        if self.task_type == "reg":
            self.y_pred[portion] = np.squeeze(self.y_logits[portion])
        elif self.task_type == "mlc":
            self.y_pred[portion] = np.where(self.y_logits[portion] > 0.25, 1, 0).astype(np.int32)
        else:
            self.y_pred[portion] = np.argmax(self.y_logits[portion], axis=-1)
            if portion == "test":
                self.y_pred[portion] = [self.id2label[p] for p in self.y_pred[portion]]

        # self.y_pred[portion] = self.y_pred[portion].tolist()

    def final_metric(self, scores):
        return scores[EVAL_METRIC[self.task_name]]

    def reset_pred(self):
        self.y_pred, self.y_logits, self.eval_counter = defaultdict(list), defaultdict(list), defaultdict(tuple)

    def compute_score_ner(self, portion):
        rint = random.randint(0, 100)
        output_file = "/tmp/ner.output.%s" % rint
        score_file = "/tmp/ner.score.%s" % rint
        fout = open(output_file, 'w')
        exp_lst = self.data_dict[portion]["exp_lst"]
        for exp_dict, g_lst, p_lst in zip(exp_lst, self.y_true[portion], self.y_pred[portion]):
            w_lst = exp_dict["tokens"]
            sentence = "\n".join(["%s %s %s" % (w, g, p) for w, g, p in zip(w_lst, g_lst, p_lst)])
            fout.write("%s\n\n" % sentence)
        fout.close()

        mode = " -r" if "pos" in self.task_name else ""
        os.system("perl %s %s < %s > %s" % ("tools/conlleval", mode, output_file, score_file))
        eval_lines = [l.rstrip() for l in open(score_file)]
        f1 = float(eval_lines[1].strip().split()[-1])  # / 100
        return {"f1": f1}

    def compute_score_qa(self, portion):
        if not hasattr(self, "metric"):
            self.metric = load_metric("squad_v2")
        return self.metric.compute(predictions=self.y_pred[portion], references=self.y_true[portion])

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
            # use_agregator = True
            rouge_types = ["rouge1", "rouge2", "rougeL"]  # , "rougeLsum"]
            scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)
            # if use_agregator:
            aggregator = scoring.BootstrapAggregator()
            # else:
            #     scores = []

            for ref, pred in zip(self.y_true[portion], self.y_pred[portion]):
                pred = rouge_postprocessor(pred)
                ref = [rouge_postprocessor(r) for r in ref] if isinstance(ref, list) else rouge_postprocessor(ref)
                if not isinstance(ref, list):
                    score = scorer.score(ref, pred)
                else:
                    lst = [(r, scorer.score(r, pred)["rougeL"]) for r in ref]
                    lst.sort(key=lambda x: x[1], reverse=True)
                    score = scorer.score(lst[0][0], pred)

                # if use_agregator:
                aggregator.add_scores(score)
                # else:
                #     scores.append(score)

            # if use_agregator:
            results = aggregator.aggregate()
            # else:
            #     result = {}
            #     for key in scores[0]:
            #         result[key] = list(score[key] for score in scores)

            # result = []#metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            # Extract a few results from ROUGE
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


def get_tag_to_id(exp_lst, task_type):
    if task_type == "cls":
        tag_to_id = _get_tag_to_id_cls(exp_lst)
    elif task_type == "ner":
        tag_to_id = _get_tag_to_id_ner(exp_lst)
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


def _get_tag_to_id_ner(exp_lst):
    counter = Counter()
    for exp in exp_lst:
        counter.update(exp["tags"])

    if "O" not in counter:
        counter["O"] = 1e9
    counter = sorted(counter.items(), key=lambda x: (x[1], x[0]), reverse=True)
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
    parser.add_argument("--bench_name",
                        default="alue",
                        choices=["alue", "orca"],
                        help="Choose which functionality to run")

    parser.add_argument(
        "--model_name",
        default="JABER",
        type=str,
        help="model_name",
    )

    args = parser.parse_args()
    if args.bench_name == "alue":
        task_lst = ALUE_TASKS
    elif args.bench_name == "orca":
        task_lst = ORCA_TASKS
    else:
        raise ValueError("Not Supported Benchmark")

    for task_name in task_lst:
        for is_gen in [1, 0]:
            args.is_gen = is_gen
            data_processor = DataProcessor(task_name, args.model_name, args.is_gen)
            if data_processor.task_type in ["ner", "reg"] and args.is_gen: continue
            if data_processor.task_type == "gen" and not is_gen: continue
            if data_processor.task_type == "gen" and args.model_name not in MODEL_ARCH_MAP["t5"]: continue
            if args.model_name not in MODEL_ARCH_MAP["t5"] and is_gen: continue
            print(task_name, args.model_name, is_gen)
            data_processor.dump_train_data()
            pkl_file = os.path.join("./raw_datasets", "dp.%s.%s.%s.pkl" % (task_name, args.is_gen, args.model_name))
            # if os.path.exists(pkl_file): continue
            with open(pkl_file, 'wb') as handle:
                pickle.dump(data_processor, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

    main()
