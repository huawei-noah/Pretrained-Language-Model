from __future__ import absolute_import, division, print_function

import argparse
import random
import copy
import time
from datetime import timedelta

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.metrics import classification_report
from tqdm import trange, tqdm

from transformer import BertForSequenceClassification, WEIGHTS_NAME, CONFIG_NAME
from transformer.modeling_quant import BertForSequenceClassification as QuantBertForSequenceClassification
from transformer import BertTokenizer
from transformer import BertAdam
from transformer import BertConfig
from utils_multiemo import *
from utils import dictionary_to_json, result_to_text_file

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()


def get_tensor_data(output_mode, features):
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_seq_lengths)
    return tensor_data, all_label_ids


def do_eval(model, task_name, eval_dataloader,
            device, output_mode, eval_labels, num_labels):
    eval_loss = 0
    nb_eval_steps = 0
    all_logits = None

    for batch_ in tqdm(eval_dataloader):
        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_
            logits, _, _ = model(input_ids, segment_ids, input_mask)

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.append(all_logits, logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    if output_mode == "regression":
        all_logits = np.squeeze(all_logits)
    result = compute_metrics(task_name, all_logits, eval_labels.numpy())
    result['eval_loss'] = eval_loss
    return result, all_logits


def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return (- targets_prob * student_likelihood).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default='data',
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_dir",
                        default='models/tinybert',
                        type=str,
                        help="The model dir.")
    parser.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        help="The models directory.")
    parser.add_argument("--student_model",
                        default=None,
                        type=str,
                        help="The models directory.")
    parser.add_argument("--task_name",
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default='output',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--weight_decay', '--wd',
                        default=0.01,
                        type=float,
                        metavar='W',
                        help='weight decay')

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument('--aug_train',
                        action='store_false',
                        help="Whether to use augmented data or not")
    parser.add_argument('--pred_distill',
                        action='store_true',
                        help="Whether to distil with task layer")
    parser.add_argument('--intermediate_distill',
                        action='store_true',
                        help="Whether to distil with intermediate layers")
    parser.add_argument('--save_fp_model',
                        action='store_true',
                        help="Whether to save fp32 model")
    parser.add_argument('--save_quantized_model',
                        action='store_true',
                        help="Whether to save quantized model")

    parser.add_argument("--weight_bits",
                        default=2,
                        type=int,
                        choices=[2, 8],
                        help="Quantization bits for weight.")
    parser.add_argument("--input_bits",
                        default=8,
                        type=int,
                        help="Quantization bits for activation.")
    parser.add_argument("--clip_val",
                        default=2.5,
                        type=float,
                        help="Initial clip value.")

    args = parser.parse_args()
    assert args.pred_distill or args.intermediate_distill, "'pred_distill' and 'intermediate_distill', at least one must be True"
    logger.info('The args: {}'.format(args))
    task_name = args.task_name.lower()
    data_dir = args.data_dir
    output_dir = os.path.join(args.output_dir, task_name)

    os.makedirs(output_dir, exist_ok=True)

    if args.student_model is None:
        args.student_model = os.path.join(args.model_dir, task_name)
    if args.teacher_model is None:
        args.teacher_model = os.path.join(args.model_dir, task_name)

    processors = {
        "multiemo": MultiemoProcessor
    }

    output_modes = {
        "multiemo": "classification"
    }

    default_params = {
        "multiemo": {"max_seq_length": 128, "batch_size": 16, "eval_step": 50}
    }

    acc_tasks = ["multiemo"]

    # Prepare devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    # Prepare seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if task_name in default_params:
        args.batch_size = default_params[task_name]["batch_size"]
        if n_gpu > 0:
            args.batch_size = int(args.batch_size * n_gpu)
        args.max_seq_length = default_params[task_name]["max_seq_length"]
        args.eval_step = default_params[task_name]["eval_step"]

    if 'multiemo' in task_name:
        _, lang, domain, kind = task_name.split('_')
        processor = MultiemoProcessor(lang, domain, kind)
    else:
        raise ValueError("Task not found: %s" % task_name)

    if 'multiemo' in task_name:
        output_mode = output_modes['multiemo']
    else:
        raise ValueError("Task not found: %s" % task_name)

    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.student_model, do_lower_case=args.do_lower_case)

    if args.aug_train:
        train_examples = processor.get_aug_examples(data_dir)
        train_features = convert_examples_to_features(train_examples, label_list,
                                                      args.max_seq_length, tokenizer, output_mode)
    else:
        train_examples = processor.get_train_examples(data_dir)
        train_features = convert_examples_to_features(train_examples, label_list,
                                                      args.max_seq_length, tokenizer, output_mode)

    num_train_optimization_steps = int(len(train_features) / args.batch_size) * args.num_train_epochs
    train_data, _ = get_tensor_data(output_mode, train_features)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    eval_examples = processor.get_dev_examples(data_dir)
    eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer,
                                                 output_mode)
    eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    teacher_model = BertForSequenceClassification.from_pretrained(args.teacher_model)
    teacher_model.to(device)
    teacher_model.eval()
    if n_gpu > 1:
        teacher_model = torch.nn.DataParallel(teacher_model)

    result, _ = do_eval(teacher_model, task_name, eval_dataloader,
                        device, output_mode, eval_labels, num_labels)

    fp32_performance = f"f1/acc:{result['f1']}/{result['acc']}"
    fp32_performance = task_name + ' fp32   ' + fp32_performance

    student_config = BertConfig.from_pretrained(
        args.teacher_model,
        quantize_act=True,
        weight_bits=args.weight_bits,
        input_bits=args.input_bits,
        clip_val=args.clip_val
    )
    student_model = QuantBertForSequenceClassification.from_pretrained(args.student_model, config=student_config,
                                                                       num_labels=num_labels)
    student_model.to(device)

    training_start_time = time.monotonic()

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    if n_gpu > 1:
        student_model = torch.nn.DataParallel(student_model)

    optimizer = get_optimizer(args, num_train_optimization_steps, student_model)
    loss_mse = MSELoss()
    global_step = 0
    best_dev_acc = 0.0
    previous_best = None
    output_eval_file = os.path.join(output_dir, "eval_results.txt")

    tr_loss = 0.
    tr_att_loss = 0.
    tr_rep_loss = 0.
    tr_cls_loss = 0.
    for epoch_ in trange(int(args.num_train_epochs)):
        nb_tr_examples, nb_tr_steps = 0, 0

        for step, batch in enumerate(tqdm(train_dataloader, f"Epoch {epoch_ + 1}: ", ascii=True)):
            student_model.train()
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
            att_loss = 0.
            rep_loss = 0.
            cls_loss = 0.
            loss = 0.

            student_logits, student_atts, student_reps = student_model(input_ids, segment_ids, input_mask)

            with torch.no_grad():
                teacher_logits, teacher_atts, teacher_reps = teacher_model(input_ids, segment_ids, input_mask)

            if args.pred_distill:
                if output_mode == "classification":
                    cls_loss = soft_cross_entropy(student_logits, teacher_logits)
                elif output_mode == "regression":
                    cls_loss = loss_mse(student_logits, teacher_logits)

                loss = cls_loss
                tr_cls_loss += cls_loss.item()

            if args.intermediate_distill:
                for student_att, teacher_att in zip(student_atts, teacher_atts):
                    student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                              student_att)
                    teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                              teacher_att)
                    tmp_loss = loss_mse(student_att, teacher_att)
                    att_loss += tmp_loss

                for student_rep, teacher_rep in zip(student_reps, teacher_reps):
                    tmp_loss = loss_mse(student_rep, teacher_rep)
                    rep_loss += tmp_loss

                loss += rep_loss + att_loss
                tr_att_loss += att_loss.item()
                tr_rep_loss += rep_loss.item()

            if n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            tr_loss += loss.item()
            nb_tr_examples += label_ids.size(0)
            nb_tr_steps += 1

        logger.info("***** Running evaluation *****")
        logger.info("  {} step of {} steps".format(global_step, num_train_optimization_steps))
        if previous_best is not None:
            logger.info(f"{fp32_performance}\nPrevious best = {previous_best}")

        student_model.eval()

        loss = tr_loss / nb_tr_steps
        cls_loss = tr_cls_loss / nb_tr_steps
        att_loss = tr_att_loss / nb_tr_steps
        rep_loss = tr_rep_loss / nb_tr_steps

        result, _ = do_eval(student_model, task_name, eval_dataloader,
                            device, output_mode, eval_labels, num_labels)

        result['epoch'] = epoch_ + 1
        result['global_step'] = global_step
        result['cls_loss'] = cls_loss
        result['att_loss'] = att_loss
        result['rep_loss'] = rep_loss
        result['loss'] = loss

        result_to_text_file(result, output_eval_file)

        save_model = False

        if result['acc'] > best_dev_acc:
            previous_best = f"f1/acc:{result['f1']}/{result['acc']}"
            best_dev_acc = result['acc']
            save_model = True

        if save_model:
            logger.info(fp32_performance)
            logger.info(previous_best)
            if args.save_fp_model:
                logger.info("******************** Save full precision model ********************")
                model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
                output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
                output_config_file = os.path.join(output_dir, CONFIG_NAME)

                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(output_dir)

            if args.save_quantized_model:
                logger.info("******************** Save quantized model ********************")
                output_quant_dir = os.path.join(output_dir, 'quant')
                if not os.path.exists(output_quant_dir):
                    os.makedirs(output_quant_dir)
                model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
                quant_model = copy.deepcopy(model_to_save)
                for name, module in quant_model.named_modules():
                    if hasattr(module, 'weight_quantizer'):
                        module.weight.data = module.weight_quantizer.apply(
                            module.weight,
                            module.weight_clip_val,
                            module.weight_bits, True
                        )

                output_model_file = os.path.join(output_quant_dir, WEIGHTS_NAME)
                output_config_file = os.path.join(output_quant_dir, CONFIG_NAME)

                torch.save(quant_model.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(output_quant_dir)

    # Measure End Time
    training_end_time = time.monotonic()

    diff = timedelta(seconds=training_end_time - training_start_time)
    diff_seconds = diff.total_seconds()

    training_parameters = vars(args)
    training_parameters['training_time'] = diff_seconds

    output_training_params_file = os.path.join(output_dir, "training_params.json")
    dictionary_to_json(training_parameters, output_training_params_file)

    #########################
    #       Test model      #
    #########################
    test_examples = processor.get_test_examples(data_dir)
    test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer,
                                                 output_mode)

    test_data, test_labels = get_tensor_data(output_mode, test_features)
    test_sampler = SequentialSampler(eval_data)
    test_dataloader = DataLoader(eval_data, sampler=test_sampler, batch_size=args.batch_size)

    logger.info("\n***** Running evaluation on test dataset *****")
    logger.info("  Num examples = %d", len(test_features))
    logger.info("  Batch size = %d", args.batch_size)

    eval_start_time = time.monotonic()
    student_model.eval()
    result, y_logits = do_eval(student_model, task_name, test_dataloader,
                               device, output_mode, test_labels, num_labels)
    eval_end_time = time.monotonic()

    diff = timedelta(seconds=eval_end_time - eval_start_time)
    diff_seconds = diff.total_seconds()
    result['eval_time'] = diff_seconds
    result_to_text_file(result, os.path.join(output_dir, "test_results.txt"))

    y_pred = np.argmax(y_logits, axis=1)
    print('\n\t**** Classification report ****\n')
    print(classification_report(test_labels.numpy(), y_pred, target_names=label_list))

    report = classification_report(test_labels.numpy(), y_pred, target_names=label_list, output_dict=True)
    report['eval_time'] = diff_seconds
    dictionary_to_json(report, os.path.join(output_dir, "test_results.json"))


def get_optimizer(args, num_train_optimization_steps, student_model):
    # Prepare optimizer
    param_optimizer = list(student_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    schedule = 'warmup_linear'
    optimizer = BertAdam(
        optimizer_grouped_parameters,
        schedule=schedule,
        lr=args.learning_rate,
        warmup=0.1,
        t_total=num_train_optimization_steps
    )
    return optimizer


if __name__ == "__main__":
    main()
