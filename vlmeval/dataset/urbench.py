import ast
import os
import os.path as osp
import re
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from ..smp import LMUDataRoot, dump, get_intermediate_file_path, load, read_ok, toliststr
from .image_base import ImageBaseDataset

# UrBench Task Configuration & Dimensions
# Sourced from "UrBench: A Comprehensive Benchmark for Evaluating Large Multimodal Models
# in Multi-View Urban Scenarios" (AAAI 2025)
URBENCH_TASKS: Dict[str, Dict[str, Any]] = {
    # 1. Geo-Localization (3 tasks)
    'camera-localization': {
        'dimension': 'Geo-Localization',
        'metric': 'multiple-choice',
        'aliases': ['camera_localization', 'camera-localization']
    },
    'city-retrieval': {
        'dimension': 'Geo-Localization',
        'metric': 'multiple-choice',
        'aliases': ['city_retrieval', 'city-retrieval']
    },
    'orientation': {
        'dimension': 'Geo-Localization',
        'metric': 'multiple-choice',
        'aliases': ['orientation']
    },
    # 2. Scene Reasoning (4 tasks)
    'road-understanding': {
        'dimension': 'Scene Reasoning',
        'metric': 'multiple-choice',
        'aliases': ['road_understanding', 'road-understanding']
    },
    'role-based-reasoning': {
        'dimension': 'Scene Reasoning',
        'metric': 'multiple-choice',
        'aliases': ['role_based_reasoning', 'role-based-reasoning', 'role_based_reasoing']
    },
    'traffic-sign-reasoning': {
        'dimension': 'Scene Reasoning',
        'metric': 'multiple-choice',
        'aliases': ['traffic_sign_reasoning', 'traffic-sign-reasoning']
    },
    'scene-comparison': {
        'dimension': 'Scene Reasoning',
        'metric': 'multiple-choice',
        'aliases': ['scene_comparison', 'scene-comparison']
    },
    # 3. Scene Understanding (3 tasks)
    'counting': {
        'dimension': 'Scene Understanding',
        'metric': 'open-ended',
        'aliases': ['counting']
    },
    'scene-recognition': {
        'dimension': 'Scene Understanding',
        'metric': 'open-ended',
        'aliases': ['scene_recognition', 'scene-recognition']
    },
    'image-retrieval': {
        'dimension': 'Scene Understanding',
        'metric': 'multiple-choice',
        'aliases': ['image_retrieval', 'image-retrieval']
    },
    # 4. Object Understanding (4 tasks)
    'object-attribute-recognition': {
        'dimension': 'Object Understanding',
        'metric': 'multiple-choice',
        'aliases': ['object_attribute_recognition', 'object-attribute-recognition']
    },
    'object-grounding': {
        'dimension': 'Object Understanding',
        'metric': 'multiple-choice',
        'aliases': ['object_grounding', 'object-grounding']
    },
    'object-matching': {
        'dimension': 'Object Understanding',
        'metric': 'multiple-choice',
        'aliases': ['object_matching', 'object-matching']
    },
    'visual-prompt-reasoning': {
        'dimension': 'Object Understanding',
        'metric': 'multiple-choice',
        'aliases': ['visual_prompt_reasoning', 'visual-prompt-reasoning']
    },
}

MULTI_CHOICE_PROMPT = "Answer with the option's letter from the given choices directly."
OPEN_ENDED_PROMPT = "Answer the question using a single word or phrase."
VISUAL_GROUNDING_PROMPT = "Answer with four numbers whose range align with the image's original resolution."
REGION_CAPTION_PROMPT = "A specific area in the image has been marked with a red bounding box."
COUNTING_PROMPT = "Answer the question using arabic number only."

IMAGE_MAX_NUM = 6


def parse_options(options) -> str:
    """Format multiple choice options into standardized A. ..., B. ... string."""
    if isinstance(options, str):
        try:
            options = ast.literal_eval(options)
        except Exception:
            pass
    if isinstance(options, (list, tuple)):
        option_letters = [chr(ord("A") + i) for i in range(len(options))]
        choices_str = "\n".join([
            f"{letter}. {str(opt).replace(letter + '.', '').strip()}"
            for letter, opt in zip(option_letters, options)
        ])
        return choices_str
    return str(options)


def construct_prompt(question: str, choices, metric: str, task_type: str) -> str:
    """Construct prompt matching official UrBench benchmark specifications."""
    if task_type == 'road-understanding' and metric == 'open-ended':
        return question

    if task_type == 'counting':
        return f"{question}\n{COUNTING_PROMPT}"

    if metric == "multiple-choice":
        parsed_opts = parse_options(choices)
        return f"{question}\n{parsed_opts}\n\n{MULTI_CHOICE_PROMPT}"
    elif metric == "open-ended":
        return f"{question}\n\n{OPEN_ENDED_PROMPT}"
    elif metric == "mIoU":
        return f"{question}\n\n{VISUAL_GROUNDING_PROMPT}"
    elif metric == "bleu-4":
        return f"{REGION_CAPTION_PROMPT}\n\n{question}"
    return question


def get_multi_choice_info(options) -> Tuple[Dict[str, str], List[str]]:
    """Return index2ans map and list of choices."""
    if isinstance(options, str):
        try:
            options = ast.literal_eval(options)
        except Exception:
            pass
    if not isinstance(options, (list, tuple)):
        return {}, []

    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        letter = chr(ord("A") + i)
        index2ans[letter] = str(option)
        all_choices.append(letter)
    return index2ans, all_choices


def parse_multi_choice_response(response: Any, all_choices: List[str], index2ans: Dict[str, str]) -> str:
    """Parse multiple choice prediction letter from generated model response."""
    response = str(response)
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if str(ans).lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if len(candidates) == 0:
        return ""
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    start_indexes.append(response.rfind(f"({can})"))
            else:
                for can in candidates:
                    start_indexes.append(response.rfind(f" {can} "))
        else:
            for can in candidates:
                start_indexes.append(response.lower().rfind(str(index2ans[can]).lower()))
        pred_index = candidates[int(np.argmax(start_indexes))]
    else:
        pred_index = candidates[0]

    return pred_index


def extract_numbers(string: str) -> List[Any]:
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"

    numbers_with_commas = re.findall(pattern_commas, string)
    numbers_scientific = re.findall(pattern_scientific, string)
    numbers_simple = re.findall(pattern_simple, string)
    return numbers_with_commas + numbers_scientific + numbers_simple


def check_is_number(string: Any) -> bool:
    try:
        float(str(string).replace(",", ""))
        return True
    except ValueError:
        return False


def normalize_str(string: Any) -> List[Any]:
    string = str(string).strip()
    if check_is_number(string):
        num = round(float(string.replace(",", "")), 2)
        return [num]
    else:
        lower = string.lower()
        if len(lower) == 1:
            return [" " + lower, lower + " "]
        return [lower]


def parse_open_response(response: Any) -> List[Any]:
    """Parse candidate strings and numbers from open-ended responses."""
    resp_text = str(response)

    def get_key_subresponses(text: str) -> List[str]:
        text = text.strip().strip(".").lower()
        sub_responses = re.split(r"\.\s(?=[A-Z])|\n", text)
        indicators_of_keys = [
            "could be ", "so ", "is ", "thus ", "therefore ",
            "final ", "answer ", "result ",
        ]
        key_responses = []
        for index, resp in enumerate(sub_responses):
            if index == len(sub_responses) - 1:
                indicators_of_keys.append("=")
            shortest_key = None
            for indicator in indicators_of_keys:
                if indicator in resp:
                    part = resp.split(indicator)[-1].strip()
                    if not shortest_key or len(part) < len(shortest_key):
                        shortest_key = part
            if shortest_key and shortest_key.strip() not in [":", ",", ".", "!", "?", ";", "'"]:
                key_responses.append(shortest_key)
        if not key_responses:
            return [text]
        return key_responses

    key_responses = get_key_subresponses(resp_text)
    pred_list = key_responses.copy()
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))

    normalized_preds = []
    for item in pred_list:
        normalized_preds.extend(normalize_str(item))
    return list(set(normalized_preds))


def extract_bounding_box(string: str) -> List[float]:
    pattern = re.compile(r'\b([-+]?\d*\.?\d+)[,\s]+([-+]?\d*\.?\d+)[,\s]+([-+]?\d*\.?\d+)[,\s]+([-+]?\d*\.?\d+)\b')
    match = pattern.search(str(string))
    if match:
        return list(map(float, match.groups()))
    return []


def compute_iou(box1: Any, box2: Any) -> float:
    if isinstance(box1, str):
        box1 = extract_bounding_box(box1)
    if isinstance(box2, str):
        box2 = extract_bounding_box(box2)
    if not isinstance(box1, (list, tuple)) or not isinstance(box2, (list, tuple)):
        return 0.0
    if len(box1) != 4 or len(box2) != 4:
        return 0.0

    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2
    if x_min1 > x_max1 or y_min1 > y_max1 or x_min2 > x_max2 or y_min2 > y_max2:
        return 0.0

    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)

    if x_max_inter > x_min_inter and y_max_inter > y_min_inter:
        area_inter = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
    else:
        area_inter = 0.0

    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    area_union = area1 + area2 - area_inter
    return float(area_inter / area_union) if area_union > 0 else 0.0


def evaluate_bleu(reference: str, candidate: str) -> float:
    try:
        from nltk.translate.bleu_score import sentence_bleu
        ref_tokens = str(reference).split()
        cand_tokens = str(candidate).split()
        return float(sentence_bleu([ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25)))
    except Exception:
        ref_tokens = set(str(reference).lower().split())
        cand_tokens = set(str(candidate).lower().split())
        if not ref_tokens or not cand_tokens:
            return 0.0
        return float(len(ref_tokens & cand_tokens) / max(len(ref_tokens), len(cand_tokens)))


def eval_multi_choice(gold: Any, pred: str) -> bool:
    if isinstance(gold, list):
        return any(str(ans).strip() == str(pred).strip() for ans in gold)
    return str(gold).strip() == str(pred).strip()


def eval_open(gold: Any, pred_list: List[Any]) -> bool:
    if isinstance(gold, list):
        norm_answers = []
        for ans in gold:
            norm_answers.extend(normalize_str(ans))
    else:
        norm_answers = normalize_str(gold)

    for pred in pred_list:
        if isinstance(pred, str):
            for norm_ans in norm_answers:
                if isinstance(norm_ans, str) and norm_ans in pred:
                    return True
        else:
            if pred in norm_answers:
                return True
    return False


class UrBench(ImageBaseDataset):
    """
    UrBench: A Comprehensive Benchmark for Evaluating Large Multimodal Models
    in Multi-View Urban Scenarios (AAAI 2025).

    Dataset Source: bczhou/UrBench (Hugging Face)
    Official Repo: opendatalab/UrBench
    """

    TYPE = 'VQA'
    MODALITY = 'IMAGE'
    HF_REPO = 'bczhou/UrBench'

    DATASET_URL = {}
    DATASET_MD5 = {}

    @classmethod
    def _parse_dataset_name(cls, dataset: str) -> Tuple[List[str], str]:
        """Parse dataset string into list of task configs and split ('val' or 'test')."""
        dname = dataset.strip()
        split = 'test' if dname.endswith('_TEST') else 'val'

        base = dname
        if base.endswith('_TEST'):
            base = base[:-5]
        elif base.endswith('_VAL'):
            base = base[:-4]

        if base in ('UrBench', 'urbench'):
            return list(URBENCH_TASKS.keys()), split

        # Check task aliases
        task_candidate = base[len('UrBench_'):] if base.startswith('UrBench_') else base
        task_candidate = task_candidate.lower().replace('_', '-')

        for task_name, info in URBENCH_TASKS.items():
            if task_candidate == task_name or task_candidate in [a.replace('_', '-') for a in info['aliases']]:
                return [task_name], split

        # Default fallback to all tasks
        return list(URBENCH_TASKS.keys()), split

    @classmethod
    def supported_datasets(cls) -> List[str]:
        names = ['UrBench', 'UrBench_VAL', 'UrBench_TEST']
        for task_name, info in URBENCH_TASKS.items():
            for alias in info['aliases']:
                names.extend([
                    f'UrBench_{alias}',
                    f'UrBench_{alias}_VAL',
                    f'UrBench_{alias}_TEST',
                ])
        return list(OrderedDict.fromkeys(names))

    def __init__(self, dataset: str = 'UrBench', skip_noimg: bool = True):
        self.data_root = LMUDataRoot()
        self.img_root = osp.join(self.data_root, 'images', 'UrBench')
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)

    def load_data(self, dataset: str) -> pd.DataFrame:
        tasks, split = self._parse_dataset_name(dataset)
        target_tsv = osp.join(self.data_root, f'{dataset}.tsv')

        if osp.exists(target_tsv):
            return load(target_tsv)

        # Build requested tasks
        dfs = []
        for task in tasks:
            task_tsv = osp.join(self.data_root, f'UrBench_{task.replace("-", "_")}_{split}.tsv')
            if osp.exists(task_tsv):
                df = load(task_tsv)
            else:
                df = self._download_and_convert_task(task, split, task_tsv)
            dfs.append(df)

        if len(dfs) == 1:
            full_df = dfs[0]
        else:
            full_df = pd.concat(dfs, ignore_index=True)
            full_df['index'] = np.arange(len(full_df))

        dump(full_df, target_tsv)
        return full_df

    def _download_and_convert_task(self, task: str, split: str, output_tsv: str) -> pd.DataFrame:
        """Download task config from Hugging Face and convert to standardized TSV."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets library via: pip install datasets")

        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

        img_dir = osp.join(self.img_root, task, split)
        os.makedirs(img_dir, exist_ok=True)

        ds = load_dataset(self.HF_REPO, task, split=split)
        dimension = URBENCH_TASKS.get(task, {}).get('dimension', 'General')

        records = []
        for idx, item in enumerate(ds):
            sample_id = str(item.get('id', idx))
            img_paths = []

            for i in range(1, IMAGE_MAX_NUM + 1):
                img_key = f'image_{i}'
                img_obj = item.get(img_key)
                if img_obj is not None:
                    im_filename = f"{sample_id}_{img_key}.png"
                    abs_im_path = osp.join(img_dir, im_filename)
                    if not read_ok(abs_im_path):
                        if isinstance(img_obj, Image.Image):
                            img_obj.convert('RGB').save(abs_im_path, format='PNG')
                        elif hasattr(img_obj, 'save'):
                            img_obj.save(abs_im_path)
                    # Use relative path from LMUDataRoot for cross-machine portability
                    rel_im_path = osp.join('images', 'UrBench', task, split, im_filename).replace('\\', '/')
                    img_paths.append(rel_im_path)

            raw_q = str(item.get('question', ''))
            metric = str(item.get('metric', 'multiple-choice'))
            choices = item.get('choices', '')
            formatted_q = construct_prompt(raw_q, choices, metric, task)

            row = {
                'index': idx,
                'id': sample_id,
                'image_path': repr(img_paths) if len(img_paths) > 1 else (img_paths[0] if img_paths else ''),
                'question': formatted_q,
                'choices': str(choices),
                'answer': str(item.get('answer', '')),
                'task_type': task,
                'view_type': str(item.get('view_type', 'unknown')),
                'metric': metric,
                'category': dimension,
            }
            records.append(row)

        df = pd.DataFrame(records)
        dump(df, output_tsv)
        return df

    def build_prompt(self, line: Any) -> List[Dict[str, Any]]:
        """
        Build multimodal prompt for VLM.
        Interleaves text with images when <image> placeholders exist in the question.
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        raw_image_path = line.get('image_path', '')
        tgt_path = toliststr(raw_image_path) if raw_image_path else []

        # Resolve to valid absolute paths
        resolved_paths = []
        for p in tgt_path:
            if not p:
                continue
            if osp.isabs(p) and read_ok(p):
                resolved_paths.append(p)
            elif read_ok(osp.join(self.data_root, p)):
                resolved_paths.append(osp.join(self.data_root, p))
            elif read_ok(osp.join(self.img_root, p)):
                resolved_paths.append(osp.join(self.img_root, p))
            else:
                resolved_paths.append(p)

        question = str(line.get('question', ''))
        clean_q = re.sub(r'<image\s*\d*>', '<image>', question)

        if '<image>' in clean_q and resolved_paths:
            parts = clean_q.split('<image>')
            msgs = []
            for i, part in enumerate(parts):
                if part.strip():
                    msgs.append(dict(type='text', value=part))
                if i < len(resolved_paths):
                    msgs.append(dict(type='image', value=resolved_paths[i]))
        else:
            msgs = [dict(type='image', value=p) for p in resolved_paths]
            msgs.append(dict(type='text', value=question))

        return msgs

    def evaluate(self, eval_file: str, **judge_kwargs) -> pd.DataFrame:
        """
        Evaluate model predictions using the official UrBench / MMMU protocol.
        Generates breakdown across 4 dimensions, 3 view types, and 14 tasks.
        """
        data = load(eval_file)
        if isinstance(data, list):
            data = pd.DataFrame(data)
        data = data.sort_values(by='index').copy()
        data['prediction'] = [str(x) if not pd.isna(x) else '' for x in data['prediction']]

        hits = []
        parsed_preds = []

        for _, row in data.iterrows():
            pred = str(row['prediction'])
            metric = str(row.get('metric', 'multiple-choice'))
            choices = row.get('choices', '')
            answer = str(row.get('answer', '')).strip()

            if metric == 'multiple-choice':
                index2ans, all_choices = get_multi_choice_info(choices)
                parsed = parse_multi_choice_response(pred, all_choices, index2ans)
                hit = 1.0 if eval_multi_choice(answer, parsed) else 0.0
            elif metric == 'open-ended':
                parsed_candidates = parse_open_response(pred)
                parsed = str(parsed_candidates)
                hit = 1.0 if eval_open(answer, parsed_candidates) else 0.0
            elif metric == 'mIoU':
                parsed_box = extract_bounding_box(pred)
                parsed = str(parsed_box)
                iou = compute_iou(answer, parsed_box)
                hit = 1.0 if iou >= 0.2 else 0.0
            elif metric == 'bleu-4':
                parsed = pred
                hit = evaluate_bleu(answer, pred)
            else:
                parsed = pred.strip()
                hit = 1.0 if parsed.lower() == answer.lower() else 0.0

            hits.append(hit)
            parsed_preds.append(parsed)

        data['parsed_pred'] = parsed_preds
        data['hit'] = hits

        # Save details file
        details_file = get_intermediate_file_path(eval_file, '_details', 'tsv')
        dump(data, details_file)

        # Build summary report
        rows = []
        overall_acc = float(np.mean(hits) * 100) if hits else 0.0
        rows.append({
            'Group': 'Overall',
            'Name': 'Overall',
            'Count': len(data),
            'Accuracy': round(overall_acc, 2)
        })

        # By Dimension
        if 'category' in data.columns:
            for dim, sub in data.groupby('category', sort=False):
                rows.append({
                    'Group': 'Dimension',
                    'Name': dim,
                    'Count': len(sub),
                    'Accuracy': round(float(sub['hit'].mean() * 100), 2)
                })

        # By View Type
        if 'view_type' in data.columns:
            for vt, sub in data.groupby('view_type', sort=False):
                rows.append({
                    'Group': 'View Type',
                    'Name': vt,
                    'Count': len(sub),
                    'Accuracy': round(float(sub['hit'].mean() * 100), 2)
                })

        # By Task
        if 'task_type' in data.columns:
            for tt, sub in data.groupby('task_type', sort=False):
                rows.append({
                    'Group': 'Task',
                    'Name': tt,
                    'Count': len(sub),
                    'Accuracy': round(float(sub['hit'].mean() * 100), 2)
                })

        summary_df = pd.DataFrame(rows)
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(summary_df, score_file)

        return summary_df
