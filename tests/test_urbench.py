import os.path as osp
import tempfile

import pandas as pd

from vlmeval.dataset import SUPPORTED_DATASETS
from vlmeval.dataset.urbench import (URBENCH_TASKS, UrBench, compute_iou, construct_prompt,
                                     eval_multi_choice, eval_open, evaluate_bleu,
                                     parse_multi_choice_response, parse_open_response,
                                     parse_options)


def test_urbench_supported_datasets():
    supported = UrBench.supported_datasets()
    assert 'UrBench' in supported
    assert 'UrBench_VAL' in supported
    assert 'UrBench_TEST' in supported

    # Check that all 14 tasks are present
    for task in URBENCH_TASKS:
        assert f'UrBench_{task}' in supported or f'UrBench_{task.replace("-", "_")}' in supported
        assert f'UrBench_{task}_VAL' in supported or f'UrBench_{task.replace("-", "_")}_VAL' in supported
        assert f'UrBench_{task}_TEST' in supported or f'UrBench_{task.replace("-", "_")}_TEST' in supported

    # Check global registration in VLMEvalKit
    for name in ['UrBench', 'UrBench_VAL', 'UrBench_TEST']:
        assert name in SUPPORTED_DATASETS


def test_urbench_parse_dataset_name():
    tasks, split = UrBench._parse_dataset_name('UrBench')
    assert len(tasks) == 14
    assert split == 'val'

    tasks, split = UrBench._parse_dataset_name('UrBench_VAL')
    assert len(tasks) == 14
    assert split == 'val'

    tasks, split = UrBench._parse_dataset_name('UrBench_TEST')
    assert len(tasks) == 14
    assert split == 'test'

    tasks, split = UrBench._parse_dataset_name('UrBench_counting_VAL')
    assert tasks == ['counting']
    assert split == 'val'

    tasks, split = UrBench._parse_dataset_name('UrBench_camera_localization_TEST')
    assert tasks == ['camera-localization']
    assert split == 'test'

    tasks, split = UrBench._parse_dataset_name('UrBench_camera-localization')
    assert tasks == ['camera-localization']
    assert split == 'val'


def test_parse_options_and_construct_prompt():
    choices = ['3', '1', '4', '2']
    formatted = parse_options(choices)
    assert 'A. 3' in formatted
    assert 'B. 1' in formatted
    assert 'C. 4' in formatted
    assert 'D. 2' in formatted

    # Multiple-choice prompt
    q = construct_prompt("What is the location?", choices, "multiple-choice", "camera-localization")
    assert "A. 3" in q
    assert "Answer with the option's letter from the given choices directly." in q

    # Counting prompt
    q_count = construct_prompt("How many cars are visible?", [], "open-ended", "counting")
    assert "Answer the question using arabic number only." in q_count

    # General open-ended prompt
    q_open = construct_prompt("What is the road type?", [], "open-ended", "scene-recognition")
    assert "Answer the question using a single word or phrase." in q_open


def test_multi_choice_parsing_and_eval():
    all_choices = ['A', 'B', 'C', 'D']
    index2ans = {'A': 'Park', 'B': 'Highway', 'C': 'Bridge', 'D': 'Tunnel'}

    # Direct letter
    assert parse_multi_choice_response('A', all_choices, index2ans) == 'A'
    assert parse_multi_choice_response('(B)', all_choices, index2ans) == 'B'
    assert parse_multi_choice_response('C.', all_choices, index2ans) == 'C'
    assert parse_multi_choice_response('Therefore, the answer is (D).', all_choices, index2ans) == 'D'
    assert parse_multi_choice_response('I think the best option is Highway because...', all_choices, index2ans) == 'B'

    # Eval multi choice
    assert eval_multi_choice('B', 'B') is True
    assert eval_multi_choice('B', 'A') is False
    assert eval_multi_choice(['B', 'C'], 'B') is True


def test_open_ended_parsing_and_eval():
    # Number extraction
    res = parse_open_response("There are approximately 42 cars in the lot.")
    assert 42.0 in res

    res2 = parse_open_response("The answer is 1,250.")
    assert 1250.0 in res2

    # String matching
    res3 = parse_open_response("The scene is an urban residential area.")
    assert eval_open("residential area", res3) is True
    assert eval_open("highway", res3) is False


def test_iou_and_bleu():
    # Overlapping boxes: [xmin, ymin, xmax, ymax]
    box1 = [0, 0, 10, 10]
    box2 = [0, 0, 10, 10]
    assert compute_iou(box1, box2) == 1.0

    box3 = [5, 5, 15, 15]
    # Inter: [5,5,10,10] = 5x5 = 25. Union: 100 + 100 - 25 = 175. IoU = 25/175 = 1/7
    assert round(compute_iou(box1, box3), 3) == round(25 / 175, 3)

    box_disjoint = [20, 20, 30, 30]
    assert compute_iou(box1, box_disjoint) == 0.0

    # BLEU
    bleu = evaluate_bleu("a red bus on the street", "a red bus on the street")
    assert bleu > 0.9


def test_build_prompt_interleaving():
    dummy_data = pd.DataFrame([{
        'index': 0,
        'image_path': repr(['dummy_1.png', 'dummy_2.png']),
        'question': 'Here is satellite: <image>\nHere is street: <image>\nWhich is it?',
        'choices': "['A', 'B']",
        'answer': 'A',
    }])

    # Instantiate UrBench with dummy data
    dataset = object.__new__(UrBench)
    dataset.data = dummy_data
    dataset.data_root = '/dummy/root'
    dataset.img_root = '/dummy/root/images/UrBench'

    msgs = dataset.build_prompt(dummy_data.iloc[0])
    # Expect interleaved: text -> image -> text -> image -> text
    types = [m['type'] for m in msgs]
    assert types == ['text', 'image', 'text', 'image', 'text']
    assert 'satellite' in msgs[0]['value']
    assert 'dummy_1.png' in msgs[1]['value']
    assert 'street' in msgs[2]['value']
    assert 'dummy_2.png' in msgs[3]['value']


def test_evaluate_end_to_end():
    sample_records = [
        {
            'index': 0,
            'question': 'Which quadrant? A. 1 B. 2',
            'choices': "['1', '2']",
            'answer': 'A',
            'prediction': 'The answer is A.',
            'task_type': 'camera-localization',
            'view_type': 'cross-view',
            'metric': 'multiple-choice',
            'category': 'Geo-Localization',
        },
        {
            'index': 1,
            'question': 'Which quadrant? A. 1 B. 2',
            'choices': "['1', '2']",
            'answer': 'B',
            'prediction': 'Option A is selected.',
            'task_type': 'camera-localization',
            'view_type': 'cross-view',
            'metric': 'multiple-choice',
            'category': 'Geo-Localization',
        },
        {
            'index': 2,
            'question': 'How many buildings?',
            'choices': '',
            'answer': '5',
            'prediction': 'There are 5 buildings.',
            'task_type': 'counting',
            'view_type': 'street-view',
            'metric': 'open-ended',
            'category': 'Scene Understanding',
        },
        {
            'index': 3,
            'question': 'What is the scene?',
            'choices': "['park', 'parking lot']",
            'answer': 'A',
            'prediction': 'The answer is (A)',
            'task_type': 'scene-comparison',
            'view_type': 'satellite-view',
            'metric': 'multiple-choice',
            'category': 'Scene Reasoning',
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        eval_file = osp.join(tmpdir, 'mock_UrBench_eval.xlsx')
        df = pd.DataFrame(sample_records)
        df.to_excel(eval_file, index=False)

        dataset = object.__new__(UrBench)
        dataset.data_root = tmpdir
        dataset.img_root = osp.join(tmpdir, 'images')

        summary_df = dataset.evaluate(eval_file)

        assert isinstance(summary_df, pd.DataFrame)
        assert 'Group' in summary_df.columns
        assert 'Name' in summary_df.columns
        assert 'Accuracy' in summary_df.columns

        overall_row = summary_df[summary_df['Group'] == 'Overall'].iloc[0]
        # 3 out of 4 correct: idx 0 (A==A), idx 2 (5==5), idx 3 (A==A), idx 1 wrong (A!=B) -> 75%
        assert overall_row['Accuracy'] == 75.0
        assert overall_row['Count'] == 4

        # Check dimension breakdown
        geo_row = summary_df[(summary_df['Group'] == 'Dimension') & (summary_df['Name'] == 'Geo-Localization')].iloc[0]
        assert geo_row['Accuracy'] == 50.0  # 1 of 2 correct

        scene_und = summary_df[
            (summary_df['Group'] == 'Dimension') & (summary_df['Name'] == 'Scene Understanding')
        ].iloc[0]
        assert scene_und['Accuracy'] == 100.0  # 1 of 1 correct

        # Verify output files generated
        acc_file = osp.join(tmpdir, 'mock_UrBench_eval_acc.csv')
        details_file = osp.join(tmpdir, 'mock_UrBench_eval_details.tsv')
        assert osp.exists(acc_file)
        assert osp.exists(details_file)
