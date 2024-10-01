import json
import copy

from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets



# packing sequence training: SUM(Len(Seq)) must > max_seq_len of LLM
def load_data(data_path, test_path, shuffle=False):
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    with open(test_path, 'r') as f:
        test_data = [json.loads(line) for line in f.readlines()]

    # [{"input": "xxx", "output": "xxx"}...]
    messages = [[
        {'role': 'user', 'content': D['input']},
        {'role': 'assistant', 'content': D['output']}
    ] for D in data]
    messages_test = [[
        {'role': 'user', 'content': D['input']},
        {'role': 'assistant', 'content': D['output']}
    ] for D in test_data]
    data_dict = {'messages': messages}
    test_data_dict = {'messages': messages_test}

    # create a dataset from a dictionary
    raw_train_datasets = Dataset.from_dict(data_dict)
    raw_test_datasets = Dataset.from_dict(test_data_dict)

    # raw_datasets = raw_train_datasets.train_test_split(test_size=0.2, seed=42)

    raw_datasets = DatasetDict({
        'train': raw_train_datasets,
        'test': raw_test_datasets,
    })

    if shuffle:
        raw_datasets['train'] = raw_datasets['train'].shuffle(seed=42)
        raw_datasets['test'] = raw_datasets['test'].shuffle(seed=42)

    return raw_datasets



if __name__ == '__main__':
    load_data('data/data.jsonl')