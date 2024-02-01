import json
from typing import List

import pandas as pd


def _create_vocab_mapping(vocab: List[str], output_file: str) -> None:
    vocab = ['unk', 'pad'] + vocab
    vocab2id = {vocab_text: idx for idx, vocab_text in enumerate(vocab)}
    with open(output_file, 'w') as json_file:
        json.dump(vocab2id, json_file, indent=2)


def _get_unique_column_values(dataframe: pd.DataFrame, column_name: str) -> List[str]:
    """
    Args:
        dataframe: DataFrame containing the data.
        column_name: Name of the column for which unique values are needed.

    Returns:
        List of unique values from the specified column.
    """
    return list(set(dataframe[column_name]))


def create_category_mapping(train_news_path: str, output_file: str) -> None:
    train_news_df = pd.read_csv(train_news_path, sep='\t',
                                names=['news_id', 'category', 'sub_category', 'title', 'abstract', 'url',
                                       'title_entities', 'abstract_entities'])
    categories = _get_unique_column_values(train_news_df, 'category')
    _create_vocab_mapping(vocab=categories, output_file=output_file)


def create_user_mapping(train_impressions_path: str, output_file: str) -> None:
    train_impressions_df = pd.read_csv(train_impressions_path, sep='\t',
                                       names=['row_index', 'user_id', 'time', 'history', 'impressions'])
    user_categories = _get_unique_column_values(train_impressions_df, 'user_id')
    _create_vocab_mapping(vocab=user_categories, output_file=output_file)


if __name__ == "__main__":
    train_news_path = "/home/benshoho/projects/RS/data/MIND_small/train/news.tsv"
    train_impressions_path = "/home/benshoho/projects/RS/data/MIND_small/train/behaviors.tsv"
    categories_vocab_output_file = "/home/benshoho/projects/RS/miner/vocabs/category2id.json"
    user_vocab_output_file = "/home/benshoho/projects/RS/miner/vocabs/user2id.json"

    create_category_mapping(train_news_path, categories_vocab_output_file)
    create_user_mapping(train_impressions_path, user_vocab_output_file)
