import logging
import os
import socket
import sys
from typing import Union, List, Optional

import dataclasses
import fire
import wandb
from transformers import TrainingArguments, is_wandb_available

from cli_model import train, DEFAULT_NUM_TRAIN_EPOCHS
from cli_specter import find_train_ids
from cli_triples import get_metadata, get_specter_triples
from gdt.models import PoolingStrategy
from gdt.triples_miner import TriplesMinerArguments
from gdt.utils import get_kwargs_for_data_classes, get_workers

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def run_specter(
        output_dir: str,
        base_model_name_or_path: str,
        scidocs_dir: str,
        s2orc_metadata_dir: str,
        specter_triples_path: str,
        graph_paper_ids_path: str,
        graph_embeddings_path: str,
        s2id_to_s2orc_input_path: str,
        train_s2orc_paper_ids: Optional[Union[str, List[str]]] = None,
        train_query_s2orc_paper_ids: Optional[Union[str, List[str]]] = None,
        graph_limit: str = 'specter',
        workers: int = 0,
        masked_language_modeling: bool = False,
        masked_language_modeling_weight: float = 1.0,
        predict_embeddings: bool = False,
        pooling_strategy: PoolingStrategy = PoolingStrategy.CLS,
        max_sequence_length: int = 512,
        val_or_test_or_both: str = 'both',
        query_n_folds: int = 0,
        query_fold_k: Union[int, List[int], str] = 0,
        query_oversampling_ratio: float = 0.0,
        sample_queries_ratio: float = 1.0,
        auto_output_dir: bool = False,
        skip_queries: bool = False,
        skip_triples: bool = False,
        skip_metadata: bool = False,
        skip_train: bool = False,
        skip_eval: bool = False,
        override_triples: bool = False,
        override_queries: bool = False,
        override_metadata: bool = False,
        override_train: bool = False,
        gzip: bool = False,
        scidocs_cuda_device: int = -1,
        **kwargs
        ):
    """

    Runs all at once (with difference hyperparameters) -> generate triples -> train -> evaluate

    - Models are saved in $EXP_DIR/model
    - Training arguments are not needed (by default SPECTER settings are used)

    Usage:

    python cli_pipeline.py run_specter $EXP_DIR \
        --base_model_name_or_path $BASE_MODEL \
        --scidocs_dir $SCIDOCS_DIR \
        --s2orc_metadata_dir $S2ORC_METADATA_DIR \
        --specter_triples_path ${SPECTER_DIR}/train_triples.csv \
        --graph_paper_ids_path ${S2ORC_PAPER_IDS} \
        --graph_embeddings_path ${S2ORC_EMBEDDINGS}  \
        --s2id_to_s2orc_input_path ${SPECTER_DIR}/s2id_to_s2orc_paper_id.json \
        --train_s2orc_paper_ids ${EXP_DIR}/s2orc_paper_ids.json \
        --train_query_s2orc_paper_ids ${EXP_DIR}/query_s2orc_paper_ids.json \
        --ann_trees 1000 \
        --triples_per_query 5 \
        --easy_positives_count 5 --easy_positives_strategy knn --easy_positives_k_min 0 --easy_positives_k_max 5 \
        --easy_negatives_count 3 --easy_negatives_strategy random \
        --hard_negatives_count 2 --hard_negatives_strategy knn --hard_negatives_k_min 498 --hard_negatives_k_max 500 \
        --workers $WORKERS

    :param override_train: Override trained model
    :param gzip: Uses gzip compression for triples.csv and metadata.json
    :param masked_language_modeling_weight: Weight factor for MLM loss
    :param predict_embeddings: Enable prediction of target embeddings as additional loss
    :param max_sequence_length: Max. tokens for training set (does not apply for test set)
    :param sample_queries_ratio: Post-sampling of query documents (performed after folds etc)
    :param output_dir:
    :param base_model_name_or_path: Base BERT-style Transformer model (see AutoModel.from_pretrained)
    :param scidocs_dir:
    :param s2orc_metadata_dir:
    :param specter_triples_path:
    :param graph_paper_ids_path:
    :param graph_embeddings_path:
    :param s2id_to_s2orc_input_path:
    :param train_s2orc_paper_ids: Path to JSON, List (default: <output_dir>/s2orc_paper_ids.json)
    :param train_query_s2orc_paper_ids: Path to JSON, List (default: <output_dir>/query_s2orc_paper_ids.json)
    :param graph_limit:
    :param workers:
    :param masked_language_modeling:
    :param pooling_strategy:
    :param val_or_test_or_both:
    :param query_n_folds:
    :param query_fold_k:
    :param query_oversampling_ratio: Pre-sampling
    :param auto_output_dir:
    :param skip_queries:
    :param skip_triples:
    :param skip_metadata:
    :param skip_train:
    :param skip_eval:
    :param override_triples: Override triples
    :param override_queries: Override queries
    :param override_metadata: Override metadata
    :return:
    """

    # Log arg settings
    # write_func_args(inspect.currentframe(), os.path.join(output_dir, 'pipeline.args.json'))

    logger.info(f'Running pipeline in {output_dir}')
    logger.info(f'Host: {socket.gethostname()}')

    triples_miner_kwargs, training_kwargs = get_kwargs_for_data_classes([TriplesMinerArguments, TrainingArguments], kwargs)
    triples_miner_args = TriplesMinerArguments(**triples_miner_kwargs)

    base_model_name = base_model_name_or_path.split('/')[-1]

    if auto_output_dir:
        # Automatically determining output dir
        auto_output_dir = os.path.join(output_dir, graph_limit)

        if query_oversampling_ratio > 0:
            auto_output_dir = os.path.join(auto_output_dir, f'oversampling_{query_oversampling_ratio}')

        if query_n_folds > 0:
            auto_output_dir = os.path.join(auto_output_dir, f'folds_{query_n_folds}', f'k_{query_fold_k}')

        auto_output_dir = os.path.join(auto_output_dir, triples_miner_args.stringify())

        # Override run name
        training_kwargs['run_name'] = auto_output_dir + f' ({base_model_name})'

        output_dir = os.path.join(output_dir, auto_output_dir)

        logger.info(f'Output directory set to: {output_dir}')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info('Created output directory')

    workers = get_workers(workers)

    # Determine model dir depending on settings
    model_dir = os.path.join(output_dir, f'model_{base_model_name}')

    if masked_language_modeling:
        logger.info('Enable masked_language_modeling')
        model_dir += f'_mlm'

        if masked_language_modeling_weight != 1.0:
            logger.info(f'--masked_language_modeling_weight = {masked_language_modeling_weight}')
            model_dir += f'_{masked_language_modeling_weight}'

    if pooling_strategy != PoolingStrategy.CLS:
        logger.info(f'PoolingStrategy {pooling_strategy}')
        model_dir += '_' + pooling_strategy

    if 'fp16' in training_kwargs:
        # Float precision
        model_dir += '_fp16'

    if predict_embeddings:
        logger.info('Enable predict_embeddings')
        model_dir += '_predict_embeddings'

    if 'warmup_ratio' in training_kwargs and training_kwargs['warmup_ratio'] > 0:
        logger.info('Custom warmup_ratio')
        model_dir += f'_warmup_ratio_{training_kwargs["warmup_ratio"]}'

    if 'num_train_epochs' in training_kwargs and training_kwargs['num_train_epochs'] != DEFAULT_NUM_TRAIN_EPOCHS:
        logger.info('Custom num_train_epochs')
        model_dir += f'_epochs_{training_kwargs["num_train_epochs"]}'

    triples_path = os.path.join(output_dir, 'train_triples.csv')
    metadata_path = os.path.join(output_dir, 'train_metadata.jsonl')

    if gzip:
        triples_path += '.gz'
        metadata_path += '.gz'

    if train_s2orc_paper_ids is None:
        train_s2orc_paper_ids = os.path.join(output_dir, 's2orc_paper_ids.json')

    if train_query_s2orc_paper_ids is None:
        train_query_s2orc_paper_ids = os.path.join(output_dir, 'query_s2orc_paper_ids.json')

    # triples_miner_args, training_args = get_data_class_args([TriplesMinerArguments, TrainingArguments], kwargs,
    #                                                         output_dir=model_dir)

    if skip_queries:
        logger.info('Skipping queries')
    else:
        if os.path.exists(train_s2orc_paper_ids) and os.path.exists(train_query_s2orc_paper_ids)\
                and not override_queries:
            logger.info('Skipping queries (output exists already)')
        else:
            logger.info('Finding query ids')

            find_train_ids(
                specter_triples_path,
                scidocs_dir,
                s2id_to_s2orc_input_path,
                s2orc_paper_ids=graph_paper_ids_path,
                output_path=train_s2orc_paper_ids,
                query_output_path=train_query_s2orc_paper_ids,
                query_n_folds=query_n_folds,
                query_fold_k=query_fold_k,
                query_oversampling_ratio=query_oversampling_ratio,
                seed=triples_miner_args.seed,
            )

    if skip_triples:
        logger.info('Skipping triples')
    else:
        if os.path.exists(triples_path) and not override_triples:
            logger.info('Skipping triples (output exists already)')
        else:
            logger.info('Generating triples')

            get_specter_triples(triples_path,
                                scidocs_dir,
                                specter_triples_path,
                                graph_paper_ids_path,
                                graph_embeddings_path,
                                s2id_to_s2orc_input_path,
                                train_s2orc_paper_ids,
                                train_query_s2orc_paper_ids,
                                sample_queries_ratio,
                                graph_limit,
                                workers,
                                **triples_miner_kwargs)

    if skip_metadata:
        logger.info('Skipping metadata')
    else:
        if os.path.exists(metadata_path) and not override_metadata:
            logger.info('Skipping metdata (exists already')
        else:
            logger.info('Generating triple metadata')

            # Use metadata JSONL if file exists (this is faster than extracting from S2ORC dump)
            train_s2orc_paper_ids_metadata_path = train_s2orc_paper_ids + '.metadata.jsonl'
            if not os.path.exists(train_s2orc_paper_ids_metadata_path):
                train_s2orc_paper_ids_metadata_path = None

            get_metadata(triples_path, metadata_path, s2orc_metadata_dir, workers,
                         jsonl_metadata_path=train_s2orc_paper_ids_metadata_path)

    if skip_train:
        logger.info('Skipping train')
    else:
        if not os.path.exists(triples_path):
            logger.error('Cannot train: triples does not exist')
            return

        if not os.path.exists(metadata_path):
            logger.error('Cannot train: triples does not exist')
            return

        if os.path.exists(model_dir) and not override_train:
            logger.error(f'Model dir exists already: {model_dir}')
            return

        logger.info('Training model')

        train(
            model_dir,
            base_model_name_or_path,
            output_dir,
            scidocs_dir,
            scidocs_cuda_device=scidocs_cuda_device,
            use_dataset_cache=True,
            abstract_only=False,
            workers=workers,
            masked_language_modeling=masked_language_modeling,
            masked_language_modeling_weight=masked_language_modeling_weight,
            predict_embeddings=predict_embeddings,
            pooling_strategy=pooling_strategy,
            do_eval=False if skip_eval else True,
            val_or_test_or_both=val_or_test_or_both,
            max_sequence_length=max_sequence_length,
            graph_paper_ids_path=graph_paper_ids_path,
            graph_embeddings_path=graph_embeddings_path,
            **training_kwargs,
            # **training_args.to_sanitized_dict()
            # output_dir=model_dir
        )

    # Log additional (to Weights & Biases)
    if is_wandb_available() and hasattr(wandb.config, 'update'):
        wandb.config.update(dataclasses.asdict(triples_miner_args), allow_val_change=True)
        wandb.config.update({
            'workers': workers,
            'graph_limit': graph_limit,
            'graph_paper_ids_path': graph_paper_ids_path,
            'graph_embeddings_path': graph_embeddings_path,
            's2id_to_s2orc_input_path': s2id_to_s2orc_input_path,
            'train_s2orc_paper_ids': train_s2orc_paper_ids,
            'train_query_s2orc_paper_ids': train_query_s2orc_paper_ids,
            'query_oversampling_ratio': query_oversampling_ratio,
            'query_fold_k': query_fold_k,
            'query_n_folds': query_n_folds,
        }, allow_val_change=True)

    # if skip_eval:
    #     logger.info('Skipping eval')
    # else:
    #     logger.info('Evaluating model')
    #
    #     evaluate(model_dir, output_dir, scidocs_dir=scidocs_dir, use_dataset_cache=True)

    logger.info('done')


if __name__ == '__main__':
    fire.Fire()
    sys.exit(0)
