import inspect
import json
import logging
import os
import pickle
import random
import re
from collections import defaultdict
from multiprocessing import Pool
from typing import Union, List, Dict

import fire
import h5py
import numpy as np
from annoy import AnnoyIndex
from smart_open import open
from tqdm.auto import tqdm
from transformers import set_seed

from cli_specter import find_train_ids
from gdt.triples_miner import TriplesMinerArguments, AnnBackend
from gdt.triples_miner.worker import worker_generate_triples
from gdt.utils import split_into_n_chunks, write_func_args, normalize_in_parallel, get_workers

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# See https://github.com/allenai/s2orc#metadata-schema
S2ORC_METADATA_FIELDS = ['paper_id', 'title', 'abstract', 'arxiv_id', 'doi']


def get_specter_triples(output_path: str,
                        scidocs_dir: str,
                        specter_triples_path: str,
                        graph_paper_ids_path: str,
                        graph_embeddings_path: str,
                        s2id_to_s2orc_input_path: str,
                        train_s2orc_paper_ids: Union[str, List[str]] = None,
                        train_query_s2orc_paper_ids: Union[str, List[str]] = None,
                        sample_queries_ratio: float = 1.0,
                        graph_limit: str = 'specter',
                        workers: int = 10,
                        triples_miner_args: TriplesMinerArguments = None,
                        **triples_miner_kwargs):
    """
    Triple mining

    python cli_triples.py get_specter_triples ./data/gdt/train_triples.csv \
        --scidocs_dir /home/mostendorff/experiments/scidocs/data \
        --specter_triples_path ./data/specter/train_triples.csv \
        --paper_ids_path ./data/s2orc/entity_names_paper_id_0.json \
        --embeddings_path ./model/s2orc/embeddings_paper_id_0.v100.h5 \
        --s2id_to_s2orc_input_path ./data/specter/s2id_to_s2orc_paper_id.json \
        --easy_negatives_count 3 \
        --hard_negatives_count 2 \
        --ann_top_k 500 \
        --workers 50

    SPECTER:
    - To train our model we use about 146K query papers -> but train.pickle = 136_820 unique query papers
    - We empirically found it helpful to use 2 hard negatives (citations of citations) and 3 easy negatives (randomly selected papers) for each query paper.
    - SPECTER paper "This process results in about 684K training triples"

    :param sample_queries_ratio: Down- or up-sample query documents (default = 1.0 = no sampling)
    :param graph_limit: Limit citation graph to either SPECTER or S2ORC (= full graph = no limit). Choices: specter,s2orc
    :param triples_miner_args:
    :param ann_metric: Metric for ANN index (default: euclidean)
    :param train_query_s2orc_paper_ids: Path to JSON or list of query paper IDs used for training
    :param train_s2orc_paper_ids: Path to JSON or list of query paper IDs used for training
    :param ann_index_path: ANN index is saved to disk at this path (default: output_path.ann)
    :param ann_top_k: The lower k the harder the negatives
    :param ann_trees: More trees gives higher precision when querying ANN
    :param easy_negatives_count: SPECTER: 3 easy negatives (randomly selected papers) for each query paper.
    :param hard_negatives_count: SPECTER 2 hard negatives (citations of citations)
    :param triples_per_query: SPECTER: up to 5 training triples comprised of a query
    :param seed: Random seed
    :param graph_embeddings_path: Graph embeddings path (.h5 file)
    :param graph_paper_ids_path: Paper IDs of graph embeddings (.json files)
    :param scidocs_dir: SciDocs evaluation data dir
    :param output_path: Saves triples as CSV (columns: query_paper_id, positive_id, negative_id)
    :param workers: Threads for building ANN and mining triplets
    :param specter_triples_path: SPECTER triples data (get training paper ids: train_triples.csv)
    :param s2id_to_s2orc_input_path:
    :return:
    """
    train_query_s2orc_paper_ids_path = None

    # Log arg settings
    write_func_args(inspect.currentframe(), output_path + '.args.json')

    workers = get_workers(workers)
    triples_miner_args = TriplesMinerArguments.args_or_kwargs(triples_miner_args, triples_miner_kwargs)

    set_seed(triples_miner_args.seed)

    # SPECTER: To train our model we use about 146K query papers -> but train.pickle = 136_820 unique query papers
    # -> probably the others are part of validation!
    # query_papers_count = 136_820 #146_000
    # train_unique_papers_count = 311_860  # Number of unique papers in training set (same as SPECTER)

    # See https://github.com/allenai/specter/blob/master/scripts/pytorch_lightning_training_script/train.py#L44
    # training_size = 684100  #TODO why is this not equal to train_n? -> SPECTER paper "This process results in about 684K training triples"
    # wc-l =>  730001 data/s2orc/train_triples.csv
    # For each query paper we con- struct up to 5 training triples comprised of a query

    with open(graph_paper_ids_path) as f:
        s2orc_paper_ids = json.load(f)  # S2ORC Ids

    # Papers in train corpus and query papers
    if train_s2orc_paper_ids is not None and train_query_s2orc_paper_ids is not None:
        if isinstance(train_s2orc_paper_ids, list) and isinstance(train_query_s2orc_paper_ids, list):
            # ids are provided as arguments
            pass
        elif isinstance(train_s2orc_paper_ids, str) and isinstance(train_query_s2orc_paper_ids, str):
            # load from path
            train_query_s2orc_paper_ids_path = train_query_s2orc_paper_ids

            with open(train_s2orc_paper_ids) as f:
                train_s2orc_paper_ids = json.load(f)
            with open(train_query_s2orc_paper_ids_path) as f:
                train_query_s2orc_paper_ids = json.load(f)
        else:
            raise ValueError(f'Train S2ORC (query) paper ids not set: {type(train_s2orc_paper_ids)}')
    else:
        # Generate new train ids
        train_s2orc_paper_ids, train_query_s2orc_paper_ids = find_train_ids(scidocs_dir, specter_triples_path,
                                                                            s2id_to_s2orc_input_path, s2orc_paper_ids)

    logger.info(f'Loading graph embeddings: {graph_embeddings_path}')

    with h5py.File(graph_embeddings_path, "r") as hf:
        if triples_miner_args.ann_index_path is None:
            # load full graph embeddings

            # reading all embeddings at once is quicker than loading with indices
            graph_embeddings = hf["embeddings"][:, :]
        else:
            logger.warning('Skipping graph embeddings because `ann_index_path` is set and ANN index will not be build.')

            graph_embeddings = hf["embeddings"][:1, :] # placeholder (only the first entry)

    logger.info(f'Graph embeddings: {graph_embeddings.shape}')

    if graph_limit.lower() == 's2orc':
        # Utilize full citation graph of S2ORC
        train_embeddings = graph_embeddings
        train_s2orc_paper_id_to_idx = {pid: idx for idx, pid in enumerate(s2orc_paper_ids)}
        train_idx_to_s2orc_paper_id = {idx: pid for idx, pid in enumerate(s2orc_paper_ids)}

    elif graph_limit.lower() == 'specter':
        # Limit citation graph to the papers that are also used by SPECTER

        # Find indices in graph embeddings and extract vectors
        s2orc_paper_id_to_paper_idx = {pid: idx for idx, pid in enumerate(s2orc_paper_ids)}

        if not isinstance(train_s2orc_paper_ids, list):
            train_s2orc_paper_ids = list(train_s2orc_paper_ids)  # python sets are unordered -> convert to list!

        logger.warning('Limiting graph embedding to SPECTER')

        if triples_miner_args.ann_index_path is None:
            train_embeddings = np.array(
                [graph_embeddings[s2orc_paper_id_to_paper_idx[s2orc_id], :] for s2orc_id in train_s2orc_paper_ids])
        else:
            train_embeddings = graph_embeddings  # do not filter if ANN exist

        train_idx_to_s2orc_paper_id = {idx: pid for idx, pid in enumerate(train_s2orc_paper_ids)}
        train_s2orc_paper_id_to_idx = {pid: idx for idx, pid in enumerate(train_s2orc_paper_ids)}
    elif graph_limit.startswith('custom_'):
        logger.info('Using custom graph limit')

        if triples_miner_args.ann_index_path is None or not os.path.exists(triples_miner_args.ann_index_path):
            raise ValueError(f'Custom graph limit needs precomputed ANN: {triples_miner_args.ann_index_path}')

        train_embeddings = graph_embeddings  # do not filter if ANN exist

        train_idx_to_s2orc_paper_id = {idx: pid for idx, pid in enumerate(train_s2orc_paper_ids)}
        train_s2orc_paper_id_to_idx = {pid: idx for idx, pid in enumerate(train_s2orc_paper_ids)}
    else:
        raise ValueError(f'Unsupported graph limit: {graph_limit}')

    # Query sampling
    if sample_queries_ratio != 1.0:
        if sample_queries_ratio < 1:
            # down sampling
            sample_n = int(len(train_query_s2orc_paper_ids) * sample_queries_ratio)
            logger.info(f'Down sampling to {sample_n} ({sample_queries_ratio})')

            train_query_s2orc_paper_ids = random.sample(train_query_s2orc_paper_ids, sample_n)

            # save to disk for reproduciblity
            if train_query_s2orc_paper_ids_path:
                with open(train_query_s2orc_paper_ids_path.replace('.json', f'.sample_{sample_queries_ratio}.json'), 'w') as f:
                    json.dump(
                        train_query_s2orc_paper_ids,
                        f
                    )
        else:
            # up sampling
            raise NotImplementedError()

    return get_generic_triples(train_s2orc_paper_id_to_idx,
                               train_idx_to_s2orc_paper_id,
                               train_query_s2orc_paper_ids,
                               train_embeddings,
                               output_path,
                               triples_miner_args=triples_miner_args,
                               workers=workers,
                               normalize_graph_embeddings=True,
                               output_csv_header='query_paper_id,positive_id,negative_id',
                               )


def get_wiki_triples(
        output_path: str,
        metadata_output_path: str,
        wikipedia_dump_path: str,
        wikidata_embeddings_path: str,
        wikimapper_path: str,
        query_documents_count: int = 100_000,
        spacy_model: str = 'en_core_web_sm',
        seed: int = 0,
        use_cache: bool = False,
        reset_cache: bool = False,
        triples_miner_args: TriplesMinerArguments = None,
        **triples_miner_kwargs,
        ):
    """

    python cli_triples.py get_wiki_triples ./data/wiki/train_triples.csv ./data/wiki/train_metadata.jsonl \
        --wikipedia_dump_path /data/datasets/wikipedia_en/dumps/enwiki-20191101-pages-articles.filtered.jsonl \
        --wikidata_embeddings_path /data/datasets/wikidata/biggraph/wikidata_translation_v1.tsv.gz \
        --wikimapper_path /data/datasets/wikimapper/index_enwiki-20190420.db \
        --query_documents_count 10000 \
        --use_cache

    /data/datasets/wikimapper/index_enwiki-20190420.db

    https://github.com/attardi/wikiextractor -> JSON
    https://github.com/jcklie/wikimapper

    :param triples_miner_args:
    :param reset_cache: Delete existing cache files
    :param use_cache: Cache Wikipedia and Wikidata input
    :param seed: Random seed
    :param spacy_model: Name of Spacy model (en_core_web_sm; used for sentence splitting)
    :param query_documents_count:
    :param wikimapper_path:
    :param wikidata_embeddings_path:
    :param wikipedia_dump_path:
    :param metadata_output_path:
    :param output_path:
    :return:
    """
    from wikimapper import WikiMapper
    import spacy

    triples_miner_args = TriplesMinerArguments.args_or_kwargs(triples_miner_args, triples_miner_kwargs)

    set_seed(seed)

    wikidata_cache_path = output_path + '.wikidata_cache.pickle'
    wikipedia_cache_path = output_path + '.wikipedia_cache.json'

    if reset_cache:
        logger.info(f'Removing cache files: {[wikidata_cache_path, wikipedia_cache_path]}')
        if os.path.exists(wikidata_cache_path):
            os.remove(wikidata_cache_path)

        if os.path.exists(wikipedia_cache_path):
            os.remove(wikipedia_cache_path)

    nlp = spacy.load(spacy_model)

    mapper = WikiMapper(wikimapper_path)

    papers = []  # qid, title, first_sentence => paper_id, title, abstract

    # TODO wiki input cache!
    if use_cache and os.path.exists(wikipedia_cache_path):
        logger.info(f'Loading Wikipedia from cache: {wikipedia_cache_path} ')

        with open(wikipedia_cache_path) as f:
            papers = json.load(f)
    else:
        logger.info(f'Parsing Wikipedia dump from: {wikipedia_dump_path}')

        with open(wikipedia_dump_path) as f:
            for i, line in enumerate(tqdm(f, total=168_584)):  # TODO
                doc = json.loads(line)
                if len(doc['section_texts']) > 0:
                    intro_text = re.sub(r' \(.*?\)', '', doc['section_texts'][0].strip().replace('\'\'\'', ''))
                    """
                    # extract first sentence with regex -> is faster but does not work that well
                    match = re.search(r'([a-z]|[0-9]|")\.(\s+)[A-Z]', intro_text)
                    if not match:
                        continue                    
                    regex_fs = intro_text[:match.start() + 2]
    
                    """
                    intro_doc = nlp(intro_text, disable=["tagger", "ner"])
                    first_sentence = None
                    for sentence in intro_doc.sents:
                        first_sentence = sentence.text.strip()
                        break

                    if first_sentence and 50 < len(first_sentence) < 10_000:
                        # Wikimapper: "The page title is the last part of a Wikipedia url **unescaped** and spaces
                        #         replaced by underscores "
                        qid = mapper.title_to_id(doc['title'].replace(' ', '_'))  # replace whitespaces with underscores

                        if qid:
                            papers.append({
                                'paper_id': qid,
                                'title': doc['title'],
                                'abstract': first_sentence
                            })

                if (i % 10_000) == 0:
                    logger.info(f'Line {i:,} completed')
        if use_cache:
            # write to disk
            with open(wikipedia_cache_path, 'w') as f:
                json.dump(papers, f)

    qids = [p['paper_id'] for p in papers]
    qids_set = {p['paper_id'] for p in papers}

    logger.info(f'Loaded {len(qids):,} QIDs')

    logger.info(f'Writing metadata to {metadata_output_path}')

    with open(metadata_output_path, 'w') as f:
        for p in papers:
            # if p['paper_id'] in qids_in_triples:
            #
            f.write(json.dumps(p) + '\n')

    # find embeddings
    if use_cache and os.path.exists(wikidata_cache_path):
        logger.info(f'Loading graph from cache: {wikidata_cache_path}')
        with open(wikidata_cache_path, 'rb') as f:
            qid_to_embedding = pickle.load(f)
    else:
        qid_to_embedding = {}

        logger.info(f'Reading graph embeddings from {wikidata_embeddings_path}')

        with open(wikidata_embeddings_path, encoding='utf-8') as fp:  # smart open can read .gz files
            for i, line in enumerate(fp):
                cols = line.split('\t')
                qid = cols[0]

                if qid.startswith('<http://www.wikidata.org/entity/Q') and qid.endswith('>'):
                    qid = qid.replace('<http://www.wikidata.org/entity/', '').replace('>', '')

                    if qid in qids_set:
                        qid_to_embedding[qid] = np.array(cols[1:]).astype(np.float)

        if use_cache:
            # write to disk
            with open(wikidata_cache_path, 'wb') as f:
                pickle.dump(qid_to_embedding, f)

    logger.info(f'Loaded {len(qid_to_embedding):,} QIDs with embeddings')

    # Filter for existing embeddings
    qids = [qid for qid in qids if qid in qid_to_embedding]

    logger.info(f'Example QIDs: {qids[:5]}')
    logger.info(f'Example embedding keys: {list(qid_to_embedding.keys())[:5]}')

    query_document_ids = random.sample(qids, query_documents_count)

    graph_embeddings = np.array([qid_to_embedding[qid] for qid in qids if qid in qid_to_embedding])

    logger.info(f'Graph embeddings: {graph_embeddings.shape}')

    document_id_to_idx = {qid: idx for idx, qid in enumerate(qids)}
    idx_to_document_id = {idx: qid for idx, qid in enumerate(qids)}

    return get_generic_triples(
        document_id_to_idx,
        idx_to_document_id,
        query_document_ids,
        graph_embeddings,
        output_path,
        triples_miner_args=triples_miner_args,
    )


def get_generic_triples(
        document_id_to_idx: Dict[str, int],
        idx_to_document_id: Dict[int, str],
        query_document_ids: List[str],
        graph_embeddings: Union[List, np.ndarray],
        output_path: str,
        # triples_per_query: int = 5,
        # hard_negatives_count: int = 2,
        # easy_negatives_count: int = 3,
        # ann_trees: int = 1000,
        # ann_top_k: int = 500,
        # ann_index_path: str = None,
        # ann_metric: str = 'euclidean',
        triples_miner_args: TriplesMinerArguments,
        workers: int = 10,
        normalize_graph_embeddings: bool = True,
        output_csv_header: str = 'query_paper_id,positive_id,negative_id',
        ):

    # assert hard_negatives_count + easy_negatives_count == triples_per_query

    logger.info(f'Query papers: {len(query_document_ids):,}')
    logger.info(f'Triples per query: {triples_miner_args.triples_per_query}')
    logger.info(f'Triple miner args: {triples_miner_args}')

    set_seed(triples_miner_args.seed)

    if triples_miner_args.ann_workers is not None and triples_miner_args.ann_workers > 0:
        workers = triples_miner_args.ann_workers

    if triples_miner_args.ann_index_path is None:
        triples_miner_args.ann_index_path = output_path + '.' + triples_miner_args.ann_backend

    graph_embedding_size = len(graph_embeddings[0])

    if os.path.exists(triples_miner_args.ann_index_path):
        # Reuse existing ANN index
        logger.info(f'Reusing existing ANN index from {triples_miner_args.ann_index_path}')
    else:
        if triples_miner_args.ann_backend == AnnBackend.ANNOY:
            # New ANN index
            ann_index = AnnoyIndex(graph_embedding_size, triples_miner_args.ann_metric)
            ann_index.set_seed(triples_miner_args.seed)

            # Length of item vector that will be indexed
            # Cosine distance is equivalent to Euclidean distance of normalized vectors = sqrt(2-2*cos(u, v))

            # normalize vectors for cosine similarity
            if normalize_graph_embeddings:
                logger.info('Normalizing graph embeddings ...')
                graph_embeddings = normalize_in_parallel(graph_embeddings, workers)

            logger.info('Adding to ANN index')
            for idx, embed in enumerate(graph_embeddings):
                ann_index.add_item(idx, embed)

            logger.info(f'Building ANN index with trees={triples_miner_args.ann_trees} and workers={workers}')

            ann_index.build(triples_miner_args.ann_trees, n_jobs=workers)

            logger.info(f'Saving ANN index to {triples_miner_args.ann_index_path}')
            ann_index.save(triples_miner_args.ann_index_path)
        else:
            raise NotImplementedError(f'Cannot build ANN index with backend = {triples_miner_args.ann_backend} '
                                      f'(use extra CLI script instead)')
    # Easy negatives = random papers
    # easy_negatives = []
    #
    # for i in range(easy_negatives_count):
    #     random_papers = list(query_document_ids)
    #     random.shuffle(random_papers)
    #
    #     easy_negatives.append(random_papers)

    if workers == 1:
        logger.info(f'Triple generation with single thread')

        triples = worker_generate_triples(0, query_document_ids, document_id_to_idx, idx_to_document_id,
                                          graph_embedding_size, triples_miner_args.seed, triples_miner_args.__dict__)

    else:
        logger.info(f'Starting {workers} workers for triple generation')

        worker_data = zip(
            list(range(workers)),  # worker ids
            # split_into_n_chunks(list(zip(query_document_ids, *easy_negatives)), workers),  # items
            split_into_n_chunks(list(query_document_ids), workers),  # items

            # static arguments (same for all workers)
            [document_id_to_idx] * workers,
            [idx_to_document_id] * workers,
            [graph_embedding_size] * workers,
            [triples_miner_args.seed] * workers,
            [triples_miner_args.__dict__] * workers,
            # [ann_index_path] * workers,
            # [ann_top_k] * workers,
            # [ann_metric] * workers,
            # [triples_per_query] * workers,
            # [easy_negatives_count] * workers,
            # [hard_negatives_count] * workers,
        )

        # Run threads
        with Pool(workers) as pool:
            # pool.map(functools.partial(print_data, first=False), test_data)
            # pool_outputs = list(pool.starmap(functools.partial(worker_generate_triples,
            #                                                    paper_id_to_idx=document_id_to_idx,
            #                                                    idx_to_paper_id=idx_to_document_id,
            #                                                    ann_vector_size=graph_embeddings,
            #                                                    args=triples_miner_args), zip(
            #     list(range(workers)),  # worker ids
            #     split_into_n_chunks(list(query_document_ids), workers)
            # )))
            pool_outputs = list(pool.starmap(worker_generate_triples, worker_data))

            # takes some time to start
            # pool_outputs = list(tqdm(pool.imap_unordered(worker.generate_triples, list(zip(train_query_s2orc_paper_ids, *easy_negatives))),
            #                          total=len(train_query_s2orc_paper_ids)))

        # Merge thread outputs
        triples = [i for batch in pool_outputs for i in batch]

    logger.info(f'Triples mined: {len(triples):,}')

    if output_path:
        # write to disk
        logger.info(f'Writing {len(triples):,} triples to {output_path}')

        with open(os.path.join(output_path), 'w') as f:
            f.write(output_csv_header + '\n')
            for query_paper_id, pos_id, neg_id in triples:
                f.write(f'{query_paper_id},{pos_id},{neg_id}\n')
    else:
        return triples


def worker_extract_metadata(worker_id, batch_fns, needed_paper_ids, s2orc_metadata_dir):
    batch_metadata = []

    for batch_fn in tqdm(batch_fns, desc=f'Worker {worker_id}'):
        with open(os.path.join(s2orc_metadata_dir, batch_fn)) as batch_f:
            for i, line in enumerate(batch_f):
                meta = json.loads(line)

                if meta['paper_id'] in needed_paper_ids:
                    batch_metadata.append({f: meta[f] for f in S2ORC_METADATA_FIELDS})

    return batch_metadata


def worker_extract_metadata_with_lines(worker_id, batch_fns_with_lines, s2orc_metadata_dir):
    batch_metadata = []

    for batch_fn, needed_lines in tqdm(batch_fns_with_lines, desc=f'Worker {worker_id}'):
        needed_lines = set(needed_lines)

        with open(os.path.join(s2orc_metadata_dir, batch_fn)) as batch_f:
            for i, line in enumerate(batch_f):
                if i in needed_lines:
                    meta = json.loads(line)
                    batch_metadata.append({f: meta[f] for f in S2ORC_METADATA_FIELDS})

    return batch_metadata


def get_plaintext_from_metadata(jsonl_metadata_path: str, output_path: str, override: bool = False):
    """
    Saves titles and abstract from papers as plaintext to disk (for language modeling).

    Format: "<title>: <abstract>\n"

    Usage:

    python cli_triples.py get_plaintext_from_metadata ${QUERY_DIR}/s2orc_paper_ids.json.metadata.jsonl ${QUERY_DIR}/s2orc_paper_ids.json.metadata.txt

    :param override: Override existing output file
    :param jsonl_metadata_path: Path to metadata JSONL file (see get_metadata)
    :param output_path: Save txt file at this location
    :return:
    """
    if os.path.exists(output_path) and not override:
        logger.error(f'Output exists already: {output_path}')
        return

    logger.info(f'Extracting metadata from JSONL: {jsonl_metadata_path}')
    logger.info(f'Writing output to {output_path}')

    with open(output_path, 'w') as out_f:
        with open(jsonl_metadata_path) as f:
            for line in f:
                paper = json.loads(line)

                out_f.write(paper['title'] + ": ")
                out_f.write((paper['abstract'] or '') + "\n")

    logger.info('done')


def get_metadata(input_path, output_path, s2orc_metadata_dir, workers: int = 10, id_mapping_path: str = None,
                 jsonl_metadata_path: str = None):
    """
    Extract meta data from S2ORC for triples

    python cli_triples.py get_metadata ${EXP_DIR}/train_triples.csv ${EXP_DIR}/train_metadata.jsonl \
        --s2orc_metadata_dir ${S2ORC_METADATA_DIR} --workers 50

    python cli_triples.py get_metadata ${OLD_DIR}/train_triples.csv ${OLD_DIR}/train_metadata.jsonl \
        --s2orc_metadata_dir ${S2ORC_METADATA_DIR} --workers 10

    python cli_triples.py get_metadata ${GRIDSEARCH_DIR}/s2orc_paper_ids.json ${GRIDSEARCH_DIR}/s2orc_paper_ids.json.metadata.jsonl \
        --s2orc_metadata_dir ${S2ORC_METADATA_DIR} --workers 50


    :param jsonl_metadata_path: If this JSONL path is provided, S2ORC is not used
    :param id_mapping_path: Path to ID Mapping JSON (default: s2orc_metadata_dir / s2orc_metadata_id_mapping.json)
    :param workers: Number of threads for parallel processing
    :param input_path: CSV with triples or JSON with IDs
    :param s2orc_metadata_dir: S2ORC metadata directory (.jsonl.gz files)
    :param output_path: Save JSONL file with metadata at this path
    :return:
    """

    # Default (see cli_s2orc.get_metadata_id_mapping)
    if id_mapping_path is None:
        id_mapping_path = os.path.join(s2orc_metadata_dir, 's2orc_metadata_id_mapping.json')

    if input_path.endswith('.json'):
        logger.info(f'Loading IDs from JSON: {input_path}')

        with open(input_path) as f:
            needed_paper_ids = set(json.load(f))

    else:
        # load triples from disk
        triples = []

        with open(input_path) as f:
            for i, line in enumerate(f):
                if i > 0:
                    triples.append(line.strip().split(','))

        logger.info(f'Loaded {len(triples):,} triples')

        needed_paper_ids = set([pid for triple in triples for pid in triple])

    logger.info(f'Needed metadata for {len(needed_paper_ids):,}')

    if jsonl_metadata_path:
        logger.info(f'Extracting metadata from JSONL: {jsonl_metadata_path}')

        train_metadata = []
        with open(jsonl_metadata_path) as f:
            for line in f:
                paper = json.loads(line)

                if paper['paper_id'] in needed_paper_ids:
                    train_metadata.append(paper)

    else:
        logger.info(f'Extracting metadata from S2ORC: {s2orc_metadata_dir}')

        # Meta data files
        batch_fns = [batch_fn for batch_fn in os.listdir(s2orc_metadata_dir) if batch_fn.endswith('.jsonl.gz')]
        logger.info(f'Files available: {len(batch_fns):,}')

        # Does a mapping file exists?
        if os.path.exists(id_mapping_path):
            # The mapping allows parsing only those JSON lines that contained needed metadata
            # (this should be faster than parsing all lines and deciding then based on ID)
            # ---
            # on serv-9212
            # +5min for reading the mapping file
            # +2min for extracting and writing metadata
            #
            # on  RTX309@slurm
            # +4:30min for reading mapping file
            # +2min for extracing and writing metadata
            #
            # ==> actually not faster use no mapping!
            logger.warning('DO NOT USE ID MAPPING SINCE IT IS SLOWER!')

            with open(id_mapping_path) as f:
                logger.info(f'Reading ID mapping from: {id_mapping_path}')
                id_mapping = json.load(f)  # batch_fn => [id, line_idx]

                batch_fn_to_needed_lines = defaultdict(list)  # batch_fn => list of line_idx

                # rewrite mapping
                paper_id_to_batch_fn_line_idx = {}
                for batch_fn, papers in id_mapping.items():
                    for paper_id, idx in papers:
                        paper_id_to_batch_fn_line_idx[paper_id] = [batch_fn, idx]

                for paper_id in needed_paper_ids:
                    batch_fn, line_idx = paper_id_to_batch_fn_line_idx[paper_id]
                    batch_fn_to_needed_lines[batch_fn].append(line_idx)

                batch_fn_to_needed_lines_list = list(batch_fn_to_needed_lines.items())

                logger.info(f'Extracting metadata with {workers} workers from {len(batch_fn_to_needed_lines_list)} files')

                # worker_id, batch_fns_with_lines, s2orc_metadata_dir
                worker_data = zip(
                    list(range(workers)),  # worker ids
                    split_into_n_chunks(batch_fn_to_needed_lines_list, workers),
                    # static arguments (same for all workers)
                    [s2orc_metadata_dir] * workers,
                )

                # Run threads
                with Pool(workers) as pool:
                    pool_outputs = list(pool.starmap(worker_extract_metadata_with_lines, worker_data))

        else:
            # Read all lines and check based on `needed_paper_ids`
            # ---
            # on serv-9212
            # + 4:30min
            #
            # on RTX309@slurm
            # + 4min
            logger.info(f'Extracting metadata with workers: {workers} (all files + lines)')

            # worker_id, batch_fns, needed_paper_ids, s2orc_metadata_dir
            worker_data = zip(
                list(range(workers)),  # worker ids
                split_into_n_chunks(batch_fns, workers),
                # static arguments (same for all workers)
                [needed_paper_ids] * workers,
                [s2orc_metadata_dir] * workers,
            )

            # Run threads
            with Pool(workers) as pool:
                pool_outputs = list(pool.starmap(worker_extract_metadata, worker_data))

        # Merge thread outputs
        train_metadata = [i for b in pool_outputs for i in b]

    logger.info(f'Metadata parsed. {len(train_metadata):,} train_metadata')

    if output_path:
        # Write to disk
        logger.info(f'Writing {len(train_metadata):,} metadata to {output_path}')

        with open(os.path.join(output_path), 'w') as f:
            for paper in train_metadata:
                f.write(json.dumps(paper) + '\n')

        logger.info('done')

    else:
        return train_metadata


def extract_ids_from_triples(triples_input_path, output_dir):
    """
    Extract original IDs used by the triples (for debugging)

     python cli_triples.py extract_ids_from_triples ./data/scigdt/specter/gdt.epoch_20_768d.easy_3.hard_2.k_500/_original__train_triples.csv ./data/scigdt/specter/gdt.epoch_20_768d.easy_3.hard_2.k_500

    :param triples_input_path: Path to train_triples.csv
    :param output_dir: Write `s2orc_paper_ids.json` and `query_s2orc_paper_ids.json` into this directory.
    :return:
    """

    # load triples from disk
    all_ids = []
    query_ids = []

    logger.info(f'Reading from {triples_input_path}')

    with open(os.path.join(triples_input_path)) as f:
        for i, line in enumerate(f):
            if i > 0:
                triple = line.strip().split(',')

                all_ids += triple
                query_ids.append(triple[0])

    # unique
    all_ids = set(all_ids)
    query_ids = set(query_ids)

    logger.info(f'All IDs: {len(all_ids):,}; Query IDs: {len(query_ids):,}')

    # write to disk
    with open(os.path.join(output_dir, 's2orc_paper_ids.json'), 'w') as f:
        json.dump(list(all_ids), f)
    with open(os.path.join(output_dir, 'query_s2orc_paper_ids.json'), 'w') as f:
        json.dump(list(query_ids), f)


if __name__ == '__main__':
    fire.Fire()
