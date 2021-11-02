import json
import logging
import os
import pickle
from collections import defaultdict
from multiprocessing import Pool
from typing import Union, List, Optional

import fire
import numpy as np
import torch
from annoy import AnnoyIndex
from gensim import matutils
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from smart_open import open
from tqdm.auto import tqdm
from transformers import set_seed

from gdt.utils import get_graph_embeddings, get_scidocs_metadata, split_into_n_chunks, get_workers

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def worker_extract_citations(batch_fp):
    batch_cits = []
    batch_paper_ids = []

    with open(batch_fp) as batch_f:
        for i, line in enumerate(batch_f):
            meta = json.loads(line)

            batch_paper_ids.append(meta['paper_id'])

            # TODO use outbound or inbound citations? or both?
            # => no difference

            if meta['has_outbound_citations']:
                for to_id in meta['outbound_citations']:
                    batch_cits.append((meta['paper_id'], to_id))

            # if meta['has_inbound_citations']:
            #    for from_id in meta['inbound_citations']:
            #        batch_cits.append((from_id, meta['paper_id']))

    return batch_paper_ids, batch_cits


def write_citations(cits, out_fp, nodes_count, col_sep, line_sep):
    logger.info(f'Writing {len(cits):,} to {out_fp}')

    header = f'''# Directed graph
    # Directed citation graph from S2ORC
    # Nodes: {nodes_count} Edges: {len(cits)}
    # FromNodeId{col_sep}ToNodeId
    '''

    with open(out_fp, 'w') as f:
        f.write(header)
        for from_id, to_id in tqdm(cits, desc='Writing to disk', total=len(cits)):
            f.write(from_id + col_sep + to_id + line_sep)


def get_citations(s2orc_metadata_dir: str, output_dir=None, workers: int = 10, test_ratio: float = 0., seed: int = 0):
    """
    Extracts citations from S2ORC metadata.

    python cli_s2orc.py extract_citations /data/datasets/s2orc/20200705v1/full/metadata ./data/s2orc


    :param seed: Set random seed for train/test split
    :param test_ratio: Ratio of citation edges that are used as test set
    :param s2orc_metadata_dir: S2ORC directory with metadata files (.jsonl.gz)
    :param output_dir: Write `citations.tsv` (`citations.train.csv` and `citations.test.csv`), `nodes.csv`
        and `paper_ids.csv` (papers that do not have any citations)
    :param workers: Number of threads for parallel processing
    :return: citations, nodes, paper_ids
    """
    set_seed(seed)

    line_sep = '\n'
    col_sep = '\t'

    # Meta data files
    batch_fps = [os.path.join(s2orc_metadata_dir, batch_fn) for batch_fn in os.listdir(s2orc_metadata_dir) if batch_fn.endswith('.jsonl.gz')]

    logger.info(f'Files available: {len(batch_fps):,}')

    # Run threads
    with Pool(workers) as pool:
        pool_outputs = list(tqdm(pool.imap_unordered(worker_extract_citations, batch_fps), total=len(batch_fps)))

    # Merge thread outputs
    cits = [i for batch_paper_ids, batch_cits in pool_outputs for i in batch_cits]
    paper_ids = [i for batch_paper_ids, batch_cits in pool_outputs for i in batch_paper_ids]

    # 467,588,220 citations (outbound only)
    # 935,176,440 citations (in- and outbound)
    logger.info(f'Metadata parsed. {len(cits):,} citations')

    paper_ids_set = set(paper_ids)

    # Paper IDs: 136,595,995
    logger.info(f'Paper IDs: {len(paper_ids):,}')

    # Papers with citations
    nodes = set([i for cit in cits for i in cit])
    logger.info(f'Unique nodes (from citations): {len(nodes):,}')

    if output_dir and os.path.exists(output_dir):

        # Papers IDs
        with open(os.path.join(output_dir, 'paper_ids.csv'), 'w') as f:
            f.write('\n'.join(paper_ids))

        #  Unique nodes (outbound only): 52,620,852
        #  Unique nodes (in- and outbound): 52,620,852

        with open(os.path.join(output_dir, 'nodes.csv'), 'w') as f:
            f.write('\n'.join(nodes))

        if test_ratio > 0:
            logger.info(f'Splitting citations into train/test set: ratio = {test_ratio}')
            train_cits, test_cits = train_test_split(cits, test_size=test_ratio)

            logger.info(f'Train: {len(train_cits):,}; Test: {len(test_cits):,}')
            write_citations(train_cits, os.path.join(output_dir, 'citations.train.tsv'), len(nodes), col_sep, line_sep)
            write_citations(test_cits, os.path.join(output_dir, 'citations.test.tsv'), len(nodes), col_sep, line_sep)
        else:
            write_citations(cits, os.path.join(output_dir, 'citations.tsv'), len(nodes), col_sep, line_sep)

        logger.info('done')

    else:
        # output dir is not set, return instead
        return cits, nodes, paper_ids


def worker_extract_inbound_citations(batch_fp):
    """
    Worker method for `get_inbound_citations_count`
    """
    batch_cits_counts = []

    with open(batch_fp) as batch_f:
        for i, line in enumerate(batch_f):
            meta = json.loads(line)

            if meta['has_inbound_citations']:
                batch_cits_counts.append((meta['paper_id'], len(meta['inbound_citations'])))

    return batch_cits_counts


def get_inbound_citations_count(s2orc_metadata_dir: str, output_path: str, workers: int = 10):
    """
    Extracts inbound citation count from S2ORC and saves id-count mapping as JSON file.

    Usage:

    python cli_s2orc.py get_inbound_citations_count ${S2ORC_METADATA_DIR} ${SPECTER_DIR}/s2orc_inbound_citations.json.gz \
        --workers ${WORKERS}

    :param s2orc_metadata_dir: Directory with S2ORC metadata (.jsonl.gz) files
    :param output_path: Save JSON to this path
    :param workers: Number of threads
    :return:
    """
    if os.path.exists(output_path):
        logger.error(f'Output exists already: {output_path}')
        return

    # Meta data files
    batch_fps = [os.path.join(s2orc_metadata_dir, batch_fn) for batch_fn in os.listdir(s2orc_metadata_dir) if batch_fn.endswith('.jsonl.gz')]

    logger.info(f'Files available: {len(batch_fps):,}')

    # Run threads
    with Pool(workers) as pool:
        pool_outputs = list(tqdm(pool.imap_unordered(worker_extract_inbound_citations, batch_fps), total=len(batch_fps)))

    cits_counts = {pid: count for b in pool_outputs for pid, count in b}

    logger.info(f'Extracted citation counts for {len(cits_counts):,} papers')

    with open(output_path, 'w') as f:
        json.dump(cits_counts, f)

    logger.info(f'Saved to {output_path}')


def get_scidocs_title_mapping(scidocs_dir, s2orc_metadata_dir, output_fp, workers: int = 10):
    """
    Find S2ORC paper ids based on title

    python cli_s2orc.py get_scidocs_title_mapping /home/mostendorff/experiments/scidocs/data \
        /data/datasets/s2orc/20200705v1/full/metadata ./data/scidocs_s2id_to_s2orc_paper_id.json

    :param scidocs_dir:
    :param s2orc_metadata_dir:
    :param output_fp:
    :param workers:
    :return:
    """

    # SciDocs Metadata
    scidocs_metadata = get_scidocs_metadata(scidocs_dir)

    scidocs_paper_ids = set([paper_id for ds, ds_meta in scidocs_metadata.items() for paper_id, paper_meta in ds_meta.items()])

    logger.info(f'scidocs_paper_ids = {len(scidocs_paper_ids):,}')

    scidocs_titles = [paper_meta['title'] for ds, ds_meta in scidocs_metadata.items() for paper_id, paper_meta in
                      ds_meta.items()]

    logger.info(f'scidocs_titles = {len(scidocs_titles):,}')

    unique_scidocs_titles = set(scidocs_titles)

    logger.info(f'unique_scidocs_titles = {len(unique_scidocs_titles):,}')

    # S2ORC Meta data files
    batch_fns = [batch_fn for batch_fn in os.listdir(s2orc_metadata_dir) if batch_fn.endswith('.jsonl.gz')]

    logger.info(f'Files available: {len(batch_fns):,}')

    def worker_extract_matching_titles(batch_fn):
        batch = []

        with open(os.path.join(s2orc_metadata_dir, batch_fn)) as batch_f:
            for i, line in enumerate(batch_f):
                meta = json.loads(line)

                if meta['title'] in unique_scidocs_titles:
                    batch.append((
                        meta['paper_id'], meta['title']
                    ))
        return batch

    # Run threads
    with Pool(workers) as pool:
        pool_outputs = list(tqdm(pool.imap_unordered(worker_extract_matching_titles, batch_fns), total=len(batch_fns)))

    # Merge thread outputs
    matching_titles = [i for o in pool_outputs for i in o]

    logger.info(f'Metadata parsed. {len(matching_titles):,} matching_titles')

    # Build mapping from titles to ids
    title_to_s2orc_paper_ids = defaultdict(list)
    for paper_id, t in matching_titles:
        title_to_s2orc_paper_ids[t].append(paper_id)

    s2id_to_s2orc_paper_id = {}

    ambiguous_s2orc_paper_ids = defaultdict(list)

    for ds, ds_meta in scidocs_metadata.items():
        for s2_id, paper_meta in ds_meta.items():

            if s2_id in s2id_to_s2orc_paper_id:
                # skip
                continue

            if paper_meta['title'] in title_to_s2orc_paper_ids:
                s2orc_paper_ids = title_to_s2orc_paper_ids[paper_meta['title']]

                # Ignore ambiguous paper ids
                if len(s2orc_paper_ids) == 1:
                    s2id_to_s2orc_paper_id[s2_id] = s2orc_paper_ids[0]
                else:
                    ambiguous_s2orc_paper_ids[s2_id] += s2orc_paper_ids

    logger.warning(f'Ambiguous paper ids / titles: {len(ambiguous_s2orc_paper_ids):,}')

    logger.info(f'Mapping for {len(s2id_to_s2orc_paper_id):,} IDs')

    if output_fp:
        # save to disk
        with open(output_fp, 'w') as f:
            json.dump(s2id_to_s2orc_paper_id, f)
    else:
        # return data
        return s2id_to_s2orc_paper_id


def get_s2orc_scidocs_mappings(
        s2id_to_paper_input_paths: Union[str, List[str]],
        s2id_to_s2orc_paper_id_input_paths: Union[str, List[str]],
        output_path: Union[None, str]
    ):
    """
    Merge mappings from S2 IDs to S2ORC IDs

    python cli_s2orc.py get_s2orc_scidocs_mappings \
        ./data/specter/id2paper.json,./data/specter_train_source_papers/id2paper.json,./data/scidocs_s2orc/id2paper.json \
        ./data/scidocs_s2id_to_s2orc_paper_id.json \
        ./data/specter/s2id_to_s2orc_paper_id.json

    :param s2id_to_paper_input_paths: List of S2 API response files (comma separated, .json)
    :param s2id_to_s2orc_paper_id_input_paths: List of S2ID-S2ORC ID mappings as JSON (comma separated, .json)
    :param output_path: Output path S2ID-S2ORC ID mapping JSON
    :return:
    """
    if isinstance(s2id_to_paper_input_paths, str):
        s2id_to_paper_input_paths = s2id_to_paper_input_paths.split(',')

    if isinstance(s2id_to_s2orc_paper_id_input_paths, str):
        s2id_to_s2orc_paper_id_input_paths = s2id_to_s2orc_paper_id_input_paths.split(',')

    # Load S2 API responses from disk
    s2_id_to_paper_list = []

    for fp in s2id_to_paper_input_paths:
        logger.info(f'Loading from {fp}')

        with open(fp) as f:
            s2_id_to_paper = json.load(f)
        s2_id_to_paper_list.append(s2_id_to_paper)

    # S2 ID to S2ORC mapping
    s2id_to_s2orc_paper_id = {}

    for s2_id_to_paper in s2_id_to_paper_list:
        for s2id, paper in s2_id_to_paper.items():
            if s2id not in s2id_to_s2orc_paper_id:
                s2id_to_s2orc_paper_id[s2id] = str(paper['corpusId'])

    # Predefined ID mappings (e.g., from titles)
    for fp in s2id_to_s2orc_paper_id_input_paths:
        logger.info(f'Loading from {fp}')
        with open(fp) as f:
            titles_s2id_to_s2orc_paper_id = json.load(f)

        # titles (last since probably inaccurate)
        for s2id, s2orc_id in titles_s2id_to_s2orc_paper_id.items():
            if s2id not in s2id_to_s2orc_paper_id:
                s2id_to_s2orc_paper_id[s2id] = s2orc_id

    logger.info(f'Mappings for {len(s2id_to_s2orc_paper_id):,} S2 IDs')

    if output_path:
        # write to disk
        with open(output_path, 'w') as f:
            json.dump(s2id_to_s2orc_paper_id, f)
    else:
        return s2id_to_s2orc_paper_id


def build_annoy(
        graph_embeddings_path: str,
        index_path: str,
        paper_ids: Optional[str] = None,
        include_paper_ids: Optional[str] = None,
        do_normalize: bool = False,
        ann_metric: str = 'euclidean',
        ann_trees: int = 100,
        ann_on_disk_build: bool = False,
        workers: int = 10,
        seed: int = 0,
        ):
    """
    Build approximate nearest neighbors index

    Usage:

    python cli_s2orc.py build_annoy ${S2ORC_EMBEDDINGS} \
        ${BASE_DIR}/data/biggraph/models/s2orc/epoch_20/index__dot__1000-trees.ann \
        --do_normalize --ann_metric dot --ann_trees 1000 --workers ${WORKERS} --ann_on_disk_build

    :param include_paper_ids: See get_graph_embeddings
    :param paper_ids: See get_graph_embeddings
    :param ann_on_disk_build:
    :param seed:
    :param graph_embeddings_path:
    :param index_path:
    :param do_normalize:
    :param ann_metric:
    :param ann_trees:
    :param workers:
    :return:
    """

    graph_embeddings = get_graph_embeddings(graph_embeddings_path, do_normalize, workers, paper_ids, include_paper_ids)
    graph_embedding_size = len(graph_embeddings[0])

    ann_index = AnnoyIndex(graph_embedding_size, ann_metric)
    ann_index.set_seed(seed)

    for idx, embed in tqdm(enumerate(graph_embeddings), desc='Adding to index', total=len(graph_embeddings)):
        ann_index.add_item(idx, embed)

    # delete to free up memory
    del graph_embeddings

    if ann_on_disk_build:
        logger.info('Building on disk')
        ann_index.on_disk_build(index_path + '.build')
    else:
        logger.info(f'Building ANN index with trees={ann_trees} and workers={workers}')
        ann_index.build(ann_trees, n_jobs=workers)

    logger.info(f'Saving ANN index to {index_path}')

    ann_index.save(index_path)


def build_faiss(
        graph_embeddings_path: str,
        index_path: str,
        string_factory: str = 'PCA64,IVF16384_HNSW32,Flat',
        metric_type: int = 0,
        do_normalize: bool = False,
        paper_ids: Optional[str] = None,
        include_paper_ids: Optional[str] = None,
        workers: int = 10,
        batch_size: int = 1000,
        train_size: int = 0,
        seed: int = 0,
        device: Union[int, str] = -1,):
    """
    Build FAISS ANN index

    # on full S2ORC
    python cli_s2orc.py build_faiss ${S2ORC_EMBEDDINGS} \
        ${BASE_DIR}/data/ann_benchmark/IVF16384_HNSW32,Flat.faiss \
        --string_factory IVF16384_HNSW32,Flat \
        --do_normalize \
        --batch_size 512 --workers ${WORKERS} --device 0

    # on SPECTER papers only
    python cli_s2orc.py build_faiss ${S2ORC_EMBEDDINGS} \
        ${OLD_DIR}/IVF16384_HNSW32,Flat.faiss \
        --string_factory IVF16384_HNSW32,Flat \
        --paper_ids ${S2ORC_PAPER_IDS} --include_paper_ids ${OLD_DIR}/s2orc_paper_ids.json \
        --do_normalize \
        --batch_size 512 --workers ${WORKERS} --device 0

    # TODO maybe limit train_size if GPU memory is not enough
    # TODO use PCA -> PCA128, PCA256?

    IndexFlat: the vectors are stored without compression
    IndexIVF:IVF16384 (2^14): The feature space is partitioned into nlist cells.
    HNSW32: M is the number of neighbors used in the graph. A larger M is more accurate but uses more memory


    :param include_paper_ids: See get_graph_embeddings
    :param paper_ids: See get_graph_embeddings
    :param batch_size:
    :param train_size:
    :param device: GPU index or "all"
    :param string_factory:  https://github.com/facebookresearch/faiss/wiki/The-index-factory
    :param metric_type: METRIC_INNER_PRODUCT = 0
    :param graph_embeddings_path:
    :param index_path: Output path
    :param do_normalize: Normalize input graph embeddings (for cosine similarity)
    :param workers:
    :return:
    """
    import faiss

    set_seed(seed)  # make reproducible

    if isinstance(string_factory, tuple) or isinstance(string_factory, list):
        string_factory = ','.join(string_factory)  # force to be string

    workers = get_workers(workers)

    graph_embeddings = get_graph_embeddings(graph_embeddings_path, do_normalize, workers, paper_ids, include_paper_ids)
    graph_embedding_size = len(graph_embeddings[0])

    index = faiss.index_factory(graph_embedding_size, string_factory, metric_type)

    if isinstance(device, int) and device > -1:
        logger.info(f'Use GPU device: {device}')
        faiss_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(faiss_res, device, index)
    elif device == 'all':
        # use all available GPUs
        logger.info(f'Use all available GPUs: {faiss.get_num_gpus()}')

        index = faiss.index_cpu_to_all_gpus(  # build the index
            index
        )

    verbose = True
    index.verbose = verbose

    if hasattr(index, "index") and index.index is not None:
        index.index.verbose = verbose
    if hasattr(index, "quantizer") and index.quantizer is not None:
        index.quantizer.verbose = verbose
    if hasattr(index, "clustering_index") and index.clustering_index is not None:
        index.clustering_index.verbose = verbose

    if train_size > 0:
        train_vecs = graph_embeddings[:train_size]
    else:
        train_vecs = graph_embeddings

    logger.info(f'Training ... train_size = {train_size:,}')

    index.train(train_vecs)

    # write to disk
    if (isinstance(device, int) and device > -1) or device == 'all':
        logger.info('Index back to CPU')

        index = faiss.index_gpu_to_cpu(index)

    for i in tqdm(range(0, len(graph_embeddings), batch_size), desc='Adding'):
        vecs = graph_embeddings[i: i + batch_size]
        index.add(vecs)

        # See https://github.com/facebookresearch/faiss/issues/1517
        # index.reclaimMemory()

    faiss.write_index(index, index_path)

    logger.info(f'Index saved at {index_path}')

    # Save config
    config_path = index_path + '.faiss_config.json'
    config = {
        'faiss_version': faiss.__version__,
        'seed': seed,
        'string_factory': string_factory,
        'device': device,
        'batch_size': batch_size,
        'train_size': train_size,
        'workers': workers,
        'train_vecs_shape': list(train_vecs.shape),
    }
    with open(config_path, 'w') as f:
        json.dump(config, f)

    logger.info(f'Config saved to {config_path}')


def get_knn(
        graph_embeddings_path: str,
        query_idxs: Union[str, List],
        output_path: Optional[str] = None,
        do_normalize: bool = False,
        batch_size: int = 100,
        k: int = 1000,
        workers: int = 10):
    """
    Get exact k nearest neighbors from graph embeddings

    python cli_s2orc.py get_knn ${S2ORC_EMBEDDINGS} --query_idxs 0,1,2,3,4 \
        --output_path ${BASE_DIR}/data/ann_benchmark/query_knns.json \
        --batch_size 3 --workers ${WORKERS} --k 1000 --do_normalize

    :param graph_embeddings_path:
    :param query_idxs:
    :param output_path:
    :param do_normalize:
    :param batch_size:
    :param k:
    :param workers:
    :return:
    """
    graph_embeddings = get_graph_embeddings(graph_embeddings_path, do_normalize, workers)

    if isinstance(query_idxs, str):
        logger.info(f'Load query indices from {query_idxs}')
        with open(query_idxs) as f:
            query_idxs = json.load(f)

    query_vecs = graph_embeddings[query_idxs, :]

    logger.info(query_vecs.shape)

    batch_out = []

    for i in tqdm(range(0, len(query_vecs), batch_size), desc='Search NN'):
        batch_query_vecs = query_vecs[i: i + batch_size]

        # cosine similarity or cosine distance?
        cosine_similarities = linear_kernel(batch_query_vecs, graph_embeddings)

        sorted_idxs = matutils.argsort(cosine_similarities, reverse=True, topn=k+1)  # in reverse order

        sorted_idxs = sorted_idxs[:, 1:k+1]  # k nearest neighbors (exclude self)

        # stack
        batch_out.append(sorted_idxs)

        del cosine_similarities

    if len(batch_out) == 1:
        query_knns = batch_out[0]
    else:
        query_knns = np.concatenate(list(batch_out), axis=0)

    if output_path:
        # write as JSON
        logger.info(f'Writing to {output_path}')
        with open(output_path, 'w') as f:
            json.dump(query_knns.tolist(), f)
    else:
        return query_knns


def build_k_means(
        graph_embeddings_path: str,
        paper_ids: Union[str, List[str]],
        output_dir: Optional[str] = None,
        seed: int = 0,
        k: int = 10,
        max_points_per_centroid: int = 1_000_000,
        min_points_per_centroid: int = 1,
        niter: int = 20,
        nredo: int = 1,
    ):
    """
    Run k-means clustering on graph embeddings (approx. 25min on two GPUs with k=10k; little GPU memory needed)

    Usage:

    export CUDA_VISIBLE_DEVICES=...
    python cli_s2orc.py build_k_means ${S2ORC_EMBEDDINGS} ${S2ORC_PAPER_IDS} ${K_MEANS_DIR} \
        -k ${K_MEANS_CENTROIDS} -niter ${K_MEANS_N_ITER} -nredo 3

    :param paper_ids: Path to JSON with paper IDs or list of paper IDs
    :param nredo: Repeat k-means n times
    :param niter: K-means iterations
    :param max_points_per_centroid:
    :param graph_embeddings_path: Path to graph embeddings
    :param output_dir: Store output files in this directory (paper_idx_to_centroid.json, centroids.npy, sorted_centroids.npy, similarities.npy)
    :param seed: Random seed
    :return: paper idx to centroid mapping, centroid positions, cosine similarities, centroids sorted by cosine similarity.
    """

    import faiss

    set_seed(seed)

    if isinstance(paper_ids, str):
        with open(paper_ids) as f:
            paper_ids = json.load(f)

    if not os.path.exists(output_dir):
        logger.info(f'Creating output dir: {output_dir}')
        os.makedirs(output_dir)

    ngpu = torch.cuda.device_count()

    if ngpu < 1:
        raise ValueError(f'No GPU available')

    logger.info(f'GPUs: {ngpu}')

    graph_embeddings = get_graph_embeddings(graph_embeddings_path, do_normalize=False, workers=10)
    graph_embedding_size = graph_embeddings.shape[1]

    clustering = faiss.Clustering(graph_embedding_size, k)

    clustering.verbose = True
    clustering.niter = niter
    clustering.nredo = nredo
    clustering.seed = seed
    clustering.max_points_per_centroid = max_points_per_centroid  # otherwise the kmeans implementation sub-samples the training set
    clustering.min_points_per_centroid = min_points_per_centroid

    # GPU setup
    res = [faiss.StandardGpuResources() for i in range(ngpu)]

    flat_config = []
    for i in range(ngpu):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)

    if ngpu == 1:
        index = faiss.GpuIndexFlatL2(res[0], graph_embedding_size, flat_config[0])
    else:
        indexes = [faiss.GpuIndexFlatL2(res[i], graph_embedding_size, flat_config[i])
                   for i in range(ngpu)]
        index = faiss.IndexProxy()
        for sub_index in indexes:
            index.addIndex(sub_index)

    logger.info('start train')
    clustering.train(graph_embeddings, index)
    logger.info(f'end train')

    # quantization_error = clustering.train(graph_embeddings, index)
    # logger.info(f'end train. quantization error  = {quantization_error}')

    # Return the assignment and centroid positions
    logger.info('start search')
    _, ids = index.search(graph_embeddings, 1)

    # centroid pairwise distance
    centroids = faiss.vector_float_to_array(clustering.centroids).reshape(k, graph_embedding_size)
    # objective = faiss.vector_float_to_array(clustering.obj)

    logger.info('Computing centroid similarity')
    similarities = cosine_similarity(centroids)

    sorted_centroids = np.argsort(-1 * similarities, axis=1)  # descending order

    idx_to_paper_id = {idx: paper_id for idx, paper_id in enumerate(paper_ids)}
    paper_id_to_centroid = {}
    centroid_to_paper_ids = defaultdict(list)

    for paper_idx, centroid in enumerate(ids[:, 0]):
        paper_id = idx_to_paper_id[paper_idx]
        paper_id_to_centroid[paper_id] = centroid
        centroid_to_paper_ids[centroid].append(paper_id)

    if output_dir:
        logger.info(f'Writing output into: {output_dir}')

        with open(os.path.join(output_dir, 'centroids.npy'), 'wb') as f:
            np.save(f, centroids)

        with open(os.path.join(output_dir, 'sorted_centroids.npy'), 'wb') as f:
            np.save(f, sorted_centroids)

        with open(os.path.join(output_dir, 'similarities.npy'), 'wb') as f:
            np.save(f, similarities)

        with open(os.path.join(output_dir, 'centroid_to_paper_ids.pickle'), 'wb') as f:
            pickle.dump(centroid_to_paper_ids, f,
                        pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(output_dir, 'paper_id_to_centroid.pickle'), 'wb') as f:
            pickle.dump(paper_id_to_centroid, f,
                        pickle.HIGHEST_PROTOCOL)


    else:
        return ids, clustering.centroids, similarities


def worker_extract_metadata_id_mapping(worker_id, batch_fns, s2orc_metadata_dir):
    batch_metadata = []

    for batch_fn in tqdm(batch_fns, desc=f'Worker {worker_id}'):
        with open(os.path.join(s2orc_metadata_dir, batch_fn)) as batch_f:
            for i, line in enumerate(batch_f):
                meta = json.loads(line)

                batch_metadata.append((
                    meta['paper_id'],
                    batch_fn,
                    i
                ))

    return batch_metadata


def get_metadata_id_mapping(s2orc_metadata_dir, output_path, workers: int = 10):
    """
    Extract id/file metadata mapping for S2ORC

    python cli_s2orc.py get_metadata_id_mapping ${S2ORC_METADATA_DIR} ${S2ORC_METADATA_DIR}/s2orc_metadata_id_mapping.json --workers 10

    :param s2orc_metadata_dir:
    :param output_path:
    :param workers:
    :return:
    """
    if os.path.exists(output_path):
        logger.error(f'Output already exists: {output_path}')
        return

    # Meta data files
    batch_fns = [batch_fn for batch_fn in os.listdir(s2orc_metadata_dir) if batch_fn.endswith('.jsonl.gz')]
    logger.info(f'Files available: {len(batch_fns):,}')

    logger.info(f'Extracting metadata with workers: {workers}')

    # worker_id, batch_fns, needed_paper_ids, s2orc_metadata_dir
    worker_data = zip(
        list(range(workers)),  # worker ids
        split_into_n_chunks(batch_fns, workers),
        [s2orc_metadata_dir] * workers,
    )

    # Run threads
    with Pool(workers) as pool:
        pool_outputs = list(pool.starmap(worker_extract_metadata_id_mapping, worker_data))

    # Merge thread outputs
    metadata_id_mapping = defaultdict(list)
    for b in pool_outputs:
        for paper_id, batch_fn, line_idx in b:
            metadata_id_mapping[batch_fn].append([
                paper_id, line_idx
            ])

    logger.info(f'Writing {len(metadata_id_mapping):,} metadata mappings to {output_path}')

    with open(output_path, 'w') as f:
        json.dump(metadata_id_mapping, f)


if __name__ == '__main__':
    fire.Fire()
