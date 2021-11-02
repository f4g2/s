# Anonymous supplemental materials

Trained models and datasets can be downloaded from GitHub releases.

## Requirements

- Python 3.6 (same as Nvidia/PyTorch Docker images)
- CUDA GPU (for Transformers)
- FAISS-GPU >= 1.7.1 (v1.7.0 leads to poor results)

## Installation

Create a new virtual environment for Python 3.7 with Conda or use our slurm-ready Docker image (see Dockerfile):

```bash
conda create -n repo python=3.6
conda activate repo
conda install nb_conda_kernels
```

Clone repository and install dependencies:

```bash
git clone https://... repo
cd repo
pip install -r requirements.txt
```

## Data preparations

Download
- [S2ORC 20200705v1](https://github.com/allenai/s2orc)
- SPECTER's original training data: [train.pickle](https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/specter/training-data/train.pkl) (see [here](https://github.com/allenai/specter/issues/2))
- [SciDocs evaluation benchmark](https://github.com/allenai/scidocs)

Set the following environment variables accordingly:
```bash
export SPECTER_DIR=
export SCIDOCS_DIR=
export S2ORC_METADATA_DIR=
export DATA_DIR=
export S2ORC_EMBEDDINGS=.h5
export S2ORC_PAPER_IDS=entity_names_paper_id_0.json
export OUTPUT_DIR=
export BASE_MODEL=scibert-scivocab-uncased

```

Extract SPECTER's training data:
```bash
python cli_specter.py extract_triples ${SPECTER_DIR}/train.pkl ${SPECTER_DIR}/train_triples.csv
```

Scrape missing training paper data:
```bash
python s2_scraper.py get_from_ids ${SPECTER_DIR}/missing_train_paper_ids.csv ${SPECTER_DIR} --save_every=1000
```

Extract data from S2ORC

```bash
# SciDocs-S2ORC mapping with titles
python cli_s2orc.py get_scidocs_title_mapping ${SCIDOCS_DIR} ${S2ORC_METADATA_DIR} ${DATA_DIR}/scidocs_s2id_to_s2orc_paper_id.json

# Merge SciDocs-S2ORC mappings (from S2 API) 
python cli_s2orc.py get_s2orc_scidocs_mappings \
    ${SPECTER_DIR}/id2paper.json,${SPECTER_DIR}/specter_train_source_papers/id2paper.json,${SCIDOCS_DIR}/scidocs_s2orc/id2paper.json \
    ${DATA_DIR}/scidocs_s2id_to_s2orc_paper_id.json \
    ${SPECTER_DIR}/s2id_to_s2orc_paper_id.json

# Extract all citations
python cli_s2orc.py get_citations ${S2ORC_METADATA_DIR} ${DATA_DIR}/s2orc

# Extract citations graph edges from S2ORC: train/test ratio = 1% 
python cli_s2orc.py get_citations ${S2ORC_METADATA_DIR} {DATA_DIR}/biggraph/s2orc_train_test --test_ratio 0.01

```

## Citation graph embeddings

Select config file:
```bash
export BIGGRAPH_CONFIG=./biggraph_configs/s2orc_768d_dot.py
```

Train and evaluate (adjust paths for full data set):
```bash
# Import TSV (train and test)
# - train Nodes: 52620852 Edges:  462 912 337
# - test Nodes: 52620852 Edges:     4 675 883
torchbiggraph_import_from_tsv --lhs-col=0 --rhs-col=1 ${BIGGRAPH_CONFIG} \
    ./data/biggraph/s2orc_train_test/citations.train.tsv \
    ./data/biggraph/s2orc_train_test/citations.test.tsv


# Train model on train set (takes 6 hrs)
torchbiggraph_train ${BIGGRAPH_CONFIG} \
    -p edge_paths=./data/biggraph/s2orc_train_test/train_partitioned


# Evaluate on test set (takes 3 min)
torchbiggraph_eval ${BIGGRAPH_CONFIG} \
    -p edge_paths=./data/biggraph/s2orc_train_test/test_partitioned

```


## Contrastive language model

Our full pipeline can be run within a standard Python environment or as Slurm job. Set `PY` variable following accordingly:
```
# standard python
export PY="python"

# slurm (adjust with your settings)
export PY="srun ... python"
```

Run full pipeline including query paper selection, ANN index creation, triple mining, metadata extraction, training, and evaluation:

```bash
${PY} cli_pipeline.py run_specter ${OUTPUT_DIR} \
    --auto_output_dir \
    --scidocs_dir ${SCIDOCS_DIR} \
    --s2orc_metadata_dir ${S2ORC_METADATA_DIR} \
    --specter_triples_path ${SPECTER_DIR}/train_triples.csv \
    --graph_paper_ids_path ${S2ORC_PAPER_IDS} \
    --graph_embeddings_path ${S2ORC_EMBEDDINGS}  \
    --s2id_to_s2orc_input_path ${SPECTER_DIR}/s2id_to_s2orc_paper_id.json \
    --graph_limit specter \
    --ann_metric inner_product --ann_workers 1 \
    --ann_backend faiss \
    --ann_index_path ${ANN_INDEX_PATH} \
    --train_s2orc_paper_ids ${QUERY_DIR}/s2orc_paper_ids.json \
    --train_query_s2orc_paper_ids ${QUERY_DIR}/query_s2orc_paper_ids.json \
    --val_or_test_or_both both --eval_steps 1 --save_steps 2 \
    --triples_per_query 5 \
    --workers ${WORKERS} --gzip \
    --base_model_name_or_path ${BASE_MODEL} \
    --easy_positives_count 5 --easy_positives_strategy knn --easy_positives_k_min 20 --easy_positives_k_max 25 \
    --easy_negatives_count 3 --easy_negatives_strategy random_without_knn \
    --hard_negatives_count 2 --hard_negatives_strategy knn  --hard_negatives_k_min 3998 --hard_negatives_k_max 4000

```

Evaluation results are reported to Weights & Biases or stored on disk.

The individual pipeline steps can be run separately or reused to save compute time by setting corresponding arguments (e.g., `--skip-triples`).

## License

This repository is only for peer-review.