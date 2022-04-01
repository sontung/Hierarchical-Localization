from pathlib import Path

from hloc import extract_features, pairs_from_retrieval

dataset = Path('../vloc_workspace_retrieval')  # change this if your dataset is somewhere else
images_dir = dataset / 'images_retrieval'

loc_pairs_dir = dataset / 'pairs.txt'  # top 20 retrieved by NetVLAD
retrieval_conf = extract_features.confs['netvlad']

global_descriptors = extract_features.main(retrieval_conf, images_dir, dataset)
pairs_from_retrieval.main(global_descriptors, loc_pairs_dir, num_matched=40, db_prefix="db", query_prefix="query")
