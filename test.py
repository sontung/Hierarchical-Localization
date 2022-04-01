import sys
from pathlib import Path
from pprint import pformat

import pycolmap

from hloc import extract_features, match_features, pairs_from_covisibility, pairs_from_retrieval
from hloc import colmap_from_nvm, triangulation, localize_sfm, visualization

dataset = Path('datasets/hblab/')  # change this if your dataset is somewhere else
images = dataset / 'images'

outputs = Path('outputs/hblab/')  # where everything will be saved
sfm_pairs = outputs / 'pairs-db-covis20.txt'  # top 20 most covisible in SIFT model
loc_pairs = outputs / 'pairs-query-netvlad20.txt'  # top 20 retrieved by NetVLAD
reference_sfm = outputs / 'sfm_superpoint+superglue'  # the SfM model we will build
results = outputs / 'Aachen_hloc_superpoint+superglue_netvlad20.txt'  # the result file
recon_output_dir = outputs / "recon"
recon_output_dir.mkdir(parents=True, exist_ok=True)

# list the standard configurations available
print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')

# pick one of the configurations for image retrieval, local feature extraction, and matching
# you can also simply write your own here!
retrieval_conf = extract_features.confs['netvlad']
feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['superglue']

features = extract_features.main(feature_conf, images, outputs)

pairs_from_covisibility.main(
    outputs / 'sfm_sift', sfm_pairs, num_matched=20)

sfm_matches = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)

try:
    print("read pre-computed reconstruction")
    reconstruction = pycolmap.Reconstruction(recon_output_dir)
except ValueError:
    print("no pre-computed reconstruction")
    reconstruction = triangulation.main(
        reference_sfm,
        outputs / 'sfm_sift',
        images,
        sfm_pairs,
        features,
        sfm_matches
    )
    reconstruction.write(recon_output_dir)


global_descriptors = extract_features.main(retrieval_conf, images, outputs)
pairs_from_retrieval.main(global_descriptors, loc_pairs, num_matched=40, db_prefix="db", query_prefix="query")
loc_matches = match_features.main(matcher_conf, loc_pairs, feature_conf['output'], outputs)
localize_sfm.main(
    reconstruction,
    dataset / 'queries.txt',
    loc_pairs,
    features,
    loc_matches,
    results,
    covisibility_clustering=False)  # not required with SuperPoint+SuperGlue
