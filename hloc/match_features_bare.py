import argparse
from typing import Union, Optional, Dict, List, Tuple
from pathlib import Path
import pprint
import collections.abc as collections
from tqdm import tqdm
import h5py
import torch

from . import matchers, logger
from .utils.base_model import dynamic_load
from .utils.parsers import names_to_pair, names_to_pair_old, parse_retrieval
from .utils.io import list_h5_names

'''
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the match file that will be generated.
    - model: the model configuration, as passed to a feature matcher.
'''
confs = {
    'superglue': {
        'output': 'matches-superglue',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 50,
        },
    },
    'superglue-fast': {
        'output': 'matches-superglue-it5',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 5,
        },
    },
    'NN-superpoint': {
        'output': 'matches-NN-mutual-dist.7',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
            'distance_threshold': 0.7,
        },
    },
    'NN-ratio': {
        'output': 'matches-NN-mutual-ratio.8',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
            'ratio_threshold': 0.8,
        }
    },
    'NN-mutual': {
        'output': 'matches-NN-mutual',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
        },
    }
}


def main(name2ref: Dict, conf: Dict,
         pairs: Path, features: Union[Path, str],
         export_dir: Optional[Path] = None,
         matches: Optional[Path] = None,
         features_ref: Optional[Path] = None,
         matching_feature_file=None,
         overwrite: bool = False) -> Path:
    if isinstance(features, Path) or Path(features).exists():
        features_q = features
        if matches is None:
            raise ValueError('Either provide both features and matches as Path'
                             ' or both as names.')
    else:
        if export_dir is None:
            raise ValueError('Provide an export_dir if features is not'
                             f' a file path: {features}.')
        features_q = Path(export_dir, features + '.h5')
        if matches is None:
            matches = Path(
                export_dir, f'{features}_{conf["output"]}_{pairs.stem}.h5')

    if features_ref is None:
        features_ref = features_q
    if isinstance(features_ref, collections.Iterable):
        features_ref = list(features_ref)
    else:
        features_ref = [features_ref]
    # match_from_paths(conf, pairs, matches, features_q, features_ref, name2ref, overwrite)
    match_from_paths_fast(conf, pairs, matches, features_q, features_ref, name2ref, matching_feature_file, overwrite)

    return matches


def main_evaluation(name2ref: Dict, conf: Dict,
                    pairs: Path, features: Union[Path, str],
                    export_dir: Optional[Path] = None,
                    matches: Optional[Path] = None,
                    features_ref: Optional[Path] = None,
                    overwrite: bool = False) -> Path:
    if isinstance(features, Path) or Path(features).exists():
        features_q = features
        if matches is None:
            raise ValueError('Either provide both features and matches as Path'
                             ' or both as names.')
    else:
        if export_dir is None:
            raise ValueError('Provide an export_dir if features is not'
                             f' a file path: {features}.')
        features_q = Path(export_dir, features + '.h5')
        if matches is None:
            matches = Path(
                export_dir, f'{features}_{conf["output"]}_{pairs.stem}.h5')

    if features_ref is None:
        features_ref = features_q
    if isinstance(features_ref, collections.Iterable):
        features_ref = list(features_ref)
    else:
        features_ref = [features_ref]
    match_from_paths(conf, pairs, matches, features_q, features_ref, name2ref, overwrite)

    return matches


def return_name2ref(features: Union[Path, str]):
    features_q = features
    if isinstance(features_q, collections.Iterable):
        features_ref = list(features_q)
    else:
        features_ref = [features_q]
    name2ref = {n: i for i, p in enumerate(features_ref)
                for n in list_h5_names(p)}
    return name2ref


def find_unique_new_pairs(pairs_all: List[Tuple[str]], match_path: Path = None):
    '''Avoid to recompute duplicates to save time.'''
    pairs = set()
    for i, j in pairs_all:
        if (j, i) not in pairs:
            pairs.add((i, j))
    pairs = list(pairs)
    if match_path is not None and match_path.exists():
        with h5py.File(str(match_path), 'r') as fd:
            pairs_filtered = []
            for i, j in pairs:
                if (names_to_pair(i, j) in fd or
                        names_to_pair(j, i) in fd or
                        names_to_pair_old(i, j) in fd or
                        names_to_pair_old(j, i) in fd):
                    continue
                pairs_filtered.append((i, j))
        return pairs_filtered
    return pairs


@torch.no_grad()
def match_from_paths(conf: Dict,
                     pairs_path: Path,
                     match_path: Path,
                     feature_path_q: Path,
                     feature_paths_refs: Path,
                     name2ref: Dict,
                     overwrite: bool = False) -> Path:
    if not feature_path_q.exists():
        raise FileNotFoundError(f'Query feature file {feature_path_q}.')
    for path in feature_paths_refs:
        if not path.exists():
            raise FileNotFoundError(f'Reference feature file {path}.')

    match_path.parent.mkdir(exist_ok=True, parents=True)

    assert pairs_path.exists(), pairs_path
    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)
    if len(pairs) > 100:
        pairs = tqdm(pairs, desc="Matching")
    for (name0, name1) in pairs:
        data = {}
        with h5py.File(str(feature_path_q), 'r') as fd:
            grp = fd[name0]
            for k, v in grp.items():
                data[k + '0'] = torch.from_numpy(v.__array__()).float().to(device)
            # some matchers might expect an image but only use its size
            data['image0'] = torch.empty((1,) + tuple(grp['image_size'])[::-1])
        with h5py.File(str(feature_paths_refs[name2ref[name1]]), 'r') as fd:
            grp = fd[name1]
            for k, v in grp.items():
                data[k + '1'] = torch.from_numpy(v.__array__()).float().to(device)
            data['image1'] = torch.empty((1,) + tuple(grp['image_size'])[::-1])
        data = {k: v[None] for k, v in data.items()}

        pred = model(data)
        pair = names_to_pair(name0, name1)

        with h5py.File(str(match_path), 'a') as fd:
            if pair in fd:
                del fd[pair]
            grp = fd.create_group(pair)
            matches = pred['matches0'][0].cpu().short().numpy()
            grp.create_dataset('matches0', data=matches)
            if 'matching_scores0' in pred:
                scores = pred['matching_scores0'][0].cpu().half().numpy()
                grp.create_dataset('matching_scores0', data=scores)


@torch.no_grad()
def match_from_paths_fast(conf: Dict,
                          pairs_path: Path,
                          match_path: Path,
                          feature_path_q: Path,
                          feature_paths_refs: Path,
                          name2ref: Dict,
                          matching_feature_file,
                          overwrite: bool = False) -> Path:
    """
    Don't use this function
    """

    if not feature_path_q.exists():
        raise FileNotFoundError(f'Query feature file {feature_path_q}.')
    for path in feature_paths_refs:
        if not path.exists():
            raise FileNotFoundError(f'Reference feature file {path}.')

    match_path.parent.mkdir(exist_ok=True, parents=True)

    assert pairs_path.exists(), pairs_path
    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)
    if len(pairs) > 100:
        pairs = tqdm(pairs, desc="Matching")

    fd_file = h5py.File(str(feature_path_q), 'r')
    fd_file2 = h5py.File(str(match_path), 'a')
    query_data = {}
    grp = fd_file["query/query.jpg"]
    for k, v in grp.items():
        query_data[k] = torch.from_numpy(v.__array__()).float().to(device)

    for (name0, name1) in pairs:
        data = {}
        if "query" in name0:
            grp = fd_file[name0]
            for k, v in grp.items():
                data[k + '0'] = query_data[k]
            data['image0'] = torch.empty((1,) + tuple(grp['image_size'])[::-1])

            grp2 = matching_feature_file[name1]
            for k, v in grp2.items():
                data[k + '1'] = v
            data['image1'] = torch.empty((1,) + tuple(fd_file[name1]['image_size'])[::-1])
        else:
            grp = matching_feature_file[name0]
            for k, v in grp.items():
                data[k + '0'] = v
            data['image0'] = torch.empty((1,) + tuple(fd_file[name0]['image_size'])[::-1])

            grp = fd_file[name1]
            for k, v in grp.items():
                data[k + '1'] = query_data[k]
            data['image1'] = torch.empty((1,) + tuple(grp['image_size'])[::-1])

        data = {k: v[None] for k, v in data.items()}

        pred = model(data)
        pair = names_to_pair(name0, name1)

        if pair in fd_file2:
            del fd_file2[pair]
        grp = fd_file2.create_group(pair)
        matches = pred['matches0'][0].cpu().short().numpy()
        grp.create_dataset('matches0', data=matches)
    fd_file.close()
    fd_file2.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--export_dir', type=Path)
    parser.add_argument('--features', type=str,
                        default='feats-superpoint-n4096-r1024')
    parser.add_argument('--matches', type=Path)
    parser.add_argument('--conf', type=str, default='superglue',
                        choices=list(confs.keys()))
    args = parser.parse_args()
    main(confs[args.conf], args.pairs, args.features, args.export_dir)
