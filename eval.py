# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Script for calculating Frechet Inception Distance (FID)."""

import os
import glob
import click
import tqdm
import numpy as np
import shutil

import torch_fidelity
import zipfile

#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
@click.command()
@click.option('--img_path',            help='Path to a directory of image zip files.', metavar='PATH',      type=str, required=True)
@click.option('--ref_path',            help='Path to a directory of reference images.', metavar='ZIP',     type=str, required=True)
@click.option('--tmp_path',            help='Path to a temporary directory.', metavar='PATH',               type=str, required=True)
@click.option('--out_file',            help='Output filename.', metavar='PATH',                                 type=str, required=True)
@click.option('--isc',                 help='Calculate Inception Score', metavar='BOOL',                    type=bool, default=True, show_default=True)
@click.option('--fid',                 help='Calculate FID Score', metavar='BOOL',                          type=bool, default=True, show_default=True)
@click.option('--batch',               help='Maximum batch size', metavar='INT',                            type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--inception_path',      help='Path for the Inception detector', metavar='PATH',              type=str, default='https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl')
def eval(img_path, ref_path, tmp_path, out_file, isc, fid, batch, inception_path):
    """Calculate metrics between two sets of images"""

    cache_dir = os.path.join(tmp_path, 'fidelity_cache')
    img_zips = sorted(glob.glob(os.path.join(img_path, '*.zip')))
    n_steps = len(img_zips)
    print('Calculating Metrics')
    print(f'Found {n_steps} image zip files')
    results = {'kimg': np.zeros(n_steps)}
    ref_tmp_path = os.path.join(tmp_path, 'ref_imgs')
    os.makedirs(ref_tmp_path, exist_ok=True)
    with zipfile.ZipFile(ref_path, 'r') as f:
        f.extractall(ref_tmp_path)

    prev_data = None
    if os.path.exists(out_file):
        prev_data = np.load(out_file)

    for i, img_zip in enumerate(tqdm.tqdm(img_zips)):
        step = os.path.split(img_zip)[-1][:-4]
        results['kimg'][i] = int(step)
        if prev_data is not None and int(step) in prev_data['kimg']:
            print(f"Found previous data for step {step}")
            mask = prev_data['kimg'] == int(step)
            for k, v in prev_data.items():
                if k not in results:
                    results[k] = np.zeros(n_steps)
                results[k][i] = v[mask][0]
        else:
            extract_dir = os.path.join(tmp_path, step)
            print(f'Extracting step {step} to {extract_dir}')
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(img_zip, 'r') as f:
                f.extractall(extract_dir)
            print(f'Running metrics for step {step}')
            metrics = torch_fidelity.calculate_metrics(
                input1=extract_dir,
                input2=ref_tmp_path,
                cuda=True,
                batch_size = batch,
                isc=isc,
                fid=fid,
                feature_extractor_weights_path=inception_path,
                samples_find_deep=True,
                cache_root=cache_dir,
                input2_cache_name='ref_imgs',
                verbose=True,
            )
            for k,v in metrics.items():
                if k not in results:
                    results[k] = np.zeros(n_steps)
                results[k][i] = v
            shutil.rmtree(extract_dir)

    np.savez(out_file, **results)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    eval()

#----------------------------------------------------------------------------
