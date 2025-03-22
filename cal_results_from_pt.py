import os
import numpy as np
import torch
from glob import glob


if __name__ == '__main__':
    eval_path = './eval_results'
    results_fn_list = glob(os.path.join(eval_path, '*.pt'))
    print("num of results.pt: ",  len(results_fn_list))
    docking_mode = 'vina_score'

    qed_all = []
    sa_all = []
    qvina_all = []
    vina_score_all = []
    vina_min_all = []
    vina_dock_all = []
    logp_all, lipinski_all, tpsa_all, hba_all, hbd_all, fsp3_all, rotb_all = [], [], [], [], [], [], []
    for rfn in results_fn_list:
        # print(torch.load(rfn))
        result_i = torch.load(rfn)['all_results']
        qed_all += [r['chem_results']['qed'] for r in result_i]
        sa_all += [r['chem_results']['sa'] for r in result_i]
        logp_all += [r['chem_results']['logp'] for r in result_i]
        lipinski_all += [r['chem_results']['lipinski'] for r in result_i]
        tpsa_all += [r['chem_results']['tpsa'] for r in result_i]
        hba_all += [r['chem_results']['hba'] for r in result_i]
        hbd_all += [r['chem_results']['hbd'] for r in result_i]
        fsp3_all += [r['chem_results']['fsp3'] for r in result_i]
        rotb_all += [r['chem_results']['rotb'] for r in result_i]


        if docking_mode == 'qvina':
            qvina_all += [r['vina'][0]['affinity'] for r in result_i]
        elif docking_mode in ['vina_dock', 'vina_score']:
            vina_score_all += [r['vina']['score_only'][0]['affinity'] for r in result_i]
            vina_min_all += [r['vina']['minimize'][0]['affinity'] for r in result_i]
            if docking_mode == 'vina_dock':
                vina_dock_all += [r['vina']['dock'][0]['affinity'] for r in result_i]

    qed_all_mean, qed_all_median = np.mean(qed_all), np.median(qed_all)
    sa_all_mean, sa_all_median = np.mean(sa_all), np.median(sa_all)
    logp_all_mean, logp_all_median = np.mean(logp_all), np.median(logp_all)
    lipinski_all_mean, lipinski_all_median = np.mean(lipinski_all), np.median(lipinski_all)
    tpsa_all_mean, tpsa_all_median = np.mean(tpsa_all), np.median(tpsa_all)
    hba_all_mean, hba_all_median = np.mean(hba_all), np.median(hba_all)
    hbd_all_mean, hbd_all_median = np.mean(hbd_all), np.median(hbd_all)
    fsp3_all_mean, fsp3_all_median = np.mean(fsp3_all), np.median(fsp3_all)
    rotb_all_mean, rotb_all_median = np.mean(rotb_all), np.median(rotb_all)

    print("qed_all_mean, qed_all_median:", qed_all_mean, qed_all_median)
    print("sa_all_mean, sa_all_median:", sa_all_mean, sa_all_median)
    print("logp_all_mean, logp_all_median:", logp_all_mean, logp_all_median)
    print("lipinski_all_mean, lipinski_all_median:", lipinski_all_mean, lipinski_all_median)
    print("tpsa_all_mean, tpsa_all_median:", tpsa_all_mean, tpsa_all_median)
    print("hba_all_mean, hba_all_median:", hba_all_mean, hba_all_median)
    print("hbd_all_mean, hbd_all_median:", hbd_all_mean, hbd_all_median)
    print("fsp3_all_mean, fsp3_all_mediann:", fsp3_all_mean, fsp3_all_median)
    print("rotb_all_mean, rotb_all_median:", rotb_all_mean, rotb_all_median)
   

    if len(qvina_all):
        qvina_all_mean, qvina_all_median = np.mean(qvina_all), np.median(qvina_all)
        print("qvina_all_mean, qvina_all_median:", qvina_all_mean, qvina_all_median)

    if len(vina_score_all):
        vina_score_all_mean, vina_score_all_median = np.mean(vina_score_all), np.median(vina_score_all)
        print("vina_score_all_mean, vina_score_all_median:", vina_score_all_mean, vina_score_all_median)

    if len(vina_min_all):
        vina_min_all_mean, vina_min_all_median = np.mean(vina_min_all), np.median(vina_min_all)
        print("vina_min_all_mean, vina_min_all_median:", vina_min_all_mean, vina_min_all_median)

    if len(vina_dock_all):
        vina_dock_all_mean, vina_dock_all_median = np.mean(vina_dock_all), np.median(vina_dock_all)
        print("qvina_all_mean, qvina_all_median:" , vina_dock_all_mean, vina_dock_all_median)

