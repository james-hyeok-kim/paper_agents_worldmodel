"""
Random tracker 체크포인트 생성 스크립트.
tracking.pt와 동일한 아키텍처(obs_shape=227, action_dim=56)로
random weight 초기화 후 동일 포맷으로 저장.
"""
import sys
import os

os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'

sys.path.insert(0, '/home/jovyan/workspace/paper_agents_worldmodel/baselines/puppeteer/puppeteer')

import torch
from omegaconf import OmegaConf
from common.parser import parse_cfg
from tdmpc2 import TDMPC2

TRACKING_PT = '/home/jovyan/workspace/paper_agents_worldmodel/baselines/puppeteer/checkpoints/tracking.pt'
RANDOM_PT   = '/home/jovyan/workspace/paper_agents_worldmodel/experiments/wip/puppeteer-acquisition-curve/random_tracker.pt'


def main():
    # tracking.pt 로드해서 아키텍처 정보 추출
    ckpt = torch.load(TRACKING_PT, map_location='cpu')
    model_sd = ckpt['model']
    obs_dim = model_sd['_encoder.state.0.weight'].shape[1]
    print(f"tracking.pt obs_dim: {obs_dim}")

    # 최소 cfg 구성 (tracking task와 동일 아키텍처)
    cfg = OmegaConf.create({
        'task': 'tracking',
        'obs': 'state',
        'obs_shape': {'state': (obs_dim,)},
        'action_dim': 56,
        'num_enc_layers': 2,
        'enc_dim': 256,
        'num_channels': 32,
        'mlp_dim': 512,
        'latent_dim': 512,
        'num_q': 5,
        'dropout': 0.01,
        'simnorm_dim': 8,
        'lr': 3e-4,
        'enc_lr_scale': 0.3,
        'vmin': -10,
        'vmax': 10,
        'num_bins': 101,
        'log_std_min': -10,
        'log_std_max': 2,
        'entropy_coef': 1e-4,
        'discount': 0.97,
        'tau': 0.01,
        'iterations': 6,
        'num_samples': 512,
        'num_elites': 64,
        'num_pi_trajs': 24,
        'horizon': 3,
        'min_std': 0.05,
        'max_std': 2,
        'temperature': 0.5,
        'mpc': True,
    })
    # bin_size 직접 계산 (parse_cfg 없이)
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)

    # random TDMPC2 인스턴스 생성
    agent = TDMPC2(cfg)

    # 키 집합 검증
    trained_keys = set(model_sd.keys())
    random_keys  = set(agent.model.state_dict().keys())
    assert trained_keys == random_keys, \
        f"Key mismatch!\n  trained-only: {trained_keys - random_keys}\n  random-only: {random_keys - trained_keys}"

    # shape 검증
    for k in trained_keys:
        t_shape = model_sd[k].shape
        r_shape = agent.model.state_dict()[k].shape
        assert t_shape == r_shape, f"Shape mismatch for {k}: trained={t_shape} vs random={r_shape}"

    print(f"Key & shape parity verified: {len(trained_keys)} tensors match.")

    # tracking.pt 포맷과 동일하게 저장
    agent.save(RANDOM_PT)
    print(f"Random tracker saved to: {RANDOM_PT}")

    # 최종 검증: 저장된 파일 다시 로드
    loaded = torch.load(RANDOM_PT, map_location='cpu')
    assert set(loaded.keys()) == set(ckpt.keys()), "Top-level key mismatch after save!"
    print(f"Top-level keys OK: {list(loaded.keys())}")


if __name__ == '__main__':
    main()
