"""Explicit reuse of the two frozen control arms in the rival-readout study."""
from pathlib import Path


def experiment_run(contract, mode, seed):
    assert mode in contract['arms'] and seed in contract['training']['seeds']
    return Path(contract['training_runs'][mode][str(seed)])


def fit_contract_path(contract, mode):
    assert mode in contract['arms']
    return Path(contract['fit_contracts'][mode])


def training_source(contract, mode):
    assert mode in contract['arms']
    return Path(contract['training_sources'][mode])


def validate_protocol(contract):
    assert contract['experiment_kind'] == 'rival_candidate_token_readout_matched'
    assert contract['arms'] == ['uniform', 'label_attention', 'rival_attention']
    assert contract['new_training_arms'] == ['rival_attention']
    assert contract['training']['seeds'] == [20261031, 20261032, 20261033, 20261034]
    assert contract['training']['steps'] == 19929 and contract['training']['epochs'] == 3
    assert contract['architecture']['trainable_parameters'] == 80001
    assert not contract['automatic_test_launch']
    assert set(contract['training_runs']) == set(contract['fit_contracts']) == set(contract['training_sources']) == set(contract['arms'])
    runs = []
    for mode in contract['arms']:
        assert set(contract['training_runs'][mode]) == set(map(str, contract['training']['seeds']))
        for seed in contract['training']['seeds']:
            path = experiment_run(contract, mode, seed)
            assert path.name == f'train_{mode}_{seed}'
            assert path.parent == fit_contract_path(contract, mode).parent
            runs.append(str(path.resolve()))
    assert len(set(runs)) == 12
    assert contract['fit_contracts']['uniform'] == contract['fit_contracts']['label_attention']
    assert contract['training_sources']['uniform'] == contract['training_sources']['label_attention']
    assert contract['fit_contracts']['rival_attention'] != contract['fit_contracts']['uniform']
    assert contract['training_sources']['rival_attention'] != contract['training_sources']['uniform']


def resources_ready(predecessors_complete, route_positive, route_test_reviewed, model_inference_active):
    assert len(predecessors_complete) == 3
    assert all(type(value) is bool for value in predecessors_complete)
    assert all(type(value) is bool for value in [route_positive, route_test_reviewed, model_inference_active])
    return all(predecessors_complete) and (not route_positive or route_test_reviewed) and not model_inference_active
