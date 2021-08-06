import pytest

from nncf import NNCFConfig
from nncf.config.extractors import extract_accuracy_aware_training_params


def test_accuracy_aware_config():
    nncf_config = NNCFConfig()
    nncf_config.update(
        {
            "accuracy_aware_training": {
                "mode": "adaptive_compression_level",
                "params": {
                    "maximal_accuracy_degradation": 1.0,
                    "maximal_total_epochs": 100,
                    "patience_epochs": 30
                }
            },
            "compression": [
                {
                    "algorithm": "filter_pruning",
                    "pruning_init": 0.1,
                    "params": {
                        "schedule": "exponential",
                        "pruning_target": 0.3,
                        "pruning_steps": 15,
                        "filter_importance": "geometric_median"
                    }
                },
                {
                    "algorithm": "quantization",
                    "initializer": {
                        "range": {
                            "num_init_samples": 850
                        }
                    }
                }
            ]
        })

    extract_accuracy_aware_training_params(nncf_config)

    nncf_config = NNCFConfig()
    nncf_config.update(
        {
            "accuracy_aware_training": {
                "mode": "adaptive_compression_level",
                "params": {
                    "maximal_accuracy_degradation": 1.0,
                    "maximal_total_epochs": 100
                }
            },
            "compression": {
                "algorithm": "quantization",
                "initializer": {
                    "range": {
                        "num_init_samples": 850
                    }
                }
            }
        })

    with pytest.raises(RuntimeError):
        extract_accuracy_aware_training_params(nncf_config)

    nncf_config = NNCFConfig()
    nncf_config.update(
        {
            "accuracy_aware_training": {
                "mode": "adaptive_compression_level",
                "params": {
                    "maximal_accuracy_degradation": 1.0,
                    "maximal_total_epochs": 100,
                    "patience_epochs": 30
                }
            },
            "compression": [
                {
                    "algorithm": "filter_pruning",
                    "pruning_init": 0.1,
                    "params": {
                        "schedule": "exponential",
                        "pruning_target": 0.3,
                        "pruning_steps": 15,
                        "filter_importance": "geometric_median"
                    }
                },
                {
                    "algorithm": "rb_sparsity",
                    "sparsity_init": 0.01,
                    "params": {
                        "sparsity_target": 0.61,
                        "sparsity_target_epoch": 5,
                        "sparsity_freeze_epoch": 10
                    }
                }
            ]
        })

    with pytest.raises(RuntimeError):
        extract_accuracy_aware_training_params(nncf_config)
