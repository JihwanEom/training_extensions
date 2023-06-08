"""Set configurable parameters for Visual Prompting."""

# Copyright (C) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from pathlib import Path
from typing import Union, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf

from otx.api.configuration.configurable_parameters import ConfigurableParameters


def get_visual_promtping_config(task_name: str, otx_config: ConfigurableParameters) -> Union[DictConfig, ListConfig]:
    """Get visual prompting configuration.

    Create an visual prompting config object that matches the values specified in the
    OTX config.

    Args:
        task_name: Task name to load configuration from Visual Prompting
        otx_config: ConfigurableParameters: OTX config object parsed from
            configuration.yaml file

    Returns:
        Visual prompting config object for the specified model type with overwritten
        default values.
    """
    config_path = Path(f"otx/algorithms/visual_prompting/configs/{task_name.lower()}/config.yaml")
    visual_prompting_config = get_configurable_parameters(model_name=task_name.lower(), config_path=config_path)
    update_visual_prompting_config(visual_prompting_config, otx_config)
    return visual_prompting_config


def get_configurable_parameters(
    model_name: Optional[str] = None,
    config_path: Optional[Union[Path, str]] = None,
    weight_file: Optional[str] = None,
    config_filename: Optional[str] = "config",
    config_file_extension: Optional[str] = "yaml",
) -> Union[DictConfig, ListConfig]:
    """Get configurable parameters.

    Args:
        model_name (Optional[str]):  (Default value = None)
        config_path (Optional[Union[Path, str]]):  (Default value = None)
        weight_file (Optional[str]): Path to the weight file
        config_filename (Optional[str]):  (Default value = "config")
        config_file_extension (Optional[str]):  (Default value = "yaml")

    Returns:
        Union[DictConfig, ListConfig]: Configurable parameters in DictConfig object.
    """
    if model_name is None is config_path:
        raise ValueError(
            "Both model_name and model config path cannot be None! "
            "Please provide a model name or path to a config file!"
        )

    if config_path is None:
        config_path = Path(f"otx/algorithms/visual_prompting/configs/{model_name}/{config_filename}.{config_file_extension}")

    config = OmegaConf.load(config_path)

    return config


def _visual_prompting_config_mapper(
    visual_prompting_config: Union[DictConfig, ListConfig],
    otx_config: ConfigurableParameters
) -> None:
    """Return mapping from learning parameters to visual prompting parameters.

    Args:
        visual_prompting_config: DictConfig: Visual prompting config object
        otx_config: ConfigurableParameters: OTX config object parsed from configuration.yaml file
    """
    parameters = otx_config.parameters
    groups = otx_config.groups
    for name in parameters:
        if name == "train_batch_size":
            visual_prompting_config.dataset["train_batch_size"] = getattr(otx_config, "train_batch_size")
        elif name == "max_epochs":
            visual_prompting_config.trainer["max_epochs"] = getattr(otx_config, "max_epochs")
        else:
            assert name in visual_prompting_config.model.keys(), f"Parameter {name} not present in visual prompting config."
            sc_value = getattr(otx_config, name)
            sc_value = sc_value.value if hasattr(sc_value, "value") else sc_value
            visual_prompting_config.model[name] = sc_value
    for group in groups:
        update_visual_prompting_config(visual_prompting_config.model[group], getattr(otx_config, group))


def update_visual_prompting_config(visual_prompting_config: Union[DictConfig, ListConfig], otx_config: ConfigurableParameters):
    """Update visual prompting configuration.

    Overwrite the default parameter values in the visual prompting config with the
    values specified in the OTX config. The function is recursively called for
    each parameter group present in the OTX config.

    Args:
        visual_prompting_config: DictConfig: Visual prompting config object
        otx_config: ConfigurableParameters: OTX config object parsed from
            configuration.yaml file
    """
    for param in otx_config.parameters:
        assert param in visual_prompting_config.keys(), f"Parameter {param} not present in visual prompting config."
        sc_value = getattr(otx_config, param)
        sc_value = sc_value.value if hasattr(sc_value, "value") else sc_value
        visual_prompting_config[param] = sc_value
    for group in otx_config.groups:
        # Since pot_parameters and nncf_optimization are specific to OTX
        if group == "learning_parameters":
            _visual_prompting_config_mapper(visual_prompting_config, getattr(otx_config, "learning_parameters"))
        elif group not in ["pot_parameters", "nncf_optimization"]:
            update_visual_prompting_config(visual_prompting_config[group], getattr(otx_config, group))
