import gin
import os
import time

import numpy as np
import ray

from jax import config

from state_space_models.configurations.utils import modify_and_parse_test_config
from building_coordinator.distributed_coordinator import DistributedCoordinator
from common.utils import save_config_file
from common.configurations.settings import Settings
from common.system_multi_zone_building import MultiZoneBuildingSimulation
from controllers.shooting_controller import ShootingController, RayShootingController
from controllers.rule_based_controller import RuleBasedController


gin.parse_config_file(f'common/configurations/settings.gin')


def parse_configuration(
    simulation_config: int,
    controller_type: str,
    coordinator_config: int,
    controller_config: int,
    model_type: str,
    model_config: list,
    base_model_config: int
) -> None:
    """Parse configurations for multi-zone building control simulation.

    Args:
        simulation_config: ID of the simulation configuration
        coordinator_config: ID of the coordinator configuration
        model_type: Type of model (SSM, RSSM, etc.)
        model_configs: List of model configuration IDs for each zone
        base_config_id: Base configuration ID
    """

    # Get the base configurations for the train_test, model, and distributed coordinator
    # Config for RayShootingController is in a JSON and not a .gin file, so this is not
    # read here
    modified_config = f"include './state_space_models/configurations/train_test/base_config{base_model_config}.gin' \n"
    modified_config += f"include './state_space_models/configurations/models/{model_type}/config{model_config}.gin'\n"
    modified_config += f"include './building_coordinator/configurations/DistributedCoordinator/config{controller_config}.gin'\n"

    # Add simulation configuration and replace the controller type
    with open(f'./common/configurations/multi_zone_configuration{simulation_config}.gin', "r") as file:
        original_config = file.readlines()

    for line in original_config:
        if line.startswith("MultiZoneBuildingSimulation.controller_class ="):
            modified_config += f"MultiZoneBuildingSimulation.controller_class = @{controller_type}\n"
        else:
            modified_config += line

    # Save the modified configuration to a temporary file and parse it
    nanoseconds = time.time_ns()
    temp_filename = f"temp_config_{nanoseconds}.gin"  # create a unique file for parallel jobs
    with open(temp_filename, "w") as temp_file:
        temp_file.write(modified_config)

    gin.parse_config_file(temp_filename)
    os.remove(temp_filename)

def main(
    simulation_configuration: int,
    controller_type: str,
    coordinator_config: int,
    controller_config: int,
    model_type: str,
    model_config: list,
    training_config: int,
    base_model_config: int,
) -> None:
    """Run multi-zone building control simulation.

    Args:
        simulation_config: ID of the simulation configuration
        coordinator_config: ID of the coordinator configuration
        model_type: Type of model to use
        model_configs: List of model configuration IDs for each zone
        base_config_id: Base configuration ID
        training_config: Training configuration ID
    """

    parse_configuration(
        simulation_configuration,
        controller_type,
        coordinator_config,
        controller_config,
        model_type,
        model_config,
        base_model_config
    )

    settings = Settings()
    config.update("jax_enable_x64", settings.jax_enable_x64)

    if settings.simulation_seed is not None:
        print("Fixing simulation seed")
        np.random.seed(settings.simulation_seed)
    else:
        print('Warning: the simulation seed is not fixed')

    # Initialize and run multi-zone building simulation
    ray.init()
    building_type = gin.query_parameter("%BUILDING_TYPE")
    simulation = MultiZoneBuildingSimulation()
    simulation.run()
    ray.shutdown()
