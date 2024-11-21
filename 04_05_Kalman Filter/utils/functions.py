import numpy as np


def create_movement(
        model: str,
        numMeasurements: int,
        samplingPeriod: int,
        varObservations: float,
        processNoiseScaling: float) -> tuple[np.ndarray, np.ndarray]:
    """Create a movement. Includes the observations from the sensor and the real position.

    Args:
        model (str): Underlying model of the movement.
        numMeasurements (int): Number of measurements.
        samplingPeriod (int): Sampling periode.
        varObservations (float): Variance in the observation model.
        processNoiseScaling (float): Variance in the process model.

    Returns:
        tuple[np.ndarray, np.ndarray]: Observations, real values
    """
    # create observations
    observation = np.empty((numMeasurements, 2))
    # create reference
    referenceCurve = np.empty((numMeasurements, 2))
    # starting point
    starting_point = 10

    # add noise to arrays
    for i in range(0, numMeasurements):
        # observation noise real
        observation[i] = np.random.multivariate_normal([0, 0], [[varObservations, 0], [0, varObservations]], (1, 1))

        # process noise real
        referenceCurve[i] = np.random.multivariate_normal([0, 0], [[processNoiseScaling, 0], [0, processNoiseScaling]], (1, 1))

    # select a model and generate observations and reference curves
    if model == 'movement_1':
        func_obs, func_ref = movement_1_model(starting_point, numMeasurements, samplingPeriod)
    elif model == 'movement_2':
        func_obs, func_ref = movement_2_model(starting_point, numMeasurements, samplingPeriod)
    elif model == 'movement_3':
        func_obs, func_ref = movement_3_model(starting_point, numMeasurements, samplingPeriod)
    else:
        func_obs = - observation
        func_ref = - referenceCurve

    observation += func_obs
    referenceCurve += func_ref

    return observation, referenceCurve


def movement_1_model(starting_point: int, numMeasurements: int, samplingPeriod: int) -> tuple[np.ndarray, np.ndarray]:
    observation = [[starting_point + (i * samplingPeriod), starting_point] for i in range(numMeasurements)]
    observation = np.array(observation)
    return observation, np.copy(observation)


def movement_2_model(starting_point: int, numMeasurements: int, samplingPeriod: int) -> tuple[np.ndarray, np.ndarray]:
    observation = [[starting_point + (i * samplingPeriod), starting_point - 10.0 + (i * samplingPeriod * 0.75)] for i in range(numMeasurements)]
    observation = np.array(observation)
    return observation, np.copy(observation)


def local_linear_model(starting_point: int, numMeasurements: int, samplingPeriod: int) -> tuple[np.ndarray, np.ndarray]:
    observation = [[starting_point + (i * samplingPeriod), starting_point - 4.0 + 8.0 / (1.0 + np.exp(- i * samplingPeriod + 10))] for i in range(numMeasurements)]
    observation = np.array(observation)
    return observation, np.copy(observation)


def movement_3_model(starting_point: int, numMeasurements: int, samplingPeriod: int) -> tuple[np.ndarray, np.ndarray]:
    observation = []
    for i in range(0, numMeasurements):
        if i < numMeasurements / 2:
            observation.append([starting_point + (i * samplingPeriod), starting_point - 5. + i * samplingPeriod * 1.0])
        else:
            observation.append([starting_point + (i * samplingPeriod), 15 - (i - 10) * samplingPeriod * 1.0])
    observation = np.array(observation)
    return observation, np.copy(observation)
