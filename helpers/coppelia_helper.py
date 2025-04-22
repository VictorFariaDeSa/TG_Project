from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time

def create_stepped_sim():
    client = RemoteAPIClient()
    sim = client.require('sim')
    sim.setStepping(True)
    return sim

def reset_sim(sim):
    sim.stopSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        time.sleep(0.1)

    sim.setStepping(True)
    sim.startSimulation()

def start_sim(sim):
        sim.startSimulation()
