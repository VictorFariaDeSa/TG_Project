import sim  # Remote API
import time

# Fecha conexões anteriores
sim.simxFinish(-1)
print('Program started')

# Conecta ao servidor remoto
clientID = sim.simxStart('127.0.0.1', 19997, True, False, 5000, 5)

if clientID == -1:
    print('Failed to connect to remote API server')
    exit(1)

print('Connected to remote API server')

# Verifica o estado da simulação
state = sim.simxGetInMessageInfo(clientID, sim.simx_headeroffset_server_state)
if state[1] & 1 == 0:  # Checa se a simulação está parada
    print('Simulation is stopped, starting simulation...')
    res = sim.simxStartSimulation(clientID, sim.simx_opmode_blocking)
    if res == sim.simx_return_ok:
        print('Simulation started successfully')
    else:
        print('Failed to start the simulation')
        sim.simxFinish(clientID)
        exit(1)
else:
    print('Simulation is already running')

# Aguarda um tempo enquanto a simulação roda
time.sleep(5)

# Para a simulação
print('Stopping simulation...')
res = sim.simxStopSimulation(clientID, sim.simx_opmode_blocking)
if res == sim.simx_return_ok:
    print('Stop signal sent')
else:
    print('Failed to send stop signal')

# Fecha a conexão com o servidor
print("test")
sim.simxFinish(clientID)
print("test2")
print('Connection closed')
