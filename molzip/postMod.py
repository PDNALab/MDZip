import argparse
import mdtraj as md
import numpy as np
from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout

def potE(context):
    return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoules_per_mole)

def simulate_trajectory(trajectory_file, topology_file, out, temperature, num_steps, step_size_fs, n_min):
    
    topology = md.load_prmtop(topology_file)
    trajectory = md.load(trajectory_file, top=topology)


    forcefield = ForceField('amber14-all.xml')
    prmtop = AmberPrmtopFile(topology_file)
    system = forcefield.createSystem(prmtop.topology)
    
    # Platform selection
    platform = Platform.getPlatformByName('CUDA')  # or 'OpenCL' or 'CPU' if CUDA is not available


    new_positions = []

    for frame in trajectory:
        integrator = LangevinIntegrator(temperature*kelvin, 1/picosecond, step_size_fs*femtosecond)

        simulation = Simulation(prmtop.topology, system, integrator)
        simulation.context.setPositions(frame.xyz[0])

        E1 = potE(simulation.context)

        # Makesure there are no overlaps -- perform least amount of minimization needed
        if E1>0:
            n = 0
            while True:
                LocalEnergyMinimizer.minimize(simulation.context, maxIterations=1)
                n+=1
                if potE(simulation.context)<0.0:
                    break
                if n == n_min:
                    break
            # LocalEnergyMinimizer.minimize(simulation.context, maxIterations=50, tolerance=100)
            # Emod = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
            # if Emod>0:
            #     LocalEnergyMinimizer.minimize(simulation.context, maxIterations=950, tolerance=100)
        else:
            LocalEnergyMinimizer.minimize(simulation.context, maxIterations=5)
        
        simulation.step(num_steps)

        # E2 = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        
        state = simulation.context.getState(getPositions=True)
        new_positions.append(state.getPositions(asNumpy=True))

        # print(f'{E1:2.3e}', f'{E2:2.3e}')

        del integrator, simulation

    new_positions = np.array([np.asarray(p) for p in new_positions])
    new_topology = trajectory.topology
    new_trajectory = md.Trajectory(new_positions, topology=new_topology)

    new_trajectory.save(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulate a trajectory using OpenMM and MDTraj.')

    
    parser.add_argument('-t','--trajectory', type=str, required=True, help='Input trajectory file (e.g. .nc file).')
    parser.add_argument('-p','--topology', type=str, required=True, help='Topology file (e.g. .prmtop file).')
    parser.add_argument('-o','--out', type=str, required=False, help='Output file name', default='refined_trajectory.nc')
    parser.add_argument('-temp','--temperature', type=float, required=False, help='Simulation temperature in Kelvin.', default=300.0)
    parser.add_argument('-n','--num_steps', type=int, required=False, help='Number of integration steps.', default=1)
    parser.add_argument('-dt','--step_size_fs', type=float, required=False, help='Step size in femtoseconds.', default=2)
    parser.add_argument('-m', '--n_min', type=int, required=False, help='Maximum energy minimization steps.', default=1000)
    

    args = parser.parse_args()

    simulate_trajectory(args.trajectory, args.topology, args.out, args.temperature, args.num_steps, args.step_size_fs, args.n_min)

    