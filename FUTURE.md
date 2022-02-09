# Imeiro RL Framework Future Work

Future work for improving the framework.

## Tasks

### High Priority Tasks

Tasks with high priority.

- TODO: Clean up the atrocious and opaque program flow in the environment module 
- TODO: Set the Main Config directory and for ABSOLUTE PATHS in config directory to stem from that directory.
    This will allow us to greatly clean up the config folder.
- TODO: Deal with canonicity constraints for agent IDs between environments with agent count modification and agent systems that also allow agent count modification.

### API Tasks (High-Priority Structural Tasks)

Tasks which affect the exposed API.

- Trajectory (with output from environment) should only be recorded by the Controller under special circumstances.
- - Add a Memory object to PolicyGroup.
- Cumulative reward should be accessed via Agent System.
- Agent System has Policy / Policies; Policy has Algorithm.
- MDPController made subclass of abstract Controller method
- Hook in callbacks via abstract method of Controller

### Structural Tasks

Tasks which affect the framework's internal organization.

- Actual __init__ functions of components should take in the config values. from_config should be a classmethod of ModuleFrame (rename to Component) which calls __init__ with the extracted parameters alongside expected input parameters.
- Config should have a function which traverses the subconfigs to find the config for the requested class.
- Support ConfigDesc (nested config) for non-class objects? (e.g. team description in gridbattle)

### QOL Tasks

Tasks which improve the ease of use and user-exposed features.

### Feature Tasks

Tasks for individual, existing components (not features) of the framework (e.g. environments, algorithms, etc.)

- Blocking agents in gridworld.
- Verify ResourceCollection

### Extension Tasks

Tasks describing additional components to be added.

- Learned Coordination (LocalStateSystem) which would need to extract local features.
- - Learn like sharedsystem with a mixture of policies over possible assigned roles, with mixture weights learned and produced by a separate Policy. Agents will be assigned to roles which fit into a coordination graph.
- Hierarchical RL
- POMDP Algorithm
- A3C (once parallel episode execution done)
- Random Network Distillation

## Issues

Issues for which a clear solution is not immediately obvious. When a solution can be formulated as a task, the issue should be moved to the Tasks section with the solution tagged.

### Unaddressed problems

- Controller support of parallel episode execution for Asynchronous Learning
- - Should the controller copy the environment or should separate workers be created?
- Fragmentation of parallizable code
- - Possible to "gather and dispatch" at init phase?
- Fragmentation of array data
- - Return observation maps as axis0-indexed whenever possible instead of Dict, preserving canonicity of agent IDs. Maybe have observation map per agentclass?
- - Related to fragmentation of parallelizability; We can eval multi-agent features as batch-like data.


