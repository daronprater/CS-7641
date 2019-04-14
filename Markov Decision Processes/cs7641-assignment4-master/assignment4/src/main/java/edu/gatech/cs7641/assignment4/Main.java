package edu.gatech.cs7641.assignment4;

import java.util.HashMap;
import java.util.List;

import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldRewardFunction;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.common.ConstantStateGenerator;
import burlap.mdp.auxiliary.common.SinglePFTF;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.oo.propositional.PropositionalFunction;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import edu.gatech.cs7641.assignment4.artifacts.Algorithm;
import edu.gatech.cs7641.assignment4.artifacts.Analysis;
import edu.gatech.cs7641.assignment4.artifacts.Hazard;
import edu.gatech.cs7641.assignment4.artifacts.Hazard.HazardType;
import edu.gatech.cs7641.assignment4.artifacts.PlannerFactory;
import edu.gatech.cs7641.assignment4.artifacts.Problem;

public class Main {

	/*
	 * Set this constant to the specific problem you want to execute. The code currently has two
	 * different problems, but you can add more problems and control which one runs by using this
	 * constant.
	 * I GET THIS
	 */
	private static int PROBLEM = 2;

	/*
	 * This class runs one algorithm at the time. You can set this constant to the specific
	 * algorithm you want to run.
	 * I GET THIS
	 */
	private final static Algorithm algorithm = Algorithm.QLearning;

	/*
	 * If you set this constant to false, the specific GUI showing the grid, rewards, and policy
	 * will not be displayed. Honestly, I never set this to false, so I'm not even sure why I added
	 * the constant, but here it is anyway.
	 * I GET THIS
	 */
	private final static boolean SHOW_VISUALIZATION = true;

	/*
	 * This is a very cool feature of BURLAP that unfortunately only works with learning algorithms
	 * (so no ValueIteration or PolicyIteration). It is somewhat redundant to the specific analysis
	 * I implemented for all three algorithms (it computes some of the same stuff), but it shows
	 * very cool charts and it lets you export all the data to external files.
	 * 
	 * At the end, I didnt't use this much, but I'm sure some people will love it. Keep in mind that
	 * by setting this constant to true, you'll be running the QLearning experiment twice (so double
	 * the time).
	 * I GET THIS
	 */
	private static boolean USE_LEARNING_EXPERIMENTER = false;

	public static void main(String[] args) {

		/*
		 * If you want to set up more than two problems, make sure you change this ternary operator.
		 */
		Problem problem = PROBLEM == 1
			? createProblem1()
			: createProblem2();

		GridWorldDomain gridWorldDomain = new GridWorldDomain(problem.getWidth(), problem.getWidth());
		gridWorldDomain.setMap(problem.getMatrix());
		gridWorldDomain.setProbSucceedTransitionDynamics(0.9);

		/*
		 * This makes sure that the algorithm finishes as soon as the agent reaches the goal. We
		 * don't want the agent to run forever, so this is kind of important.
		 * 
		 * You could set more than one goal if you wish, or you could even set hazards that end the
		 * game (and penalize the agent with a negative reward). But this is not this code...
		 */
		

		GridWorldRewardFunction rewardFunction = new GridWorldRewardFunction(problem.getWidth(), problem.getWidth(), problem.getDefaultReward());

		/*
		 * This sets the reward for the cell representing the goal. Of course, we want this reward
		 * to be positive and juicy (unless we don't want our agent to reach the end, which will
		 * probably be mean).
		 */
		rewardFunction.setReward(problem.getGoal().x, problem.getGoal().y, problem.getGoalReward());
		TerminalFunction terminalFunction = new SinglePFTF(PropositionalFunction.findPF(gridWorldDomain.generatePfs(), GridWorldDomain.PF_AT_LOCATION));

		
		/*
		 * This sets up all the rewards associated with the different hazards specified on the
		 * surface of the grid.
		 */
		for (Hazard hazard : problem.getHazards()) {
			rewardFunction.setReward(hazard.getLocation().x, hazard.getLocation().y, hazard.getReward());
		}

		gridWorldDomain.setTf(terminalFunction);
		gridWorldDomain.setRf(rewardFunction);

		OOSADomain domain = gridWorldDomain.generateDomain();

		/*
		 * This sets up the initial position of the agent, and the goal.
		 */
		GridWorldState initialState = new GridWorldState(new GridAgent(problem.getStart().x, problem.getStart().y), new GridLocation(problem.getGoal().x, problem.getGoal().y, "loc0"));

		SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();

		Analysis analysis = new Analysis();

		/*
		 * Depending on the specific algorithm that we want to run, I call the magic method and
		 * specify the corresponding planner.
		 */
		switch (algorithm) {
			case ValueIteration:
				runAlgorithm(analysis, problem, domain, hashingFactory, initialState, new PlannerFactory() {

					@Override
					public Planner createPlanner(int episodeIndex, SADomain domain, HashableStateFactory hashingFactory, SimulatedEnvironment simulatedEnvironment) {
						return new ValueIteration(domain, 0.99, hashingFactory, 0.001, episodeIndex);
					}
				}, algorithm);
				break;
			case PolicyIteration:
				runAlgorithm(analysis, problem, domain, hashingFactory, initialState, new PlannerFactory() {

					@Override
					public Planner createPlanner(int episodeIndex, SADomain domain, HashableStateFactory hashingFactory, SimulatedEnvironment simulatedEnvironment) {

						/*
						 * For PolicyIteration we need to specify the number of iterations of
						 * ValueIteration that the algorithm will use internally to compute the
						 * corresponding values. By default, the code is using the same number of
						 * iterations specified for the ValueIteration algorithm.
						 * 
						 * A recommended modification is to change this value to the actual number
						 * of iterations that it takes ValueIteration to converge. This will
						 * considerably improve the runtime of the algorithm (assuming that
						 * ValueIteration converges faster than the number of configured
						 * iterations).
						 */
						//return new PolicyIteration(domain, 0.95, hashingFactory, 0.01, problem.getNumberOfIterations(Algorithm.ValueIteration), episodeIndex);
						return new PolicyIteration(domain, 0.99, hashingFactory, 0.001, 30, episodeIndex);
					}
				}, algorithm);
				break;
			default:
				runAlgorithm(analysis, problem, domain, hashingFactory, initialState, new PlannerFactory() {

					@Override
					public Planner createPlanner(int episodeIndex, SADomain domain, HashableStateFactory hashingFactory, SimulatedEnvironment simulatedEnvironment) {
						QLearning agent = new QLearning(domain, 0.95, hashingFactory, 0.3, 0.8);
						for (int i = 0; i < episodeIndex; i++) {
							agent.runLearningEpisode(simulatedEnvironment);
							simulatedEnvironment.resetEnvironment();
						}

						agent.initializeForPlanning(1);

						return agent;
					}
				}, algorithm);
				break;
		}

		analysis.print();
	}

	/**
	 * Here is where the magic happens. In this method is where I loop through the specific number
	 * of episodes (iterations) and run the specific algorithm. To keep things nice and clean, I use
	 * this method to run all three algorithms. The specific details are specified through the
	 * PlannerFactory interface.
	 * 
	 * This method collects all the information from the algorithm and packs it in an Analysis
	 * instance that later gets dumped on the console.
	 */
	private static void runAlgorithm(Analysis analysis, Problem problem, SADomain domain, HashableStateFactory hashingFactory, State initialState, PlannerFactory plannerFactory, Algorithm algorithm) {
		ConstantStateGenerator constantStateGenerator = new ConstantStateGenerator(initialState);
		SimulatedEnvironment simulatedEnvironment = new SimulatedEnvironment(domain, constantStateGenerator);
		Planner planner = null;
		Policy policy = null;
		for (int episodeIndex = 1; episodeIndex <= problem.getNumberOfIterations(algorithm); episodeIndex++) {
			System.out.println(episodeIndex);
			long startTime = System.nanoTime();
			planner = plannerFactory.createPlanner(episodeIndex, domain, hashingFactory, simulatedEnvironment);
			policy = planner.planFromState(initialState);

			/*
			 * If we haven't converged, following the policy will lead the agent wandering around
			 * and it might never reach the goal. To avoid this, we need to set the maximum number
			 * of steps to take before terminating the policy rollout. I decided to set this maximum
			 * at the number of grid locations in our map (width * width). This should give the
			 * agent plenty of room to wander around.
			 * 
			 * The smaller this number is, the faster the algorithm will run.
			 */
			int maxNumberOfSteps = problem.getWidth() * problem.getWidth();

			Episode episode = PolicyUtils.rollout(policy, initialState, domain.getModel(), maxNumberOfSteps);

			State final_state = episode.state(episode.rewardSequence.size());
			int final_x_coord = (int) final_state.get("agent:x");
			int final_y_coord = (int) final_state.get("agent:y");
			if ((final_x_coord == 10) && (final_y_coord == 3)) {
				episode.addReward(problem.getGoalReward());
			}
			analysis.add(episodeIndex, episode.rewardSequence, episode.numTimeSteps(), (long) (System.nanoTime() - startTime) / 1000000);
		}

		if (algorithm == Algorithm.QLearning && USE_LEARNING_EXPERIMENTER) {
			learningExperimenter(problem, (LearningAgent) planner, simulatedEnvironment);
		}

		if (SHOW_VISUALIZATION && planner != null && policy != null) {
			visualize(problem, (ValueFunction) planner, policy, initialState, domain, hashingFactory, algorithm.getTitle());
		}
	}

	/**
	 * Runs a learning experiment and shows some cool charts. Apparently, this is only useful for
	 * Q-Learning, so I only call this method when Q-Learning is selected and the appropriate flag
	 * is enabled.
	 */
	private static void learningExperimenter(Problem problem, LearningAgent agent, SimulatedEnvironment simulatedEnvironment) {
		LearningAlgorithmExperimenter experimenter = new LearningAlgorithmExperimenter(simulatedEnvironment, 10, problem.getNumberOfIterations(Algorithm.QLearning), new LearningAgentFactory() {

			public String getAgentName() {
				return Algorithm.QLearning.getTitle();
			}

			public LearningAgent generateAgent() {
				return agent;
			}
		});

		/*
		 * Try different PerformanceMetric values below to display different charts.
		 */
		experimenter.setUpPlottingConfiguration(500, 250, 2, 1000, TrialMode.MOST_RECENT_AND_AVERAGE, PerformanceMetric.CUMULATIVE_STEPS_PER_EPISODE, PerformanceMetric.AVERAGE_EPISODE_REWARD);
		experimenter.startExperiment();
	}

	/**
	 * This method takes care of visualizing the grid, rewards, and specific policy on a nice
	 * BURLAP-predefined GUI. I found this very useful to understand how the algorithm was working.
	 */
	private static void visualize(Problem map, ValueFunction valueFunction, Policy policy, State initialState, SADomain domain, HashableStateFactory hashingFactory, String title) {
		List<State> states = StateReachability.getReachableStates(initialState, domain, hashingFactory);
		ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(states, map.getWidth(), map.getWidth(), valueFunction, policy);
		gui.setTitle(title);
		gui.setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
		gui.initGUI();
	}

	private static Problem createProblem1() {
		/*
		 * The surface can be described as follows:
		 * 
		 * X — The starting point of the agent.
		 * 0 — Represents a safe cell where the agent can move.
		 * 1 — Represents a wall. The agent can't move to this cell.
		 * G — Represents the goal that the agent wants to achieve.
		 * S — Represents a small hazard. The agent will be penalized.
		 * M — Represents a medium hazard. The agent will be penalized.
		 * L — Represents a large hazard. The agent will be penalized.
		 */
		String[] map = new String[] {
				"X0011111",
				"00000000",
				"0L000000",
				"0LL00000",
				"0LL000S0",
				"0LL000S0",
				"0LL00000",
				"G0000000",
		};

		/*
		 * Make sure to specify the specific number of iterations for each algorithm. If you don't
		 * do this, I'm still nice and use 100 as the default value, but that wouldn't make sense
		 * all the time.
		 */
		HashMap<Algorithm, Integer> numIterationsHashMap = new HashMap<Algorithm, Integer>();
		numIterationsHashMap.put(Algorithm.ValueIteration, 100);
		numIterationsHashMap.put(Algorithm.PolicyIteration, 100);
		numIterationsHashMap.put(Algorithm.QLearning, 100);

		/*
		 * These are the specific rewards for each one of the hazards. Here you can be creative and
		 * play with different values as you see fit.
		 */
		HashMap<HazardType, Double> hazardRewardsHashMap = new HashMap<HazardType, Double>();
		hazardRewardsHashMap.put(HazardType.SMALL, -1.0);
		hazardRewardsHashMap.put(HazardType.MEDIUM, -2.0);
		hazardRewardsHashMap.put(HazardType.LARGE, -3.0);

		/*
		 * Notice how I specify below the specific default reward for cells with nothing on them (we
		 * want regular cells to have a small penalty that encourages our agent to find the goal),
		 * and the reward for the cell representing the goal (something nice and large so the agent
		 * is happy).
		 */
		return new Problem(map, numIterationsHashMap, -0.1, 100, hazardRewardsHashMap);
	}

	private static Problem createProblem2() {
		String[] map = new String[] {
				"X000000000000",
				"00100010001L0",
				"0111101011110",
				"0000000000000",
				"1101011101011",
				"0L00L0100L000",
				"0000000000000",
				"0110011100110",
				"0L100000001L0",
				"0001011101G00",
				"0111001L01110",
				"0000000000000",
				"0000000000000",
		};

		HashMap<Algorithm, Integer> numIterationsHashMap = new HashMap<Algorithm, Integer>();
		numIterationsHashMap.put(Algorithm.ValueIteration, 10);
		numIterationsHashMap.put(Algorithm.PolicyIteration, 100);
		numIterationsHashMap.put(Algorithm.QLearning, 500);
		
		HashMap<HazardType, Double> hazardRewardsHashMap = new HashMap<HazardType, Double>();
		hazardRewardsHashMap.put(HazardType.SMALL, 1.0);
		hazardRewardsHashMap.put(HazardType.MEDIUM, 2.0);
		hazardRewardsHashMap.put(HazardType.LARGE, -20.0);

		return new Problem(map, numIterationsHashMap, -2, 100, hazardRewardsHashMap);
	}

}
