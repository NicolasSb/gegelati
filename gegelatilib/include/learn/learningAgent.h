#ifndef LEARNING_AGENT_H
#define LEARNING_AGENT_H

#include <map>

#include "instructions/set.h"
#include "environment.h"
#include "archive.h"
#include "tpg/tpgGraph.h"
#include "tpg/tpgExecutionEngine.h"
#include "mutator/mutationParameters.h"

#include "learn/evaluationResult.h"
#include "learn/learningParameters.h"
#include "learn/learningEnvironment.h"

namespace Learn {

	/**
	* \brief Class used to control the learning steps of a TPGGraph within
	* a given LearningEnvironment.
	*/
	class LearningAgent {
	protected:
		/// LearningEnvironment with which the LearningAgent will interact.
		LearningEnvironment& learningEnvironment;

		/// Environment for executing Program of the LearningAgent
		Environment env;

		/// Archive used during the training process
		Archive archive;

		///Learning for the learning process
		LearningParameters params;

		/// TPGGraph built during the learning process.
		TPG::TPGGraph tpg;

		/// Random Number Generator for this Learning Agent
		Mutator::RNG rng;

	public:
		/**
		* \brief Constructor for LearningAgent.
		*
		* \param[in] le The LearningEnvironment for the TPG.
		* \param[in] iSet Set of Instruction used to compose Programs in the
		*            learning process.
		* \param[in] p The LearningParameters for the LearningAgent.
		* \param[in] nbRegs The number of registers for the execution
		*                   environment of Program.
		*/
		LearningAgent(LearningEnvironment& le, const Instructions::Set& iSet, const LearningParameters& p, const unsigned int nbRegs = 8) :
			learningEnvironment{ le },
			env(iSet, le.getDataSources(), nbRegs),
			tpg(this->env),
			params{ p },
			archive(p.archiveSize, p.archivingProbability)
		{
			// override the number of actions from the parameters.
			this->params.mutation.tpg.nbActions = this->learningEnvironment.getNbActions();
		};

		/**
		* \brief Getter for the TPGGraph built by the LearningAgent.
		*
		* \return Get a reference to the TPGGraph.
		*/
		TPG::TPGGraph& getTPGGraph();

		/**
		* \brief Getter for the Archive filled by the LearningAgent
		*
		* \return a const reference to the Archive.
		*/
		const Archive& getArchive() const;

		/**
		* \brief Getter for the RNG used by the LearningAgent.
		*
		* \return Get a reference to the RNG.
		*/
		Mutator::RNG& getRNG();

		/**
		* \brief Initialize the LearningAgent.
		*
		* Calls the TPGMutator::initRandomTPG function.
		* Initialize the Mutator::RNG with the given seed.
		* Clears the Archive.
		*
		* \param[in] seed the seed given to the TPGMutator.
		*/
		void init(uint64_t seed = 0);

		/**
		* \brief Evaluates policy starting from the given root.
		*
		* The policy, that is, the TPGGraph execution starting from the given
		* TPGVertex is evaluated nbIteration times. The generationNumber is
		* combined with the current iteration number to generate a set of
		* seeds for evaluating the policy.
		* The method returns the average score for this policy.
		*
		* The method is const to enable potential parallel calls to it.
		*
		* \param[in] tee The TPGExecutionEngine to use.
		* \param[in] root the TPGVertex from which the policy evaluation starts.
		* \param[in] generationNumber the integer number of the current generation.
		* \param[in] mode the LearningMode to use during the policy evaluation.
		* \param[in] le Reference to the LearningEnvironment to use during the
		* policy evaluation (may be different from the attribute of the class in
		* child LearningAgentClass).
		*/
		virtual std::shared_ptr<EvaluationResult> evaluateRoot(TPG::TPGExecutionEngine& tee, const TPG::TPGVertex& root, uint64_t generationNumber, LearningMode mode, LearningEnvironment& le) const;

		/**
		* \brief Evaluate all root TPGVertex of the TPGGraph.
		*
		* This method calls the evaluateRoot method for every root TPGVertex
		* of the TPGGraph. The method returns a sorted map associating each root
		* vertex to its average score, in ascending order or score.
		*
		* \param[in] generationNumber the integer number of the current generation.
		* \param[in] mode the LearningMode to use during the policy evaluation.
		*/
		virtual std::multimap<std::shared_ptr<EvaluationResult>, const TPG::TPGVertex*> evaluateAllRoots(uint64_t generationNumber, LearningMode mode);

		/**
		* \brief Train the TPGGraph for one generation.
		*
		* Training for one generation includes:
		* - Populating the TPGGraph according to given MutationParameters.
		* - Evaluating all roots of the TPGGraph. (call to evaluateAllRoots)
		* - Removing from tge TPGGraph the worst performing root TPGVertex.
		*
		* \param[in] generationNumber the integer number of the current generation.
		*/
		virtual void trainOneGeneration(uint64_t generationNumber);

		/**
		* \brief Removes from the TPGGraph the root TPGVertex with the worst
		* results.
		*
		* \param[in] results a multimap containing root TPGVertex associated
		* to their score during an evaluation.
		*/
		virtual void decimateWorstRoots(std::multimap<std::shared_ptr<EvaluationResult>, const TPG::TPGVertex*>& results);

		/**
		* \brief Train the TPGGraph for a given number of generation.
		*
		* The method trains the TPGGraph for a given number of generation,
		* unless the referenced boolean value becomes false (evaluated at each
		* generation).
		* Optionally, a simple progress bar can be printed within the terminal.
		* The TPGGraph is NOT (re)initialized before starting the training.
		*
		* \param[in] altTraining a reference to a boolean value that can be
		* used to halt the training process before its completion.
		* \param[in] printProgressBar select whether a progress bar will be printed
		* in the console.
		* \return the number of completed generations.
		*/
		uint64_t train(volatile bool& altTraining, bool printProgressBar);

		/**
		* \brief This method evaluates all roots and only keeps the one
		* leading to the best average score in the TPGGraph.
		*
		* The LearningMode::VALIDATION is used to select the best root.
		*/
		void keepBestPolicy();

	};
};

#endif