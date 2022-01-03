/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2019 - 2020) :
 *
 * Karol Desnos <kdesnos@insa-rennes.fr> (2019 - 2020)
 * Pierre-Yves Le Rolland-Raumer <plerolla@insa-rennes.fr> (2020)
 *
 * GEGELATI is an open-source reinforcement learning framework for training
 * artificial intelligence based on Tangled Program Graphs (TPGs).
 *
 * This software is governed by the CeCILL-C license under French law and
 * abiding by the rules of distribution of free software. You can use,
 * modify and/ or redistribute the software under the terms of the CeCILL-C
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty and the software's author, the holder of the
 * economic rights, and the successive licensors have only limited
 * liability.
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading, using, modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean that it is complicated to manipulate, and that also
 * therefore means that it is reserved for developers and experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or
 * data to be ensured and, more generally, to use and operate it in the
 * same conditions as regards security.
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL-C license and that you accept its terms.
 */
#ifndef GEGELATI_IMBALANCEDLEARNINGAGENT_H
#define GEGELATI_IMBALANCEDLEARNINGAGENT_H

#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <queue>
#include <inttypes.h>
#include <iostream>

#include "learn/learningEnvironment.h"
#include "learn/learningAgent.h"
#include "learn/parallelLearningAgent.h"
#include <data/hash.h>
#include "mutator/rng.h"
#include "mutator/tpgMutator.h"
#include "tpg/tpgExecutionEngine.h"
#include "log/laLogger.h"


namespace Learn {
    /**
     * \brief LearningAgent specialized for LearningEnvironments representing an
     * imbalanced problem.
     *
     * The key difference between this ImbalancedLearningAgent and the base
     * LearningAgent is the way roots are selected for decimation after each
     * generation. In this agent, the roots are decimated based on a comparison
     * of the trust Interval associated with the teams results instead of
     * decimating roots based on their global average score (over all classes)
     * during the last evaluation.
     * By doing so, the roots providing the best score in each class are
     * not necessarily preserved which increases the chances of the results
     * not regressing when a class is rare.
     *
     * In this context, it is assumed that each action of the
     * LearningEnvironment represents a class of the classification problem.
     *
     * The BaseLearningAgent template parameter is the LearningAgent from which
     * the ClassificationLearningAgent inherits. This template notably enable
     * selecting between the classical and the ParallelLearningAgent.
     */
    template <class BaseLearningAgent = ParallelLearningAgent>
    class ImbalancedLearningAgent : public BaseLearningAgent
    {
        static_assert(
            std::is_convertible<BaseLearningAgent*, LearningAgent*>::value);

      public:
        /**
         * \brief Constructor for LearningAgent.
         *
         * \param[in] le The LearningEnvironment for the TPG.
         * \param[in] iSet Set of Instruction used to compose Programs in the
         *            learning process.
         * \param[in] p The LearningParameters for the LearningAgent.
         */
        ImbalancedLearningAgent(LearningEnvironment& le,
                                const Instructions::Set& iSet,
                                const LearningParameters& p)
            : BaseLearningAgent(le, iSet, p){};

        /**
         * \brief Specialization of the evaluateJob method for classification
         * purposes.
         *
         * This method returns a ClassificationEvaluationResult for the
         * evaluated root instead of the usual EvaluationResult. The score per
         * root corresponds to the F1 score for this class.
         */
        virtual std::shared_ptr<EvaluationResult> evaluateJob(
            TPG::TPGExecutionEngine& tee, const Job& root,
            uint64_t generationNumber, LearningMode mode,
            LearningEnvironment& le) const override;

        /**
         * \brief Evaluate all root TPGVertex of the TPGGraph.
         *
         * This method differs from the basic EvaluateALLRoots method by
         * providing a custom comparator in the return multimap.
         *
         * \param[in] generationNumber the integer number of the current
         * generation. \param[in] mode the LearningMode to use during the policy
         * evaluation.
         */
        virtual std::multimap<std::shared_ptr<EvaluationResult>,
                              const TPG::TPGVertex*, std::less<std::shared_ptr<EvaluationResult>>>
                         evaluateAllRoots(uint64_t generationNumber,
                         LearningMode mode);

        /**
         * \brief Method for evaluating all roots with parallelism.
         *
         * The work is delegated in two distinct methods (this structure is
         * made for inheritance purpose) : evaluateAllRootsInParallelExecute and
         * evaluateAllRootsInParallelCompileResults.
         *
         * \param[in] generationNumber the integer number of the current
         * generation. \param[in] mode the LearningMode to use during the policy
         * evaluation. \param[in] results Map to store the resulting score of
         * evaluated roots.
         */
        virtual void evaluateAllRootsInParallel(
            uint64_t generationNumber, LearningMode mode,
            std::multimap<std::shared_ptr<EvaluationResult>,
                          const TPG::TPGVertex*, std::less<std::shared_ptr<EvaluationResult>>>& results);

        /**
         * \brief Subfunction of evaluateAllRootsInParallel which handles the
         * gathering of results and the merge of the archives.
         *
         * This method just emplaces results from resultsPerJobMap, as each
         * job only contains 1 root is is quite easy.
         * The archive is merged with the mergeArchiveMap method.
         *
         * @param[in] resultsPerJobMap map linking the job number with its
         * results and itself.
         * @param[out] results map linking single results to their root vertex.
         * @param[in,out] archiveMap map linking the job number with its
         * gathered archive. These archive swill later be merged with the ones
         * of the other jobs.
         */
        virtual void evaluateAllRootsInParallelCompileResults(
            std::map<uint64_t, std::pair<std::shared_ptr<EvaluationResult>,
                                         std::shared_ptr<Job>>>&
                resultsPerJobMap,
            std::multimap<std::shared_ptr<EvaluationResult>,
                          const TPG::TPGVertex*, std::less<std::shared_ptr<EvaluationResult>>>& results,
            std::map<uint64_t, Archive*>& archiveMap);

        /**
         * \brief Train the TPGGraph for one generation.
         *
         * Training for one generation includes:
         * - Populating the TPGGraph according to given MutationParameters.
         * - Evaluating all roots of the TPGGraph. (call to evaluateAllRoots)
         * - Removing from the TPGGraph the worst performing root TPGVertex.
         *
         * \param[in] generationNumber the integer number of the current
         * generation.
         */
        virtual void trainOneGeneration(uint64_t generationNumber) override;

        /**
         * \brief Removes from the TPGGraph the root TPGVertex with the worst
         * results.
         *
         * The given multimap is updated by removing entries corresponding to
         * decimated vertices.
         *
         * The resultsPerRoot attribute is updated to remove results associated
         * to removed vertices.
         *
         * \param[in,out] results a multimap containing root TPGVertex
         * associated to their score during an evaluation.
         */
        virtual void decimateWorstRoots(
            std::multimap<std::shared_ptr<EvaluationResult>,
                          const TPG::TPGVertex*,
                          std::less<std::shared_ptr<EvaluationResult>>>&results);
    };

    template <class BaseLearningAgent>
    inline std::shared_ptr<EvaluationResult> ImbalancedLearningAgent<
        BaseLearningAgent>::evaluateJob(TPG::TPGExecutionEngine& tee,
                                        const Job& job,
                                        uint64_t generationNumber,
                                        LearningMode mode,
                                        LearningEnvironment& le) const
    {
        // Only consider the first root of jobs as we are not in adversarial
        // mode
        const TPG::TPGVertex* root = job.getRoot();

        // Skip the root evaluation process if enough evaluations were already
        // performed. In the evaluation mode only.
        std::shared_ptr<Learn::EvaluationResult> previousEval;
        if (mode == Learn::LearningMode::TRAINING &&
            this->isRootEvalSkipped(*root, previousEval)) {
            return previousEval;
        }

        // Separate testing batch in nb_batches and extract one point for each batch
        uint32_t nb_batches=this->params.nbTestingBatches;

        // Init results
        std::vector<double> tmp_result(nb_batches,
                                       0.0);

        for(uint32_t batch; batch < nb_batches; batch++) {
            uint64_t nbActions = 0;
            // Compute a Hash
            Data::Hash<uint64_t> hasher;
            uint64_t hash = hasher(generationNumber) ^ hasher(0);
            // Reset the learning Environment
            le.reset(hash, mode);

            while (!le.isTerminal() &&
                 nbActions < this->params.maxNbActionsPerEval/nb_batches) {
                 // Get the action
                 uint64_t actionID =
                     ((const TPG::TPGAction*)tee.executeFromRoot(*root)
                            .back())->getActionID();
                 // Do it
                 le.doAction(actionID);
                 // Count actions
                 nbActions++;
            }

            // Update results
            tmp_result.at(batch) = le.getScore();
        }

        // compute empirical Mean and Variance
        double mean = std::accumulate(tmp_result.begin(),
                                      tmp_result.end(), 0)/(double)nb_batches;
        double variance = std::accumulate(tmp_result.begin(), tmp_result.end(),
                                          0.0,[&mean, &nb_batches]
                                          (double accumulator,
                                           const double& val)
                                          {
                                              return accumulator +
                                                  (pow(val - mean,2)/
                                                      (nb_batches - 1));
                                          });

        //std::cout << "score : " << mean << " "  << variance <<  std::endl;
        //std::cout << "low bound at : " << mean -2*sqrt(variance) << std::endl;
        //getchar();


        // Create the EvaluationResult
        //the "magic" occurs here : define the evaluation result depending on the comparison made later.
        auto evaluationResult = std::shared_ptr<EvaluationResult>(
            new EvaluationResult((mean+2*sqrt(variance)*variance),
                                 this->params.maxNbActionsPerEval));

        return evaluationResult;
    }

    template <class BaseLearningAgent>
    std::multimap<std::shared_ptr<Learn::EvaluationResult>, const TPG::TPGVertex*,
                  std::less<std::shared_ptr<EvaluationResult>>>
    Learn::ImbalancedLearningAgent<BaseLearningAgent>::evaluateAllRoots
                                                    (uint64_t generationNumber,
                                                    Learn::LearningMode mode)
    {
            std::multimap<std::shared_ptr<EvaluationResult>,
                           const TPG::TPGVertex*,
                      std::less<std::shared_ptr<EvaluationResult>>>
                results;
            if (this->maxNbThreads <= 1 || !this->learningEnvironment.isCopyable()) {
                // Sequential mode

                // Create the TPGExecutionEngine
                TPG::TPGExecutionEngine tee(this->env, (mode == LearningMode::TRAINING)
                                                           ? &this->archive
                                                           : NULL);

                // Execute for all root
                for (int i = 0; i < this->tpg.getNbRootVertices(); i++) {
                    auto job = ParallelLearningAgent::makeJob(i, mode);

                    this->archive.setRandomSeed(job->getArchiveSeed());

                    std::shared_ptr<EvaluationResult> avgScore = this->evaluateJob(
                        tee, *job, generationNumber, mode, this->learningEnvironment);
                    results.emplace(avgScore, (*job).getRoot());
                }
            }
            else {
                // Parallel mode
                evaluateAllRootsInParallel(generationNumber, mode, results);
            }

            return results;
    }

    template <class BaseLearningAgent>
    void Learn::ImbalancedLearningAgent<BaseLearningAgent>::evaluateAllRootsInParallel(
        uint64_t generationNumber, LearningMode mode,
        std::multimap<std::shared_ptr<EvaluationResult>, const TPG::TPGVertex*, std::less<std::shared_ptr<EvaluationResult>>>&
            results)
    {
        // Create Archive Map
        std::map<uint64_t, Archive*> archiveMap;
        // Create Map for results
        std::map<uint64_t,
                 std::pair<std::shared_ptr<EvaluationResult>, std::shared_ptr<Job>>>
            resultsPerJobMap;

        ParallelLearningAgent::evaluateAllRootsInParallelExecute(generationNumber, mode, resultsPerJobMap,
                                          archiveMap);

        evaluateAllRootsInParallelCompileResults(resultsPerJobMap, results,
                                                 archiveMap);
    }

    template <class BaseLearningAgent>
    void Learn::ImbalancedLearningAgent<BaseLearningAgent>::evaluateAllRootsInParallelCompileResults(
        std::map<uint64_t, std::pair<std::shared_ptr<EvaluationResult>,
                                     std::shared_ptr<Job>>>& resultsPerJobMap,
        std::multimap<std::shared_ptr<EvaluationResult>, const TPG::TPGVertex*, std::less<std::shared_ptr<EvaluationResult>>>&
            results,
        std::map<uint64_t, Archive*>& archiveMap)
    {
        // Merge the results
        for (auto& resultPerRoot : resultsPerJobMap) {
            results.emplace(resultPerRoot.second.first,
                            (*resultPerRoot.second.second).getRoot());
        }

        // Merge the archives
        this->mergeArchiveMap(archiveMap);
    }

    template <class BaseLearningAgent>
    void Learn::ImbalancedLearningAgent<BaseLearningAgent>::trainOneGeneration(uint64_t generationNumber)
    {
        for (auto logger : this->loggers) {
            logger.get().logNewGeneration(generationNumber);
        }

        // Populate Sequentially
        Mutator::TPGMutator::populateTPG(this->tpg, this->archive,
                                         this->params.mutation, this->rng,
                                         this->maxNbThreads);
        for (auto logger : this->loggers) {
            logger.get().logAfterPopulateTPG();
        }

        // Evaluate
        auto results =
            this->evaluateAllRoots(generationNumber, LearningMode::TRAINING);
        for (auto logger : this->loggers) {
            logger.get().logAfterEvaluate(results);
        }

        // Remove worst performing roots
        decimateWorstRoots(results);
        // Update the best
        this->updateEvaluationRecords(results);

        for (auto logger : this->loggers) {
            logger.get().logAfterDecimate();
        }

        // Does a validation or not according to the parameter doValidation
        if (this->params.doValidation) {
            auto validationResults =
                evaluateAllRoots(generationNumber, Learn::LearningMode::VALIDATION);
            for (auto logger : this->loggers) {
                logger.get().logAfterValidate(validationResults);
            }
        }

        for (auto logger : this->loggers) {
            logger.get().logEndOfTraining();
        }
    }

    template<class BaseLearningAgent>
    void Learn::ImbalancedLearningAgent<BaseLearningAgent>::decimateWorstRoots(
        std::multimap<std::shared_ptr<EvaluationResult>, const TPG::TPGVertex*>&
            results)
    {
        // Some actions may be encountered but not removed while scanning the
        // results map they should be re-inserted to the list before leaving the
        // method.
        std::multimap<std::shared_ptr<EvaluationResult>, const TPG::TPGVertex*>
            preservedActionRoots;

        auto i = 0;
        while (i < floor(this->params.ratioDeletedRoots *
                         (double)this->params.mutation.tpg.nbRoots) &&
               results.size() > 0) {
            // If the root is an action, do not remove it!
            const TPG::TPGVertex* root = results.begin()->second;
            if (typeid(*root) != typeid(TPG::TPGAction)) {
                this->tpg.removeVertex(*results.begin()->second);
                // Removed stored result (if any)
                this->resultsPerRoot.erase(results.begin()->second);
            }
            else {
                preservedActionRoots.insert(*results.begin());
                i--; // no vertex was actually removed
            }
            results.erase(results.begin());

            // Increment loop counter
            i++;
        }

        // Restore root actions
        results.insert(preservedActionRoots.begin(), preservedActionRoots.end());
    }
}//namespace Learn

#endif
