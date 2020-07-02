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

#include <algorithm>
#include <iterator>
#include <mutex>
#include <queue>
#include <thread>

#include "mutator/rng.h"
#include "mutator/tpgMutator.h"
#include "tpg/tpgExecutionEngine.h"

#include "learn/evaluationResult.h"
#include "learn/parallelLearningAgent.h"

std::multimap<std::shared_ptr<Learn::EvaluationResult>, const TPG::TPGVertex*>
Learn::ParallelLearningAgent::evaluateAllRoots(uint64_t generationNumber,
                                               Learn::LearningMode mode)
{
    std::multimap<std::shared_ptr<EvaluationResult>, const TPG::TPGVertex*>
        results;

    if (this->maxNbThreads <= 1 || !this->learningEnvironment.isCopyable()) {
        // Sequential mode

        // Create the TPGExecutionEngine
        TPG::TPGExecutionEngine tee(this->env, (mode == LearningMode::TRAINING)
                                                   ? &this->archive
                                                   : NULL);

        // Execute for all root
        for (const TPG::TPGVertex* root : this->tpg.getRootVertices()) {
            // Set the seed of the archive for this root.
            if (mode == LearningMode::TRAINING) {
                this->archive.setRandomSeed(
                    this->rng.getUnsignedInt64(0, UINT64_MAX));
            }
            std::shared_ptr<EvaluationResult> avgScore = this->evaluateRoot(
                tee, *root, generationNumber, mode, this->learningEnvironment);
            results.emplace(avgScore, root);
        }
    }
    else {
        // Parallel mode
        evaluateAllRootsInParallel(generationNumber, mode, results);
    }

    return results;
}

void Learn::ParallelLearningAgent::slaveEvalRootThread(
    uint64_t generationNumber, LearningMode mode,
    std::queue<std::pair<uint64_t, const TPG::TPGVertex*>>& rootsToProcess,
    std::mutex& rootsToProcessMutex,
    std::map<uint64_t, std::pair<std::shared_ptr<EvaluationResult>,
                                 const TPG::TPGVertex*>>& resultsPerRootMap,
    std::mutex& resultsPerRootMapMutex,
    std::map<uint64_t, size_t>& archiveSeeds,
    std::map<uint64_t, Archive*>& archiveMap, std::mutex& archiveMapMutex)
{

    // Clone learningEnvironment
    LearningEnvironment* privateLearningEnvironment =
        this->learningEnvironment.clone();

    // Create a TPGExecutionEngine
    Environment privateEnv(this->env.getInstructionSet(),
                           privateLearningEnvironment->getDataSources(),
                           this->env.getNbRegisters());
    TPG::TPGExecutionEngine tee(privateEnv, NULL);

    // Pop a job
    while (!rootsToProcess.empty()) { // Thread safe access to size
        bool doProcess = false;
        std::pair<uint64_t, const TPG::TPGVertex*> rootToProcess;
        { // Mutuel exclusion zone
            std::lock_guard<std::mutex> lock(rootsToProcessMutex);
            if (!rootsToProcess.empty()) { // Additional verification after lock
                rootToProcess = rootsToProcess.front();
                rootsToProcess.pop();
                doProcess = true;
            }
        } // End of mutual exclusion zone

        // Processing to do?
        if (doProcess) {
            doProcess = false;
            // Dedicated archive for the root
            Archive* temporaryArchive = NULL;
            if (mode == LearningMode::TRAINING) {
                temporaryArchive =
                    new Archive(params.archiveSize, params.archivingProbability,
                                archiveSeeds.at(rootToProcess.first));
            }
            tee.setArchive(temporaryArchive);

            std::shared_ptr<EvaluationResult> avgScore =
                this->evaluateRoot(tee, *rootToProcess.second, generationNumber,
                                   mode, *privateLearningEnvironment);

            { // Store result Mutual exclusion zone
                std::lock_guard<std::mutex> lock(resultsPerRootMapMutex);
                resultsPerRootMap.emplace(
                    rootToProcess.first,
                    std::make_pair(avgScore, rootToProcess.second));
            }

            if (mode == LearningMode::TRAINING) {
                { // Insertion archiveMap update mutual exclusion zone
                    std::lock_guard<std::mutex> lock(archiveMapMutex);
                    archiveMap.insert({rootToProcess.first, temporaryArchive});
                }
            }
        }
    }

    // Clean up
    delete privateLearningEnvironment;
}

void Learn::ParallelLearningAgent::mergeArchiveMap(
    std::map<uint64_t, Archive*>& archiveMap)
{
    // Scan the archives backward, starting from the last to identify the
    // last params.archiveSize recordings to keep (or less).
    auto reverseIterator = archiveMap.rbegin();

    uint64_t nbRecordings = 0;
    while (nbRecordings < this->params.archiveSize &&
           reverseIterator != archiveMap.rend()) {
        nbRecordings += reverseIterator->second->getNbRecordings();
        reverseIterator++;
    }

    // Insert identified recordings into this->archive
    while (reverseIterator != archiveMap.rbegin()) {
        reverseIterator--;

        auto i = reverseIterator->first;

        // Skip recordings in the first archive if needed
        uint64_t recordingIdx = 0;
        while (nbRecordings > this->params.archiveSize) {
            recordingIdx++;
            nbRecordings--;
        }

        // Insert remaining recordings
        while (recordingIdx < reverseIterator->second->getNbRecordings()) {
            // Access in reverse order
            const ArchiveRecording& recording =
                reverseIterator->second->at(recordingIdx);
            // forced Insertion
            this->archive.addRecording(
                recording.prog,
                reverseIterator->second->getDataHandlers().at(
                    recording.dataHash),
                recording.result, true);
            recordingIdx++;
        }
    }

    // delete all archives
    reverseIterator = archiveMap.rbegin();
    while (reverseIterator != archiveMap.rend()) {
        delete reverseIterator->second;
        reverseIterator++;
    }
}

void Learn::ParallelLearningAgent::evaluateAllRootsInParallel(
    uint64_t generationNumber, LearningMode mode,
    std::multimap<std::shared_ptr<EvaluationResult>, const TPG::TPGVertex*>&
        results)
{
    // Create and fill the queue for distributing work among threads
    // each root is associated to its number in the list for enabling the
    // determinism of stochastic archive storage.
    std::queue<std::pair<uint64_t, const TPG::TPGVertex*>> rootsToProcess;
    uint64_t idx = 0;

    // Fill also a map for seeding the Archive for each root
    std::map<uint64_t, size_t> archiveSeeds;

    for (const TPG::TPGVertex* root : this->tpg.getRootVertices()) {
        rootsToProcess.push({idx, root});
        if (mode == LearningMode::TRAINING) {
            archiveSeeds.insert(
                {idx, this->rng.getUnsignedInt64(0, UINT64_MAX)});
        }
        idx++;
    }

    // Create Archive Map
    std::map<uint64_t, Archive*> archiveMap;
    // Create Map for results
    std::map<uint64_t, std::pair<std::shared_ptr<EvaluationResult>,
                                 const TPG::TPGVertex*>>
        resultsPerRootMap;

    // Create mutexes
    std::mutex rootsToProcessMutex;
    std::mutex resultsPerRootMutex;
    std::mutex archiveMapMutex;

    // Create the threads
    std::vector<std::thread> threads;
    for (auto i = 0; i < (this->maxNbThreads - 1); i++) {
        threads.emplace_back(std::thread(
            &ParallelLearningAgent::slaveEvalRootThread, this, generationNumber,
            mode, std::ref(rootsToProcess), std::ref(rootsToProcessMutex),
            std::ref(resultsPerRootMap), std::ref(resultsPerRootMutex),
            std::ref(archiveSeeds), std::ref(archiveMap),
            std::ref(archiveMapMutex)));
    }

    // Work in the main thread also
    this->slaveEvalRootThread(generationNumber, mode, rootsToProcess,
                              rootsToProcessMutex, resultsPerRootMap,
                              resultsPerRootMutex, archiveSeeds, archiveMap,
                              archiveMapMutex);

    // Join the threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Merge the results
    for (auto& resultPerRoot : resultsPerRootMap) {
        results.emplace(resultPerRoot.second);
    }

    // Merge the archives
    this->mergeArchiveMap(archiveMap);
}
