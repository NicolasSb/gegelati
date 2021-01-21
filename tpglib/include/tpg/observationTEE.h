/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2019) :
 *
 * Karol Desnos <kdesnos@insa-rennes.fr> (2019)
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

#ifndef OBSERVATION_TEE_H
#define OBSERVATION_TEE_H

#include <set>
#include <vector>

#include "archive.h"
#include "program/programExecutionEngine.h"

#include "tpgGraph.h"

#include "tpg/tpgExecutionEngine.h"

namespace TPG {
    /**
     * Class in charge of executing a TPGGraph.
     *
     * This first implementation is purely sequential and does not parallelize
     * Program execution, nor executions of the TPG starting from several roots.
     */
    class ObservationTEE : public TPGExecutionEngine
    {
      protected:
        /**
         *  \brief intrinsic reward for the evaluation of the current root team
         */
        double intrinsic_reward;

        /**
         * \brief number of executed programs for the current root team
         */
        uint64_t nb_program;

      public:
        /**
         * \brief Main constructor of the class.
         *
         * \param[in] env Environment in which the Program of the TPGGraph will
         *                be executed.
         * \param[in] arch pointer to the Archive for storing recordings of
         *                 the Program Execution. By default, a NULL pointer is
         *                 given, meaning that no recording of the execution
         *                 will be made.
         */
        ObservationTEE(const Environment& env, Archive* arch = NULL)
            : TPGExecutionEngine(env, arch){};

        /**
         * \brief Evaluate all the Program of the outgoing TPGEdge of the
         *        TPGTeam.
         *
         * This method evaluates the Programs of all outgoing TPGEdge of the
         * TPGTeam, and returns the reference to the TPGEdge providing the
         * largest evaluation.
         * TPGEdge leading to a TPGTeam in the excluded set will not be
         * evaluated.
         *
         * \param[in] team the TPGTeam whose outgoing TPGEdge are evaluated.
         * \param[in] excluded the TPGTeam pointers that must be avoided when
         *            TPGEdge lead to them.
         * \return the reference to the TPGEdge evaluated with the the highest
         *         double value (and not excluded).
         *
         * \throw std::runtime_error in case the TPGTeam has no outgoing edge
         *        after excluding all edges leading to TPGVertex from the
         *        excluded set. This should not happen in a correctly
         *        constructed TPGGraph where each TPGTeam must be connected to
         *        at least one TPGAction, to ensure that all cycles have an
         *        exit.
         */
        const TPG::TPGEdge& evaluateTeam(
            const TPGTeam& team, const std::vector<const TPGVertex*>& excluded) override;

        /**
         * \brief Execute the TPGGraph starting from the given TPGVertex.
         *
         * This method browse the graph by successively evaluating Teams and
         * following the TPGEdge proposing the best bids.
         *
         * \param[in] root the TPGVertex from which the execution will start.
         * \return a vector containing all the TPGVertex traversed during the
         *         evaluation of the TPGGraph. The TPGAction resulting from the
         *         TPGGraph execution is at the end of the returned vector.
         */
        const std::pair<std::vector<const TPG::TPGVertex*>, double> executeFromRoot(
            const TPGVertex& root) override;
    };
}; // namespace TPG

#endif
