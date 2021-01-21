/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2019 - 2020) :
 *
 * Karol Desnos <kdesnos@insa-rennes.fr> (2019 - 2020)
 * Nicolas Sourbier <nsourbie@insa-rennes.fr> (2020)
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

#ifndef OBSERVATION_PROGRAM_H
#define OBSERVATIONPROGRAM_H

#include <algorithm>
#include <vector>

#include "data/constantHandler.h"
#include "environment.h"
#include "program/line.h"
#include "program/program.h"

namespace Program {
    /**
     * \brief The Program class contains a list of program lines that can be
     * executed within a well defined Environment.
     */
    class ObservationProgram : public Program
    {
      protected:
        /**
         *   \brief Target of the Program
         **/
        double target;

        /// Delete the default constructor.
        ObservationProgram() = delete;

      public:
        /**
         * \brief Main constructor of the Program.
         *
         * \param[in] e the reference to the Environment that will be referenced
         * in the Program attributes.
         */
        ObservationProgram(const Environment& e)
            : Program{e}, target{0}
        {
            constants.resetData(); // force all constant to 0 at first.
        };

        /**
         * \brief Copy constructor of the Program.
         *
         * This copy constructor realises a deep copy of the Line of the given
         * Program, instead of the default shallow copy.
         *
         * \param[in] other a const reference the the copied Program.
         */
        ObservationProgram(const ObservationProgram& other)
            : Program{other}, target{other.target}
        {
            // Replace lines with their copy
            // Keep intro info
            std::transform(
                lines.begin(), lines.end(), lines.begin(),
                [](std::pair<Line*, bool>& otherLine)
                    -> std::pair<Line*, bool> {
                    return {new Line(*(otherLine.first)), otherLine.second};
                });
        };

        /**
         * \brief accessor to the target of the Observation Program
         *
         * \return the value of the target
         */
        inline const double getTargetValue() const {return this->target;};

        /**
         * \brief mutator of the target of the Observation Program
         *
         * \param[in] val  the new value of the target
         */
        inline void setTargetValue(double val) {this->target = val;};

    };
} // namespace Program
#endif
