/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2019 - 2020) :
 *
 * Karol Desnos <kdesnos@insa-rennes.fr> (2019 - 2020)
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

#include "instructions/instruction.h"

#include <iostream>

using namespace Instructions;

Instruction::Instruction() : operandTypes()
{
}

const std::vector<std::reference_wrapper<const std::type_info>>& Instruction::
    getOperandTypes() const
{
    return this->operandTypes;
}

unsigned int Instructions::Instruction::getNbOperands() const
{
    return (unsigned int)this->operandTypes.size();
}

bool Instruction::checkOperandTypes(
    const std::vector<Data::UntypedSharedPtr>& arguments) const
{
    if (arguments.size() != this->operandTypes.size()) {
        return false;
    }

    for (int i = 0; i < arguments.size(); i++) {
        if (arguments.at(i).getType() != this->operandTypes.at(i).get()) {
            return false;
        }
    }
    return true;
}

double Instruction::execute(
    const std::vector<Data::UntypedSharedPtr>& arguments) const
{
#ifndef NDEBUG
    if (!this->checkOperandTypes(arguments)) {
        return 0.0;
    }
    else {
        return 1.0;
    }
#else
    return 1.0;
#endif
}
