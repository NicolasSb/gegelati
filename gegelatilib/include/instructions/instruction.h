/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2019 - 2021) :
 *
 * Karol Desnos <kdesnos@insa-rennes.fr> (2019 - 2020)
 * Nicolas Sourbier <nsourbie@insa-rennes.fr> (2019 - 2020)
 * Thomas Bourgoin <tbourgoi@insa-rennes.fr> (2021)
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

#ifndef INSTRUCTION_H
#define INSTRUCTION_H

#include <functional>
#include <memory>
#include <typeinfo>
#include <vector>

#include "data/untypedSharedPtr.h"

namespace Instructions {
    /**
     * \brief This abstract class is the base class for any instruction to be
     * used in a Program.
     *
     * An Instruction declares a list of operands it needs to be executed.
     * This information will be used to fetch the required operands from
     * any ProgramLine, and to ensure the compatibility of the type of the
     * fetched operands with the instruction before executing it.
     */
    class Instruction
    {

#ifdef CODE_GENERATION
      public:
        /**
         * \brief function saying if an Instruction can be printed in a C file
         * during the code gen.
         *
         * @return true if the printTemplate of the Instruction is not empty,
         * false in other case.
         */
        virtual bool isPrintable() const;

        /**
         * \brief This function return the printTemplate of the line of code
         * used to represent the instruction.
         *
         * \return The printTemplate of the line of code used to represent the
         * instruction in the C program.
         */
        virtual const std::string& getPrintTemplate() const;

        /**
         * \brief Retrieve the primitive data type of the operand
         *
         * \param[in] opIdx const uint64_t reference to the index of the operand
         * of the instruction
         *
         * \return The type of the operand. If the operand is an array return
         * the type of the element of the array.
         */
        virtual std::string getPrintablePrimitiveOperandType(
            const uint64_t& opIdx) const;

      protected:
        /**
         * \brief Protected constructor to force the class abstract nature.
         *
         * The definition of this constructor initialize the string
         * printTemplate with the format given as parameter.
         *
         * \param[in] printTemplate of the line used to represent the
         * instruction in the C files generated.
         */
        Instruction(std::string printTemplate);

        /**
         * printTemplate of the instruction used to generate the code. The
         * result of the function is represented with $0. The first parameter
         * correspond to $1 the second to $2...
         */
        std::string printTemplate;

      private:
        /**
         * This regular expression is used to extract the primitive type from
         *the type of an instruction. It removes the const definition and any
         *array declaration from the type.
         *
         * Explanation:
         * \code{.unparse}
         * (const )?
         * \endcode
         * match \c const if it is in the operand type
         * \code{.unparse}
         * [\\w:\\*]*
         * \endcode
         * match 1 to infinite words separated by \c : . Allow to match \c
         *Data::Constant. \code{.unparse}
         * ((struct )?[\\w:\\*]*)
         * \endcode
         * match the primitive type of the operand. For MSVC Data::Constant is a
         *struct so we need to match this possibility too. \code{.unparse} [ ]?
         * \endcode
         * match if the operand is a 1D array without a static size. This
         *pattern can only appear 0 or 1 time in the C/Cpp semantic.
         * \code{.unparse}
         * (\\d)+
         * \endcode
         * match any digit 1 to infinite times.
         * \code{.unparse}
         * (\\[(\\d)+\\])*
         * \endcode
         * match 0 to infinite dimensions that have static size.
         **/
        inline static const std::string GET_PRINT_PRIMITIVE_OPERAND_TYPE{
            "(const )?((struct )?[\\w:\\*]*)[ ]?(\\[(\\d)+\\])*"};

#endif // CODE_GENERATION

      public:
        /// Default virtual destructor for polyphormism.
        virtual ~Instruction() = default;

        /**
         * \brief Get the list of operand types needed by the Instruction.
         *
         * \return a const reference on the list of operand type_info of the
         * Instruction.
         */
        const std::vector<std::reference_wrapper<const std::type_info>>&
        getOperandTypes() const;

        /**
         * \brief Get the number of operands required to execute the
         * Instruction.
         *
         * \return an unsigned int value corresponding to the number of operands
         * required by the Instruction.
         */
        unsigned int getNbOperands() const;

        /**
         * \brief Check if a given vector contains elements whose types
         * corresponds to the types of the Instruction operands.
         *
         * \param[in] arguments a const list of shared pointers to any type of
         * object. (not doable at compile time)
         */
        virtual bool checkOperandTypes(
            const std::vector<Data::UntypedSharedPtr>& arguments) const;

        /**
         * \brief Execute the Instruction for the given arguments.
         *
         * Derived class should implement their own behavior for this method. In
         * case of invalid arguments, for type or number or value
         * reason, this method should always return 0.0.
         *
         * \param[in] args the vector of UntypedSharedPtr passed to the
         * Instruction.
         * \return the default implementation of the Instruction
         * class returns 0.0 if the given params or arguments are not valid.
         * Otherwise, 1.0 is returned.
         */
        virtual double execute(
            const std::vector<Data::UntypedSharedPtr>& args) const = 0;

      protected:
#ifndef CODE_GENERATION
        /**
         * \brief Protected constructor to force the class abstract nature.
         *
         * The definition of this constructor initialize an empty operandType
         * list and sets the number of required parameters to 0.
         */
        Instruction();
#endif // CODE_GENERATION
        /**
         * \brief List of the types of the operands needed to execute the
         * instruction.
         */
        std::vector<std::reference_wrapper<const std::type_info>> operandTypes;
    };

} // namespace Instructions
#endif
