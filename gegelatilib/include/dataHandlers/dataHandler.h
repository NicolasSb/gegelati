#ifndef DATA_HANDLER_H
#define DATA_HANDLER_H

#include <typeinfo>
#include <vector>
#include <functional>

#include "supportedTypes.h"

namespace std {
	/*
	* \brief  Equality operator to std to enable use of standard algorithm on vectors of reference_wrapper of const std::type_info.
	*/
	bool operator==(const std::reference_wrapper<const std::type_info>& r0, const std::reference_wrapper<const std::type_info>& r1);
}

namespace DataHandlers {
	/**
	* \brief Base class for all sources of data to be accessed by a TPG Instruction executed within a Program.
	*/
	class DataHandler {

	protected:

		/**
		* \brief Static count used to initialize the id of each DataHandler.
		*/
		static size_t count;

		/**
		* \brief Identifier of each DataHandler.
		*
		* This identifier should be used as a seed for the initialization
		* of the hash calculation.
		* Two DataHandler resulting from a copy should thus have the same id.
		*/
		const size_t id;

		/**
		* \brief List of the types of the operands needed to execute the instruction.
		*
		* Because std::unordered_set was too complex to use (because it does not support std::reference_wrapper easily), std::vector is used instead.*
		* Adding the same type several type to the list of providedType will lead to undefined behavior.
		*/
		std::vector<std::reference_wrapper<const std::type_info>> providedTypes;

		/**
		* \brief Cached value returned by the getHash() function.
		*
		* The value of the hash is updated whenever the updateHash function is called.
		*/
		mutable size_t cachedHash;

		/**
		* \brief Boolean value indicating whether the current cachedValue is
		* valid, or not.
		*
		* When getting the value of the hash, it shall be automatically updated
		* when invalidCachedHash is set to true. Whenever the data contained in
		* a DataHandler is modifier, the invalidCachedHash attribute shall be
		* set to true.
		*
		*/
		mutable bool invalidCachedHash;

		/**
		* \brief Update the cachedHash value.
		*
		* This methods trigger an update of the cachedHash value and
		* returns the new value.
		*
		* \return the new value of the cachedhash attribute.
		*/
		virtual size_t updateHash() const = 0;

	public:
		/**
		* \brief Default constructor of the DataHandler class.
		*/
		DataHandler();

		/// Default destructor
		virtual ~DataHandler() = default;

		/**
		* \brief Default copy constructor.
		*/
		DataHandler(const DataHandler& other) = default;

		/**
		* \brief Return a copy of the DataHandler (with all its content).
		*
		* The returned copy shuold always have the same polymorphic type
		* as the original object, and give the same hash and data until
		* the original or the copy is modified.
		*
		* \return a pointer to the clone.
		*/
		virtual DataHandler* clone() const = 0;

		/**
		* \brief Get the ID of the DataHandler.
		*
		* Two DataHandler should have the same ID only if the are copy
		* from each other, possibly holding different data.
		* This property will be used to simplify check that two different
		* DataHandler have the exact same characteristics (handled types,
		* addressSpace, ..)
		*/
		size_t getId() const;

		/**
		* \brief Get the current value of the hash for this DataHandler.
		*
		* This method returns the value of the hash, and updates it if
		* necessary.
		*
		* \return the cached value of the Hash.
		*/
		size_t getHash() const;

		/**
		* \brief Check a given DataHandler can handle data for the given data type.
		*
		* \param[in] type the std::type_info whose availability in the DataHandler is being tested.
		* \return true if the DataHandler can handle data for the given data type, and false otherwise.
		*/
		bool canHandle(const std::type_info& type) const;

		/**
		* \brief Retrieve the set of types provided by the DataHandler.
		*
		* \return a const reference to the data type set provided by the DataHandler.
		*/
		const std::vector<std::reference_wrapper<const std::type_info>>& getHandledTypes() const;

		/**
		* \brief Get the getAddressSpace size for the given data type.
		*
		* Since a single DataHandler may be able to provide data of different types, the addressable space may vary depending
		* on the accessed data type. This method returns the size of addressable data for each type of data.
		*
		* \param[in] type the std::type_info of data whose address space is retrieved.
		* \return the size of the retrieved address space, or 0 if the data type is not handled by the DataHandler.
		*/
		virtual size_t getAddressSpace(const std::type_info& type) const = 0;

		/**
		* \brief Get the largest AddressSpace for all data types handled by the DataHandler.
		*
		* This method relies on the getAddressSpace and getHandledTypes methods
		* to compute the size of the largest addressSpace required by the dataHandler.
		* \return the size of the largest addressSpace.
		*/
		size_t getLargestAddressSpace() const;

		/**
		* \brief Generic method for DataHandler to reset their data.
		*
		* Method used to reset the data handled by a DataHandler. Each
		* DataHandler can implement a custom behavior, or even no behavior at
		* all for this method.
		*
		* This method shall invalidate the cachedHash.
		*
		*/
		virtual void resetData() = 0;

		/**
		* \brief Get data of the given type, from the given address.
		*
		* \param[in] type the std::type_info of data retrieved.
		* \param[in] address the location of the data to retrieve.
		* \throws std::invalid_argument if the given data type is not provided by the DataHandler.
		* \throws std::out_of_range if the given address is invalid for the given data type.
		* \return a const reference to the requested data.
		*/
		virtual const SupportedType& getDataAt(const std::type_info& type, const size_t address) const = 0;
	};
}

#endif