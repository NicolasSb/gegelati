# GEGELATI Changelog

## Release version 0.1.0
_2020.04.07_

### New features
* Possibility to import a TPGGraph and its programs with the File::TPGGraphDotImporter class.
* New Data::Hash class providing a portable hash mechanism in replacement of std::hash.
* Use of Data::UntypedSharedPtr instead of std::reference_wrapper for fetching operands in DataHandler. This enables fetching "composite" operands, that is operands built on request from native data type in the data handler, and destroyed after use. Data::SupportedType and Data::PrimitiveType no longer needed after this change.
* Adding support for C-style 1D arrays of primitive types in LambdaInstruction.

### Changes
* Reorganization
  * Renaming the Exporter namespace into File.
  * Renaming DataHandlers namespace into Data.
* Switch from transfer.sh to file.io for supporting deployment.
* Update Data::DataHandler, Program::Program, Mutator::LineMutator to take composite operands into account.

### Bug fix
* Training and mutation process were not portable on multiple OSes and compilers because of the diverse implementations of std::hash.


## Release version 0.0.0
_2020.01.14_

### New features
* Implementation of TPG execution & evolution as described in [Stephen Kelly PhD thesis](http://stephenkelly.ca/research_files/Kelly-Stephen-PhD-CSCI-June-2018.pdf).

* Instructions
  * Customized instructions: Instructions used to build programs of the TPG can be customized for each learning process,
    * through specialization of the `Instructions::Instruction` class,
    * through c++ lambda function with the `Instructions::LambdaInstruction` template.
  * Constant arguments: In addition to data sources provided by the learning environment, instructions can take constant as arguments. These constant arguments are subject to mutations during the evolutionary process of the TPG. Class example: `Instructions::MultByConstParam`.

* Learn
  * Parallel TPG execution and evolution: Using the `Learn::ParallelLearningAgent`, a multithreaded execution is implemented for the evaluation of policies starting from several root vertices of the TPG, and for the evolution process of the TPG. To benefit from this parallelism, the `Learn::LearningEnvironment` given to the learning agent must be copyable (see `Learn::LearningEnvironment::isCopyable()` documentation).
  * Determinist learning: Learning process, that is TPG execution, archive management, and evolution process, is fully deterministic and portable based on a given seed and pseudo-random number generators. Determinism is also preserved in the parallel learning process.
  * Classification-oriented learning process: For learning environment representing a classification problem and specializing the `Learn::ClassificationLearningEnvironment` class, a dedicated (and hopefully more efficient) `Learn::ClassificationLearningAgent` is provided. Usage example: MNIST application on [GEGELATI-apps](https://github.com/gegelati/gegelati-apps).

* Exporter
  * DOT exporter: TPG graph resulting from a learning process can be exported for visualization in the dot format.

* Continuous Integration
  * Unit Tests: Unit tests implemented with the GoogleTest framework ensure full coverage of the library code.
  * Automated CI: Travis configuration for building the library and running unit tests under Windows MSVC19 and Linux GCC7, for all branches.
  * Neutral builds: All commits on the develop branch are deployed on the [Neutral builds page](https://gegelati.github.io/neutral-builds).
  * Applications CI: Applications on [GEGELATI-apps](https://github.com/gegelati/gegelati-apps) are built and run automatically for each new [Neutral build](https://gegelati.github.io/neutral-builds).

### Changes

### Bug fix
