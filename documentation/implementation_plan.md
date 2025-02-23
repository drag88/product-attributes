Below is the step‐by‐step implementation plan for the unified attribute generation framework.

## Phase 1: Environment Setup

1.  **Create Project Directories**

    *   Action: Create a project directory structure with the following folders:

        *   `/dags` – for Apache Airflow DAGs
        *   `/modules` – for core processing modules (config loader, validators, generators, etc.)
        *   `/cli` – for the command-line interface entry
        *   `/config` – for JSON/YAML configuration files
        *   `/logs` – for log files

    *   Reference: PRD Section 1 & 3

2.  **Initialize Python Virtual Environment**

    *   Action: Initialize a Python virtual environment using Python (recommended Python 3.11.4 for consistency with our ecosystem).
    *   Command: `python3 -m venv venv`
    *   Reference: Tech Stack Document (Python)

3.  **Install Required Python Packages**

    *   Action: Activate the virtual environment and install required libraries:

        *   Apache Airflow
        *   Pydantic
        *   PyMongo (for MongoDB)
        *   redis
        *   fuzzywuzzy (or rapidfuzz for fuzzy matching)
        *   tenacity (for API/task-level retry logic)

    *   Command: `pip install apache-airflow pydantic pymongo redis fuzzywuzzy tenacity`

    *   Reference: PRD Section 5 & Tech Stack Document

4.  **Set Up Local MongoDB Instance**

    *   Action: Install and launch a local MongoDB instance to serve as the storage backend.
    *   Validation: Connect using the Mongo shell or a client to verify it is running.
    *   Reference: PRD Section 5

## Phase 2: Backend Development

1.  **Develop the Airflow DAG for Attribute Generation**

    *   Action: Create a new file `/dags/attribute_generation_dag.py` to define the main Airflow DAG. Include tasks for configuration loading, extraction, processing, and validation.
    *   Note: Each task should use Airflow’s native retry parameters (e.g., `retries=3` and `retry_delay`) and optional exponential backoff.
    *   Reference: PRD Section 3 & 4

2.  **Implement Configuration Loader Module**

    *   Action: Create `/modules/config_loader.py` that loads JSON or YAML configuration files (defining product types, attributes, enums, etc.).
    *   Validation: Write a test to ensure that a sample config file loads correctly.
    *   Reference: PRD Section 3 & Q&A on adding/modifying product types

3.  **Implement Attribute Validation Module**

    *   Action: Create `/modules/attributes_validator.py` with Pydantic models and custom validators (including enum and fuzzy matching using the function `find_best_enum_match`).
    *   Validation: Run sample validations comparing invalid inputs to expected fallback values (e.g., converting typos to default enum values).
    *   Reference: PRD Section 4 & Q&A: Validation Logic

4.  **Implement Attribute Generation Module**

    *   Action: Create `/modules/attribute_generator.py` to process product attributes. This module should read configuration data, generate attributes by distinguishing common vs. product-specific features and invoke validators.
    *   Validation: Simulate processing a mock product record and verify the output meets the expected model.
    *   Reference: PRD Section 4

5.  **Set Up Logging Module**

    *   Action: Create `/modules/logger_setup.py` to configure Python’s logging module to use JSON-formatted logs and output to both console and `/logs/attribute_generation.log`.
    *   Validation: Run a test log call and check the log file for proper formatting.
    *   Reference: PRD Section 4 & Logging/Monitoring Recommendations

6.  **Integrate MongoDB Connectivity**

    *   Action: Create `/modules/db_connector.py` to manage connections to the local MongoDB instance and provide utility functions for saving generated attribute data.
    *   Validation: Write a simple script to insert and retrieve a document.
    *   Reference: PRD Section 5 & Tech Stack Document

7.  **Integrate Redis Caching for Fuzzy Matching**

    *   Action: Create `/modules/cache_manager.py` that connects to a local Redis instance and caches common attribute lookups to reduce processing time.
    *   Validation: Verify caching by caching a lookup value and retrieving it.
    *   Reference: PRD Section 4 & Performance Targets

8.  **Write Unit Tests for Core Modules**

    *   Action: Create a `/tests` directory and add tests (e.g., `/tests/test_config_loader.py`, `/tests/test_attributes_validator.py`) to ensure each module behaves as expected.
    *   Validation: Run `pytest` and verify that all tests pass 100%.
    *   Reference: PRD Section 8

## Phase 3: CLI/API Integration

1.  **Develop CLI Entry Point**

    *   Action: Create a CLI file `/cli/main.py` that provides commands to load configuration and run the attribute generation workflow manually.
    *   Functionality: Include at least two commands: one to validate the configuration file and another to trigger the pipeline execution.
    *   Reference: PRD Section 3 & Q&A about API/CLI interaction

2.  **Implement CLI Command for Configuration Validation**

    *   Action: In `/cli/main.py`, add a function (e.g., `load_config_command`) which loads and validates the configuration file using `/modules/config_loader.py`.
    *   Validation: Run the CLI command with a sample config file in `/config/sample_config.yaml` and check for error-free output.
    *   Reference: PRD Section 3 & Q&A

3.  **Implement CLI Command to Trigger Pipeline Execution**

    *   Action: Add a second command (e.g., `run_pipeline`) that either triggers the Airflow DAG (by programmatically calling Airflow’s CLI) or directly invokes the modules for attribute processing.
    *   Validation: Run the command and check that the processing workflow executes and logs are generated.
    *   Reference: PRD Section 3

4.  **Error Handling in the CLI**

    *   Action: Ensure that the CLI catches exceptions during configuration loading or pipeline execution and logs detailed error messages using the logging module.
    *   Reference: PRD Section 4 & Q&A on error handling

## Phase 4: Integration

1.  **Integrate Modules into the Airflow DAG**

    *   Action: Update `/dags/attribute_generation_dag.py` to call functions from `/modules/config_loader.py`, `/modules/attributes_validator.py`, and `/modules/attribute_generator.py` as separate tasks.
    *   Note: Configure each task with built-in retry logic and an exponential backoff where needed (using Airflow parameters and tenacity for lower-level API tasks).
    *   Reference: PRD Section 4 & Q&A: Error Handling

2.  **Integrate Logging Across Tasks**

    *   Action: Modify each module’s entry point in the DAG to import and use the logger from `/modules/logger_setup.py`, ensuring consistent logging of progress and errors.
    *   Validation: Trigger a run and inspect `/logs/attribute_generation.log` for correct log entries.
    *   Reference: PRD Section 4

3.  **Integrate Database and Cache Modules in the Workflow**

    *   Action: Within your DAG tasks, after attribute generation, call `/modules/db_connector.py` to save outputs and `/modules/cache_manager.py` for caching frequently used attribute lookups.
    *   Validation: Confirm successful insertion in MongoDB and appropriate cache hit responses in Redis.
    *   Reference: PRD Section 5 & Performance Targets

4.  **Perform End-to-End Testing**

    *   Action: Run the complete pipeline (via the CLI command or directly in Airflow) using a sample configuration file from `/config/sample_config.yaml`.
    *   Validation: Verify that processing is completed within expected performance limits, logs are detailed, and all attributes (common and specific) are correctly generated and stored.
    *   Reference: PRD Sections 3 and 8

## Phase 5: Deployment

1.  **Create a Local Deployment Script**

    *   Action: Write a shell script `/run_local.sh` that activates the virtual environment, starts the local Airflow scheduler and webserver, and (optionally) ensures the local MongoDB and Redis instances are running.
    *   Validation: Run the script to confirm all components start without error.
    *   Reference: PRD Sections 6 & 7

2.  **Prepare a README with Setup Instructions**

    *   Action: Create a `README.md` file at the project root that outlines:

        *   How to set up the development environment
        *   How to install dependencies
        *   How to run the local deployment script
        *   How to use the CLI commands

    *   Reference: PRD Section 7

3.  **Final Manual Test of the Full Workflow**

    *   Action: Execute the pipeline from configuration loading through attribute generation using the CLI, and then verify all outputs (logs, database entries, and cache entries) are as expected.
    *   Validation: Ensure that for a sample set, product processing completes in under 30 seconds per product and results are stored in MongoDB.
    *   Reference: PRD Section 7 (Performance Targets)

This concludes the implementation plan. Each step follows strict requirements from the PRD and additional documents. Subsequent testing phases (unit, integration, and end-to-end) ensure all modules and integrations perform as specified.
