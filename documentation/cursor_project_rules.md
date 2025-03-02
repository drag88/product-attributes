## Project Overview

*   **Type:** Cursor Project Rules

*   **Description:**

    *   A unified framework is being developed to streamline the generation of attributes for various e-commerce fashion products, including Sarees, Kurtas, and Saree Blouses.
    *   This framework consolidates existing attribute pipelines, preserving all current validation logic and processing.
    *   The goal is to maintain flexibility to integrate new products and attributes as trends evolve.
    *   The framework will intelligently differentiate between unique attributes specific to individual products and commonalities across different product categories, such as shirts or trousers.

*   **Primary Goal:**

    *   Build a unified attribute generation framework that consolidates current processing methods.
    *   Ensure validations are intact while allowing easy configuration through JSON/YAML files.
    *   Include robust error handling and workflow orchestration using Apache Airflow.

## Key Developer Note

*   **Implementation Insight for LLM Developers:**

    *   Developers should familiarize themselves with existing data structures for Sarees, Kurtas, and Saree Blouses.
    *   Review current codebase to understand and extract business logic.
    *   Adapt and integrate these logics into the new framework, ensuring the consolidation retains essential functionality and supports upcoming enhancements.

## Project Structure

### Framework-Specific Routing

*   **Directory Rules:**

    *   **Python 3.x & Apache Airflow (v2.x):** Segregate application code from workflow definitions.

    *   Workflow definitions (DAGs) reside in a `dags/` directory; core business logic, validations, and configurations are under `src/`.

    *   Examples include:

        *   `dags/`: Contains Apache Airflow workflow definitions and task scripts.
        *   `src/`: Hosts core functionality such as data processing, configuration parsing, and attribute validation using Pydantic models and custom validators.
        *   `config/`: Stores JSON/YAML files defining product types, attributes, and validation rules.

### Key Files

*   **Stack-Versioned Patterns:**

    *   `dags/main_dag.py`**:** Manages the central orchestration of the attribute generation workflow using Apache Airflow.
    *   `src/main.py`**:** Serves as the entry point for initiating processing, ensuring configuration files are accurately loaded and validated.
    *   `config/products.yaml`**:** Provides an example configuration file outlining product types and enumerated attribute definitions.

## Tech Stack Rules

*   **Version Enforcement:**

    *   <Python@3.x> - Follow Python best practices, with a focus on module separation and asynchronous processing when applicable.
    *   Apache <Airflow@2.x> - Full utilization of Airflow’s DAG construction with retries, exponential backoff, and proper logging.
    *   MongoDB - Leverage schema flexibility for dynamic, cloud-based storage of product attributes.
    *   Pydantic & Enum - Implement strict type validation and enum-based fuzzy matching for standardizing attribute inputs.
    *   Redis - Implement caching for frequently accessed attribute lookups to boost performance.

## PRD Compliance

*   **Mandatory Compliance:**

    *   The framework will consolidate existing attribute pipelines while preserving validation logic and offering flexibility to accommodate new product types and attributes as they evolve.
    *   This compliance ensures that both existing actions and future updates integrate seamlessly without requiring manual reconfigurations.

## App Flow Integration

*   **CLI & API Integration:**

    *   Users initiate processing by loading configuration files through CLI or an API endpoint, triggering the Airflow DAG (`dags/main_dag.py`).
    *   Ensures product-specific attribute generation follows a structured workflow encompassing configuration loading, validation via Pydantic, processing, error handling, and precise logging.

## Best Practices

*   **Python**

    *   Maintain clear module separations (`src/`, `dags/`, `config/`).
    *   Employ virtual environments (e.g., pipenv or poetry) for consistent builds.
    *   Develop comprehensive unit and integration tests using pytest.

*   **Apache Airflow**

    *   Define clear, modular DAGs with task retries and SLAs.
    *   Use Airflow’s logging and monitoring features for real-time observability.
    *   Ensure idempotency in tasks to enable graceful retries.

*   **MongoDB**

    *   Follow schema best practices while allowing NoSQL flexibility for consistency.
    *   Implement indexing and sharding for scalability.
    *   Regularly back up and monitor database performance.

*   **Pydantic & Enum**

    *   Use Pydantic models for type validation and data parsing.
    *   Develop custom validators for handling fuzzy matching and enforcing enum consistency.

*   **Redis**

    *   Use caching judiciously to reduce computational loads.
    *   Monitor cache performance metrics to optimize caching strategies.
    *   Implement eviction policies to maintain performance.

## Rules

*   Follow folder/file patterns derived from defined tech stack and project guidelines.
*   Use dedicated `dags/` directory for Apache Airflow workflow definitions.
*   Ensure distinct separation of configuration files (`config/`) from executable code (`src/` and `dags/`).
*   Keep validation logic consistent with Pydantic and enum-based checks.
*   Implement robust error handling through built-in Airflow retries and complementary tools like tenacity for granular management.
*   Adhere to a configuration-driven customization model, allowing new attributes and product types to be added primarily via editable JSON/YAML files.
*   Avoid mixing structural patterns, such as placing workflow definitions within the `src/` directory.
