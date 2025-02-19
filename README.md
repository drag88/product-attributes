# Product Attribute Generation Framework

A robust and scalable framework for generating product attributes for various clothing items like sarees, blouses, and kurtas. The system uses AI-powered attribute generation with a unified, configuration-driven approach.

## Features

- Unified base structure for all product types
- Configuration-driven product attribute generation
- Efficient API integration with rate limiting and caching
- Extensible design for easy addition of new product types
- Comprehensive error handling and validation

## Project Structure

```
product_attributes/
├── config/           # Configuration files for different product types
├── src/             # Source code
│   ├── base/        # Base classes and shared components
│   ├── models/      # Product-specific models
│   ├── factories/   # Factory pattern implementation
│   ├── utils/       # Utility functions and helpers
│   └── services/    # Core services (API, attribute generation)
└── main.py          # Main application entry point
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
- Copy `.env.example` to `.env`
- Update the values as needed

## Usage

Run the main script:
```bash
python main.py
```

## License

MIT 