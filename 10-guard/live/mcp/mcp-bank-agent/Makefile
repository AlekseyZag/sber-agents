.PHONY: help install run info clean

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "📋 Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies via uv
	@echo "📦 Installing dependencies..."
	uv sync
	@echo "✅ Dependencies installed successfully!"

run: ## Start HTTP MCP server on port 8000
	@echo "🚀 Starting Bank Agent MCP server..."
	@echo "📍 Server will be available at: http://localhost:8000/mcp"
	@echo "⏹️  Press Ctrl+C to stop"
	@echo ""
	uv run server.py

run-dev: ## Start MCP server with auto-reload (development mode)
	@echo "🔄 Starting Bank Agent MCP server in DEV mode (auto-reload)..."
	@echo "📍 Server will be available at: http://localhost:8000/mcp"
	@echo "📝 Changes to .py files will trigger automatic reload"
	@echo "⏹️  Press Ctrl+C to stop"
	@echo ""
	uv run dev_server.py

info: ## Show server information
	@echo "📊 Bank Agent MCP Server Information"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Name:          mcp-bank-agent"
	@echo "  Transport:     streamable-http"
	@echo "  Port:          8000"
	@echo "  URL:           http://localhost:8000/mcp"
	@echo "  Database:      data/bank_products.json"
	@echo ""
	@if [ -f data/bank_products.json ]; then \
		echo "  📊 Database status: ✅ Ready"; \
		uv run python -c "import json; data = json.load(open('data/bank_products.json')); print(f'  📈 Total products:  {len(data)}')"; \
	else \
		echo "  📊 Database status: ❌ Not found"; \
	fi
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

clean: ## Clean cache and temporary files
	@echo "🧹 Cleaning cache and temporary files..."
	@rm -rf __pycache__
	@rm -rf .pytest_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete!"

