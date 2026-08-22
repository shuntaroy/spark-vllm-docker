#!/bin/bash
set -e

SITE_PACKAGES=/usr/local/lib/python3.12/dist-packages

# The parser does a bare `from llmjp4_harmony import ...`. vLLM loads
# --reasoning-parser-plugin via importlib.spec_from_file_location, which does
# NOT add the plugin's directory to sys.path, so the helper module must live
# somewhere importable.
echo "Installing llmjp4_harmony.py into $SITE_PACKAGES"
cp llmjp4_harmony.py "$SITE_PACKAGES/llmjp4_harmony.py"

cp llmjp4_reasoning_parser.py "$WORKSPACE_DIR/llmjp4_reasoning_parser.py"
echo "=======> to apply the LLM-jp-4 reasoning parser, use:"
echo "         --reasoning-parser-plugin $WORKSPACE_DIR/llmjp4_reasoning_parser.py --reasoning-parser llmjp4"
