#!/usr/bin/bash

# Ask user to set OPENAI_API_KEY if not already set
if [ -z "$OPENAI_API_KEY" ]; then
    read -p "Enter your OpenAI API key: " OPENAI_API_KEY
    export OPENAI_API_KEY=$OPENAI_API_KEY
else
    echo "OPENAI_API_KEY is already set"
fi

python openevolve-run.py projects/eplb/initial_program.py projects/eplb/evaluator.py \
    --config projects/eplb/config.yaml \
    --checkpoint projects/eplb/openevolve_output/checkpoints/checkpoint_310/
    --iterations 650
