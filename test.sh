#!/bin/bash

# pytorch fwd
uv run pytest  ./tests/test_attention.py -k test_flash_forward_pass_pytorch

uv run pytest  ./tests/test_attention.py -k test_flash_backward_pytorch


