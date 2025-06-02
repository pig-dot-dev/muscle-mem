# Muscle Mem
Thanks for interest in contributing to Muscle Mem! 

The project is relatively early and exploratory, so the most helpful thing you could do is engage in the [discord](https://discord.gg/s84dXDff3K) or file an issue/discussion. Unsolicited PRs will likely be rejected if they touch the API surface area or substantially change the implementation. Please file an issue first to discuss your proposal!

## Setup
```bash
pip install -e ".[dev]"
```

## Testing
Tests may be run with:
```bash
python3 -m pytest tests/test_caching.py -v --capture=no
```
The `--capture=no` is to show the metrics capture, for performance reasons. If any changes are made, please manually ensure that performance has not regressed. 

It's also recommended to test the examples:
```bash
cd examples
python3 greeter.py
python3 cua.py
```
For CUA, you'll need an OpenAI API key exported.