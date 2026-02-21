# Discrete Math Project

This repository contains image-processing scripts for:
- Sobel edge detection
- Edge detection with Gaussian noise reduction
- Polygon extraction from detected edges

## Project Structure

```text
src/
  main.py
  gussianFilter.py
  shapes.py
docs/
  Project.pdf
assets/
  samples/
  results/
testCase/
tests/TestCases/
```

## Requirements

Install dependencies from:

```bash
pip install -r requirement.txt
```

## How to Run

Run commands from the repository root:

```bash
python src/main.py
python src/gussianFilter.py
python src/shapes.py
```

## Notes

- `src/main.py` uses input: `testCase/4.png`
- `src/gussianFilter.py` uses input: `table.jpg`
- `src/shapes.py` uses input: `4.jpg`
- Output images are generated in the current working directory.

