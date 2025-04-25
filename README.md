markdown

# rs-ttt-rl

A Rust port of the [ttt-rl](https://github.com/antirez/ttt-rl) project, originally written in C by antirez. This is an incomplete implementation of a Tic-Tac-Toe game using reinforcement learning with a neural network.

## Project Status

This is a work-in-progress port. The following components have been implemented:

- Neural network structure with input, hidden, and output layers
- Forward pass with ReLU activation for hidden layers and softmax for output
- Weight and bias initialization using a custom pseudo-random number generator (`CRand`)
- Basic game state representation
- Unit tests for ReLU, softmax, forward pass, and random weight generation

### Missing Features (To Be Implemented)

- Backpropagation and training logic
- Game logic (move validation, win/draw conditions, etc.)
- Reinforcement learning integration
- Main game loop and interaction with the neural network
- Comprehensive game state tests
- Command-line interface for playing the game

## Code Structure

- **Neural Network** (`NeuralNetwork` struct):
  - Supports a feedforward architecture with 18 inputs, 100 hidden neurons, and 9 outputs
  - Uses ReLU activation for hidden layers and softmax for output probabilities
  - Weights and biases initialized with random values in [-0.5, 0.5) using `CRand`
  - Stores intermediate activations for future backpropagation

- **Game State** (`GameState` struct):
  - Represents the Tic-Tac-Toe board as a 9-element array (`'.'`, `'X'`, or `'O'`)
  - Tracks the current player (0 for human `'X'`, 1 for computer `'O'`)

- **Pseudo-Random Generator** (`CRand` struct):
  - Implements a linear congruential generator (LCG) compatible with C's `rand()`
  - Provides `rand()` for integers (0 to 32767) and `rand_float()` for floats (0 to 1)
  - Seeds with system time (seconds XOR nanoseconds) for uniqueness

- **Tests**:
  - Unit tests for ReLU and its derivative
  - Softmax numerical stability and correctness
  - Neural network forward pass with manual verification
  - Random weight generation within expected bounds
  - Basic game state initialization

## Building and Running

Requires Rust (stable). To compile and run tests:

```bash
cargo test
```

The main function is currently empty, as the game logic is not yet implemented.

## Dependencies

Uses only the Rust standard library (std)

## Notes

- The CRand implementation aims to mimic C's rand() so that we only use the standard library
- The port prioritizes clarity and correctness over performance at this stage.