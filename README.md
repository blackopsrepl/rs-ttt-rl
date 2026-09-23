# rs-ttt-rl

A Rust port of the [ttt-rl](https://github.com/antirez/ttt-rl) project, originally written in C by antirez. This is a fully functional implementation of a Tic-Tac-Toe game using reinforcement learning with a neural network.

### Key Features Implemented

- ✅ **Neural Network Architecture**: Single hidden layer with 18 inputs, 100 hidden neurons, and 9 outputs
- ✅ **Reinforcement Learning**: Backpropagation with reward scaling and temporal importance weighting
- ✅ **Game Logic**: Complete move validation, win/draw detection, and game flow
- ✅ **Training System**: Neural network learns from gameplay with configurable training cycles
- ✅ **Interactive Play**: Stable command-line interface for human vs. AI gameplay
- ✅ **Visualization**: Clear board display with position guides and neural network probability grids

### Training Results

With the current learning loop:
- **Learning Works**: The network's win rate against a random opponent rises with training — a typical 50,000-game run ends with the network winning about two thirds of the games in the final reporting window
- **Stable Learning**: Neural network performance increases with training
- **Robust Gameplay**: No crashes during interactive play or training sessions

## Code Structure

- **Neural Network** (`NeuralNetwork` struct):
  - Supports a feedforward architecture with 18 inputs, 100 hidden neurons, and 9 outputs
  - Uses ReLU activation for hidden layers and softmax for output probabilities
  - Weights and biases initialized with random values in [-0.5, 0.5) using `CRand`
  - Stores intermediate activations for backpropagation
  - Pre-allocated delta arrays for efficient training

- **Game State** (`GameState` struct):
  - Represents the Tic-Tac-Toe board as a 9-element array (`'.'`, `'X'`, or `'O'`)
  - Tracks the current player (0 for human `'X'`, 1 for computer `'O'`)
  - Complete win/tie detection with proper game flow logic

- **Pseudo-Random Generator** (`CRand` struct):
  - Implements a linear congruential generator (LCG) compatible with C's `rand()`
  - Provides `rand()` for integers (0 to 32767) and `rand_float()` for floats (0 to 1)
  - Seeds with system time (seconds XOR nanoseconds) for uniqueness
  - Optimized for reuse across neural network initialization

- **Reinforcement Learning**:
  - **Reward System**: +1.0 for wins, +0.3 for ties, -2.0 for losses
  - **Temporal Scaling**: Later moves weighted more heavily during learning
  - **Probability Distribution**: Winner-take-all for positive rewards, uniform distribution over the other legal moves for negative rewards

## Building and Running

Requires Rust (stable). To compile and run:

```bash
# Run tests
cargo test

# Build an optimized binary
cargo build --release

# Train 150000 games (the default) against random moves, then play
# interactively as X against the network (O)
cargo run --release -- 150000

# Skip training and play with a freshly initialized network
cargo run --release -- 0
```

The binary takes a single optional positional argument: the number of
training games (default: 150000). There are no subcommands. The trained
network is kept in memory only — every run trains from scratch.

A `Makefile` wraps the common cargo commands:

```bash
make                  # format, lint, build, test
make lint             # cargo clippy with warnings denied
make run              # train GAMES games, then play (default GAMES=150000)
make run GAMES=50000  # shorter training session
```

## Dependencies

Uses only the Rust standard library (std)

## Notes

- The `CRand` implementation aims to mimic C's rand() to maintain compatibility with the original
- The port balances performance optimization with code clarity and maintainability
- Neural network architecture and learning parameters match the original C implementation