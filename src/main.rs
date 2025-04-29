/*
rs-ttt-rl
Rewrite in Rust of ttt-rl, originally written by antirez - https://github.com/antirez/ttt-rl
*/

use rs_ttt_rl::CRand;
use std::f32;

// Neural network parameters
const NN_INPUT_SIZE: usize = 18;
const NN_HIDDEN_SIZE: usize = 100;
const NN_OUTPUT_SIZE: usize = 9;
const LEARNING_RATE: f32 = 0.1;

// Game board representation
struct GameState {
    board: [char; 9],      // Can be '.' (empty) or 'X', 'O'
    current_player: usize, // 0 for player (X), 1 for computer (O)
}

struct NeuralNetwork {
    // Weights and biases
    weights_ih: Vec<f32>,
    weights_ho: Vec<f32>,
    biases_h: Vec<f32>,
    biases_o: Vec<f32>,

    // Activations are part of the structure for simplicity
    inputs: Vec<f32>,
    hidden: Vec<f32>,
    raw_logits: Vec<f32>, // Outputs before softmax()
    outputs: Vec<f32>,    // Outputs after softmax()
}

// ReLU activation function
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

// Derivative of ReLU acctivation function
fn relu_derivative(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

/* Initialize a neural network with random weights.
Since we are committing to using the standard libary only,
we will implement CRand as a simple linear congruential generator
compatible with standard C rnd(), from a new random_weight!() macro. */
#[macro_export]
macro_rules! random_weight {
    () => {{
        let mut rng: CRand = CRand::new();
        rng.rand_float() - 0.5
    }};
}

impl Default for NeuralNetwork {
    fn default() -> Self {
        NeuralNetwork::new()
    }
}

impl NeuralNetwork {
    fn new() -> Self {
        let mut weights_ih: Vec<f32> = vec![0.0; NN_INPUT_SIZE * NN_HIDDEN_SIZE];
        let mut weights_ho: Vec<f32> = vec![0.0; NN_HIDDEN_SIZE * NN_OUTPUT_SIZE];
        let mut biases_h: Vec<f32> = vec![0.0; NN_HIDDEN_SIZE];
        let mut biases_o: Vec<f32> = vec![0.0; NN_OUTPUT_SIZE];

        /* Initialize weights with random values between -0.5 and 0.5
        Since random_weight!() uses the current timestamp as a seed for
        CRand.rand_float(), we need to use iterators to avoid initializing
        duplicate values */
        weights_ih.iter_mut().for_each(|w| *w = random_weight!());
        weights_ho.iter_mut().for_each(|w| *w = random_weight!());
        biases_h.iter_mut().for_each(|b| *b = random_weight!());
        biases_o.iter_mut().for_each(|b| *b = random_weight!());

        NeuralNetwork {
            weights_ih,
            weights_ho,
            biases_h,
            biases_o,
            inputs: vec![0.0; NN_INPUT_SIZE],
            hidden: vec![0.0; NN_HIDDEN_SIZE],
            raw_logits: vec![0.0; NN_OUTPUT_SIZE],
            outputs: vec![0.0; NN_OUTPUT_SIZE],
        }
    }

    // Apply softmax activation function
    fn softmax(&mut self) {
        // Find maximum value for numerical stability
        let mut max_val: f32 = f32::NEG_INFINITY;
        for i in 0..NN_OUTPUT_SIZE {
            if self.raw_logits[i] > max_val {
                max_val = self.raw_logits[i];
            }
        }

        // Calculate exp(x_i - max) for each element and sum
        let mut sum: f32 = 0.0;
        for i in 0..NN_OUTPUT_SIZE {
            self.outputs[i] = (self.raw_logits[i] - max_val).exp();
            sum += self.outputs[i];
        }

        // Normalize to get probabilities
        if sum > 0.0 {
            for i in 0..NN_OUTPUT_SIZE {
                self.outputs[i] /= sum;
            }
        } else {
            // Fallback for numerical issues - uniform distribution
            for i in 0..NN_OUTPUT_SIZE {
                self.outputs[i] = 1.0 / NN_OUTPUT_SIZE as f32;
            }
        }
    }

    /* Neural network foward pass (inference). We store the activations
    so we can also do backpropagation later. */
    fn forward_pass(&mut self, inputs: &[f32]) {
        self.inputs.copy_from_slice(&inputs[0..NN_INPUT_SIZE]);

        // Input to hidden layer
        for i in 0..NN_HIDDEN_SIZE {
            let mut sum: f32 = self.biases_h[i];
            for j in 0..NN_INPUT_SIZE {
                sum += self.inputs[j] * self.weights_ih[i * NN_INPUT_SIZE + j];
            }
            self.hidden[i] = relu(sum);
        }

        // Hidden to output (raw logits)
        for i in 0..NN_OUTPUT_SIZE {
            self.raw_logits[i] = self.biases_o[i];
            for j in 0..NN_HIDDEN_SIZE {
                self.raw_logits[i] += self.hidden[j] * self.weights_ho[j * NN_OUTPUT_SIZE + i];
            }
        }

        self.softmax();
    }

    /* Backpropagation function.
    The only difference here from vanilla backprop is that we have
    'reward_scaling' argument that makes the output error more/less
    dramatic, so that we can adjust the weights proportionally to the
    reward we want to provide. */
    fn backprop(&mut self, target_probs: &[f32], learning_rate: f32, reward_scaling: f32) {
        let mut output_deltas: Vec<f32> = vec![0.0; NN_OUTPUT_SIZE];
        let mut hidden_deltas: Vec<f32> = vec![0.0; NN_HIDDEN_SIZE];

        /* === STEP 1: Compute deltas === */

        /* Calculate output layer deltas:
        Note what's going on here: we are technically using softmax
        as output function and cross entropy as loss, but we never use
        cross entropy in practice since we check the progresses in terms
        of winning the game.

        Still calculating the deltas in the output as:

            output[i] - target[i]

        Is exactly what happens if you derivate the deltas with
        softmax and cross entropy.

        LEARNING OPPORTUNITY: This is a well established and fundamental
        result in neural networks, you may want to read more about it. */
        for i in 0..NN_OUTPUT_SIZE {
            /* reward_scaling is used with an absolute value (reward_scaling.abs()),
            which ensures that scaling is always positive, preventing weights from
            being updated in the wrong direction. */
            output_deltas[i] = (self.outputs[i] - target_probs[i]) * reward_scaling.abs();
        }

        // Backpropagate error to hidden layer
        hidden_deltas
            .iter_mut()
            .enumerate()
            .for_each(|(i, hidden_delta)| {
                let error: f32 = output_deltas
                    .iter()
                    .enumerate()
                    .map(|(j, delta)| delta * self.weights_ho[i * NN_OUTPUT_SIZE + j])
                    .sum::<f32>();

                *hidden_delta = error * relu_derivative(self.hidden[i]);
            });

        /* === STEP 2: Weights updating === */

        // *** Original porting ***
        //
        // Output layer weights and biases
        // for i in 0..NN_HIDDEN_SIZE {
        //     for j in 0..NN_OUTPUT_SIZE {
        //         self.weights_ho[i * NN_OUTPUT_SIZE + j] -=
        //             learning_rate * output_deltas[j] * self.hidden[i];
        //     }
        // }

        // for j in 0..NN_OUTPUT_SIZE {
        //     self.biases_o[j] -= learning_rate * output_deltas[j];
        // }

        // // Hidden layer weights and biases
        // for j in 0..NN_HIDDEN_SIZE {
        //     for i in 0..NN_INPUT_SIZE {
        //         self.weights_ih[j * NN_INPUT_SIZE + i] -=
        //             learning_rate * hidden_deltas[j] * self.inputs[i];
        //     }
        // }

        // for j in 0..NN_HIDDEN_SIZE {
        //     self.biases_h[j] -= learning_rate * hidden_deltas[j];
        // }

        // Instead of C-style nested loops, we create a cartesian product of coordinates using flat_map
        (0..NN_HIDDEN_SIZE)
            .flat_map(|i| {
                // For each hidden neuron i, process all outputs j
                (0..NN_OUTPUT_SIZE).map(move |j| (i, j))
            })
            .for_each(|(i, j)| {
                // Update each weight using the same formula as before
                self.weights_ho[i * NN_OUTPUT_SIZE + j] -=
                    learning_rate * output_deltas[j] * self.hidden[i];
            });

        // Update output biases using iterator
        self.biases_o.iter_mut().enumerate().for_each(|(j, bias)| {
            *bias -= learning_rate * output_deltas[j];
        });

        // Hidden layer weights and biases - similar pattern
        (0..NN_HIDDEN_SIZE)
            .flat_map(|j| {
                // For each hidden neuron j, process all inputs i
                (0..NN_INPUT_SIZE).map(move |i| (j, i))
            })
            .for_each(|(j, i)| {
                self.weights_ih[j * NN_INPUT_SIZE + i] -=
                    learning_rate * hidden_deltas[j] * self.inputs[i];
            });

        // Update hidden biases using iterator
        self.biases_h.iter_mut().enumerate().for_each(|(j, bias)| {
            *bias -= learning_rate * hidden_deltas[j];
        });
    }
}

impl GameState {
    // Initialize game state with an empty board
    fn new() -> Self {
        GameState {
            board: ['.'; 9],
            current_player: 0,
        }
    }

    // Show board on screen in ASCII "art"
    fn display_board(&self) {
        for row in 0..3 {
            //Display the board symbols
            println!(
                "{} {} {}\n",
                self.board[row * 3],
                self.board[row * 3 + 1],
                self.board[row * 3 + 2]
            )
        }
        println!();
    }

    /* Convert board state to neural network inputs. Note that we use
     a peculiar encoding I descrived here:
     https://www.youtube.com/watch?v=EXbgUXt8fFU

     Instead of one-hot encoding, we can represent N different categories
     as different bit patterns. In this specific case it's trivial:

     00 = empty
     10 = X
     01 = O

    Two inputs per symbol instead of 3 in this case, but in the general case
    this reduces the input dimensionality A LOT.

    LEARNING OPPORTUNITY: You may want to learn (if not already aware) of
    different ways to represent non scalar inputs in neural networks:
    One hot encoding, learned embeddings, and even if it's just my random
    exeriment this "permutation coding" that I'm using here. */
    fn board_to_inputs(&self, inputs: &mut [f32]) {
        for i in 0..9 {
            if self.board[i] == '.' {
                inputs[i * 2] = 0.0;
                inputs[i * 2 + 1] = 0.0;
            } else if self.board[i] == 'X' {
                inputs[i * 2] = 1.0;
                inputs[i * 2 + 1] = 0.0;
            } else {
                // '0'
                inputs[i * 2] = 0.0;
                inputs[i * 2 + 1] = 1.0;
            }
        }
    }

    /* Check if the game is over (win or tie).
    Very brutal but fast enough. */
    fn check_game_over(&self, winner: &mut char) -> bool {
        // Check rows
        for i in 0..3 {
            if self.board[i * 3] != '.'
                && self.board[i * 3] == self.board[i * 3 + 1]
                && self.board[i * 3 + 1] == self.board[i * 3 + 2]
            {
                *winner = self.board[i * 3];
                return true;
            }
        }

        // Check columns
        for i in 0..3 {
            if self.board[i] != '.'
                && self.board[i] == self.board[i + 3]
                && self.board[i + 3] == self.board[i + 6]
            {
                *winner = self.board[i];
                return true;
            }
        }

        // Check diagonals
        if self.board[0] != '.' && self.board[0] == self.board[4] && self.board[4] == self.board[8]
        {
            *winner = self.board[0];
            return true;
        }

        if self.board[2] != '.' && self.board[2] == self.board[4] && self.board[4] == self.board[6]
        {
            *winner = self.board[2];
            return true;
        }

        // Check for tie (no free tiles left)
        let mut empty_tiles: usize = 0;
        for i in 0..9 {
            if self.board[i] == '.' {
                empty_tiles += 1;
            }
            if empty_tiles == 0 {
                *winner = 'T'; // tie
                return true;
            }
        }
        false
    }

    /* Get the best move for the computer using the neural network.
    Note that there is no complex sampling at all, we just get
    the output with the highest value THAT has an empty tile. */
    fn get_computer_move(&self, nn: &mut NeuralNetwork, display_probs: bool) -> i32 {
        let mut inputs = vec![0.0; NN_INPUT_SIZE];

        self.board_to_inputs(&mut inputs);
        nn.forward_pass(&inputs);

        // Find the highest probability valie and best legal move
        let mut highest_prob: f32 = -1.0;
        let mut highest_prob_idx: usize = 0;
        let mut best_move: i32 = -1;
        let mut best_legal_prob: f32 = -1.0;

        for i in 0..9 {
            // Track highest probability overall
            if nn.outputs[i] > highest_prob {
                highest_prob = nn.outputs[i];
                highest_prob_idx = i
            }

            // Track best legal move
            if self.board[i] == '.' && best_move == -1 || nn.outputs[i] > best_legal_prob {
                best_move = i as i32;
                best_legal_prob = nn.outputs[i];
            }
        }

        /* That's just for debugging. It's interesting to show to user
        in the first iterations of the game, since you can see how initially
        the net picks illegal moves as best, and so forth. */
        if display_probs {
            println!("Neural Network move probabilities");
            for row in 0..3 {
                for col in 0..3 {
                    let pos: usize = row * 3 + col;

                    // Print probability as percentage
                    print!("{:5.1}%", nn.outputs[pos] * 100.0);

                    // Add markers
                    if pos == highest_prob_idx {
                        print!("*"); // Highest probability overall
                    }

                    if pos as i32 == best_move {
                        print!("#") // Selected move (highest valid probability)
                    }
                    println!();
                }

                // Sum of probabilities should be 1.0
                let mut total_prob = 0.0;
                for i in 0..9 {
                    total_prob += nn.outputs[i];
                }
                println!("Sum of probabilities: {:.2}", total_prob);
            }
        }
        best_move
    }
}

fn main() {}

/* === TESTS === */

/* These tests are not very meaningful by themselves, but since we are porting from C to Rust on the fly,
it's handy to be able to test if each function works as intended. */
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_and_derivative() {
        assert_eq!(relu(-1.0), 0.0);
        assert_eq!(relu(0.0), 0.0);
        assert_eq!(relu(1.0), 1.0);

        assert_eq!(relu_derivative(-1.0), 0.0);
        assert_eq!(relu_derivative(0.0), 0.0);
        assert_eq!(relu_derivative(1.0), 1.0);
    }

    #[test]
    fn test_game_state() {
        // Initialize new game state
        let game_state = GameState::new();

        assert_eq!(game_state.current_player, 0);

        game_state.display_board();

        assert_eq!(
            game_state.board,
            ['.', '.', '.', '.', '.', '.', '.', '.', '.']
        );
    }

    #[test]
    fn test_neural_network_constructor() {
        let nn = NeuralNetwork::new();

        // Print the weights and biases
        println!("weights_ih: {:?}", nn.weights_ih);
        println!("weights_ho: {:?}", nn.weights_ho);
        println!("biases_h: {:?}", nn.biases_h);
        println!("biases_o: {:?}", nn.biases_o);
        println!("inputs: {:?}", nn.inputs);
        println!("hidden: {:?}", nn.hidden);
        println!("raw_logits: {:?}", nn.raw_logits);
        println!("outputs: {:?}", nn.outputs);
    }

    // test board_to_inputs
    #[test]
    fn test_board_to_inputs() {
        let mut nn = NeuralNetwork::new();
        let mut inputs = vec![0.0; NN_INPUT_SIZE];

        // crete game state
        let mut game_state = GameState::new();

        // Test case 1: Empty board
        game_state.board_to_inputs(&mut inputs);
        nn.inputs = inputs.clone();

        assert!(
            nn.inputs.iter().all(|&x| x == 0.0),
            "All inputs should be 0.0, got {:?}",
            nn.inputs
        );

        // Test case 2: non-empty board
        game_state.board[0] = 'X';
        game_state.board[1] = 'O';
        game_state.board_to_inputs(&mut inputs);
        nn.inputs = inputs.clone();
        assert_eq!(nn.inputs[0], 1.0);
        assert_eq!(nn.inputs[1], 0.0);
        assert_eq!(nn.inputs[2], 0.0);
        assert_eq!(nn.inputs[3], 1.0);
        assert_eq!(nn.inputs[4], 0.0);
        assert_eq!(nn.inputs[5], 0.0);
        assert_eq!(nn.inputs[6], 0.0);
        assert_eq!(nn.inputs[7], 0.0);
        assert_eq!(nn.inputs[8], 0.0);
        // TODO: study results
    }

    // test softmax
    #[test]
    fn test_softmax() {
        let mut nn = NeuralNetwork::new();

        // Test case 1: Standard input
        nn.raw_logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        // Apply softmax
        nn.softmax();

        // Print the outputs
        println!("outputs: {:?}", nn.outputs);

        /* The softmax function is designed to transform a vector of real numbers (logits) into a probability distribution.
        A valid probability distribution has two key properties:
        1. Non-negative values: Each output must be greater than or equal to 0.
        2. Sums to 1: The sum of all outputs must equal 1, ensuring the outputs represent probabilities that collectively cover all possible outcomes. */

        // Check sum to 1
        let sum: f32 = nn.outputs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Sum of softmax outputs should be 1, got {}",
            sum
        );

        // Check non-negative outputs
        assert!(
            nn.outputs.iter().all(|&x| x >= 0.0),
            "All softmax outputs must be non-negative"
        );

        // Test case 2: Equal logits (should produce uniform distribution)
        nn.raw_logits = vec![1.0; NN_OUTPUT_SIZE];

        // Apply softmax
        nn.softmax();

        println!("outputs: {:?}", nn.outputs);

        // Check sum to 1
        let sum: f32 = nn.outputs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Sum of softmax outputs should be 1, got {}",
            sum
        );

        // Check that each output of the softmax function is approximately equal to 1/9 (~0.1111)
        let uniform_prob = 1.0 / NN_OUTPUT_SIZE as f32;
        nn.outputs.iter().enumerate().for_each(|(i, &x)| {
            assert!(
                (x - uniform_prob).abs() < 1e-6,
                "Output[{}] = {}, expected {}",
                i,
                x,
                uniform_prob
            );
        });

        // Test case 3: Large negative logits (numerical stability)
        nn.raw_logits = vec![-1000.0; NN_OUTPUT_SIZE];

        // Apply softmax
        nn.softmax();

        println!("outputs: {:?}", nn.outputs);

        // Check sum to 1
        let sum: f32 = nn.outputs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Sum of softmax outputs should be 1, got {}",
            sum
        );

        // Check that each output of the softmax function is approximately equal to 1/9 (~0.1111)
        nn.outputs.iter().enumerate().for_each(|(i, &x)| {
            assert!(
                (x - uniform_prob).abs() < 1e-6,
                "Output[{}] = {}, expected {}",
                i,
                x,
                uniform_prob
            );
        });
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn test_forward_pass() {
        // Create a minimal test network with predetermined weights and biases
        let mut nn = NeuralNetwork::new();

        // Set weights and biases to simple values for easy manual verification
        // For example, set weights_ih to 0.1, weights_ho to 0.2, biases_h to 0.01, biases_o to 0.02
        for i in 0..NN_HIDDEN_SIZE {
            nn.biases_h[i] = 0.01;
            for j in 0..NN_INPUT_SIZE {
                nn.weights_ih[i * NN_INPUT_SIZE + j] = 0.1;
            }
        }

        for i in 0..NN_OUTPUT_SIZE {
            nn.biases_o[i] = 0.02;
            for j in 0..NN_HIDDEN_SIZE {
                nn.weights_ho[j * NN_OUTPUT_SIZE + i] = 0.2;
            }
        }

        // Prepare simple input data (e.g., all 1.0)
        let inputs = vec![1.0; NN_INPUT_SIZE];

        // Run forward pass
        nn.forward_pass(&inputs);

        // Calculate expected outputs manually
        // For hidden layer: bias + sum(input * weight) = 0.01 + (1.0 * 0.1 * NN_INPUT_SIZE) = 0.01 + 0.1*N
        let expected_hidden = relu(0.01 + 0.1 * NN_INPUT_SIZE as f32);

        // For output layer (before softmax): bias + sum(hidden * weight) = 0.02 + (expected_hidden * 0.2 * NN_HIDDEN_SIZE)
        let expected_raw_logit = 0.02 + expected_hidden * 0.2 * NN_HIDDEN_SIZE as f32;

        // Check hidden layer values
        for h in nn.hidden.iter() {
            assert!((h - expected_hidden).abs() < 1e-5);
        }

        // Manually calculate softmax output
        let mut expected_output = [0.0; NN_OUTPUT_SIZE];
        let mut sum_exp = 0.0;
        for i in 0..NN_OUTPUT_SIZE {
            expected_output[i] = expected_raw_logit.exp();
            sum_exp += expected_output[i];
        }
        for i in 0..NN_OUTPUT_SIZE {
            expected_output[i] /= sum_exp;
        }

        // Check output values
        for i in 0..NN_OUTPUT_SIZE {
            assert!((nn.outputs[i] - expected_output[i]).abs() < 1e-5);
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn test_backprop_deterministic() {
        /* Create a new neural network instance with random initial weights and biases.
        NeuralNetwork::new initializes weights_ih, weights_ho, biases_h, and biases_o
        with random values, but we’ll override most for determinism. */
        let mut nn = NeuralNetwork::new();

        /* Randomize weights_ih to ensure varied hidden layer activations.
        weights_ih connects NN_INPUT_SIZE (18) inputs to NN_HIDDEN_SIZE (100) hidden nodes.
        Random values prevent uniform hidden values (e.g., 0.22 from weights_ih = 0.1). */
        for i in 0..NN_INPUT_SIZE * NN_HIDDEN_SIZE {
            nn.weights_ih[i] = random_weight!();
        }

        /* Set weights_ho to deterministic values (0.2 + i * 0.001) for reproducibility.
        weights_ho connects NN_HIDDEN_SIZE (100) hidden nodes to NN_OUTPUT_SIZE (9) outputs.
        Variations (0.2, 0.201, ...) ensure non-zero hidden_deltas. This avoids situations
        where uniform weights_ho produced near-zero deltas (~1e-8). */
        for i in 0..NN_HIDDEN_SIZE * NN_OUTPUT_SIZE {
            nn.weights_ho[i] = 0.2 + (i as f32 * 0.001);
        }

        /* Set hidden layer biases to a fixed value (0.01) for consistency.
        biases_h (size NN_HIDDEN_SIZE) adds a constant to each hidden node’s sum
        before ReLU activation in forward_pass. */
        for i in 0..NN_HIDDEN_SIZE {
            nn.biases_h[i] = 0.01;
        }

        /* Set output layer biases to a fixed value (0.02) for consistency.
        biases_o (size NN_OUTPUT_SIZE) adds a constant to each output node’s raw logits
        before softmax in forward_pass. */
        for i in 0..NN_OUTPUT_SIZE {
            nn.biases_o[i] = 0.02;
        }

        /* Store the initial weights_ih to compare with updated weights after backprop.
        This allows us to check if backprop modifies weights_ih as expected. */
        let orig_weights_ih = nn.weights_ih.clone();

        /* Create an input vector of size NN_INPUT_SIZE (18) initialized to zeros.
        This represents a tic-tac-toe board state for the network to process. */
        let mut inputs = vec![0.0; NN_INPUT_SIZE];

        /* Set the first five inputs to non-zero values (0.5, 0.3, 0.7, 0.2, 0.4).
        This ensures the first five weights_ih updates are non-zero, as the gradient
        is hidden_deltas[j] * inputs[i]. Zero inputs produce zero updates. */
        let len = inputs.len().min(5);
        inputs[..len].copy_from_slice(&[0.5, 0.3, 0.7, 0.2, 0.4][..len]);

        // Print inputs for debugging, showing the board state being tested.
        println!("Input values: {:?}", inputs);

        /* Run the forward pass to compute hidden and output values.
        forward_pass:
        1. Copies inputs to nn.inputs.
        2. Computes hidden values: sum = biases_h[i] + sum(inputs[j] * weights_ih[i * NN_INPUT_SIZE + j]), then applies ReLU.
        3. Computes raw_logits: biases_o[i] + sum(hidden[j] * weights_ho[j * NN_OUTPUT_SIZE + i]).
        4. Applies softmax to get output probabilities. */
        nn.forward_pass(&inputs);

        /* Print hidden values to verify they vary due to random weights_ih.
        These are the ReLU-activated sums for each of the 100 hidden nodes. */
        println!("Hidden values after forward pass: {:?}", nn.hidden);

        /* Print output values to verify they vary due to weights_ho variations.
        These are the softmax probabilities for each of the 9 output nodes. */
        println!("Output values after forward pass: {:?}", nn.outputs);

        /* Create a target probability vector of size NN_OUTPUT_SIZE (9), all zeros.
        This represents the desired output (e.g., the correct move). */
        let mut target_probs = vec![0.0; NN_OUTPUT_SIZE];

        /* Set the first output to 1.0, simulating a target where the first move is correct.
        This creates a clear error for backprop to learn from. */
        target_probs[0] = 1.0;

        // Print target probabilities for debugging, showing the expected output.
        println!("Target probabilities: {:?}", target_probs);

        /* Define the learning rate (0.1) for weight updates in backprop.
        This scales the gradient: weight -= learning_rate * hidden_deltas[j] * inputs[i]. */
        let learning_rate = 0.1;

        /* Define reward scaling (1.0) for backprop.
        This scales the output error: output_deltas[i] = (outputs[i] - target_probs[i]) * reward_scaling.abs(). */
        let reward_scaling = 1.0;

        // Print learning rate and reward scaling for debugging.
        println!(
            "Running backprop with learning_rate={}, reward_scaling={}",
            learning_rate, reward_scaling
        );

        /* Run backpropagation to update weights and biases based on the error.
        backprop:
        1. Computes output_deltas: (outputs[i] - target_probs[i]) * reward_scaling.abs().
        2. Computes hidden_deltas: error = sum(output_deltas[j] * weights_ho[i * NN_OUTPUT_SIZE + j]), then hidden_deltas[i] = error * relu_derivative(hidden[i]).
        3. Updates weights_ho, biases_o, weights_ih, and biases_h using gradients. */
        nn.backprop(&target_probs, learning_rate, reward_scaling);

        /* Print the first five weight changes to verify backprop updates weights_ih.
        Changes should be non-zero and proportional to inputs[i] (0.5, 0.3, 0.7, 0.2, 0.4). */
        println!("Weight changes (first 5 input-hidden weights):");
        for i in 0..5.min(nn.weights_ih.len()) {
            let change = nn.weights_ih[i] - orig_weights_ih[i];
            println!(
                "  weights_ih[{}]: {} -> {} (change: {})",
                i, orig_weights_ih[i], nn.weights_ih[i], change
            );
        }

        /* Check if any weights_ih were updated significantly (> 1e-10).
        This ensures backprop works, as non-zero inputs and weights_ho variations
        produce detectable changes. */
        let mut any_weight_updated = false;
        for i in 0..nn.weights_ih.len() {
            if (nn.weights_ih[i] - orig_weights_ih[i]).abs() > 1e-10 {
                any_weight_updated = true;
                break;
            }
        }

        // Assert that at least one weight was updated, failing if backprop didn’t work.
        assert!(any_weight_updated, "Input-hidden weights should be updated");
    }
}
