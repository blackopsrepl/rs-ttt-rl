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
        let mut rng = CRand::new();
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
        let mut weights_ih = vec![0.0; NN_INPUT_SIZE * NN_HIDDEN_SIZE];
        let mut weights_ho = vec![0.0; NN_HIDDEN_SIZE * NN_OUTPUT_SIZE];
        let mut biases_h = vec![0.0; NN_HIDDEN_SIZE];
        let mut biases_o = vec![0.0; NN_OUTPUT_SIZE];

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
        let mut max_val = f32::NEG_INFINITY;
        for i in 0..NN_OUTPUT_SIZE {
            if self.raw_logits[i] > max_val {
                max_val = self.raw_logits[i];
            }
        }
        
        // Calculate exp(x_i - max) for each element and sum
        let mut sum = 0.0;
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
}

fn main() {}

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
        // TODO: simulate game state change

        let game_state = GameState {
            board: ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
            current_player: 0,
        };

        assert_eq!(
            game_state.board,
            ['.', '.', '.', '.', '.', '.', '.', '.', '.']
        );
        assert_eq!(game_state.current_player, 0);
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
        assert!((sum - 1.0).abs() < 1e-6, "Sum of softmax outputs should be 1, got {}", sum);

        // Check non-negative outputs
        assert!(nn.outputs.iter().all(|&x| x >= 0.0), "All softmax outputs must be non-negative");


        // Test case 2: Equal logits (should produce uniform distribution)
        nn.raw_logits = vec![1.0; NN_OUTPUT_SIZE];

        // Apply softmax
        nn.softmax();

        println!("outputs: {:?}", nn.outputs);

        // Check sum to 1
        let sum: f32 = nn.outputs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Sum of softmax outputs should be 1, got {}", sum);
        
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
        assert!((sum - 1.0).abs() < 1e-6, "Sum of softmax outputs should be 1, got {}", sum);
        
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
}