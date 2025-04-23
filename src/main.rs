/* 
rs-ttt-rl
Rewrite in Rust of ttt-rl, originally written by antirez - https://github.com/antirez/ttt-rl
*/

use std::f32;
use rs_ttt_rl::CRand;

// Neural network parameters
const NN_INPUT_SIZE: usize = 18;
const NN_HIDDEN_SIZE: usize = 100;
const NN_OUTPUT_SIZE: usize = 9;
const LEARNING_RATE: f32 = 0.1;

// Game board representation
struct GameState {
    board: [char; 9],        // Can be '.' (empty) or 'X', 'O'
    current_player: usize,   // 0 for player (X), 1 for computer (O)
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
    raw_logits: Vec<f32>,    // Outputs before softmax()
    outputs: Vec<f32>,       // Outputs after softmax()
}

// ReLU activation function
fn relu(x: f32) -> f32 {
    return x.max(0.0);
}

// Derivative of ReLU acctivation function
fn relu_derivative(x: f32) -> f32 {
    return if x > 0.0 { 1.0 } else { 0.0 };
}

/* Initialize a neural network with random weights.
Since we are committing to using the standard libary only,
we will implement CRand as a simple linear congruential generator
compatible with standard C rnd(), from a new random_weight!() macro. See lib.rs. */
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

        // Initialize weights with random values between -0.5 and 0.5
        for i in 0..NN_INPUT_SIZE * NN_HIDDEN_SIZE {
            weights_ih[i] = random_weight!();
        }
        for i in 0..NN_HIDDEN_SIZE * NN_OUTPUT_SIZE {
            weights_ho[i] = random_weight!();
        }
        for i in 0..NN_HIDDEN_SIZE {
            biases_h[i] = random_weight!();
        }
        for i in 0..NN_OUTPUT_SIZE {
            biases_o[i] = random_weight!();
        }

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

        assert_eq!(game_state.board, ['.', '.', '.', '.', '.', '.', '.', '.', '.']);
        assert_eq!(game_state.current_player, 0);
    }

    #[test]
    fn test_neural_network() {
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
}
