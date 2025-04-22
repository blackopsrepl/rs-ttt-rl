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

fn main() {}