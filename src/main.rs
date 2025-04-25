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
        for i in 0..NN_HIDDEN_SIZE {
            let mut error: f32 = 0.0;
            for j in 0..NN_OUTPUT_SIZE {
                error += output_deltas[j] * self.weights_ho[i * NN_OUTPUT_SIZE + j];
            }
            hidden_deltas[i] = error * relu_derivative(self.hidden[i]);
        }

        /* === STEP 2: Weights updating === */

        // Output layer weights and biases
        for i in 0..NN_HIDDEN_SIZE {
            for j in 0..NN_OUTPUT_SIZE {
                self.weights_ho[i * NN_OUTPUT_SIZE + j] -=
                    learning_rate * output_deltas[j] * self.hidden[i];
            }
        }

        for j in 0..NN_OUTPUT_SIZE {
            self.biases_o[j] -= learning_rate * output_deltas[j];
        }

        // Hidden layer weights and biases
        for j in 0..NN_HIDDEN_SIZE {
            for i in 0..NN_INPUT_SIZE {
                self.weights_ih[j * NN_INPUT_SIZE + i] -=
                    learning_rate * hidden_deltas[j] * self.inputs[i];
            }
        }

        for j in 0..NN_HIDDEN_SIZE {
            self.biases_h[j] -= learning_rate * hidden_deltas[j];
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
        let mut expected_output = vec![0.0; NN_OUTPUT_SIZE];
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
    fn test_backprop() {
        let mut nn = NeuralNetwork::new();
        for i in 0..NN_INPUT_SIZE * NN_HIDDEN_SIZE {
            nn.weights_ih[i] = random_weight!();
        }
        for i in 0..NN_HIDDEN_SIZE * NN_OUTPUT_SIZE {
            nn.weights_ho[i] = 0.2 + (i as f32 * 0.001);
        }
        for i in 0..NN_HIDDEN_SIZE {
            nn.biases_h[i] = 0.01;
        }
        for i in 0..NN_OUTPUT_SIZE {
            nn.biases_o[i] = 0.02;
        }
        let orig_weights_ih = nn.weights_ih.clone();

        let mut inputs = vec![0.0; NN_INPUT_SIZE];
        for i in 0..inputs.len().min(5) {
            inputs[i] = [0.5, 0.3, 0.7, 0.2, 0.4][i]; // Non-zero for first 5
        }
        println!("Input values: {:?}", inputs);
        nn.forward_pass(&inputs);
        println!("Hidden values after forward pass: {:?}", nn.hidden);
        println!("Output values after forward pass: {:?}", nn.outputs);

        let mut target_probs = vec![0.0; NN_OUTPUT_SIZE];
        target_probs[0] = 1.0;
        println!("Target probabilities: {:?}", target_probs);

        let learning_rate = 0.1;
        let reward_scaling = 1.0;
        println!(
            "Running backprop with learning_rate={}, reward_scaling={}",
            learning_rate, reward_scaling
        );
        nn.backprop(&target_probs, learning_rate, reward_scaling);

        println!("Weight changes (first 5 input-hidden weights):");
        for i in 0..5.min(nn.weights_ih.len()) {
            let change = nn.weights_ih[i] - orig_weights_ih[i];
            println!(
                "  weights_ih[{}]: {} -> {} (change: {})",
                i, orig_weights_ih[i], nn.weights_ih[i], change
            );
        }

        let mut any_weight_updated = false;
        for i in 0..nn.weights_ih.len() {
            if (nn.weights_ih[i] - orig_weights_ih[i]).abs() > 1e-10 {
                any_weight_updated = true;
                break;
            }
        }
        assert!(any_weight_updated, "Input-hidden weights should be updated");
    }
}
