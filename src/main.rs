/*
rs-ttt-rl
Rewrite in Rust of ttt-rl, originally written by antirez - https://github.com/antirez/ttt-rl
*/

use rs_ttt_rl::CRand;
use std::env;
use std::f32;
use std::io;
use std::io::Write;

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

    // Pre-allocated vectors for backprop to avoid allocations
    output_deltas: Vec<f32>,
    hidden_deltas: Vec<f32>,

    // Single RNG instance to avoid expensive SystemTime calls
    rng: CRand,
}

// ReLU activation function
#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

// Derivative of ReLU activation function
#[inline]
fn relu_derivative(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

/* Neural network with optimized performance */

impl Default for NeuralNetwork {
    fn default() -> Self {
        NeuralNetwork::new()
    }
}

impl NeuralNetwork {
    fn new() -> Self {
        let mut rng = CRand::new();

        let mut weights_ih: Vec<f32> = vec![0.0; NN_INPUT_SIZE * NN_HIDDEN_SIZE];
        let mut weights_ho: Vec<f32> = vec![0.0; NN_HIDDEN_SIZE * NN_OUTPUT_SIZE];
        let mut biases_h: Vec<f32> = vec![0.0; NN_HIDDEN_SIZE];
        let mut biases_o: Vec<f32> = vec![0.0; NN_OUTPUT_SIZE];

        // Initialize weights with random values between -0.5 and 0.5 using single RNG
        for w in weights_ih.iter_mut() {
            *w = rng.rand_float() - 0.5;
        }
        for w in weights_ho.iter_mut() {
            *w = rng.rand_float() - 0.5;
        }
        for b in biases_h.iter_mut() {
            *b = rng.rand_float() - 0.5;
        }
        for b in biases_o.iter_mut() {
            *b = rng.rand_float() - 0.5;
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
            output_deltas: vec![0.0; NN_OUTPUT_SIZE],
            hidden_deltas: vec![0.0; NN_HIDDEN_SIZE],
            rng,
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
        /* === STEP 1: Compute deltas === */

        // Calculate output layer deltas
        for i in 0..NN_OUTPUT_SIZE {
            self.output_deltas[i] = (self.outputs[i] - target_probs[i]) * reward_scaling.abs();
        }

        // Backpropagate error to hidden layer
        for i in 0..NN_HIDDEN_SIZE {
            let mut error: f32 = 0.0;
            for j in 0..NN_OUTPUT_SIZE {
                error += self.output_deltas[j] * self.weights_ho[i * NN_OUTPUT_SIZE + j];
            }
            self.hidden_deltas[i] = error * relu_derivative(self.hidden[i]);
        }

        /* === STEP 2: Weights updating === */

        // Output layer weights and biases
        for i in 0..NN_HIDDEN_SIZE {
            for j in 0..NN_OUTPUT_SIZE {
                self.weights_ho[i * NN_OUTPUT_SIZE + j] -=
                    learning_rate * self.output_deltas[j] * self.hidden[i];
            }
        }

        for j in 0..NN_OUTPUT_SIZE {
            self.biases_o[j] -= learning_rate * self.output_deltas[j];
        }

        // Hidden layer weights and biases
        for j in 0..NN_HIDDEN_SIZE {
            for i in 0..NN_INPUT_SIZE {
                self.weights_ih[j * NN_INPUT_SIZE + i] -=
                    learning_rate * self.hidden_deltas[j] * self.inputs[i];
            }
        }

        for j in 0..NN_HIDDEN_SIZE {
            self.biases_h[j] -= learning_rate * self.hidden_deltas[j];
        }
    }

    fn learn_from_game(
        &mut self,
        move_history: &[usize],
        num_moves: usize,
        nn_moves_even: bool,
        winner: char,
    ) {
        // Determine reward based on game outcome
        let reward: f32;
        let nn_symbol = if nn_moves_even { 'O' } else { 'X' };

        if winner == 'T' {
            reward = 0.3; // small reward for draw
        } else if winner == nn_symbol {
            reward = 1.0; // positive reward for win
        } else {
            reward = -2.0; // negative reward for loss
        }

        // Process each move the neural network made
        for move_idx in 0..num_moves {
            // Skip if this wasn't a move by the neural network
            if nn_moves_even && move_idx % 2 != 1 || !nn_moves_even && move_idx % 2 != 0 {
                continue;
            }

            // Recreate board state BEFORE this move was made
            let mut state = GameState::new();
            for i in 0..move_idx {
                let symbol = if i % 2 == 0 { 'X' } else { 'O' };
                state.board[move_history[i]] = symbol;
            }

            // Convert board to inputs
            state.board_to_inputs(&mut self.inputs);
            assert_eq!(self.inputs.len(), NN_INPUT_SIZE, "Inputs length mismatch");

            // Pass the array as a slice to forward_pass; no borrow of self.inputs remains
            let mut inputs_copy = [0.0; NN_INPUT_SIZE];
            inputs_copy.copy_from_slice(&self.inputs[..NN_INPUT_SIZE]);

            // Do forward pass
            self.forward_pass(&inputs_copy);

            /* The move that was actually made by the NN, that is
            the one we want to reward (positively or negatively). */
            let mv: usize = move_history[move_idx];

            /* Here we can't really implement temporal difference in the strict
            reinforcement learning sense, since we don't have an easy way to
            evaluate if the current situation is better or worse than the
            previous state in the game.

            However "time related" we do something that is very effective in
            this case: we scale the reward according to the move time, so that
            later moves are more impacted (the game is less open to different
            solutions as we go forward).

            We give a fixed 0.5 importance to all the moves plus
            a 0.5 that depends on the move position.

            NOTE: this makes A LOT of difference. Experiment with different
            values.

            LEARNING OPPORTUNITY: Temporal Difference in Reinforcement Learning
            is a very important result, that was worth the Turing Award in
            2024 to Sutton and Barto. You may want to read about it. */
            let move_importance: f32 = 0.5 + 0.5 * (move_idx as f32 / num_moves as f32);
            let scaled_reward: f32 = reward * move_importance;

            /* Create target probability distribution:
            let's start with the logits all set to 0. */
            let mut target_probs: [f32; 9] = [0.0; NN_OUTPUT_SIZE];

            // Set target for chosen move based on reward
            if scaled_reward >= 0.0 {
                /* For positive reward, set probability of the chosen move to
                1, with all the rest set to 0. */
                target_probs[mv] = 1.0;
            } else {
                /* For negative reward, distribute probability to OTHER
                valid moves, which is conceptually the same as discouraging
                the move that we want to discourage. */

                // Count actual empty squares on the board (excluding the move we made)
                let mut valid_moves_count = 0;
                for i in 0..9 {
                    if state.board[i] == '.' && i != mv {
                        valid_moves_count += 1;
                    }
                }

                if valid_moves_count > 0 {
                    let other_prob: f32 = 1.0 / valid_moves_count as f32;
                    for i in 0..9 {
                        if state.board[i] == '.' && i != mv {
                            target_probs[i] = other_prob;
                        }
                    }
                }
            }

            /* Call the generic backpropagation function, using
            our target logits as target. */
            self.backprop(&target_probs, LEARNING_RATE, scaled_reward);
        }
    }

    /* Get the best move for the computer using the neural network.
    Note that there is no complex sampling at all, we just get
    the output with the highest value THAT has an empty tile. */
    fn get_computer_move(&mut self, game_state: &mut GameState, display_probs: bool) -> i32 {
        let mut inputs = vec![0.0; NN_INPUT_SIZE];

        game_state.board_to_inputs(&mut inputs);
        self.forward_pass(&inputs);

        // Find the highest probability valie and best legal move
        let mut highest_prob: f32 = -1.0;
        let mut highest_prob_idx: usize = 0;
        let mut best_move: i32 = -1;
        let mut best_legal_prob: f32 = -1.0;

        for i in 0..9 {
            // Track highest probability overall
            if self.outputs[i] > highest_prob {
                highest_prob = self.outputs[i];
                highest_prob_idx = i
            }

            // Track best legal move
            if game_state.board[i] == '.' && (best_move == -1 || self.outputs[i] > best_legal_prob)
            {
                best_move = i as i32;
                best_legal_prob = self.outputs[i];
            }
        }

        /* That's just for debugging. It's interesting to show to user
        in the first iterations of the game, since you can see how initially
        the net picks illegal moves as best, and so forth. */
        if display_probs {
            println!("Neural network move probabilities:");
            for row in 0..3 {
                for col in 0..3 {
                    let pos: usize = row * 3 + col;

                    // Print probability as percentage
                    print!("{:5.1}%", self.outputs[pos] * 100.0);

                    // Add markers
                    if pos == highest_prob_idx {
                        print!("*"); // Highest probability overall
                    }

                    if pos as i32 == best_move {
                        print!("#"); // Selected move (highest valid probability)
                    }
                    print!(" ");
                }
                println!();
            }

            // Sum of probabilities should be 1.0
            let mut total_prob = 0.0;
            for i in 0..9 {
                total_prob += self.outputs[i];
            }
            println!("Sum of all probabilities: {:.2}\n", total_prob);
        }
        best_move
    }

    /* Get a random valid move, this is used for training
    against a random opponent. Note: this function will loop forever
    if the board is full, but here we want simple code. Use CRand*/
    fn get_random_move(&mut self, state: &GameState) -> i32 {
        loop {
            let mv: u32 = self.rng.rand_int(9);
            if state.board[mv as usize] == '.' {
                return mv as i32;
            }
        }
    }

    fn play_game(&mut self) {
        let mut state: GameState = GameState::new();
        let mut winner: char = 'T';
        let mut move_history: Vec<i32> = vec![0; 9];
        let mut num_moves: i32 = 0;

        println!("Welcome to Tic Tac Toe! You are X, the computer is O.");
        println!("Enter positions as numbers from 0 to 8 (see picture).");

        while !state.check_game_over(&mut winner) {
            state.display_board();

            if state.current_player == 0 {
                // Human turn
                let mut input = String::new();
                print!("Your move (0-8): ");
                io::stdout().flush().unwrap();
                io::stdin().read_line(&mut input).unwrap();

                let mvc = input.trim();
                match mvc.parse::<usize>() {
                    Ok(mv) => {
                        // Check if move is valid
                        if mv > 8 || state.board[mv] != '.' {
                            println!("Invalid move! Try again.");
                            continue;
                        }

                        state.board[mv] = 'X';
                        move_history[num_moves as usize] = mv as i32;
                        num_moves += 1;
                    }
                    Err(_) => {
                        println!("Invalid move! Try again.");
                        continue;
                    }
                }
            } else {
                // Computer's turn
                println!("Computer's move:");
                let mv: usize = self.get_computer_move(&mut state, true) as usize;
                state.board[mv] = 'O';

                println!("Computer placed O at position {}", mv);

                move_history[num_moves as usize] = mv as i32;
                num_moves += 1;
            }

            // Safety check: A tic-tac-toe game should never exceed 9 moves
            if num_moves >= 9 {
                eprintln!("ERROR: Game reached maximum moves (9), forcing game over check");
                state.check_game_over(&mut winner);
                break;
            }

            // Switch player
            state.current_player = 1 - state.current_player;
        }

        state.display_board();

        match winner {
            'X' => println!("You win!"),
            'O' => println!("Computer wins!"),
            _ => println!("It's a tie!"),
        }

        // Collect history to [usize]
        let mhistory: Vec<usize> = move_history[..num_moves as usize]
            .iter()
            .map(|&x| x as usize)
            .collect();

        // Learn from this game - neural network plays as O (second player) so moves are at odd indices
        self.learn_from_game(&mhistory, num_moves as usize, false, winner);
    }

    /* Train the neural network against random moves.*/
    fn train_against_random(&mut self, num_games: usize) {
        let mut move_history = Vec::with_capacity(9);
        let mut num_moves: usize;
        let mut wins = 0;
        let mut losses = 0;
        let mut ties = 0;

        println!(
            "Training neural network against {} random games...",
            num_games
        );

        let mut played_games = 0;
        for i in 0..num_games {
            // Prepare for a new game
            move_history.clear();
            move_history.resize(9, 0);
            num_moves = 0;

            // Play a random game
            let mut state = GameState::new();
            let mut winner: char = '?'; // Initialize to invalid value instead of 'T'

            // Play until game over
            while !state.check_game_over(&mut winner) {
                let mv: i32;
                if state.current_player == 0 {
                    // Random move for player X
                    mv = self.get_random_move(&state);
                } else {
                    // Neural network move for player O
                    mv = self.get_computer_move(&mut state, false);
                }

                // Safety check: A tic-tac-toe game should never exceed 9 moves
                if num_moves >= 9 {
                    state.check_game_over(&mut winner);
                    break;
                }

                // Make the move
                let symbol = if state.current_player == 0 { 'X' } else { 'O' };
                state.board[mv as usize] = symbol;
                move_history[num_moves] = mv;
                num_moves += 1;

                // Switch player
                state.current_player = 1 - state.current_player;
            }

            played_games += 1;

            // Update statistics based on winner
            match winner {
                'O' => wins += 1,   // Neural network won
                'X' => losses += 1, // Random player won
                _ => ties += 1,     // Tie
            }

            // Convert move history to usize vector for learning
            let mhistory: Vec<usize> = move_history[..num_moves]
                .iter()
                .map(|&x| x as usize)
                .collect();

            // Learn from this game - neural network moves are odd-indexed (player O)
            self.learn_from_game(&mhistory, num_moves, false, winner);

            // Show progress periodically
            if (i + 1) % 10000 == 0 {
                println!(
                    "Games: {}, Wins: {} ({:.1}%), Losses: {} ({:.1}%), Ties: {} ({:.1}%)",
                    i + 1,
                    wins,
                    (wins as f32 * 100.0) / played_games as f32,
                    losses,
                    (losses as f32 * 100.0) / played_games as f32,
                    ties,
                    (ties as f32 * 100.0) / played_games as f32
                );

                // Reset counters for next batch
                played_games = 0;
                wins = 0;
                losses = 0;
                ties = 0;
            }
        }

        println!("\nTraining complete!");
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
            // Display the board symbols
            print!(
                "{}{}{} ",
                self.board[row * 3],
                self.board[row * 3 + 1],
                self.board[row * 3 + 2]
            );

            // Display the position numbers for this row, for the poor human
            println!("{}{}{}", row * 3, row * 3 + 1, row * 3 + 2);
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
        }
        if empty_tiles == 0 {
            *winner = 'T'; // tie
            return true;
        }

        false
    }
}

fn main() {
    // Default number of training games
    let mut random_games = 150000; // Fast and enough to play in a decent way

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        random_games = args[1].parse::<i32>().unwrap_or(15000000) as usize;
    }

    // Initialize neural network
    let mut nn = NeuralNetwork::new();

    // Train against random moves
    if random_games > 0 {
        nn.train_against_random(random_games);
    }

    // Play game with human and learn more
    loop {
        nn.play_game();

        print!("Play again? (y/n): ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();

        let play_again = input.trim().to_lowercase();
        if play_again != "y" {
            break;
        }
    }
}

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
        with random values, but we'll override most for determinism. */
        let mut nn = NeuralNetwork::new();

        /* Randomize weights_ih to ensure varied hidden layer activations.
        weights_ih connects NN_INPUT_SIZE (18) inputs to NN_HIDDEN_SIZE (100) hidden nodes.
        Random values prevent uniform hidden values (e.g., 0.22 from weights_ih = 0.1). */
        for i in 0..NN_INPUT_SIZE * NN_HIDDEN_SIZE {
            nn.weights_ih[i] = nn.rng.rand_float() - 0.5;
        }

        /* Set weights_ho to deterministic values (0.2 + i * 0.001) for reproducibility.
        weights_ho connects NN_HIDDEN_SIZE (100) hidden nodes to NN_OUTPUT_SIZE (9) outputs.
        Variations (0.2, 0.201, ...) ensure non-zero hidden_deltas. This avoids situations
        where uniform weights_ho produced near-zero deltas (~1e-8). */
        for i in 0..NN_HIDDEN_SIZE * NN_OUTPUT_SIZE {
            nn.weights_ho[i] = 0.2 + (i as f32 * 0.001);
        }

        /* Set hidden layer biases to a fixed value (0.01) for consistency.
        biases_h (size NN_HIDDEN_SIZE) adds a constant to each hidden node's sum
        before ReLU activation in forward_pass. */
        for i in 0..NN_HIDDEN_SIZE {
            nn.biases_h[i] = 0.01;
        }

        /* Set output layer biases to a fixed value (0.02) for consistency.
        biases_o (size NN_OUTPUT_SIZE) adds a constant to each output node's raw logits
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

        // Assert that at least one weight was updated, failing if backprop didn't work.
        assert!(any_weight_updated, "Input-hidden weights should be updated");
    }
}
