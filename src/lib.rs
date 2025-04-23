/*
Simple pseudo-random generator, aims to be similar to the standard C rand() function

The `rand()` method implements a linbear congruential generator (LCG) calculation and aims to be compatible
with the standard C `rand()` function, using a linear congruential generator (LCG) calculation. 
It updates the state by multiplying it with a constant multiplier, adding an increment 
and discarding the lower 16 bits to produce a random number between 0 and 32767.

The `rand_float()` method returns a float between 0 and 1 by dividing the result of `rand()` by 32767.
*/
use std::time::{SystemTime, UNIX_EPOCH};

pub struct CRand {
    state: u32,
}

impl CRand {
    const MULTIPLIER: u32 = 1103515245;
    const INCREMENT: u32 = 12345;
    const RAND_MAX: u32 = 0x7fff;

    // Initializes with current time as seed
    pub fn new() -> Self {
        let start = SystemTime::now();
        let since_epoch = start
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        CRand {
            /* This uses bitwise XOR to combine seconds and nanoseconds for a higher precision seeds.
            This ensures unique seeds - even for multiple calls within one second, which would else get identical seeds */
            state: since_epoch.as_secs() as u32 ^ (since_epoch.subsec_nanos() as u32),
        }
    }

    // Implements the same algorithm as standard C rand()
    pub fn rand(&mut self) -> u32 {
        // runs LCG calculation
        self.state = self
            .state
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(Self::INCREMENT);
        // discards lower 16 bits
        (self.state >> 16) & Self::RAND_MAX
    }

    // Returns a float between 0 and 1
    pub fn rand_float(&mut self) -> f32 {
        self.rand() as f32 / Self::RAND_MAX as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_weight() {
        for _ in 0..20 {
            let mut rng = CRand::new();
            let weight = rng.rand_float() - 0.5;
            assert!(
                weight >= -0.5 && weight < 0.5,
                "Weight should be in [-0.5, 0.5)"
            );
            print!("{}\n", weight)
        }
    }
}
