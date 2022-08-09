use rand::RngCore;

use crate::neuron::Neuron;

#[derive(Debug,Clone)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
}

pub struct LayerTopology {
    pub neurons: usize,
}

impl Layer {
    pub fn random(rng: &mut dyn RngCore, input_neurons: usize, output_neurons: usize) -> Self {
        let neurons = (0..output_neurons)
            .map(|_| Neuron::random(rng, input_neurons))
            .collect();

        Self { neurons }
    }

    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.neurons
            .iter()
            .map(|neuron| neuron.propagate(&inputs))
            .collect()
    }

    pub fn from_weights(
        input_size: usize,
        output_size: usize,
        weights: &mut dyn Iterator<Item = f32>,
    ) -> Self {
        let neurons = (0..output_size)
            .map(|_| Neuron::from_weights(input_size, weights))
            .collect();

        Self { neurons }
    }
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::Layer;

    mod random {
        use super::*;

        #[test]
        fn test() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let layer = Layer::random(&mut rng, 4, 3);

            assert_eq!(layer.neurons.len(), 3)
        }
    }

    mod propagate {
        use super::*;

        #[test]
        fn test() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let layers = Layer::random(&mut rng, 3, 2);
            let inputs = vec![0.0, 0.0, 0.0];
            let outputs = layers.propagate(inputs);
            assert_relative_eq!(outputs.as_slice(), [0.0, 0.5238807].as_ref());
        }
    }
}
