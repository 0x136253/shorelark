use rand::RngCore;

use crate::layer::{Layer, LayerTopology};

mod layer;
mod neuron;

pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn random(rng: &mut dyn RngCore, layers: &[LayerTopology]) -> Self {
        assert!(layers.len() > 1);

        let layers = layers
            .windows(2)
            .map(|layers| Layer::random(rng, layers[0].neurons, layers[1].neurons))
            .collect();

        Self { layers }
    }

    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers
            .iter()
            .fold(inputs, |inputs, layer| layer.propagate(inputs))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use rand::SeedableRng;

    use super::LayerTopology;
    use super::Network;

    fn random_network() -> Network {
        let mut rng = rand_chacha::ChaCha8Rng::from_seed(Default::default());
        let network = Network::random(
            &mut rng,
            &[
                LayerTopology { neurons: 4 },
                LayerTopology { neurons: 3 },
                LayerTopology { neurons: 4 },
            ],
        );

        network
    }

    mod random {
        use super::*;

        #[test]
        fn test() {
            let network = random_network();
            assert_eq!(network.layers.len(), 2);
        }
    }

    mod propagate {
        use super::*;

        #[test]
        fn test() {
            let network = random_network();

            let outputs = network.propagate(vec![0.0, 0.0, 0.0, 0.0]);
            assert_relative_eq!(outputs.as_slice(), [0.0, 0.35662687, 0.0, 0.0].as_ref());
        }
    }
}
