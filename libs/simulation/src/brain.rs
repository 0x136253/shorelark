use lib_neural_network as nn;
use rand::RngCore;
use crate::Eye;
use lib_genetic_algorithm::chromosome::Chromosome;

#[derive(Debug)]
pub struct Brain {
    pub(crate) nn: nn::Network,
}

impl Brain {
    pub fn random(rng: &mut dyn RngCore, eye: &Eye) -> Self {
        Self {
            nn: nn::Network::random(rng, &Self::topology(eye)),
        }
    }

    pub(crate) fn as_chromosome(&self) -> Chromosome {
        self.nn.weights().collect()
    }

    pub(crate) fn from_chromosome(
        chromosome: Chromosome,
        eye: &Eye,
    ) -> Self {
        Self {
            nn: nn::Network::from_weights(
                &Self::topology(eye),
                chromosome,
            ),
        }
    }
    
    fn topology(eye: &Eye) -> [nn::LayerTopology; 3] {
        [
            nn::LayerTopology {
                neurons: eye.cells(),
            },
            nn::LayerTopology {
                neurons: 2 * eye.cells(),
            },
            nn::LayerTopology { neurons: 2 },
        ]
    }
}