use rand::RngCore;

use crate::chromosome::Chromosome;

pub trait MutationMethod {
    fn mutate(&self, rng:&mut dyn RngCore,child: &mut Chromosome);
}

#[derive(Clone,Debug)]
pub struct GaussianMutation {
     /// Probability of changing a gene:
    /// - 0.0 = no genes will be touched
    /// - 1.0 = all genes will be touched
    chance: f32,

    /// Magnitude of that change:
    /// - 0.0 = touched genes will not be modified
    /// - 3.0 = touched genes will be += or -= by at most 3.0
    coeff: f32,
}

impl GaussianMutation {
    pub fn new(chance: f32, coeff:f32) -> Self {
        assert!(chance >= 0.0 && chance <=1.0);

        Self { chance, coeff}
    }
}

impl MutationMethod for GaussianMutation {
    fn mutate(&self, rng:&mut dyn RngCore,child: &mut Chromosome) {
        todo!()
    }
}