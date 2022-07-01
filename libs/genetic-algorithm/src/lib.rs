use crossover::CrossoverMethod;
use rand::RngCore;
use selection::SelectionMethod;

use crate::individual::Individual;

pub mod individual;
pub mod selection;
pub mod chromosome;
pub mod crossover;

pub struct GeneticAlgorithm<S,C> {
    selection_method: S,
    crossover_method: C,
}

impl<S,C> GeneticAlgorithm<S,C> 
where
    S: SelectionMethod,
    C: CrossoverMethod,
{
    pub fn new(
        selection_method: S,
        crossover_method: C,
    ) -> Self {
        Self{
            selection_method,
            crossover_method
        }
    }

    pub fn evolve<I>(
        &self,
        rng: &mut dyn RngCore,
        population: &[I],
    ) -> Vec<I>
    where
        I: Individual,
    {
        assert!(!population.is_empty());

        (0..population.len())
            .map(|_| {
                let parent_a = self
                    .selection_method
                    .select(rng, population)
                    .chromosome();

                let parent_b = self
                    .selection_method
                    .select(rng, population)
                    .chromosome();
                
                let mut child = self
                    .crossover_method
                    .crossover(rng, parent_a, parent_b);
                
                // TODO mutation
                // TODO convert `Chromosome` back into `Individual`
                todo!()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
