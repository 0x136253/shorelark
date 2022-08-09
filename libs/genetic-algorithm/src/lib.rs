use crossover::CrossoverMethod;
use mutation::MutationMethod;
use rand::RngCore;
use selection::SelectionMethod;
use statistics::Statistics;
use individual::Individual;

pub mod individual;
pub mod selection;
pub mod chromosome;
pub mod crossover;
pub mod mutation;
pub mod statistics;

pub struct GeneticAlgorithm<S,C,M> {
    selection_method: S,
    crossover_method: C,
    mutation_method: M,
}

impl<S,C,M> GeneticAlgorithm<S,C,M> 
where
    S: SelectionMethod,
    C: CrossoverMethod,
    M: MutationMethod,
{
    pub fn new(
        selection_method: S,
        crossover_method: C,
        mutation_method: M,
    ) -> Self {
        Self{
            selection_method,
            crossover_method,
            mutation_method
        }
    }

    pub fn evolve<I>(
        &self,
        rng: &mut dyn RngCore,
        population: &[I],
    ) -> (Vec<I>, Statistics)
    where
        I: Individual,
    {
        assert!(!population.is_empty());

        let new_population = (0..population.len())
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
                
                self.mutation_method.mutate(rng, &mut child);
                I::create(child)
            })
            .collect();

        let stats = Statistics::new(population);

        (new_population, stats)
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use crate::{individual::{TestIndividual, Individual}, GeneticAlgorithm, selection::RouletteWheelSelection, crossover::UniformCrossover, mutation::GaussianMutation};


    fn individual(genes: &[f32]) -> TestIndividual {
        let chromosome = genes.iter().cloned().collect();

        TestIndividual::create(chromosome)
    }

    #[test]
    fn test() {
        let mut rng = ChaCha8Rng::from_seed(Default::default());

        let ga = GeneticAlgorithm::new(
            RouletteWheelSelection::default(),
            UniformCrossover::new(),
            GaussianMutation::new(0.5,0.5)
        );

        let mut population = vec![
            individual(&[0.0, 0.0, 0.0]), // fitness = 0.0
            individual(&[1.0, 1.0, 1.0]), // fitness = 3.0
            individual(&[1.0, 2.0, 1.0]), // fitness = 4.0
            individual(&[1.0, 2.0, 4.0]), // fitness = 7.0
        ];

        for _ in 0..10 {
            population = ga.evolve(&mut rng, &population).0;
        }

        let expected_population = vec![
            individual(&[0.44769490, 2.0648358, 4.3058133]),
            individual(&[1.21268670, 1.5538777, 2.8869110]),
            individual(&[1.06176780, 2.2657390, 4.4287640]),
            individual(&[0.95909685, 2.4618788, 4.0247330]),
        ];

        assert_eq!(population, expected_population);
    }
}
